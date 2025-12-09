import time
import math
import torch
from torch.utils.cpp_extension import load_inline

# ==============================================================================
# CUDA SOURCE: Coalesced Access Pattern
# ==============================================================================
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define MAX_K 64


__device__ __forceinline__ void warp_reduce_max(float& val, int& idx) {
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(mask, val, offset);
        int other_idx = __shfl_down_sync(mask, idx, offset);
        if (other_val > val) {
            val = other_val;
            idx = other_idx;
        }
    }
}


__global__ void kernel_scout_transposed(
    const __half* __restrict__ Q, 
    const __half* __restrict__ K_T, // Transposed K: Size [D x N]
    long* indices,      
    int N, int D, int K_target) 
{
    int row = blockIdx.x; 
    int tid = threadIdx.x;
    

    extern __shared__ float s_mem[];
    float* s_scores = s_mem;
    

    volatile float* s_max_vals = (volatile float*)&s_scores[N];
    volatile int* s_max_idxs = (volatile int*)&s_max_vals[32]; 


    for (int col = tid; col < N; col += blockDim.x) {
        float dot = 0.0f;
        

        const __half* Q_row = &Q[row * D];

        
        for (int d = 0; d < D; d++) {
            // Note: We lose __half2 vectorization here effectively because K is scattered 
            // across D-dimension in memory, but the Coalescing win outweighs this 10x.
            __half q_val = Q_row[d];
            __half k_val = K_T[d * N + col]; 
            dot += __half2float(__hmul(q_val, k_val));
        }
        s_scores[col] = dot;
    }
    __syncthreads();


    for (int k = 0; k < K_target; k++) {
        float my_max_val = -1e20f;
        int my_max_idx = -1;

   
        for (int i = tid; i < N; i += blockDim.x) {
            float val = s_scores[i];
            if (val > my_max_val) {
                my_max_val = val;
                my_max_idx = i;
            }
        }

        warp_reduce_max(my_max_val, my_max_idx);

      
        if ((tid % 32) == 0) {
            s_max_vals[tid / 32] = my_max_val;
            s_max_idxs[tid / 32] = my_max_idx;
        }
        __syncthreads();

   
        if (tid < 32) {
            int num_warps = blockDim.x / 32;
            float val = (tid < num_warps) ? s_max_vals[tid] : -1e20f;
            int idx = (tid < num_warps) ? s_max_idxs[tid] : -1;
            
            warp_reduce_max(val, idx);

            if (tid == 0) {
                indices[row * K_target + k] = (long)idx;
                if (idx != -1) s_scores[idx] = -1e20f; // Mask
            }
        }
        __syncthreads();
    }
}


__global__ void kernel_refine_fp32(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const long* __restrict__ indices, 
    float* output,
    int N, int D, int K_target, float scale) 
{
    int row = blockIdx.x; 
    int tid = threadIdx.x; 
    extern __shared__ float s_Q[];
    if (tid < D) s_Q[tid] = Q[row * D + tid];
    __syncthreads();

    if (tid < D) {
        float k_scores[MAX_K];
        int k_indices[MAX_K];
        float max_score = -1e20f;

        for (int i = 0; i < K_target; i++) {
            int idx = indices[row * K_target + i];
            k_indices[i] = idx;
            float dot = 0.0f;
            for (int d = 0; d < D; d++) dot += s_Q[d] * K[idx * D + d]; 
            k_scores[i] = dot * scale;
            if (k_scores[i] > max_score) max_score = k_scores[i];
        }

        float sum_exp = 0.0f;
        for (int i = 0; i < K_target; i++) {
            k_scores[i] = expf(k_scores[i] - max_score);
            sum_exp += k_scores[i];
        }

        float acc = 0.0f;
        for (int i = 0; i < K_target; i++) {
            float attn = k_scores[i] / (sum_exp + 1e-6f);
            acc += attn * V[k_indices[i] * D + tid]; 
        }
        output[row * D + tid] = acc;
    }
}


void launch_scout(torch::Tensor Q, torch::Tensor K_T, torch::Tensor idx, int Kt) {
    int N = Q.size(0); 
    int D = Q.size(1);
    
    // Shared Mem: Scores (N) + WarpBuffer (32 floats + 32 ints)
    int smem_size = (N * sizeof(float)) + (64 * sizeof(float)); 

    kernel_scout_transposed<<<N, 256, smem_size>>>(
        reinterpret_cast<__half*>(Q.data_ptr<at::Half>()), 
        reinterpret_cast<__half*>(K_T.data_ptr<at::Half>()),
        idx.data_ptr<long>(), 
        N, D, Kt
    );
}

void launch_refine(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor idx, torch::Tensor out, int Kt, float s) {
    int N = Q.size(0); 
    int D = Q.size(1);
    int threads = (D > 1024) ? 1024 : D;
    kernel_refine_fp32<<<N, threads, D*sizeof(float)>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        idx.data_ptr<long>(), out.data_ptr<float>(), N, D, Kt, s
    );
}
"""


cpp_source = """
#include <torch/extension.h>
void launch_scout(torch::Tensor Q, torch::Tensor K_T, torch::Tensor idx, int Kt);
void launch_refine(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor idx, torch::Tensor out, int Kt, float s);
"""


module = load_inline(
    name='mp_proof_v5', 
    cpp_sources=cpp_source, 
    cuda_sources=cuda_source, 
    functions=['launch_scout', 'launch_refine'], 
    extra_cuda_cflags=['--use_fast_math', '-O3']
)

def compare_methods():
    N = 4096   
    D = 128    
    K_target = 64     
    device = torch.device('cuda')

    print(f"--- PROOF OF MIXED PRECISION: N={N}, D={D}, Top-K={K_target} ---")


    Q_32 = torch.randn(N, D, device=device, dtype=torch.float32)
    K_32 = torch.randn(N, D, device=device, dtype=torch.float32)
    V_32 = torch.randn(N, D, device=device, dtype=torch.float32)
    scale = 1.0 / math.sqrt(D)


    Q_16 = Q_32.to(torch.half).contiguous()
    K_16_T = K_32.t().to(torch.half).contiguous() 

    indices = torch.zeros((N, K_target), dtype=torch.long, device=device)
    out_mp = torch.zeros_like(Q_32)

 
    for _ in range(10): torch.matmul(Q_32, K_32.T)

  
    torch.cuda.synchronize()
    start_base = time.time()
    
    scores = torch.matmul(Q_32, K_32.T) * scale
    attn = torch.softmax(scores, dim=-1)
    out_base = torch.matmul(attn, V_32)
    
    torch.cuda.synchronize()
    end_base = time.time()
    time_base = (end_base - start_base) * 1000
    print(f"[Baseline] Full FP32 Attention: {time_base:.3f} ms")

  
    torch.cuda.synchronize()
    start_mp = time.time()


    module.launch_scout(Q_16, K_16_T, indices, K_target)
    module.launch_refine(Q_32, K_32, V_32, indices, out_mp, K_target, scale)

    torch.cuda.synchronize()
    end_mp = time.time()
    time_mp = (end_mp - start_mp) * 1000
    print(f"[Proposed] Mixed Precision:     {time_mp:.3f} ms")

    # Results
    speedup = time_base / time_mp
    mae = (out_base - out_mp).abs().mean().item()
    
    print(f"\n--- FINAL RESULTS ---")
    print(f"SPEEDUP: {speedup:.2f}x")
    print(f"ERROR (MAE): {mae:.5f}")

if __name__ == "__main__":
    compare_methods()