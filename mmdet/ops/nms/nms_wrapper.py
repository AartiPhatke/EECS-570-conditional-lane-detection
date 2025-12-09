import numpy as np
import torch

from . import nms_ext


def nms_mixed_precision(dets, iou_thr, device_id=None, top_k_ratio=0.2, use_mixed_precision=True):

    if not use_mixed_precision or dets.shape[0] < 10:
   
        return nms(dets, iou_thr, device_id)
    

    if isinstance(dets, torch.Tensor):
        is_numpy = False
        dets_th = dets
    elif isinstance(dets, np.ndarray):
        is_numpy = True
        device = 'cpu' if device_id is None else f'cuda:{device_id}'
        dets_th = torch.from_numpy(dets).to(device)
    else:
        raise TypeError('dets must be either a Tensor or numpy array, '
                        f'but got {type(dets)}')
    
    if dets_th.shape[0] == 0:
        inds = dets_th.new_zeros(0, dtype=torch.long)
        return dets[inds, :], inds
    

    dets_fp16 = dets_th.to(torch.float16)
    
    inds_coarse = nms_ext.nms(dets_fp16, iou_thr)
    
    keep_count = max(1, int(len(inds_coarse) * top_k_ratio))
    scores_coarse = dets_th[inds_coarse, -1]  # scores from original FP32
    
    _, top_k_idx = torch.topk(scores_coarse, k=min(keep_count, len(scores_coarse)))
    inds_topk = inds_coarse[top_k_idx]
    
    dets_topk = dets_th[inds_topk, :]
    
    if dets_topk.shape[0] > 0:
        inds_refined = nms_ext.nms(dets_topk, iou_thr)
        inds_final = inds_topk[inds_refined]
    else:
        inds_final = inds_coarse
    
    refined_set = set(inds_final.cpu().numpy().tolist())
    coarse_set = set(inds_coarse.cpu().numpy().tolist())
    

    other_inds = [idx for idx in coarse_set if idx not in 
                  set(inds_topk.cpu().numpy().tolist())]
    
    inds_final_merged = torch.cat([
        inds_final,
        dets_th.new_tensor(list(other_inds), dtype=torch.long)
    ]) if other_inds else inds_final
    
    inds_final_merged, _ = torch.sort(inds_final_merged)
    
    if is_numpy:
        inds_final_merged = inds_final_merged.cpu().numpy()
    
    return dets[inds_final_merged, :], inds_final_merged


def nms(dets, iou_thr, device_id=None, use_mixed_precision=False):
    
    if use_mixed_precision:
        return nms_mixed_precision(dets, iou_thr, device_id)
    

    if isinstance(dets, torch.Tensor):
        is_numpy = False
        dets_th = dets
    elif isinstance(dets, np.ndarray):
        is_numpy = True
        device = 'cpu' if device_id is None else f'cuda:{device_id}'
        dets_th = torch.from_numpy(dets).to(device)
    else:
        raise TypeError('dets must be either a Tensor or numpy array, '
                        f'but got {type(dets)}')

    if dets_th.shape[0] == 0:
        inds = dets_th.new_zeros(0, dtype=torch.long)
    else:
        if dets_th.is_cuda:
            inds = nms_ext.nms(dets_th, iou_thr)
        else:
            inds = nms_ext.nms(dets_th, iou_thr)

    if is_numpy:
        inds = inds.cpu().numpy()
    return dets[inds, :], inds


def soft_nms(dets, iou_thr, method='linear', sigma=0.5, min_score=1e-3, use_mixed_precision=False):
    
    if isinstance(dets, torch.Tensor):
        is_tensor = True
        dets_t = dets.detach().cpu()
    elif isinstance(dets, np.ndarray):
        is_tensor = False
        dets_t = torch.from_numpy(dets)
    else:
        raise TypeError('dets must be either a Tensor or numpy array, '
                        f'but got {type(dets)}')

    method_codes = {'linear': 1, 'gaussian': 2}
    if method not in method_codes:
        raise ValueError(f'Invalid method for SoftNMS: {method}')
    

    if use_mixed_precision and dets_t.shape[0] > 10:
        dets_t = dets_t.to(torch.float16)
    
    results = nms_ext.soft_nms(dets_t, iou_thr, method_codes[method], sigma,
                               min_score)

    new_dets = results[:, :5]
    inds = results[:, 5]

    if is_tensor:
        return new_dets.to(
            device=dets.device, dtype=dets.dtype), inds.to(
                device=dets.device, dtype=torch.long)
    else:
        return new_dets.numpy().astype(dets.dtype), inds.numpy().astype(
            np.int64)


def batched_nms(bboxes, scores, inds, nms_cfg):
    
    max_coordinate = bboxes.max()
    offsets = inds.to(bboxes) * (max_coordinate + 1)
    bboxes_for_nms = bboxes + offsets[:, None]
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = eval(nms_type)
    dets, keep = nms_op(
        torch.cat([bboxes_for_nms, scores[:, None]], -1), **nms_cfg_)
    bboxes = bboxes[keep]
    scores = dets[:, -1]
    return torch.cat([bboxes, scores[:, None]], -1), keep
