import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init

from mmdet.core import auto_fp16
from ..builder import NECKS


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self,
                 num_pos_feats=64,
                 temperature=10000,
                 normalize=False,
                 scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mask):
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(
            self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

def build_position_encoding(hidden_dim, shape):
    mask = torch.zeros(shape, dtype=torch.bool)
    pos_module = PositionEmbeddingSine(hidden_dim // 2)
    pos_embs = pos_module(mask)
    return pos_embs


class AttentionLayer(nn.Module):
    """ Position attention module with mixed precision (FP16 coarse -> FP32 refinement)"""

    def __init__(self, in_dim, out_dim, ratio=4, stride=1, top_k_ratio=0.2):
        super(AttentionLayer, self).__init__()
        self.chanel_in = in_dim
        self.top_k_ratio = top_k_ratio  # Fraction of positions to refine in FP32
        norm_cfg = dict(type='BN', requires_grad=True)
        act_cfg = dict(type='ReLU')
        self.pre_conv = ConvModule(
            in_dim,
            out_dim,
            kernel_size=3,
            stride=stride,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            inplace=False)
        self.query_conv = nn.Conv2d(
            in_channels=out_dim, out_channels=out_dim // ratio, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=out_dim, out_channels=out_dim // ratio, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=out_dim, out_channels=out_dim, kernel_size=1)
        self.final_conv = ConvModule(
            out_dim,
            out_dim,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x, pos=None):
        """
            Mixed-precision attention:
            - FP16 coarse attention for all positions
            - FP32 refinement only for Top-K important positions
        """
        
        target_dtype = self.pre_conv.conv.weight.dtype
        
        if x.dtype != target_dtype:
            x = x.to(target_dtype)
            
        if pos is not None and pos.dtype != target_dtype:
            pos = pos.to(target_dtype)

        x = self.pre_conv(x)
        B, C, H, W = x.shape
        N = H * W

        if pos is not None:
            x = x + pos

        proj_query_fp32 = self.query_conv(x).view(B, -1, N).permute(0, 2, 1)  # [B, HW, C//ratio]
        proj_key_fp32   = self.key_conv(x).view(B, -1, N)                      # [B, C//ratio, HW]
        proj_value_fp32 = self.value_conv(x).view(B, -1, N)                    # [B, C, HW]

        proj_query_fp16 = proj_query_fp32.to(torch.float16)
        proj_key_fp16   = proj_key_fp32.to(torch.float16)
        proj_value_fp16 = proj_value_fp32.to(torch.float16)

        energy_coarse = torch.bmm(proj_query_fp16, proj_key_fp16).float()  # [B, HW, HW]
        attention_coarse = self.softmax(energy_coarse).to(torch.float16)    # [B, HW, HW]

        attention_importance = attention_coarse.sum(dim=1)                  # [B, HW]
        top_k_count = max(1, int(N * self.top_k_ratio))
        _, top_k_indices = torch.topk(attention_importance, k=top_k_count, dim=-1)
        
        top_k_mask = torch.zeros(B, N, device=x.device, dtype=torch.bool)
        top_k_mask.scatter_(1, top_k_indices, True)
        top_k_mask = top_k_mask.view(B, 1, H, W)


        attention_coarse_t = attention_coarse.permute(0, 2, 1)             # [B, HW, HW]
        out_fp16 = torch.bmm(proj_value_fp16, attention_coarse_t).view(B, C, H, W)



        if top_k_count > 0:
            top_k_flat = top_k_indices.view(B, -1)

            proj_query_topk = torch.gather(
                proj_query_fp32, dim=1,
                index=top_k_flat.unsqueeze(-1).expand(-1, -1, proj_query_fp32.shape[-1])
            ) 

            proj_key_topk = torch.gather(
                proj_key_fp32, dim=2,
                index=top_k_flat.unsqueeze(1).expand(-1, proj_key_fp32.shape[1], -1)
            ) 

            energy_refined_topk = torch.bmm(proj_query_topk, proj_key_topk)     # [B, K, K]
            attention_refined_topk = self.softmax(energy_refined_topk)          # [B, K, K]

            proj_value_topk = torch.gather(
                proj_value_fp32, dim=2,
                index=top_k_flat.unsqueeze(1).expand(-1, C, -1)
            )  # [B, C, K]

            out_fp32_topk = torch.bmm(proj_value_topk, attention_refined_topk.permute(0, 2, 1))  # [B, C, K]

            out_fp32 = out_fp16.to(torch.float32).clone()
            out_fp32_flat = out_fp32.view(B, C, -1)
            out_fp32_flat.scatter_(2, top_k_flat.unsqueeze(1).expand(-1, C, -1), out_fp32_topk)
            out_fp32 = out_fp32_flat.view(B, C, H, W)
        else:
            out_fp32 = out_fp16.to(torch.float32)

        out_fp16 = self.gamma * out_fp16 + x.to(torch.float16)
        out_fp32 = self.gamma * out_fp32 + x

        out_fp16 = self.final_conv(out_fp16.to(x.dtype)).to(torch.float16)
        out_fp32 = self.final_conv(out_fp32)

        return out_fp16, out_fp32, top_k_mask

class TransConvEncoderModule(nn.Module):
    def __init__(self, in_dim, attn_in_dims, attn_out_dims, strides, ratios, downscale=True, pos_shape=None):
        super(TransConvEncoderModule, self).__init__()
        if downscale:
            stride = 2
        else:
            stride = 1
        # self.first_conv = ConvModule(in_dim, 2*in_dim, kernel_size=3, stride=stride, padding=1)
        # self.final_conv = ConvModule(attn_out_dims[-1], attn_out_dims[-1], kernel_size=3, stride=1, padding=1)
        attn_layers = []
        for dim1, dim2, stride, ratio in zip(attn_in_dims, attn_out_dims, strides, ratios):
            attn_layers.append(AttentionLayer(dim1, dim2, ratio, stride))
        if pos_shape is not None:
            self.attn_layers = nn.ModuleList(attn_layers)
        else:
            self.attn_layers = nn.Sequential(*attn_layers)
        self.pos_shape = pos_shape
        self.pos_embeds = []
        if pos_shape is not None:
            for dim in attn_out_dims:
                pos_embed = build_position_encoding(dim, pos_shape).cuda()
                self.pos_embeds.append(pos_embed)
    
    def forward(self, src):
        # src = self.first_conv(src)
        src_fp16 = src
        src_fp32 = src
        top_k_mask = None
        
        if self.pos_shape is None:
            src = self.attn_layers(src)
            return src, None, None  
        else:
            for layer, pos in zip(self.attn_layers, self.pos_embeds):
                src_fp16, src_fp32, top_k_mask = layer(src_fp16, pos.to(src.device))
        # src = self.final_conv(src)
        return src_fp16, src_fp32, top_k_mask

@NECKS.register_module()
class TransConvFPN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 trans_idx=-1,
                 trans_cfg=None,
                 attention=True):
        super(TransConvFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.trans_cfg = trans_cfg
        self.trans_idx = trans_idx
        self.attention = attention
        if self.attention:
            self.trans_head = TransConvEncoderModule(**trans_cfg)

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
                self.fpn_convs.append(extra_fpn_conv)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    
    @auto_fp16()
    def forward(self, src):
        assert len(src) >= len(self.in_channels)
        src = list(src)

        expected_channels = self.trans_cfg['in_dim']
        trans_input = None
        trans_src_idx = None  

        if self.trans_idx is not None:
            try:
                idx = self.trans_idx
                if idx < 0:
                    idx += len(src)
                
                candidate = src[idx]
                if hasattr(candidate, 'shape') and len(candidate.shape) >= 2:
                    if candidate.shape[1] == expected_channels:
                        trans_input = candidate
                        trans_src_idx = idx
            except (IndexError, TypeError):
                pass

        if trans_input is None:
            for i in range(len(src) - 1, -1, -1):
                feat = src[i]
                if hasattr(feat, 'shape') and len(feat.shape) >= 2:
                    if feat.shape[1] == expected_channels:
                        trans_input = feat
                        trans_src_idx = i
                        break
        
        if trans_input is None:
            idx = self.trans_idx if self.trans_idx is not None else -1
            if idx < 0:
                idx += len(src)
            trans_input = src[idx]
            trans_src_idx = idx


        if self.attention:
            trans_feat_fp16, trans_feat_fp32, top_k_mask = self.trans_head(trans_input)

            if trans_feat_fp32 is None or top_k_mask is None:
                trans_feat = trans_feat_fp16
                return tuple(outs), trans_feat # Note: This line assumes outs exists, ensure fallback logic is complete if this case is possible.


            inputs_fp16 = list(src)
            
            if trans_src_idx is not None and trans_src_idx < len(inputs_fp16):
                inputs_fp16[trans_src_idx] = trans_feat_fp16
            else:
                inputs_fp16[-1] = trans_feat_fp16

            if len(inputs_fp16) > len(self.in_channels):
                for _ in range(len(inputs_fp16) - len(self.in_channels)):
                    del inputs_fp16[0]

            laterals_fp16 = []
            for i, lateral_conv in enumerate(self.lateral_convs):
                feat = inputs_fp16[i + self.start_level]
                
                target_dtype = lateral_conv.conv.weight.dtype
                if feat.dtype != target_dtype:
                    feat = feat.to(dtype=target_dtype)
                
                res = lateral_conv(feat)
                laterals_fp16.append(res.to(torch.float16))

            used_backbone_levels = len(laterals_fp16)
            
            for i in range(used_backbone_levels - 1, 0, -1):
                prev_shape = laterals_fp16[i - 1].shape[2:]
                laterals_fp16[i - 1] = laterals_fp16[i - 1] + F.interpolate(
                    laterals_fp16[i], size=prev_shape, mode='nearest'
                )

            outs_fp16 = []
            for i in range(used_backbone_levels):
                feat = laterals_fp16[i]
                target_dtype = self.fpn_convs[i].conv.weight.dtype
                if feat.dtype != target_dtype:
                    feat = feat.to(target_dtype)
                res = self.fpn_convs[i](feat)
                outs_fp16.append(res.to(torch.float16))

            if self.num_outs > len(outs_fp16):
                if not self.add_extra_convs:
                    for _ in range(self.num_outs - used_backbone_levels):
                        outs_fp16.append(F.max_pool2d(outs_fp16[-1], 1, stride=2))
                else:
                    if self.extra_convs_on_inputs:
                        orig_idx = self.backbone_end_level - 1 - (len(src) - len(inputs_fp16))
                        orig = inputs_fp16[orig_idx]
                    else:
                        orig = outs_fp16[-1]
                        
                    extra_conv_idx = used_backbone_levels
                    target_dtype = self.fpn_convs[extra_conv_idx].conv.weight.dtype
                    if orig.dtype != target_dtype:
                        orig = orig.to(target_dtype)
                        
                    res = self.fpn_convs[extra_conv_idx](orig)
                    outs_fp16.append(res.to(torch.float16))

                    for i in range(used_backbone_levels + 1, self.num_outs):
                        inp = outs_fp16[-1]
                        if self.relu_before_extra_convs:
                            inp = F.relu(inp)
                        target_dtype = self.fpn_convs[i].conv.weight.dtype
                        if inp.dtype != target_dtype:
                            inp = inp.to(target_dtype)
                        res = self.fpn_convs[i](inp)
                        outs_fp16.append(res.to(torch.float16))



            lateral_idx = trans_src_idx - self.start_level


            if 0 <= lateral_idx < used_backbone_levels:
                
                transformed_lateral_conv = self.lateral_convs[lateral_idx]
                target_dtype = transformed_lateral_conv.conv.weight.dtype
                
                if trans_feat_fp32.dtype != target_dtype:
                    trans_feat_fp32 = trans_feat_fp32.to(target_dtype)
                    
                lateral_trans_fp32 = transformed_lateral_conv(trans_feat_fp32)

                if lateral_idx < used_backbone_levels - 1:
                    feat_above = laterals_fp16[lateral_idx + 1]
                    top_down_signal = F.interpolate(
                        feat_above, 
                        size=lateral_trans_fp32.shape[2:], 
                        mode='nearest'
                    ).to(target_dtype)
                    
                    lateral_trans_fp32 = lateral_trans_fp32 + top_down_signal

                fpn_conv_for_level = self.fpn_convs[lateral_idx]
                out_trans_level_fp32 = fpn_conv_for_level(lateral_trans_fp32)

                mask_resized_level = F.interpolate(
                    top_k_mask.float(), 
                    size=out_trans_level_fp32.shape[2:], 
                    mode='nearest'
                ).to(dtype=torch.bool)

                base_fp16 = outs_fp16[lateral_idx]
                
                out_replaced = torch.where(
                    mask_resized_level, 
                    out_trans_level_fp32.to(dtype=base_fp16.dtype), 
                    base_fp16
                )

                outs_fp16[lateral_idx] = out_replaced

            outs_fp32 = [out.to(torch.float32) for out in outs_fp16]

            return tuple(outs_fp32), trans_feat_fp32
        
        else:
            pass