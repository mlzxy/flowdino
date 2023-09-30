# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import numpy as np
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from dino_training.utils import trunc_normal_
from dino_training.flow_utils import distilled_cross_entropy, make_feature_sim_grid
from dataset.optical_flow import make_optical_flow_grid
from einops import rearrange
import torch.distributed as dist


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False, both=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention and not both:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attention:
            return x, attn
        else:
            return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, return_feats=False, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.return_feats = return_feats
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        if self.use_transhead:
            self.optical_flow_head = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=1.0, qkv_bias=False, qk_scale=None,
                drop=0, attn_drop=0, drop_path=0, norm_layer=norm_layer)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

        self.apply(self._init_weights)

    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x):
        feat_h = x.shape[2] // self.patch_size
        feat_w = x.shape[3] // self.patch_size
        bs = len(x)
        x = self.prepare_tokens(x)

        blocks = self.blocks
        norm = self.norm

        for blk_id, blk in enumerate(blocks):
            if blk_id == len(blocks) - 1:
                x, attn = blk(x, return_attention=True, both=True)
                if self.return_feats:
                    attn = attn[:, :, 0, 1:]
                    saliency = attn.sum(dim=1)
                    saliency = saliency.view(bs, feat_h, feat_w, 1)
                else:
                    saliency = None
            else:
                x = blk(x)

        x = norm(x)
        cls_feat = x[:, 0]

        if self.return_feats:
            spatial_feat = x[:, 1:, :]
            spatial_feat = spatial_feat.reshape(x.shape[0], feat_h, feat_w, -1)
            spatial_feat = torch.cat([spatial_feat, saliency], dim=-1)
            return cls_feat, spatial_feat
        else:
            return cls_feat


    def get_intermediate_layers(self, x, n=1):
        assert n == 1
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output
    
    
    def forward_backbone(self, x, last_self_attention=False):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                x = blk(x, return_attention=last_self_attention, both=last_self_attention)
        if last_self_attention:
            x, attn = x
        x = self.norm(x)
        if last_self_attention:
            return x, attn[:, :, 0, 1:]
        return x

    
    
def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_large(patch_size=16, **kwargs):
    model = VisionTransformer(patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4,  qkv_bias=True,  norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model



class DINOHead(nn.Module): 
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)

        if norm_last_layer:
            self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False)) 
            self.last_layer.weight_g.data.fill_(1)
            self.last_layer.weight_g.requires_grad = False
        else:
             self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)      
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class FlowDINOHead(nn.Module): 
    def __init__(self, dim, norm_last_layer=True, nlayers=2, input_has_sal=True, head_size=-1):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.input_has_sal = input_has_sal
        self.dim = dim
        if nlayers == 1:
            self.mlp = nn.Linear(dim, dim)
        else:
            layers = [nn.Linear(dim, dim)]
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(dim, dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(dim, dim))
            self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights)

        if head_size > 0:
            self.last_layer = nn.utils.weight_norm(nn.Linear(dim, head_size, bias=False))        
        else:
            self.last_layer = nn.utils.weight_norm(nn.Linear(dim, dim, bias=False)) 
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.input_has_sal:
            assert x.shape[-1] == (self.dim + 1)
            if len(x.shape) == 4:
                sal = x[:, :, :, -1:]
                x = x[:, :, :, :-1]
            else:
                sal = x[:, :, -1:]
                x = x[:, :, :-1]

        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        if self.input_has_sal:
            return torch.cat([x, sal], dim=-1)
        else:
            return x


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.center_buffer = []
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch, masks=None, update_center=True):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]

        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1) 
                if masks is None:
                    total_loss += loss.mean()
                else:
                    mask = masks[iq] * masks[v]
                    total_loss += (loss * mask).sum() / (mask.sum() + 1e-4)
                n_loss_terms += 1
        total_loss /= n_loss_terms
        if update_center:
            self.update_center(teacher_output)

        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        注意，提供的 checkpoint 里时没有这个 center 的!
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)




def entropy(x, dim=-1):
    return  - torch.sum(F.softmax(x, dim=dim) * F.log_softmax(x, dim=dim), dim=dim)



class FlowLoss(nn.Module):
    
    def __init__(self, flow_temp=0.1, radius=0.7, kernel_size=5, stride=2, loss_weight_mode='norm',
                loss_weight_margin=0.01, static_threshold=0.1):
        super().__init__()
        self.flow_temp = flow_temp
        self.radius = radius
        self.kernel_size = kernel_size
        self.stride = stride
        self.loss_weight_margin = loss_weight_margin
        self.static_threshold = static_threshold
        self.loss_weight_mode = loss_weight_mode

        self.maximum_entropy = entropy(torch.zeros(1, kernel_size ** 2), dim=-1).item()
    
    
    def forward(self, feat, flow, max_norm, ref_feat=None, mask=None):
        H, W = feat.shape[1], feat.shape[2]
        saliency = feat[:, :, :, -1]
        feat = feat[:, :, :, :-1]
        saliency = saliency.detach()
        saliency = saliency[:, None, :, :]
        sal_blocks = F.unfold(saliency, kernel_size=self.kernel_size, stride=self.stride)
        sal_blocks = rearrange(sal_blocks, 'N w L -> (N L) w')
        saliency_mask = sal_blocks.argmax(dim=1)

        flow_grid = make_optical_flow_grid(flow, target_size=(H, W), stride=self.stride, radius=self.radius,
                            normalize=False, kernel_size=self.kernel_size, saliency_mask=saliency_mask,
                            maximum_norm=max_norm, static_threshold=self.static_threshold, device=feat.device)
        h, w = flow_grid.shape[1:3]
        # flow_grid = flow_grid.cuda()
        feat = feat.permute(0, 3, 1, 2) # -> N, C, H, W
        flow_grid = flow_grid.flatten(3) # N, h, w, kh*kw
        feat_grid = make_feature_sim_grid(feat, kernel_size=self.kernel_size, stride=self.stride, saliency_mask=saliency_mask,
                                        reference_features=ref_feat.permute(0, 3, 1, 2) if ref_feat is not None else None)
        # (N, h*w)
        if self.loss_weight_mode == 'norm':
            flow_w = flow_grid.mean(dim=-1).flatten(1)
            flow_w = F.relu(flow_w - self.loss_weight_margin, inplace=True)
            eps = 1e-6
            flow_w.div_(flow_w.sum(dim=1, keepdim=True) + eps)
        elif self.loss_weight_mode == 'entropy':
            assert self.loss_weight_mode == 'entropy'
            flow_w = flow_grid.view(-1, h * w, self.kernel_size ** 2)
            flow_w = F.relu(self.maximum_entropy - entropy(flow_w, dim=-1)) # N, (h, w)
            eps = 1e-6
            flow_w.div_(flow_w.sum(dim=1, keepdim=True) + eps)
        else:
            flow_w = None
        
        if mask is not None:
            mask_blocks = F.unfold(mask, kernel_size=self.kernel_size, stride=self.stride)
            mask_blocks = rearrange(mask_blocks, 'N k (h w) -> N h w k', h=h, w=w)
        else:
            mask_blocks = None

        # (N, h*w)
        flat_corr_loss = distilled_cross_entropy(flow_grid, feat_grid.flatten(3), tl=self.flow_temp, tp=self.flow_temp, mask=mask_blocks).flatten(1) 
        if flow_w is None:
            flat_corr_loss = flat_corr_loss.mean(dim=1)
        else:
            flat_corr_loss = (flat_corr_loss * flow_w).sum(dim=1)

        return torch.nan_to_num(flat_corr_loss).mean() 
