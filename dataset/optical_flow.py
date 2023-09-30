import numpy as np
from PIL import Image
import torch
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large
from torchvision.utils import flow_to_image, save_image
import torch.nn.functional as F
from einops import rearrange


@torch.no_grad()
def make_optical_flow_grid(flow, kernel_size=9, stride=None,
                           static_threshold=0.1, target_size=None, radius=1.0, return_norm_map=False,
                           normalize=True, maximum_norm=None, return_norm_blocks=False, eps=1e-6,
                           saliency_mask=None, device=None):
    # flow: [N, 2, H, W]
    stride = stride or (kernel_size // 3)

    if target_size is not None:
        flow = F.interpolate(flow, target_size, mode='bicubic')
    
    if device is not None:
        flow = flow.to(device)

    H, W = flow.shape[2:]
    if normalize:
        rescale_flow = flow - flow.flatten(2).mean(dim=2)[..., None, None]  # N, 2, 1, 1
    else:
        rescale_flow = flow
    rescale_flow_norm = rescale_flow.norm(dim=1, keepdim=True)
    rescale_flow_norm /= (eps + (rescale_flow_norm.flatten(1).max(dim=1, keepdim=True).values[..., None, None] if maximum_norm is None else maximum_norm.view(-1)[:, None, None, None])).to(rescale_flow_norm.device)
    flow_norm = F.relu(rescale_flow_norm - static_threshold)

    if return_norm_map:
        return flow_norm 

    flow_blocks = F.unfold(flow, kernel_size=kernel_size, stride=stride)
    flow_norm_blocks = F.unfold(flow_norm, kernel_size=kernel_size, stride=stride)
    
    flow_blocks = rearrange(flow_blocks, 'N (c w) L -> (N L) w c', c=2)
    flow_norm_blocks = rearrange(flow_norm_blocks, 'N w L -> (N L) w')
    center_ind = flow_blocks.shape[1] // 2

    if saliency_mask is None:
        flow_centers = flow_blocks[:, center_ind, :][:, None, :]
    else:
        if len(saliency_mask.shape) > 1:
            flow_centers = (flow_blocks * saliency_mask).sum(dim=1, keepdim=True)
        else:
            flow_centers = flow_blocks[torch.arange(0, len(flow_blocks), device=flow_blocks.device), saliency_mask, :][:, None, :]

    weight = torch.exp(- torch.abs(1 - F.cosine_similarity(flow_centers, flow_blocks, dim=2).clamp_(0.0, 1.0)) / radius)
    grid = weight * flow_norm_blocks
    h=int((H-1-(kernel_size-1)) / stride + 1)
    w=int((W-1-(kernel_size-1)) / stride + 1)
    grid = rearrange(grid, '(N h w) (k1 k2) -> N h w k1 k2', k1=kernel_size, k2=kernel_size, h=h, w=w)
                
    if return_norm_blocks:
        return grid, rearrange(flow_norm_blocks, '(N h w) (k1 k2) -> N h w k1 k2', k1=kernel_size, k2=kernel_size, h=h, w=w) 
    else:
        return grid


def unpack_flow_to_float(I):
    # I: [H, W], return: [2, H, W]
    I = I.view(dtype='uint32')
    SHIFT = 2 ** 16
    x = I // SHIFT
    y = I % SHIFT
    x = unpack_channel_to_float(x)
    y = unpack_channel_to_float(y)
    return np.concatenate((x[None, ...], y[None, ...]), axis=0)

def unpack_channel_to_float(a):
    SHIFT = 2 ** 5
    a = a.astype(dtype='uint16').view(dtype='int16')
    a = a.astype('float32') / SHIFT
    return a

def pack_channel_as_uint16(a):
    SHIFT = 2 ** 5
    a = (a * SHIFT).astype('int16')
    return a.view(dtype='uint16')

def pack_flow_to_int(flow):
    # flow: [2, H, W], numpy array
    x = pack_channel_as_uint16(flow[0, :, :]).astype('uint32')
    y = pack_channel_as_uint16(flow[1, :, :]).astype('uint32') 

    SHIFT = 2 ** 16
    I = x * SHIFT + y
    return I.view(dtype='int32')


