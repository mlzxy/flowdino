from operator import itemgetter

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from einops import rearrange
from torchvision import datasets, transforms
import albumentations as A
from dino_training.utils import GaussianBlur, Solarization


class DataAugmentationDINOFlow(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, flow_mode=False, single_image_mode=False):
        self.flow_mode = flow_mode
        self.single_image_mode = single_image_mode
        color_jitter = transforms.Compose([
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        if flow_mode:
            self.global_spatial = A.Compose([
                A.RandomResizedCrop(224, 224, scale=global_crops_scale, interpolation=Image.Resampling.BICUBIC),
                A.HorizontalFlip(p=0.5)
            ], additional_targets={'flow': 'image'})
            self.local_spatial = A.Compose([
                A.RandomResizedCrop(96, 96, scale=local_crops_scale, interpolation=Image.Resampling.BICUBIC),
                A.HorizontalFlip(p=0.5),
            ], additional_targets={'flow': 'image'})
        else:
            self.global_spatial = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.Resampling.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5)
            ])
            self.local_spatial = transforms.Compose([
                transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.Resampling.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
            ])

        # first global crop
        self.global_pixel_1 = transforms.Compose([
            color_jitter,
            GaussianBlur(1.0),
            normalize,
        ])
        # second global crop

        self.global_pixel_2 = transforms.Compose([
            color_jitter,
            GaussianBlur(0.1),
            Solarization(0.2),
            normalize,
        ])

        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_pixel = transforms.Compose([
            color_jitter,
            GaussianBlur(p=0.5),
            normalize,
        ])

        self.to_tensor = lambda v: torch.from_numpy(v).permute(2, 0, 1)
        self.to_pil = transforms.ToPILImage()

    def __call__(self, image, flow=None):
        """
        numpy h x w x c [0, 255]
        flow  h x w x c
        """
        if self.flow_mode:
            assert flow is not None
            _ = itemgetter('image', 'flow')
            crops = []
            flow_crops = []
            assert isinstance(image, np.ndarray) and isinstance(flow, np.ndarray)

            im1, flow1 = _(self.global_spatial(image=image, flow=flow))
            crops.append(self.global_pixel_1(self.to_pil(im1)))
            flow_crops.append(self.to_tensor(flow1))

            if self.single_image_mode:
                return crops[0], flow_crops[0]

            im2, flow2 = _(self.global_spatial(image=image, flow=flow))
            crops.append(self.global_pixel_2(self.to_pil(im2)))
            flow_crops.append(self.to_tensor(flow2))

            for i in range(self.local_crops_number):
                im3, flow3 = _(self.local_spatial(image=image, flow=flow))
                crops.append(self.local_pixel(self.to_pil(im3)))
                flow_crops.append(self.to_tensor(flow3))
            
            return crops, flow_crops
        else:
            crops = []
            if isinstance(image, (np.ndarray, torch.Tensor)):
                image = self.to_pil(image)
            crops.append(self.global_pixel_1(self.global_spatial(image)))
            crops.append(self.global_pixel_2(self.global_spatial(image)))
            for i in range(self.local_crops_number):
                crops.append(self.local_pixel(self.local_spatial(image)))
            return crops
        
        
def distilled_cross_entropy(l, p, tl=1.0, tp=1.0, mask=None):
    # KL divergence
    """
    l: [*, d]
    p: [*, d]
    return: [*]
    """
    if mask is not None:
        l = F.softmax(l / tl * mask, dim=-1) * mask
        p = F.log_softmax(p / tp * mask, dim=-1) * mask
        loss = torch.sum(-l * p, dim=-1)
        return loss
    else:
        l = F.softmax(l / tl, dim=-1)
        p = F.log_softmax(p / tp, dim=-1)
        loss = torch.sum(-l * p, dim=-1)
        return loss


def make_feature_sim_grid(features, kernel_size=9, stride=None, reference_features=None, 
                        saliency_mask=None, feat_center_detach=False, feat_center_smooth=False):
    # features: [N, C, H, W]
    stride = stride or (kernel_size // 3)
    C, H, W = features.shape[1:]

    features = F.normalize(features, dim=1)
    feat_blocks = F.unfold(features, kernel_size=kernel_size, stride=stride)
    
    feat_blocks = rearrange(feat_blocks, 'N (c w) L -> (N L) w c', c=C)
    center_ind = feat_blocks.shape[1] // 2

    if reference_features is None:
        if saliency_mask is None:
            feat_centers = feat_blocks[:, center_ind, :][:, None, :]  # (N L) 1 c
        else:
            # N, H, W -> N, 1, H, W
            if len(saliency_mask.shape) > 1:
                feat_centers = (feat_blocks * saliency_mask).sum(dim=1, keepdim=True)
            else:
                feat_centers = feat_blocks[torch.arange(0, len(feat_blocks), device=feat_blocks.device), saliency_mask, :][:, None, :]
    else:
        assert saliency_mask is None
        reference_features = F.normalize(reference_features, dim=1)
        ref_feat_blocks = F.unfold(reference_features, kernel_size=kernel_size, stride=stride)
        ref_feat_blocks = rearrange(ref_feat_blocks, 'N (c w) L -> (N L) w c', c=C)
        feat_centers = ref_feat_blocks[:, center_ind, :][:, None, :]  # (N L) 1 c
    
    if feat_center_detach:
        feat_centers = feat_centers.detach()
        
    grid = (feat_centers * feat_blocks).sum(dim=-1) # dot product
    if feat_center_smooth:
        idx = torch.arange(0, len(feat_blocks), device=feat_blocks.device)
        grid[idx, saliency_mask] = 0
        grid[idx, saliency_mask] = grid.max(dim=1).values

    grid = rearrange(grid, '(N h w) (k1 k2) -> N h w k1 k2', k1=kernel_size, k2=kernel_size, 
                     h=int((H-1-(kernel_size-1)) / stride + 1), w=int((W-1-(kernel_size-1)) / stride + 1))
    return grid