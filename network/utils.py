import torch
import torch.nn.functional as F


def freeze_params(net, learnable=False):
    for param in net.parameters():
        param.requires_grad = learnable
    return net

    
    
def trainable_params(net):
    return [p for p in net.parameters() if p.requires_grad]

    
    
def resize_to(input, target, mode='bilinear'):
    h, w = input.shape[-2:]
    ht, wt = target.shape[-2:]
    if h != ht and w != wt:  # usually upsample labels
        input = F.interpolate(input, size=(ht, wt), mode=mode, align_corners=True)
        resized_input = input
    else:
        resized_input = input
    return resized_input