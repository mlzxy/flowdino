import torch
import torch.nn as nn
import dino_training.vision_transformer as vits
from network.utils import freeze_params


class ResNetFeaturizer(nn.Module):

    def __init__(self, net, patch_size):
        super().__init__()
        self.model = net
        self.patch_size = patch_size
    
    def forward(self, img):
        with torch.no_grad():
            feat = self.model.get_intermediate_layers(img)[0]
            return feat


class DinoFeaturizer(nn.Module):

    def __init__(self, model, patch_size):
        super().__init__()
        self.model = model
        self.patch_size = patch_size
        self.return_attn = False
            
    def forward(self, img, n=1):
        with torch.no_grad():
            if not self.return_attn:
                feat = self.model.get_intermediate_layers(img, n=n)[0]
            else:
                feat, attn = self.model.forward_backbone(img, last_self_attention=True)

            feat_h = img.shape[2] // self.patch_size
            feat_w = img.shape[3] // self.patch_size
            image_feat = feat[:, 1:, :].reshape(feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)

            if self.return_attn:
                attn = attn.sum(dim=1).reshape(-1, 1, feat_h, feat_w)
                return image_feat, attn
            else:
                return image_feat


def _make_dino(patch_size=8, arch="vit_base"):
    model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
    model.eval()
    url = None
    if arch == "vit_small" and patch_size == 16:
        url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
    elif arch == "vit_small" and patch_size == 8:
        url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
    elif arch == "vit_base" and patch_size == 16:
        url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
    elif arch == "vit_base" and patch_size == 8:
        url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
    assert url is not None
    state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
    model.load_state_dict(state_dict, strict=True)
    n_features = 384 if arch == "vit_small" else 768
    return model, n_features

def default(task="semantic"):
    net, n_features = _make_dino(8, "vit_small")
    net = freeze_params(net)
    net.eval()
    net = DinoFeaturizer(net, 8)
    return net, n_features

    
def sanitize_state_dict(sd):
    nd = {}
    for k,v in sd.items():
        if k.startswith('module'):
            k = k[len('module.'):]
        if k.startswith('backbone'):
            k = k[len('backbone.'):]
        if k.startswith('model'):
            k = k[len('model.'):]
        if k.startswith('head') or 'flow_head' in k:
            continue
        if 'projection' in k or 'prototype' in k:
            continue
        if 'decoder' in k or 'mask_token' in k:
            continue
        nd[k] = v
    return nd



def vit(checkpoint, patch_size=16, arch='base', ext=-1):
    model = vits.__dict__["vit_" + arch](patch_size=patch_size, num_classes=0, ext=ext)
    checkpoint = checkpoint or "/common/users/xz653/Workspace/nips2023/log/models/dino_vitbase16_pretrain.pth"
    print(f'load from checkpoint: {checkpoint}')
    states = torch.load(checkpoint)
    if 'teacher' in states: states = states['teacher']
    if 'model' in states: states = states['model']
    state_dict = sanitize_state_dict(states)
    state_dict = {k: v for k,v in state_dict.items() if 'optical_flow_head' not in k}
    
    model.load_state_dict(state_dict, strict=True)
    model = freeze_params(model)
    model.eval()
    net = DinoFeaturizer(model, patch_size)
    embed_sizes = {
        'small': 384,
        'base': 768,
        'large': 1024,
    }    
    return net, embed_sizes[arch]


def vit_small_8(checkpoint): return vit(checkpoint=checkpoint, patch_size=8, arch='small')

def vit_base_8(checkpoint): return vit(checkpoint=checkpoint, patch_size=8, arch='base')

def vit_base_16(checkpoint, **kwargs): return vit(checkpoint=checkpoint, patch_size=16, arch='base', **kwargs)

def vit_small_16(checkpoint, **kwargs): return vit(checkpoint=checkpoint, patch_size=16, arch='small', **kwargs)

def vit_large_16(checkpoint): return vit(checkpoint=checkpoint, patch_size=16, arch='large')