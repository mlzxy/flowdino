import numpy as np
import pickle
from pathlib import Path
from io import BytesIO
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor
    y: [A, B, C, ...]
    return: [A, B, C, ..., num_classes] 
    """
    oshape = y.shape
    one_hot = torch.eye(num_classes, dtype='uint8')[y.flatten()]
    return one_hot.reshape(*oshape, num_classes)


def resize_shortest_and_center_crop(image, size, mode='bilinear', return_ratio=False):
    """ 
    image: [N, 3, H, W] or [N, 1, H, W]
    return [N, 3, S, S] or [N, 1, S, S]
    """
    
    # initialize the dimensions of the image to be resized and
    # grab the image size
    h, w = image.shape[2:]

    if h < w:
        r = size / float(h)
        dim = (size, int(w * r))
    else:
        r = size / float(w)
        dim = (int(h * r), size)

    image = F.interpolate(image, dim, mode=mode)
    image = tvF.center_crop(image, size) 
    if return_ratio:
        return image, r
    else:
        return image

def image_resize(image, size, **kwargs):
    return resize_shortest_and_center_crop(image[None, ...], size, **kwargs)[0]

    
def mask_resize(mask, size, **kwargs):
    return resize_shortest_and_center_crop(mask[None, None, ...], size, mode='nearest-exact', **kwargs)[0, 0]


class CacheOpen:
    
    def __init__(self, size=0):
        self.cache = {}
        self.limit = size
    
    def __call__(self, path):
        path = str(path)
        if path in self.cache:
            bs = self.cache[path]
        else:
            bs = Path(path).read_bytes()
            if len(self.cache) < self.limit:
                self.cache[path] = bs

        with BytesIO(bs) as f:
            if path.endswith('.npy'):
                return np.load(f)
            elif path.endswith('.pkl'):
                return pickle.load(f)
            else:
                return np.array(Image.open(f))  # h,w,c in uint8
            

if __name__ == "__main__":
    im = torch.rand(1, 1, 100, 380)
    print(resize_shortest_and_center_crop(im, 80).shape)
