import torch
import torchvision.datasets as tvd
import torchvision.transforms.functional as F
from dataset.helper import image_resize, mask_resize

PASCAL_VOC = '/filer/tmp1/xz653/Datasets/PascalVOC' 


classes = ['aeroplane',
           'bicycle',
           'bird',
           'boat',
           'bottle',
           'bus',
           'car',
           'cat',
           'chair',
           'cow',
           'diningtable',
           'dog',
           'horse',
           'motorbike',
           'person',
           'pottedplant',
           'sheep',
           'sofa',
           'train',
           'tvmonitor']
        
classes_name2id = {c:i for i,c in enumerate(classes)}



def color_normalize(x, mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]):
    for t, m, s in zip(x, mean, std):
        t.sub_(m)
        t.div_(s)
    return x


class Segmentation(tvd.VOCSegmentation):

    def __init__(self, root=PASCAL_VOC, year='2012', split='val', center_crop=-1, **kwargs):
        super().__init__(root=root, year=year, image_set=split, **kwargs)
        self.center_crop = center_crop

    def __getitem__(self, index):
        im, mask = super().__getitem__(index)
        # during training, we may reset mask 255 to 0
        # during evaluation, we don't care
        im, mask = F.pil_to_tensor(im), F.pil_to_tensor(mask)
        im = im.float() / 255.0
        im = color_normalize(im)
        if self.center_crop > 0:
            im = image_resize(im, self.center_crop)
            mask = mask_resize(mask[0].float(), self.center_crop)[None, ...]
        return im, mask[0].long()
    
    @property
    def num_classes(self):
        return 21


if __name__ == "__main__":
    from tqdm import trange
    D = Segmentation()
    for i in trange(len(D)):
        im, mask = D[i]
        print(tuple(im.shape), tuple(mask.shape), set(mask.flatten().tolist()))
