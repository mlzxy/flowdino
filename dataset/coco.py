from typing import Tuple
import torch
import os.path as osp
from glob import glob
from pathlib import Path
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import scipy.io as sio

from dataset.helper import image_resize, mask_resize, CacheOpen
from pycocotools.coco import COCO


COCO_SUPER_CATEGORY = ['accessory',
                       'animal',
                       'appliance',
                       'electronic',
                       'food',
                       'furniture',
                       'indoor',
                       'kitchen',
                       'outdoor',
                       'person',
                       'sports',
                       'vehicle']

COCO_CATEGORY = ['handbag',
                 'mouse',
                 'cup',
                 'hot dog',
                 'bear',
                 'skis',
                 'bicycle',
                 'apple',
                 'tv',
                 'cow',
                 'traffic light',
                 'chair',
                 'toothbrush',
                 'fork',
                 'skateboard',
                 'backpack',
                 'carrot',
                 'truck',
                 'motorcycle',
                 'sports ball',
                 'cat',
                 'potted plant',
                 'cake',
                 'toaster',
                 'zebra',
                 'person',
                 'airplane',
                 'stop sign',
                 'sink',
                 'cell phone',
                 'couch',
                 'dog',
                 'laptop',
                 'parking meter',
                 'pizza',
                 'snowboard',
                 'donut',
                 'banana',
                 'bed',
                 'bottle',
                 'scissors',
                 'tie',
                 'car',
                 'tennis racket',
                 'umbrella',
                 'kite',
                 'vase',
                 'bus',
                 'baseball glove',
                 'remote',
                 'microwave',
                 'giraffe',
                 'orange',
                 'toilet',
                 'book',
                 'boat',
                 'wine glass',
                 'sheep',
                 'surfboard',
                 'broccoli',
                 'dining table',
                 'keyboard',
                 'train',
                 'clock',
                 'oven',
                 'suitcase',
                 'teddy bear',
                 'fire hydrant',
                 'baseball bat',
                 'sandwich',
                 'knife',
                 'bird',
                 'horse',
                 'refrigerator',
                 'spoon',
                 'hair drier',
                 'bench',
                 'bowl',
                 'elephant',
                 'frisbee']

dirname = osp.dirname(__file__)


def color_normalize(x, mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]):
    for t, m, s in zip(x, mean, std):
        t.sub_(m)
        t.div_(s)
    return x



class CocoDataset(Dataset):
    def __init__(self, root='/common/users/xz653/Dataset/COCO/mscoco', cache_size=0, split='val', year='2017', k20=False, include_crowd=False,
                 semantic_only=False, use_super_category=True, center_crop=-1):
        if k20:
            split = 'train'
            year = '2014'
        self.coco = COCO(f'{root}/annotations/instances_{split}{year}.json')
        self.root = osp.join(root, split + str(year))
        self.include_crowd = include_crowd
        self.use_super_category = use_super_category
        self.semantic_only = semantic_only
        self.center_crop = center_crop

        def to_id(s):
            bname = osp.basename(s).split('.')[0]
            return int(bname.split('_')[-1])

        self.img_ids = list(map(to_id, (Path(dirname) / 'coco_20k_filenames.txt').read_text(
        ).splitlines(keepends=False))) if k20 else self.coco.getImgIds()
        self.image_open = CacheOpen(cache_size)
        self.category2classidx = {c['id']: (COCO_SUPER_CATEGORY.index(c['supercategory']) if use_super_category else COCO_CATEGORY.index(c['name'])) + 1
                                  for i, c in enumerate(self.coco.loadCats(self.coco.getCatIds()))}

    def __len__(self) -> int:
        return len(self.img_ids)

    @property
    def num_classes(self):
        return 1 + len(self.category2classidx)

    def __getitem__(self, i: int):
        meta = self.coco.loadImgs(self.img_ids[i])[0]
        image = self.image_open(
            osp.join(self.root, meta['file_name']))
        if len(image.shape) == 2:  # gray scale image
            image = image[..., None].repeat(3, axis=-1) 

        smask = np.zeros(image.shape[:-1])
        annIds = self.coco.getAnnIds(
            imgIds=self.img_ids[i], iscrowd=self.include_crowd)
        anns = self.coco.loadAnns(annIds)
        imasks = []
        icategories = []

        image = torch.from_numpy(image).permute(2, 0, 1)
        image = (image.float() / 255.0 - 0.5) * 2 # need a proper transform here

        for a in anns:
            mask = self.coco.annToMask(a)
            imasks.append(mask)
            class_ind = self.category2classidx[a['category_id']]
            icategories.append(class_ind)
            smask = np.maximum(smask, mask * class_ind)

        smask = torch.from_numpy(smask)

        if self.center_crop > 0:
            image = image_resize(image, self.center_crop)
            smask = mask_resize(smask, self.center_crop)

        if self.semantic_only:
            return image, smask.long()
        else:
            imasks = torch.from_numpy(np.array(imasks))
            if self.center_crop > 0:
                imasks = image_resize(imasks, self.center_crop, mode='nearest-exact')
            icategories = torch.from_numpy(np.array(icategories))
            keep_masks = imasks.flatten(1).max(dim=1).values > 0
            imasks = imasks[keep_masks]
            icategories = icategories[keep_masks]
            return image, smask.long(), imasks, icategories



class CocoStuff(Dataset):

    Fine2Coarse = {0: 9, 1: 11, 2: 11, 3: 11, 4: 11, 5: 11, 6: 11, 7: 11, 8: 11, 9: 8, 10: 8, 11: 8, 12: 8,
                   13: 8, 14: 8, 15: 7, 16: 7, 17: 7, 18: 7, 19: 7, 20: 7, 21: 7, 22: 7, 23: 7, 24: 7,
                   25: 6, 26: 6, 27: 6, 28: 6, 29: 6, 30: 6, 31: 6, 32: 6, 33: 10, 34: 10, 35: 10, 36: 10,
                   37: 10, 38: 10, 39: 10, 40: 10, 41: 10, 42: 10, 43: 5, 44: 5, 45: 5, 46: 5, 47: 5, 48: 5,
                   49: 5, 50: 5, 51: 2, 52: 2, 53: 2, 54: 2, 55: 2, 56: 2, 57: 2, 58: 2, 59: 2, 60: 2,
                   61: 3, 62: 3, 63: 3, 64: 3, 65: 3, 66: 3, 67: 3, 68: 3, 69: 3, 70: 3, 71: 0, 72: 0,
                   73: 0, 74: 0, 75: 0, 76: 0, 77: 1, 78: 1, 79: 1, 80: 1, 81: 1, 82: 1, 83: 4, 84: 4,
                   85: 4, 86: 4, 87: 4, 88: 4, 89: 4, 90: 4, 91: 17, 92: 17, 93: 22, 94: 20, 95: 20, 96: 22,
                   97: 15, 98: 25, 99: 16, 100: 13, 101: 12, 102: 12, 103: 17, 104: 17, 105: 23, 106: 15,
                   107: 15, 108: 17, 109: 15, 110: 21, 111: 15, 112: 25, 113: 13, 114: 13, 115: 13, 116: 13,
                   117: 13, 118: 22, 119: 26, 120: 14, 121: 14, 122: 15, 123: 22, 124: 21, 125: 21, 126: 24,
                   127: 20, 128: 22, 129: 15, 130: 17, 131: 16, 132: 15, 133: 22, 134: 24, 135: 21, 136: 17,
                   137: 25, 138: 16, 139: 21, 140: 17, 141: 22, 142: 16, 143: 21, 144: 21, 145: 25, 146: 21,
                   147: 26, 148: 21, 149: 24, 150: 20, 151: 17, 152: 14, 153: 21, 154: 26, 155: 15, 156: 23,
                   157: 20, 158: 21, 159: 24, 160: 15, 161: 24, 162: 22, 163: 25, 164: 15, 165: 20, 166: 17,
                   167: 17, 168: 22, 169: 14, 170: 18, 171: 18, 172: 18, 173: 18, 174: 18, 175: 18, 176: 18,
                   177: 26, 178: 26, 179: 19, 180: 19, 181: 24}

    # /common/users/xz653/Dataset/COCO/cocostuff-full/
    # "/filer/tmp1/xz653/Datasets/cocostuff-full/"
    # /filer/tmp1/xz653/
    def __init__(self, root="/filer/tmp1/xz653/Datasets/cocostuff-full/", split="train", cache_size=0, use_super_category=True, k10=False, center_crop=-1,
                 aug=None
                 ):
        assert split in ('train', 'all', 'test', 'val')
        self.k10 = k10 
        if k10:
            if split == 'val':
                split = 'test'
            self.img_ids = (Path(root) / "imageLists" / (split + '.txt')).read_text().splitlines(keepends=False)
        else:
            self.img_ids = [osp.splitext(osp.basename(f))[0] for f in glob(osp.join(root, "annotations", split + "2017", "*.png"))]
        self.image_open = CacheOpen(cache_size)
        self.split = split
        self.use_super_category = use_super_category
        self.center_crop = center_crop
        self.aug = aug


        if use_super_category:
            # here we count 255 as an extra class (a bug introduced quite early)
            # but this won't change the IOU calculation (tested)
            # so the results are still legit and we don't change this then
            self.num_classes = len(set(self.Fine2Coarse.values())) + 1
            self.class_mapping = np.zeros((256, ))
            for f_cid, c_cid in self.Fine2Coarse.items():
                self.class_mapping[f_cid] = c_cid
            self.class_mapping[255] = 255
        else:
            self.num_classes = len(set(self.Fine2Coarse.keys())) + 1           
        self.root = root

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, index): # still 255 means unlabel/invalid/background
        image_id = self.img_ids[index]
        if self.k10:
            image_path = osp.join(self.root, "images", image_id + ".jpg")
            label_path = osp.join(self.root, "annotations", image_id + ".mat")
            im = self.image_open(image_path)
            mask = sio.loadmat(label_path)["S"]
        else:
            image_path = osp.join(self.root, self.split + "2017", image_id + '.jpg')
            label_path = osp.join(self.root, "annotations", self.split + "2017", image_id + ".png")
            im = self.image_open(image_path)
            mask = self.image_open(label_path)
        mask_shape = mask.shape
        if self.use_super_category:
            mask = self.class_mapping[mask.flatten()].reshape(mask_shape)
        if len(im.shape) == 2:  # gray scale image
            im = im[..., None].repeat(3, axis=-1)
        
        
        # apply augmentation here
        if self.aug is not None:
            _ = self.aug(image=im, mask=mask)
            im = _['image']
            mask = _['mask']
        
        im, mask = torch.from_numpy(im).permute(2, 0, 1), torch.from_numpy(mask)
        im = im / 255.0
        im = color_normalize(im)
        # im = (im.float() / 255.0 - 0.5) * 2

        if self.center_crop > 0 and im.shape[-1] != self.center_crop:
            im = image_resize(im, self.center_crop)
            mask = mask_resize(mask, self.center_crop)

        return im, mask.long()
        
            
class CocoStuff10k(CocoStuff):
    def __init__(self, root="/common/users/xz653/Dataset/COCO/cocostuff-10k/", split="train", cache_size=0, use_super_category=True, k10=True):
        super().__init__(root=root, split=split, cache_size=cache_size, use_super_category=use_super_category, k10=k10)


if __name__ == "__main__":
    from tqdm import trange
    import random

    def print_shapes(*args):
        print([tuple(a.shape) for a in args])

    D = CocoStuff()
    label = set()
    for i in trange(5000):
        im, mask = D[random.randint(0, len(D)-1)]
        label |= set(mask.numpy().flatten().tolist())
    print(label, len(label))
