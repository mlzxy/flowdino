import click
import pandas as pd
import os
import os.path as osp
import torch
import torch.nn as nn
import pytorch_lightning as pl
import random
import torchvision.transforms as T
import torchvision.transforms.functional as tvF
import torch.nn.functional as F
from glob import glob
from importlib import import_module

import fire

from datetime import datetime
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms.functional import InterpolationMode
from typing import List, Any, Tuple

from typing import Optional, Callable
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import VisionDataset
from typing import Tuple, Any


################### Metrics ###################

import numpy as np
import os
import time
import torch
import torch.nn as nn

from collections import defaultdict
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from torchmetrics import Metric
from typing import Optional, List, Tuple, Dict


class PredsmIoU(Metric):
    """
    Subclasses Metric. Computes mean Intersection over Union (mIoU) given ground-truth and predictions.
    .update() can be called repeatedly to add data from multiple validation loops.
    """

    def __init__(self,
                 num_pred_classes: int,
                 num_gt_classes: int):
        """
        :param num_pred_classes: The number of predicted classes.
        :param num_gt_classes: The number of gt classes.
        """
        super().__init__(dist_sync_on_step=False, compute_on_step=False)
        self.num_pred_classes = num_pred_classes
        self.num_gt_classes = num_gt_classes
        self.add_state("gt", [])
        self.add_state("pred", [])
        self.n_jobs = -1

    def update(self, gt: torch.Tensor, pred: torch.Tensor) -> None:
        self.gt.append(gt)
        self.pred.append(pred)

    def compute(self, is_global_zero: bool, many_to_one: bool = False,
                precision_based: bool = False, linear_probe: bool = False) -> Tuple[float, List[np.int64],
                                                                                    List[np.int64], List[np.int64],
                                                                                    List[np.int64], float]:
        """
        Compute mIoU with optional hungarian matching or many-to-one matching (extracts information from labels).
        :param is_global_zero: Flag indicating whether process is rank zero. Computation of metric is only triggered
        if True.
        :param many_to_one: Compute a many-to-one mapping of predicted classes to ground truth instead of hungarian
        matching.
        :param precision_based: Use precision as matching criteria instead of IoU for assigning predicted class to
        ground truth class.
        :param linear_probe: Skip hungarian / many-to-one matching. Used for evaluating predictions of fine-tuned heads.
        :return: mIoU over all classes, true positives per class, false negatives per class, false positives per class,
        reordered predictions matching gt,  percentage of clusters matched to background class. 1/self.num_pred_classes
        if self.num_pred_classes == self.num_gt_classes.
        """
        if is_global_zero:
            pred = torch.cat(self.pred).cpu().numpy().astype(int)
            gt = torch.cat(self.gt).cpu().numpy().astype(int)
            assert len(np.unique(pred)) <= self.num_pred_classes
            assert np.max(pred) <= self.num_pred_classes
            return self.compute_miou(gt, pred, self.num_pred_classes, self.num_gt_classes, many_to_one=many_to_one,
                                     precision_based=precision_based, linear_probe=linear_probe)

    def compute_miou(self, gt: np.ndarray, pred: np.ndarray, num_pred: int, num_gt: int,
                     many_to_one=False, precision_based=False, linear_probe=False) -> Tuple[float, List[np.int64], List[np.int64], List[np.int64],
                                                                                            List[np.int64], float]:
        """
        Compute mIoU with optional hungarian matching or many-to-one matching (extracts information from labels).
        :param gt: numpy array with all flattened ground-truth class assignments per pixel
        :param pred: numpy array with all flattened class assignment predictions per pixel
        :param num_pred: number of predicted classes
        :param num_gt: number of ground truth classes
        :param many_to_one: Compute a many-to-one mapping of predicted classes to ground truth instead of hungarian
        matching.
        :param precision_based: Use precision as matching criteria instead of IoU for assigning predicted class to
        ground truth class.
        :param linear_probe: Skip hungarian / many-to-one matching. Used for evaluating predictions of fine-tuned heads.
        :return: mIoU over all classes, true positives per class, false negatives per class, false positives per class,
        reordered predictions matching gt,  percentage of clusters matched to background class. 1/self.num_pred_classes
        if self.num_pred_classes == self.num_gt_classes.
        """
        assert pred.shape == gt.shape
        print(f"seg map preds have size {gt.shape}")
        tp = [0] * num_gt
        fp = [0] * num_gt
        fn = [0] * num_gt
        jac = [0] * num_gt

        if linear_probe:
            reordered_preds = pred
            matched_bg_clusters = {}
        else:
            if many_to_one:
                match = self._original_match(
                    num_pred, num_gt, pred, gt, precision_based=precision_based)
                # remap predictions
                reordered_preds = np.zeros(len(pred))
                for target_i, matched_preds in match.items():
                    for pred_i in matched_preds:
                        reordered_preds[pred == int(pred_i)] = int(target_i)
                matched_bg_clusters = len(match[0]) / num_pred
            else:
                match = self._hungarian_match(num_pred, num_gt, pred, gt)
                # remap predictions
                reordered_preds = np.zeros(len(pred))
                for target_i, pred_i in zip(*match):
                    reordered_preds[pred == int(pred_i)] = int(target_i)
                # merge all unmatched predictions to background
                for unmatched_pred in np.delete(np.arange(num_pred), np.array(match[1])):
                    reordered_preds[pred == int(unmatched_pred)] = 0
                matched_bg_clusters = 1/num_gt

        # tp, fp, and fn evaluation
        for i_part in range(0, num_gt):
            tmp_all_gt = (gt == i_part)
            tmp_pred = (reordered_preds == i_part)
            tp[i_part] += np.sum(tmp_all_gt & tmp_pred)
            fp[i_part] += np.sum(~tmp_all_gt & tmp_pred)
            fn[i_part] += np.sum(tmp_all_gt & ~tmp_pred)

        # Calculate IoU per class
        for i_part in range(0, num_gt):
            jac[i_part] = float(
                tp[i_part]) / max(float(tp[i_part] + fp[i_part] + fn[i_part]), 1e-8)

        print("IoUs computed")
        return np.mean(jac), tp, fp, fn, reordered_preds.astype(int).tolist(), matched_bg_clusters

    @staticmethod
    def get_score(flat_preds: np.ndarray, flat_targets: np.ndarray, c1: int, c2: int, precision_based: bool = False) \
            -> float:
        """
        Calculates IoU given gt class c1 and prediction class c2.
        :param flat_preds: flattened predictions
        :param flat_targets: flattened gt
        :param c1: ground truth class to match
        :param c2: predicted class to match
        :param precision_based: flag to calculate precision instead of IoU.
        :return: The score if gt-c1 was matched to predicted c2.
        """
        tmp_all_gt = (flat_targets == c1)
        tmp_pred = (flat_preds == c2)
        tp = np.sum(tmp_all_gt & tmp_pred)
        fp = np.sum(~tmp_all_gt & tmp_pred)
        if not precision_based:
            fn = np.sum(tmp_all_gt & ~tmp_pred)
            jac = float(tp) / max(float(tp + fp + fn), 1e-8)
            return jac
        else:
            prec = float(tp) / max(float(tp + fp), 1e-8)
            return prec

    def compute_score_matrix(self, num_pred: int, num_gt: int, pred: np.ndarray, gt: np.ndarray,
                             precision_based: bool = False) -> np.ndarray:
        """
        Compute score matrix. Each element i, j of matrix is the score if i was matched j. Computation is parallelized
        over self.n_jobs.
        :param num_pred: number of predicted classes
        :param num_gt: number of ground-truth classes
        :param pred: flattened predictions
        :param gt: flattened gt
        :param precision_based: flag to calculate precision instead of IoU.
        :return: num_pred x num_gt matrix with A[i, j] being the score if ground-truth class i was matched to
        predicted class j.
        """
        print("Parallelizing iou computation")
        start = time.time()
        score_mat = Parallel(n_jobs=self.n_jobs)(delayed(self.get_score)(pred, gt, c1, c2, precision_based=precision_based)
                                                 for c2 in range(num_pred) for c1 in range(num_gt))
        print(f"took {time.time() - start} seconds")
        score_mat = np.array(score_mat)
        return score_mat.reshape((num_pred, num_gt)).T

    def _hungarian_match(self, num_pred: int, num_gt: int, pred: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray,
                                                                                                      np.ndarray]:
        # do hungarian matching. If num_pred > num_gt match will be partial only.
        iou_mat = self.compute_score_matrix(num_pred, num_gt, pred, gt)
        match = linear_sum_assignment(1 - iou_mat)
        print("Matched clusters to gt classes:")
        print(match)
        return match

    def _original_match(self, num_pred, num_gt, pred, gt, precision_based=False) -> Dict[int, list]:
        score_mat = self.compute_score_matrix(
            num_pred, num_gt, pred, gt, precision_based=precision_based)
        preds_to_gts = {}
        preds_to_gt_scores = {}
        # Greedily match predicted class to ground-truth class by best score.
        for pred_c in range(num_pred):
            for gt_c in range(num_gt):
                score = score_mat[gt_c, pred_c]
                if (pred_c not in preds_to_gts) or (score > preds_to_gt_scores[pred_c]):
                    preds_to_gts[pred_c] = gt_c
                    preds_to_gt_scores[pred_c] = score
        gt_to_matches = defaultdict(list)
        for k, v in preds_to_gts.items():
            gt_to_matches[v].append(k)
        print("matched clusters to gt classes:")
        return gt_to_matches



#########################################################


################### Augmentation ###################

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            img = tvF.hflip(img)
            target = tvF.hflip(target)
        return img, target


class RandomResizedCrop(object):
    def __init__(self, size, scale, ratio=(3. / 4., 4. / 3.)):
        self.rrc_transform = T.RandomResizedCrop(
            size=size, scale=scale, ratio=ratio)

    def __call__(self, img, target=None):
        y1, x1, h, w = self.rrc_transform.get_params(
            img, self.rrc_transform.scale, self.rrc_transform.ratio)
        img = tvF.resized_crop(
            img, y1, x1, h, w, self.rrc_transform.size, tvF.InterpolationMode.BILINEAR)
        target = tvF.resized_crop(
            target, y1, x1, h, w, self.rrc_transform.size, tvF.InterpolationMode.NEAREST)
        return img, target


class ToTensor(object):
    def __call__(self, img, target):
        return tvF.to_tensor(img), tvF.to_tensor(target)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = tvF.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

############################################################################


############################## Dataset ################################

class VOCDataset(VisionDataset):

    def __init__(
            self,
            root: str,
            image_set: str = "trainaug",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
            return_masks: bool = True
    ):
        super(VOCDataset, self).__init__(
            root, transforms, transform, target_transform)
        self.image_set = image_set
        if self.image_set == "trainaug" or self.image_set == "train":
            seg_folder = "SegmentationClassAug"
        elif self.image_set == "val":
            seg_folder = "SegmentationClass"
        else:
            raise ValueError(f"No support for image set {self.image_set}")
        seg_dir = os.path.join(root, seg_folder)
        image_dir = os.path.join(root, 'JPEGImages')
        if not os.path.isdir(seg_dir) or not os.path.isdir(image_dir) or not os.path.isdir(root):
            raise RuntimeError('Dataset not found or corrupted.')
        splits_dir = os.path.join(root, 'ImageSets/Segmentation')
        split_f = os.path.join(
            splits_dir, self.image_set.rstrip('\n') + '.txt')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(seg_dir, x + ".png") for x in file_names]
        self.return_masks = return_masks

        assert all([Path(f).is_file() for f in self.masks]) and all(
            [Path(f).is_file() for f in self.images])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = Image.open(self.images[index]).convert('RGB')
        if self.image_set == "val":
            mask = Image.open(self.masks[index])
            if self.transforms:
                img, mask = self.transforms(img, mask)
            return img, mask
        elif "train" in self.image_set:
            if self.transforms:
                if self.return_masks:
                    mask = Image.open(self.masks[index])
                    res = self.transforms(img, mask)
                else:
                    res = self.transforms(img)
                return res
            return img

    def __len__(self) -> int:
        return len(self.images)


class VOCDataModule(pl.LightningDataModule):

    CLASS_IDX_TO_NAME = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                        'train', 'tvmonitor']

    def __init__(self,
                 data_dir: str = "/common/users/xz653/Dataset/PascalVOC/VOCdevkit/VOC2012",
                 train_split: str = "trainaug",
                 val_split: str = "val",
                 train_image_transform: Optional[Callable] = None,
                 val_image_transform: Optional[Callable] = None,
                 val_target_transform: Optional[Callable] = None,
                 batch_size: int = 60,
                 num_workers: int = 12,
                 shuffle: bool = True,
                 return_masks: bool = True,
                 drop_last: bool = True):
        """
        Data module for PVOC data. "trainaug" and "train" are valid train_splits.
        If return_masks is set train_image_transform should be callable with imgs and masks or None.
        """
        super().__init__()
        self.root = data_dir
        self.train_split = train_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_image_transform = train_image_transform
        self.val_image_transform = val_image_transform
        self.val_target_transform = val_target_transform
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.return_masks = return_masks
        self.num_classes = len(self.CLASS_IDX_TO_NAME)

        # Set up datasets in __init__ as we need to know the number of samples to init cosine lr schedules
        assert train_split == "trainaug" or train_split == "train"
        self.voc_train = VOCDataset(root=self.root, image_set=train_split, transforms=self.train_image_transform,
                                    return_masks=self.return_masks)
        self.voc_val = VOCDataset(root=self.root, image_set=val_split, transform=self.val_image_transform,
                                  target_transform=self.val_target_transform)

    def __len__(self):
        return len(self.voc_train)

    def class_id_to_name(self, i: int):
        return self.CLASS_IDX_TO_NAME[i]

    def setup(self, stage: Optional[str] = None):
        print(f"Train size {len(self.voc_train)}")
        print(f"Val size {len(self.voc_val)}")

    def train_dataloader(self):
        return DataLoader(self.voc_train, batch_size=self.batch_size,
                          shuffle=self.shuffle, num_workers=self.num_workers,
                          drop_last=self.drop_last, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.voc_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                        drop_last=self.drop_last, pin_memory=True)



class CocoStuffDataModule(pl.LightningDataModule):

    def __init__(self,
                 data_dir: str = "/common/users/xz653/Dataset/COCO/cocostuff-full/",
                 train_split: str = "train",
                 val_split: str = "val",
                 train_image_transform: Optional[Callable] = None,
                 val_image_transform: Optional[Callable] = None,
                 val_target_transform: Optional[Callable] = None,
                 batch_size: int = 60,
                 num_workers: int = 12,
                 shuffle: bool = True,
                 num_samples: int= -1,
                 drop_last: bool = True):
        super().__init__()
        self.root = data_dir
        self.train_split = train_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_image_transform = train_image_transform
        self.val_image_transform = val_image_transform
        self.val_target_transform = val_target_transform
        self.shuffle = shuffle
        self.drop_last = drop_last


        # Set up datasets in __init__ as we need to know the number of samples to init cosine lr schedules
        self.train_dataset = CocoStuff(root=self.root, split=train_split, transforms=self.train_image_transform)
        self.val_dataset = CocoStuff(root=self.root, split=val_split, transform=self.val_image_transform,
                                target_transform=self.val_target_transform)
        self.num_classes = self.train_dataset.num_classes


        if num_samples > 0:
            LEN = len(self.train_dataset)
            self.train_dataset = Subset(self.train_dataset, np.random.choice(range(LEN), size=min(num_samples, LEN), replace=False))


    def __len__(self):
        return len(self.train_dataset)

    def setup(self, stage: Optional[str] = None):
        print(f"Train size {len(self.train_dataset)}")
        print(f"Val size {len(self.val_dataset)}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=self.shuffle, num_workers=self.num_workers,
                          drop_last=self.drop_last, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                        drop_last=self.drop_last, pin_memory=True)



class CocoStuff(VisionDataset):

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

    def __init__(self, root="/common/users/xz653/Dataset/COCO/cocostuff-full/", split="train", use_super_category=True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None):
        super(CocoStuff, self).__init__(
            root, transforms, transform, target_transform)
        assert split in ('train', 'all', 'test', 'val')
        self.img_ids = [osp.splitext(osp.basename(f))[0] for f in glob(osp.join(root, "annotations", split + "2017", "*.png"))]
        self.split = split
        self.use_super_category = use_super_category

        if use_super_category:
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

    def __getitem__(self, index):  # still 255 means unlabel/invalid/background
        image_id = self.img_ids[index]

        image_path = osp.join(self.root, self.split + "2017", image_id + '.jpg')
        label_path = osp.join(self.root, "annotations", self.split + "2017", image_id + ".png")

        im = Image.open(image_path).convert('RGB')
        mask = Image.open(label_path)

        if self.use_super_category:
            mask = np.array(mask)
            mask_shape = mask.shape
            mask = self.class_mapping[mask.flatten()].reshape(mask_shape)
            mask = Image.fromarray(mask)

        if self.transforms:
            im, mask = self.transforms(im, mask)
        return im, mask


############################################################################

# vit_small_8, vit_small_16
# vit_base_8, vit_base_16
def main(net=None, from_ckpt=None, save_ckpt='/filer/tmp1/xz653/tmp/ckpt', dataset="voc", num_workers=12,
         input_size=448, mask_size=100, seed=-1, decay_rate=0.1, val_iters=8, drop_at=20, lr=0.01, max_epochs=25, num_samples=-1,
         batch_size=25, opt='sgd', scheduler='step'):

    if 'vit_small' in net:
        embed_dim = 384
    elif 'vit_base' in net:
        embed_dim = 786
    elif 'vit_large' in net:
        embed_dim = 1024
    else:
        raise NotImplementedError()

    assert dataset in ("voc", )
    if seed == -1: seed = random.randint(0, 1000)
    seed_everything(seed)

    print('input_size', input_size, 'mask_size', mask_size)

    # Init transforms and train data
    train_transforms = Compose([
        RandomResizedCrop(size=input_size, scale=(0.8, 1.)),
        RandomHorizontalFlip(p=0.5),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_image_transforms = T.Compose([T.Resize((input_size, input_size)),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    val_target_transforms = T.Compose([T.Resize((mask_size, mask_size), interpolation=InterpolationMode.NEAREST),
                                       T.ToTensor()])

    if dataset == "voc":
        data = VOCDataModule(train_image_transform=train_transforms, num_workers=num_workers, batch_size=batch_size,
                            val_image_transform=val_image_transforms, val_target_transform=val_target_transforms)
    elif dataset == "cocostuff":
        data = CocoStuffDataModule(train_image_transform=train_transforms, num_workers=num_workers, batch_size=batch_size,
                            num_samples=num_samples,
                            val_image_transform=val_image_transforms, val_target_transform=val_target_transforms)
    else:
        raise NotImplementedError

    net_module, net_func = net.split(":")
    feature_net = getattr(import_module(net_module), net_func)(checkpoint=from_ckpt)
    if isinstance(feature_net, tuple):
        feature_net, embed_dim = feature_net

    model = LinearFinetune(
        model=feature_net,
        embed_dim=embed_dim,
        num_classes=data.num_classes,
        lr=lr,
        input_size=input_size,
        mask_size=mask_size,
        val_iters=val_iters,
        decay_rate=decay_rate,
        drop_at=drop_at,
        opt=opt,
        scheduler=scheduler,
        ignore_index=255
    )



    os.makedirs(save_ckpt, exist_ok=True)
    callbacks = [ModelCheckpoint(
        dirpath=save_ckpt,
        monitor='miou_val',
        filename='ckp-{epoch:02d}-{miou_val:.4f}',
        save_top_k=3,
        mode='max',
        verbose=True,
    )]


    trainer = Trainer(
        num_sanity_val_steps=val_iters,
        max_epochs=max_epochs,
        accelerator='gpu',
        fast_dev_run=False,
        log_every_n_steps=50,
        benchmark=True,
        deterministic=False,
        auto_select_gpus=True,
        amp_backend='native',
        callbacks=callbacks,
        logger=False
    )
    trainer.fit(model, datamodule=data)
    print('best iou: ', callbacks[0].best_model_score)


class LinearFinetune(pl.LightningModule):

    def __init__(self, model, embed_dim: int, num_classes: int, lr: float, val_iters: int, input_size: int, mask_size:int,
                 drop_at: int, decay_rate: float = 0.1, ignore_index: int = 255, opt='sgd', scheduler='step'):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.embed_dim = embed_dim

        self.finetune_head = nn.Conv2d(embed_dim, num_classes, 1)

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.miou_metric = PredsmIoU(num_classes, num_classes)
        self.num_classes = num_classes

        self.lr = lr
        self.scheduler = scheduler
        self.opt = opt

        self.val_iters = val_iters
        self.drop_at = drop_at
        self.ignore_index = ignore_index
        self.decay_rate = decay_rate
        self.input_size = input_size
        self.mask_size = mask_size

    def on_after_backward(self):
        # Freeze all layers of backbone
        for param in self.model.parameters():
            param.requires_grad = False

    def configure_optimizers(self):
        if self.opt == 'sgd':
            optimizer = torch.optim.SGD(self.finetune_head.parameters(), weight_decay=0.0001,
                                    momentum=0.9, lr=self.lr)
        else:
            optimizer = torch.optim.AdamW(self.finetune_head.parameters(), weight_decay=0.0001, lr=self.lr)

        if self.scheduler == 'step':
            scheduler = StepLR(optimizer, gamma=self.decay_rate, step_size=self.drop_at)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-6)
        return [optimizer], [scheduler]


    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        imgs, masks = batch
        res = imgs.size(3)
        assert res == self.input_size
        self.model.eval()

        with torch.no_grad():
            features = self.model(imgs)
        mask_preds = self.finetune_head(features)
        masks = F.interpolate(masks, size=features.shape[2:], mode='nearest')

        masks *= 255
        loss = self.criterion(mask_preds, masks.long().squeeze())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        if batch_idx < self.val_iters:
            with torch.no_grad():
                imgs, masks = batch
                features = self.model(imgs)
                mask_preds = self.finetune_head(features)
                mask_preds = F.interpolate(mask_preds, size=(
                    self.mask_size, self.mask_size), mode='bilinear')

                # downsample masks and preds
                gt = masks * 255
                # mask to remove object boundary class
                valid = (gt != self.ignore_index)
                mask_preds = torch.argmax(mask_preds, dim=1).unsqueeze(1)
                # update metric
                self.miou_metric.update(gt[valid].cpu(), mask_preds[valid].cpu())

    def validation_epoch_end(self, outputs: List[Any]):
        miou = self.miou_metric.compute(
            True, many_to_one=False, linear_probe=True)[0]
        self.miou_metric.reset()
        print(miou)
        self.log('miou_val', round(miou, 6))


if __name__ == "__main__":
    fire.Fire(main)
