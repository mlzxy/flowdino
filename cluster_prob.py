
import os
import fire
import os.path as osp
import torch
import numpy as np

from tqdm import trange, tqdm
import torch.nn.functional as F
from importlib import import_module
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from dataset import get_eval_dataset
from dataset.metrics.iou import RunningIOU

import warnings
warnings.filterwarnings("ignore")


def main(device="cuda:0", network="network.dino:default", checkpoint=None):
    net_module, net_func = network.split(":")
    Dtest = get_eval_dataset(name="voc", center_crop=-1, split="trainval")
    num_classes = Dtest.num_classes

    device = 'cuda:0'
    model, n_features = getattr(import_module(net_module), net_func)(checkpoint=checkpoint)
    model = model.eval()
    model.return_attn = True
    model = model.to(device)

    LEN = len(Dtest)

    input_size = (448, 448)
    target_size = (224, 224)

    all_gt = torch.zeros(LEN, *target_size)
    all_pixels = torch.zeros(LEN, *target_size)
    all_prototypes = torch.zeros(LEN, n_features)

    for i in trange(LEN, desc='collecting proto features'):
        image, mask = Dtest[i]
        image = image[None, ...].to(device)
        mask = mask[None, None, ...]

        image = F.interpolate(image, input_size, mode='bilinear')
        mask = F.interpolate(mask.float(), target_size, mode='nearest')
        all_gt[i] = mask[:]

        feat, sal = model(image)
        sal = (sal - sal.mean()) / (sal.std() + 1e-5)
        sal = (torch.sigmoid(sal) > 0.5).float()

        proto = (feat * sal).flatten(2).mean(dim=2)
        proto = F.normalize(proto)

        all_prototypes[i] = proto.cpu()
        sal = F.interpolate(sal, target_size, mode='nearest')
        all_pixels[i, :, :] = sal.cpu()
    
    print('running kmeans') 

    pca = PCA(n_components= 64, whiten = True)
    all_prototypes = all_prototypes.numpy()
    all_prototypes = pca.fit_transform(all_prototypes)
    kmeans_res = []
    T = 10

    for t in trange(T, desc=f'running clustering algo for {T} times'):
        iou = RunningIOU(num_classes, compute_hungarian=True, has_background_at_0=False)
        kmeans = KMeans(n_clusters=num_classes - 1)
        prediction_kmeans = kmeans.fit_predict(all_prototypes)

        # print('creating pred mask')
        all_pixels_for_pred = all_pixels.clone()
        for i in range(LEN):
            pred = prediction_kmeans[i] + 1
            all_pixels_for_pred[i] = all_pixels[i] * pred

        # print('computing IOU')
        iou.update(all_gt.long(), all_pixels_for_pred.long())
        scores = iou.compute()

        m_iou = scores['mean_iou']
        kmeans_res.append(m_iou)
        tqdm.write(f'{t} - {m_iou}')
    
    all_mean = np.mean(kmeans_res)
    all_std = np.std(kmeans_res)
    print(f'IOU: mean {all_mean}, std {all_std}')


if __name__ == "__main__":
    fire.Fire(main)