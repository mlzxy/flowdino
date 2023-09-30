
import os
import torch
import random
import numpy as np
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import trange, tqdm
import fire
from importlib import import_module


import dataset as dataset_util
from dataset.metrics.iou import RunningIOU

from network.utils import trainable_params
from network.probe import LinearProbe, ClusterProbe



PROBE_LR = 5e-3
PROBE_STEPS = 10000 # 10000


def main(dataset="cocostuff", device="cuda:0", network="network.dino:default", skip_probe=False, center_crop=320, split="val", 
        steps=PROBE_STEPS, lr=5e-3, protocol="linear", num_train_samples=5000, checkpoint=None, use_super_category=True):
    use_super_category = os.environ.get('use_super_category', '1') == '1'        

    if not use_super_category:
        print("use_super_category=0")

    assert protocol in ("linear", "cluster")
    if not torch.cuda.is_available():
        device = "cpu"
    net_module, net_func = network.split(":")
    Dtrain = dataset_util.get_eval_dataset(name=dataset, center_crop=center_crop, split="train", use_super_category=use_super_category)
    Dtest = dataset_util.get_eval_dataset(name=dataset, center_crop=center_crop, split=split, use_super_category=use_super_category)
    print(f'|validation set| = {len(Dtest)}')
    num_classes = Dtest.num_classes
    model, n_features = getattr(import_module(net_module), net_func)(checkpoint=checkpoint)
    
    if skip_probe:
        semantic_probe = model
        semantic_probe.to(device)
    else:
        if protocol == "linear":
            semantic_probe = LinearProbe(model, n_features, num_classes)
        else:
            semantic_probe = ClusterProbe(model, n_features, num_classes)           

        semantic_probe.to(device)
        opt = optim.AdamW(trainable_params(semantic_probe), PROBE_LR)
        scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-6)
        
        running_iou = RunningIOU(num_classes, compute_hungarian=protocol == "cluster", has_background_at_0=False)

        LEN = len(Dtrain)
        if num_train_samples <= 0:
            num_train_samples = LEN
        train_indexes = np.random.choice(range(LEN), size=min(num_train_samples, LEN), replace=False)

        for s in trange(steps, desc="probing"):
            opt.zero_grad()
            i = train_indexes[random.randint(0, len(train_indexes)-1)]
            data_tuple = Dtrain[i]
            image, smask = data_tuple[0].to(device), data_tuple[1].to(device)
            if protocol == "linear":
                loss, prob, _ = semantic_probe(image[None, ...], smask[None, ...])
            else:
                loss, prob, _ = semantic_probe(image[None, ...])
            pred_smask = prob.argmax(dim=1)
            running_iou.update(smask, pred_smask)
            loss.backward()
            opt.step()
            scheduler.step()

            if s % 20 == 0:
                scores = running_iou.compute()
                tqdm.write(f"({s}) [probing] mean_iou = {scores['mean_iou']}, mean_acc = {scores['mean_acc']}")
    
    semantic_probe.eval()
    running_iou = RunningIOU(num_classes, compute_hungarian=protocol == "cluster", has_background_at_0=False)
    for i in trange(len(Dtest), desc="semantic evaluating"):
        data_tuple = Dtest[i]
        image, smask = data_tuple[0].to(device), data_tuple[1].to(device)
        prob, _ = semantic_probe(image[None, ...])
        running_iou.update(smask, prob.argmax(dim=1))
    
    scores = running_iou.compute()
    print(scores)


if __name__ == "__main__":
    fire.Fire(main)