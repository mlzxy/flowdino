# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
from torchmetrics import Metric
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment


class RunningIOU(Metric):
    def __init__(self, n_classes: int, compute_hungarian: bool=False,
                 dist_sync_on_step=True, has_background_at_0=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.n_classes = n_classes
        self.compute_hungarian = compute_hungarian
        self.has_background_at_0 = has_background_at_0
        self.add_state("stats", default=torch.zeros(n_classes, n_classes, dtype=torch.int64),
                       dist_reduce_fx="sum")


    def update(self, preds: torch.Tensor, target: torch.Tensor):
        with torch.no_grad():
            actual = target.reshape(-1)
            preds = preds.reshape(-1)
            mask = (actual >= 0) & (actual < self.n_classes) & (preds >= 0) & (preds < self.n_classes)
            actual = actual[mask]
            preds = preds[mask]
            self.stats += torch.bincount(
                self.n_classes * actual + preds,
                minlength=self.n_classes * self.n_classes) \
                .reshape(self.n_classes, self.n_classes).t().to(self.stats.device)

    def map_clusters(self, clusters):
        return torch.tensor(self.assignments[1])[clusters]


    def compute(self):
        if self.compute_hungarian:
            self.assignments = linear_sum_assignment(self.stats.detach().cpu(), maximize=True)
            self.histogram = self.stats[np.argsort(self.assignments[1]), :]
        else:
            self.assignments = (torch.arange(self.n_classes).unsqueeze(1),
                                torch.arange(self.n_classes).unsqueeze(1))
            self.histogram = self.stats

        tp = torch.diag(self.histogram)
        fp = torch.sum(self.histogram, dim=0) - tp
        fn = torch.sum(self.histogram, dim=1) - tp

        iou = tp / (tp + fp + fn)
        prc = tp / (tp + fn)
        opc = torch.sum(tp) / torch.sum(self.histogram)

        if self.has_background_at_0:
            iou = iou[1:]

        metric_dict = {"mean_iou": iou[~torch.isnan(iou)].mean().item(),
                        "mean_acc": opc.item()}
        result = {k: 100 * v for k, v in metric_dict.items()}
        result['iou_per_cls'] = iou
        return result

    

if __name__ == "__main__":
    ciou = RunningIOU(n_classes=3, compute_hungarian=False)
    label = torch.tensor([0,0,0,1,1,2,0])
    pred = torch.tensor([0,0,0,1,1,1,0])

    for i in range(5):
        ciou.update(pred, label)
    
    print(ciou.compute())