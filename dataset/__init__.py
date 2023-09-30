import dataset.coco as coco
import dataset.voc as voc
from torch.utils.data import Dataset


def get_eval_dataset(name="voc", task="semantic", **kwargs) -> Dataset:
    assert task in ("semantic", "instance")
    d = None
    if task == "semantic":
        if name == "voc":
            d = voc.Segmentation(**kwargs)
        elif name == "coco":
            d = coco.CocoDataset(semantic_only=True, **kwargs)
        elif name == "coco20k":
            d = coco.CocoDataset(semantic_only=True, k20=True, **kwargs)
        elif name == "cocostuff10k":
            d = coco.CocoStuff10k(**kwargs)
        elif name == "cocostuff":
            d = coco.CocoStuff(**kwargs)
    else:
        if name == "coco":
            d = coco.CocoDataset(semantic_only=False, **kwargs)
        elif name == "coco20k":
            d = coco.CocoDataset(semantic_only=False, k20=True, **kwargs)

    if d is None:
        raise NotImplementedError("no dataset for given requirements is found")
    return d
