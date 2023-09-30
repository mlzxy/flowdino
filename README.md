# Optical Flow boosts Unsupervised Localization and Segmentation (IROS 2023)

The evaluation code is in `cluster_prob.py`, `linear_prob_cocostuff.py` and `linear_prob_voc.py`. The entry point of self supervised training with flow is `dino_training/train.py`. The overall code is refactored a bit but shall be all right, but I haven't got the time to fully organize the training data, will do it later.

To run evaluation, download [pretrained weights](https://drive.google.com/drive/folders/1dHj1u0RnpZZ9qkETRrlF7KD6JzD7g2r4?usp=sharing), then 

```bash
# Linear Probing on COCOStuff and VOC
python3 ./linear_prob_cocostuff.py --network "network.dino:vit_base_8" \
 --checkpoint ./weights/b8_flow_dino.pth
python3 ./linear_prob_voc.py --network "network.dino:vit_small_8"  \
--checkpoint ./weights/s8_flow_dino.pth


# Cluster Probing on VOC
python3 ./cluster_prob.py --network "network.dino:vit_small_16" \
 --checkpoint ./weights/s16_flow_dino.pth
```