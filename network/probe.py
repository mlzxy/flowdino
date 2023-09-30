import torch
import torch.nn as nn
import torch.nn.functional as F
from network.utils import resize_to


def cross_entropy2d(input, target, weight=None, ignore_index=255):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()
    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=weight, reduction='mean', ignore_index=ignore_index
    )
    return loss



class ClusterProbe(nn.Module):
    def __init__(self, net, dim, num_classes, semantic_feature_position=-1):
        super(ClusterProbe, self).__init__()
        self.net = net
        self.semantic_feature_position = semantic_feature_position
        self.num_classes = num_classes
        self.clusters = torch.nn.Parameter(torch.randn(num_classes, dim))
    
    def reset_parameters(self):
        with torch.no_grad():
            self.clusters.copy_(torch.randn(self.n_classes, self.dim))

    def forward(self, image):
        normed_clusters = F.normalize(self.clusters, dim=1)

        code = self.net(image)        
        if self.semantic_feature_position >= 0:
            code = code[self.semantic_feature_position]

        normed_features = F.normalize(code, dim=1)

        inner_products = torch.einsum("bchw,nc->bnhw", normed_features, normed_clusters)

        cluster_probs = F.one_hot(torch.argmax(inner_products, dim=1), self.clusters.shape[0]) \
                .permute(0, 3, 1, 2).to(torch.float32)
        # cluster_probs = F.softmax(inner_products, dim=1)

        if self.training:
            cluster_loss = -(cluster_probs * inner_products).sum(1).mean()
            return cluster_loss, resize_to(cluster_probs, image), code
        else:
            return resize_to(cluster_probs, image), code


class LinearProbe(nn.Module):
    def __init__(self, net, dim, num_classes, semantic_feature_position=-1):
        super(LinearProbe, self).__init__()
        self.net = net
        self.semantic_feature_position = semantic_feature_position
        self.num_classes = num_classes
        self.conv = nn.Conv2d(dim, num_classes, 1, 1)
    
    
    def forward(self, image, label=None):
        code = self.net(image)        
        if self.semantic_feature_position >= 0:
            semantic_code = code[self.semantic_feature_position]
        else:
            semantic_code = code

        logits = self.conv(semantic_code)
        logits = resize_to(logits, image)
        if label is not None:
            loss = cross_entropy2d(logits, label)
            return loss, F.softmax(logits, dim=1), code
        else:
            return logits, code