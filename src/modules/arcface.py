"""ArcFace (additive angular margin) classifier head.

Faithful to `ArcMarginProduct` in github.com/liuyoude/STgram-MFN (`net.py`).
The training pretext is self-supervised classification by machine ID; ArcFace
pushes embeddings of the same machine together and different machines apart,
which is what makes the embedding usable for one-class anomaly scoring.

`sub` is the sub-center count from ArcFace's later formulation: when >1 the
weight matrix holds `out_features * sub` centers and the logits take the max
over each class's centers. The reference default (`sub=1`) is plain ArcFace.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from src.config import STgramMFNConfig


class ArcMarginProduct(nn.Module):
    def __init__(self, cfg: STgramMFNConfig):
        super().__init__()
        self.in_features = cfg.embed_dim
        self.out_features = cfg.num_classes
        self.s = cfg.arcface_s
        self.m = cfg.arcface_m
        self.sub = cfg.arcface_sub
        self.weight = Parameter(
            torch.Tensor(self.out_features * self.sub, self.in_features)
        )
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = cfg.arcface_easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        # cos(pi - m): threshold below which the margin is not applied, so
        # cos(theta + m) stays monotonically decreasing over [0, pi].
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        if self.sub > 1:
            cosine = cosine.view(-1, self.out_features, self.sub)
            cosine, _ = torch.max(cosine, dim=2)
        # clamp_min(0) on the radicand is a numerical guard only (the geometry
        # is unchanged): normalize can push |cosine| a hair past 1 in fp32/fp16,
        # which would otherwise make sqrt negative and poison the loss with NaN.
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp_min(0))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=x.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"s={self.s}, m={self.m}, sub={self.sub}"
        )
