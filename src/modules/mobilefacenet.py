"""MobileFaceNet backbone for STgram-MFN.

Adapted from github.com/Xiaoccer/MobileFaceNet_Pytorch via the reference
STgram-MFN `net.py`, with the DCASE 2022 Task 2 Top-1 bottleneck setting (see
`src.config.DEFAULT_BOTTLENECK_SETTING`). The input is the 2-channel
spectral-temporal STgram tensor (Sgram channel 0, Tgram channel 1); the output
is the 128-d embedding the ArcFace head consumes.

The reference hardcodes two spatial shapes for a 128x313 input: the depthwise
collapsing conv's kernel `(8, 20)` and the LayerNorm-free classifier width
`128`. Both are lifted into `STgramMFNConfig` (`spatial_size`, `embed_dim`) so
a frontend change surfaces as a clear config error rather than a silent
shape mismatch.
"""

import math

import torch
import torch.nn as nn

from src.config import STgramMFNConfig


class Bottleneck(nn.Module):
    """Inverted-residual block: 1x1 expand -> depthwise -> 1x1 project, with a
    residual connection only when the shape is preserved (`stride == 1 and
    inp == oup`).
    """

    def __init__(self, inp: int, oup: int, stride: int, expansion: int):
        super().__init__()
        self.connect = stride == 1 and inp == oup
        self.conv = nn.Sequential(
            # pointwise expand
            nn.Conv2d(inp, inp * expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # depthwise
            nn.Conv2d(
                inp * expansion,
                inp * expansion,
                3,
                stride,
                1,
                groups=inp * expansion,
                bias=False,
            ),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # pointwise project
            nn.Conv2d(inp * expansion, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.connect:
            return x + self.conv(x)
        return self.conv(x)


class ConvBlock(nn.Module):
    """Conv -> BN -> PReLU (or Conv -> BN when `linear=True`). `dw=True` makes
    the convolution depthwise.
    """

    def __init__(
        self,
        inp: int,
        oup: int,
        k,
        s,
        p,
        dw: bool = False,
        linear: bool = False,
    ):
        super().__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(self.conv(x))
        if self.linear:
            return x
        return self.prelu(x)


class MobileFaceNet(nn.Module):
    def __init__(self, cfg: STgramMFNConfig, in_channels: int = 2):
        super().__init__()
        self.cfg = cfg
        setting = cfg.bottleneck_setting

        self.conv1 = ConvBlock(in_channels, 64, 3, 2, 1)
        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)

        self.inplanes = 64
        self.blocks = self._make_layer(Bottleneck, setting)

        self.conv2 = ConvBlock(setting[-1][1], 512, 1, 1, 0)
        self.linear7 = ConvBlock(512, 512, cfg.spatial_size, 1, 0, dw=True, linear=True)
        self.linear1 = ConvBlock(512, cfg.embed_dim, 1, 1, 0, linear=True)

        self.fc_out = nn.Linear(cfg.embed_dim, cfg.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                layers.append(block(self.inplanes, c, s if i == 0 else 1, t))
                self.inplanes = c
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, label=None):
        """x: (B, 2, n_mels, n_frames). Returns (logits (B, num_classes),
        feature (B, embed_dim)). `label` is accepted for reference-signature
        parity but unused here — ArcFace consumes it downstream.
        """
        x = self.conv1(x)
        x = self.dw_conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.linear7(x)
        x = self.linear1(x)
        feature = x.view(x.size(0), -1)
        out = self.fc_out(feature)
        return out, feature
