"""STgram-MFN — spectral-temporal fusion + MobileFaceNet + ArcFace.

Faithful to the reference `STgramMFN` (github.com/liuyoude/STgram-MFN, `net.py`):

    x_wav -> TgramNet -------------------------.
                                                v
    x_mel (log-mel Sgram) -----------------> concat (B, 2, n_mels, T)
                                                v
                                         MobileFaceNet -> feature (B, 128)
                                                v
                                      (optional) ArcFace -> logits

The two branches are concatenated CHANNEL-wise, which is why MobileFaceNet is
constructed with `in_channels=2`. `forward` accepts an optional `label` and
returns `(logits, feature)`; at test time `logits` feed
`src.modules.loss.anomaly_score`.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.config import STgramMFNConfig
from src.modules.arcface import ArcMarginProduct
from src.modules.mobilefacenet import MobileFaceNet
from src.modules.tgramnet import TgramNet


class STgramMFN(nn.Module):
    def __init__(self, cfg: STgramMFNConfig):
        super().__init__()
        self.cfg = cfg
        self.use_arcface = cfg.use_arcface
        self.arcface = ArcMarginProduct(cfg) if cfg.use_arcface else None
        self.tgramnet = TgramNet(cfg)
        self.mobilefacenet = MobileFaceNet(cfg, in_channels=2)

    def get_tgram(self, x_wav: torch.Tensor) -> torch.Tensor:
        """Tgram branch alone — used by quantization calibration / diagnostics
        that need the time-domain feature without the full forward.
        """
        return self.tgramnet(x_wav.unsqueeze(1))

    def forward(
        self,
        x_wav: torch.Tensor,
        x_mel: torch.Tensor,
        label: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """x_wav: (B, T) waveform. x_mel: (B, n_mels, n_frames) log-mel.
        label: (B,) machine-ID labels; required when `use_arcface` is True
        (ArcFace's angular margin is label-dependent).

        Returns (logits (B, num_classes), feature (B, embed_dim)).
        """
        if x_wav.dim() == 2:
            x_wav = x_wav.unsqueeze(1)
        if x_mel.dim() == 3:
            x_mel = x_mel.unsqueeze(1)

        x_t = self.tgramnet(x_wav).unsqueeze(1)
        x = torch.cat((x_mel, x_t), dim=1)

        out, feature = self.mobilefacenet(x, label)
        if self.arcface is not None:
            if label is None:
                raise ValueError("ArcFace requires `label` at forward time")
            out = self.arcface(feature, label)
        return out, feature
