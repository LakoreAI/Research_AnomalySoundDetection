"""mn01 swap model: EfficientAT mn01 embedder + ArcFace, and its scorer.

Shared by `scripts/training/train_mn01.py` (training) and the ONNX exporters
(edge). The ArcFace input width is taken from the loaded embedder, so the same
class works for any EfficientAT width (mn01 = 96-d, mn02 = 192-d, ...).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import MN01Config, STgramMFNConfig
from src.modules.arcface import ArcMarginProduct
from src.modules.mn01 import load_mn01


class MN01ArcFace(nn.Module):
    """mn01 embedder + ArcFace head (both trainable when fine-tuning)."""

    def __init__(
        self,
        mn_cfg: MN01Config,
        num_classes: int,
        checkpoint: str = None,
        arcface_m: float = 0.7,
        arcface_s: float = 30.0,
    ):
        super().__init__()
        self.cfg_mn = mn_cfg
        self.embedder = load_mn01(mn_cfg, checkpoint=checkpoint, map_location="cpu")
        self.embed_dim = self.embedder.embed_dim
        head_cfg = STgramMFNConfig(
            num_classes=num_classes,
            embed_dim=self.embed_dim,
            arcface_m=arcface_m,
            arcface_s=arcface_s,
        )
        self.arcface = ArcMarginProduct(head_cfg)
        # Optional SpecAugment, applied to the mel only while training.
        self.augment = None

    def forward(self, wav: torch.Tensor, label: torch.Tensor):
        if self.augment is not None and self.training:
            melspec = self.embedder.frontend(wav)
            melspec = self.augment(melspec)
            _, emb = self.embedder.net(melspec.unsqueeze(1))
        else:
            emb = self.embedder(wav)
        return self.arcface(emb, label), emb


class MN01Scorer(nn.Module):
    """`(x_wav, label) -> (feature, score)`. Includes the 32 kHz frontend, so
    the raw-waveform graph contains a complex STFT and cannot be ONNX-exported
    by the legacy exporter; use `MN01MelScorer` for export.
    """

    def __init__(self, model: MN01ArcFace):
        super().__init__()
        self.model = model

    def forward(self, x_wav: torch.Tensor, label: torch.Tensor):
        logits, feature = self.model(x_wav, label)
        log_probs = F.log_softmax(logits, dim=1)
        score = -log_probs.gather(1, label.view(-1, 1).long()).squeeze(1)
        return feature, score


class MN01MelScorer(nn.Module):
    """Frontend-free mn01 scorer for export: input is mn01's 32 kHz log-mel.

    `x_mel` is `(B, 1, n_mels, n_frames)`, produced by `Mn01MelFrontend`
    (kept outside the graph, as with STgram-MFN's Sgram).
    """

    def __init__(self, model: MN01ArcFace):
        super().__init__()
        self.model = model

    def forward(self, x_mel: torch.Tensor, label: torch.Tensor):
        # Compute the pooled feature directly (features -> avgpool -> flatten)
        # instead of `net(x_mel)`: the backbone's `_forward_impl` ends in
        # `.squeeze()`, which collapses the batch dim when B == 1 and bakes a
        # static Squeeze into the ONNX graph. `flatten(1)` is identical for
        # B > 1 and export-safe.
        x = x_mel
        for layer in self.model.embedder.net.features:
            x = layer(x)
        feature = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        logits = self.model.arcface(feature, label)
        log_probs = F.log_softmax(logits, dim=1)
        score = -log_probs.gather(1, label.view(-1, 1).long()).squeeze(1)
        return feature, score
