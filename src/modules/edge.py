"""Edge-deployment wrappers for STgram-MFN.

`STgramMFNScorer` bundles the encoder, ArcFace logits, and the anomaly score
into one `(x_wav, x_mel, label) -> (feature, score)` graph. Exporters and
quantizers target this instead of the training `forward`, so the deployed
artifact has the same scoring semantics as `src.pipelines.eval`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.model import STgramMFN


class STgramMFNScorer(nn.Module):
    def __init__(self, model: STgramMFN):
        super().__init__()
        self.model = model

    def forward(
        self, x_wav: torch.Tensor, x_mel: torch.Tensor, label: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (feature (B, embed_dim), score (B,)) where score is
        `-log_softmax(logits)[label]` — higher means more anomalous.
        """
        logits, feature = self.model(x_wav, x_mel, label)
        log_probs = F.log_softmax(logits, dim=1)
        score = -log_probs.gather(1, label.view(-1, 1).long()).squeeze(1)
        return feature, score
