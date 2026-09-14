"""Training loss and the anomaly score derived from its logits.

STgram-MFN's pretext is self-supervised classification by machine ID. Training
uses ordinary cross-entropy on ArcFace logits (the reference's `ASDLoss`).
At test time there is no "anomaly" class — the anomaly score is the NEGATIVE
log-probability the model assigns to the sample's own machine ID: a normal
recording of machine `id_XX` should be confidently classified as `id_XX`, an
anomalous one should not. Higher score = more anomalous.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASDLoss(nn.Module):
    """Cross-entropy over machine-ID logits (reference `ASDLoss`)."""

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, labels)


@torch.no_grad()
def anomaly_score(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-sample anomaly score: `-log_softmax(logits)[label]`.

    Matches the reference `Trainer.test` scoring (it averages the negative
    log-prob over the batch, which is a no-op at inference batch size 1).
    Returns a (B,) tensor; larger means more anomalous.
    """
    log_probs = F.log_softmax(logits, dim=1)
    return -log_probs.gather(1, labels.view(-1, 1).long()).squeeze(1)
