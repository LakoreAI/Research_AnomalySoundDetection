"""TgramNet — the learned time-domain branch of STgram.

A large-kernel strided 1-D convolution whose kernel/stride/padding are set to
the same window/hop as the log-mel transform, so its output shares the mel
spectrogram's time axis. Three small-kernel 1-D conv blocks then refine it.
The reference applies `LayerNorm(313)` — normalizing over the TIME axis of the
(B, C, T) tensor — which this implementation generalizes to `cfg.n_frames`.

Faithful to `TgramNet` in github.com/liuyoude/STgram-MFN (`net.py`).
"""

import torch
import torch.nn as nn

from src.config import STgramMFNConfig


class TgramNet(nn.Module):
    def __init__(self, cfg: STgramMFNConfig):
        super().__init__()
        self.conv_extractor = nn.Conv1d(
            1, cfg.c_dim, cfg.win_len, cfg.hop_len, cfg.win_len // 2, bias=False
        )
        self.conv_encoder = nn.Sequential(
            *[
                nn.Sequential(
                    nn.LayerNorm(cfg.n_frames),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv1d(cfg.c_dim, cfg.c_dim, 3, 1, 1, bias=False),
                )
                for _ in range(cfg.tgram_num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, T) raw waveform. Returns (B, c_dim, n_frames)."""
        return self.conv_encoder(self.conv_extractor(x))
