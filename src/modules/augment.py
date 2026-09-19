"""Training-time augmentation, off unless a config enables it.

Deliberately small and parameter-light: additive noise, random gain, and a
time shift on the waveform; frequency/time masking (SpecAugment) on the
log-mel. Applied identically to whatever backbone is being trained so the
A-vs-B comparison stays fair. Nothing here runs at evaluation time.
"""

from typing import Optional

import torch
import torch.nn as nn


class WaveAugment(nn.Module):
    """Additive Gaussian noise + random gain + circular time shift."""

    def __init__(
        self, noise_std: float = 0.0, gain_db: float = 0.0, time_shift: float = 0.0
    ):
        super().__init__()
        self.noise_std = noise_std
        self.gain_db = gain_db
        self.time_shift = time_shift

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if self.gain_db > 0:
            g = torch.empty(1).uniform_(-self.gain_db, self.gain_db).item()
            wav = wav * (10.0 ** (g / 20.0))
        if self.noise_std > 0:
            wav = wav + torch.randn_like(wav) * self.noise_std
        if self.time_shift > 0:
            shift = int(
                torch.empty(1).uniform_(-self.time_shift, self.time_shift).item()
                * wav.shape[-1]
            )
            if shift:
                wav = torch.roll(wav, shift, dims=-1)
        return wav


class SpecAugment(nn.Module):
    """Frequency and time masking on a log-mel tensor `(..., n_mels, n_frames)`."""

    def __init__(
        self,
        freq_mask: int = 0,
        time_mask: int = 0,
        n_freq: int = 2,
        n_time: int = 2,
    ):
        super().__init__()
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.n_freq = n_freq
        self.n_time = n_time

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        mel = mel.clone()
        n_mels, n_frames = mel.shape[-2], mel.shape[-1]
        for _ in range(self.n_freq):
            if self.freq_mask > 0:
                f = int(torch.randint(0, self.freq_mask + 1, (1,)).item())
                if f:
                    f0 = int(torch.randint(0, max(1, n_mels - f), (1,)).item())
                    mel[..., f0 : f0 + f, :] = 0.0
        for _ in range(self.n_time):
            if self.time_mask > 0:
                t = int(torch.randint(0, self.time_mask + 1, (1,)).item())
                if t:
                    t0 = int(torch.randint(0, max(1, n_frames - t), (1,)).item())
                    mel[..., :, t0 : t0 + t] = 0.0
        return mel


def build_augment(cfg: Optional[dict]):
    """Return `(wave_aug, spec_aug)` from a TrainingConfig `augment` dict, or
    `(None, None)` when disabled."""
    if not cfg or not cfg.get("enabled", False):
        return None, None
    wave = WaveAugment(
        noise_std=cfg.get("noise_std", 0.0),
        gain_db=cfg.get("gain_db", 0.0),
        time_shift=cfg.get("time_shift", 0.0),
    )
    spec = SpecAugment(
        freq_mask=cfg.get("spec_freq_mask", 0),
        time_mask=cfg.get("spec_time_mask", 0),
        n_freq=cfg.get("spec_n_freq", 2),
        n_time=cfg.get("spec_n_time", 2),
    )
    return wave, spec
