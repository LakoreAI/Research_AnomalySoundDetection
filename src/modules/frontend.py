"""Waveform -> log-mel frontend.

The reference implementation computes the log-mel spectrogram (the "Sgram"
branch of STgram) with `torchaudio.transforms.MelSpectrogram` followed by
`AmplitudeToDB(stype="power")`. TgramNet produces the complementary "Tgram"
branch directly from the raw waveform; the two are concatenated channel-wise
inside `src.modules.model.STgramMFN`.

Keeping the mel transform in one place (rather than inline in the dataset, as
the reference does) means the dataset, inference, and any future quantization
calibration all derive features the same way.
"""

import torch
import torch.nn as nn
import torchaudio.transforms as T

from src.config import STgramMFNConfig


def frames_for_seconds(secs: float, sample_rate: int, hop_length: int) -> int:
    """Number of STFT frames for a `secs`-long clip with center=True padding:
    `1 + floor(secs*sample_rate / hop_length)`. Single source of truth for the
    TgramNet LayerNorm width and the MobileFaceNet spatial size.
    """
    return 1 + int(secs * sample_rate) // hop_length


class Wave2Mel(nn.Module):
    """Waveform (T,) -> log-mel (n_mels, n_frames). No normalization: the
    reference feeds `AmplitudeToDB` output straight into the network, and the
    quantization study needs the same unnormalized scale for both branches.
    """

    def __init__(self, cfg: STgramMFNConfig):
        super().__init__()
        self.mel = T.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            win_length=cfg.win_length,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            power=cfg.power,
        )
        self.to_db = T.AmplitudeToDB(stype="power")

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform: (T,) or (B, T) mono. Returns (n_mels, n_frames) or
        (B, n_mels, n_frames).
        """
        return self.to_db(self.mel(waveform))


class FeatureExtractor(nn.Module):
    """One clip -> (x_wav, x_mel) as the model expects them.

    Crops/pads to exactly `cfg.secs` seconds so every sample has the same
    (n_mels, n_frames) shape and no collate-time padding is needed (the
    reference truncates to `sr*secs`; short clips are zero-padded here rather
    than allowed to produce a shorter time axis, which would break TgramNet's
    fixed-width LayerNorm).
    """

    def __init__(self, cfg: STgramMFNConfig):
        super().__init__()
        self.cfg = cfg
        self.wav2mel = Wave2Mel(cfg)
        self.clip_samples = int(cfg.secs * cfg.sample_rate)

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """waveform: (T,) mono, already resampled to cfg.sample_rate.
        Returns (x_wav (clip_samples,), x_mel (n_mels, n_frames)).
        """
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=0)
        n = waveform.shape[0]
        if n < self.clip_samples:
            waveform = torch.nn.functional.pad(waveform, (0, self.clip_samples - n))
        else:
            waveform = waveform[: self.clip_samples]
        return waveform, self.wav2mel(waveform)
