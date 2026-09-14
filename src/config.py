"""Model and audio architecture for STgram-MFN.

This is the ARCHITECTURE-ONLY config — model layers, audio preprocessing, and
loss geometry. Training-loop hyperparameters (epochs, lr, data dirs, callback
wiring) live in `src.pipelines.config.TrainingConfig`, mirroring the split
used by the reference codebase this project is modelled on.

Provenance: the architecture follows the DCASE 2022 Task 2 Top-1 system
(Liu et al., "Anomalous Sound Detection using Spectral-Temporal Information
Fusion", ICASSP 2022 / arXiv:2201.05510). The reference implementation is
github.com/liuyoude/STgram-MFN; this module is a faithful re-expression of
`net.py` from that repo with the hardcoded shapes lifted into config.
"""

from dataclasses import dataclass, field
from typing import Tuple

# DCASE 2022 Task 2 Top-1 bottleneck setting, quoted verbatim from the
# reference `net.py` (it supersedes the original MobileFaceNet setting and the
# MobileNetV2 setting, both left commented out in the reference). Each entry is
# (expansion t, out channels c, repeats n, stride s).
DEFAULT_BOTTLENECK_SETTING: Tuple[Tuple[int, int, int, int], ...] = (
    (2, 128, 2, 2),
    (4, 128, 2, 2),
    (4, 128, 2, 2),
)


@dataclass
class STgramMFNConfig:
    # --- classifier head ---
    num_classes: int = 0  # set at runtime from the machine-id metadata
    embed_dim: int = 128  # MobileFaceNet's penultimate feature width

    # --- spectral-temporal frontend ---
    # `c_dim` is BOTH the number of mel bins fed to the Sgram branch and the
    # number of 1-D channels TgramNet produces, so the two branches stack
    # cleanly into the 2-channel (B, 2, n_mels, n_frames) input MobileFaceNet
    # expects.
    c_dim: int = 128
    win_len: int = 1024  # TgramNet's large-kernel 1-D conv kernel/stride
    hop_len: int = 512
    tgram_num_layers: int = 3  # reference default
    # Number of time frames a `secs`-long clip yields at this hop. Kept explicit
    # because TgramNet's LayerNorm is applied over the TIME axis (the reference
    # hardcodes 313); deriving it keeps the module correct if the frontend
    # changes instead of silently shape-erroring.
    n_frames: int = 313
    # Spatial size MobileFaceNet's depthwise collapsing conv sees after the
    # three stride-2 stages. Reference hardcodes (8, 20) for a 128x313 input.
    spatial_size: Tuple[int, int] = (8, 20)

    bottleneck_setting: Tuple[Tuple[int, int, int, int], ...] = field(
        default_factory=lambda: DEFAULT_BOTTLENECK_SETTING
    )

    # --- ArcFace geometry ---
    use_arcface: bool = True
    arcface_m: float = 0.7  # additive angular margin
    arcface_s: float = 30.0  # logit scale
    arcface_sub: int = 1  # sub-center count (1 = plain ArcFace)
    arcface_easy_margin: bool = False

    # --- audio frontend ---
    sample_rate: int = 16000
    n_fft: int = 1024
    n_mels: int = 128
    win_length: int = 1024
    hop_length: int = 512
    power: float = 2.0
    secs: float = 10.0  # clip length used for both training and scoring

    def __post_init__(self) -> None:
        if self.n_mels != self.c_dim:
            raise ValueError(
                f"n_mels ({self.n_mels}) must equal c_dim ({self.c_dim}) — "
                "the Sgram and Tgram branches are concatenated channel-wise"
            )
        if self.win_length != self.win_len or self.hop_length != self.hop_len:
            raise ValueError(
                "win_length/hop_length (audio) must match win_len/hop_len "
                "(TgramNet) — the reference ties them so Tgram and Sgram share "
                "a time axis"
            )
