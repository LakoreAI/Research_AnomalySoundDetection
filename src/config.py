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
from typing import Optional, Tuple


def _downsample2(n: int) -> int:
    """Output of a kernel-3, stride-2, pad-1 conv: ceil(n / 2)."""
    return (n - 1) // 2 + 1


def backbone_spatial_size(
    c_dim: int,
    n_frames: int,
    bottleneck_setting: Tuple[Tuple[int, int, int, int], ...],
) -> Tuple[int, int]:
    """Spatial (freq, time) size MobileFaceNet's depthwise collapsing conv sees.

    The input is `(c_dim, n_frames)`; `conv1` halves both, then each bottleneck
    entry with stride 2 halves both again. Reference: (128, 313) -> (8, 20).
    """
    f, t = _downsample2(c_dim), _downsample2(n_frames)
    for entry in bottleneck_setting:
        if entry[3] == 2:
            f, t = _downsample2(f), _downsample2(t)
    return f, t


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
    # MobileFaceNet's internal collapsing-conv width (conv2/linear7/linear1).
    # 512 is the reference; scale down for the small tiers.
    mfn_width: int = 512

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
    # Derived in __post_init__ from (c_dim, n_frames, bottleneck_setting):
    # the spatial size MobileFaceNet's depthwise collapsing conv must cover.
    # Reference: (128, 313) -> (8, 20).
    spatial_size: Optional[Tuple[int, int]] = None

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
        self.spatial_size = backbone_spatial_size(
            self.c_dim, self.n_frames, self.bottleneck_setting
        )


# EfficientAT `mn01_as` — MobileNetV3 width_mult=0.1, AudioSet mAP 29.8.
# Pinned 2026-09-16 (see docs/NOTES.md): 123,783 params, 96-d pooled output,
# and it must be fed through its OWN 32 kHz frontend, not STgram's 16 kHz one.
MN01_AS_URL = (
    "https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/"
    "mn01_as_mAP_298.pt"
)


@dataclass
class MN01Config:
    """EfficientAT mn01 (MobileNetV3 width 0.1) AudioSet embedder.

    Unlike STgram-MFN this is a *pretrained* network, so the frontend is part
    of the contract: the AudioSet weights only make sense when the input is a
    32 kHz waveform mel-transformed exactly the way EfficientAT does it
    (pre-emphasis, `win_length=800` / `hop_length=320`, Kaldi mel banks, then
    `log(x + 1e-5)` and `(x + 4.5) / 5`). MIMII is 16 kHz, so the mn01 row of
    the comparison matrix must resample up to `sample_rate` here.
    """

    # --- backbone ---
    width_mult: float = 0.1
    num_classes: int = 527  # AudioSet classes; the MLP head is dropped for ASD

    # --- frontend (EfficientAT `AugmentMelSTFT`, eval settings) ---
    sample_rate: int = 32000
    n_mels: int = 128
    n_fft: int = 1024
    win_length: int = 800
    hop_length: int = 320
    fmin: float = 0.0
    fmax: float = 15000.0
    secs: float = 10.0

    pretrained_url: str = MN01_AS_URL

    @property
    def n_frames(self) -> int:
        """STFT frames for a `secs`-long clip at this hop (shape bookkeeping)."""
        return 1 + int(self.secs * self.sample_rate) // self.hop_length


# --- mn01 family (Stage 8 tier ladder) --------------------------------------
# EfficientAT width multipliers (helpers/utils.py:NAME_TO_WIDTH) and the
# AudioSet checkpoints on the v0.0.1 release. Only widths that actually ship an
# `*_as` checkpoint are listed (mn06/mn08/mn12/... are ImageNet-only there).
MN01_WIDTHS = {
    "mn01": 0.1,
    "mn02": 0.2,
    "mn04": 0.4,
    "mn05": 0.5,
    "mn10": 1.0,
    "mn20": 2.0,
    "mn30": 3.0,
    "mn40": 4.0,
}
MN01_AS_ASSETS = {
    "mn01": "mn01_as_mAP_298.pt",
    "mn02": "mn02_as_mAP_378.pt",
    "mn04": "mn04_as_mAP_432.pt",
    "mn05": "mn05_as_mAP_443.pt",
    "mn10": "mn10_as_mAP_471.pt",
    "mn20": "mn20_as_mAP_478.pt",
    "mn30": "mn30_as_mAP_482.pt",
    "mn40": "mn40_as_mAP_484.pt",
}
MN01_RELEASE_URL = "https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/"


def mn01_config(name: str = "mn01") -> MN01Config:
    """Build an `MN01Config` for an EfficientAT width name (mn01..mn40).

    Raises for widths without an AudioSet checkpoint (the weights are width-
    specific, so a mismatched checkpoint would fail `strict=True` loading).
    """
    if name not in MN01_WIDTHS:
        raise ValueError(
            f"unknown mn01 name {name!r}; choose from {sorted(MN01_WIDTHS)}"
        )
    if name not in MN01_AS_ASSETS:
        raise ValueError(
            f"no AudioSet checkpoint for {name!r}; available: {sorted(MN01_AS_ASSETS)}"
        )
    return MN01Config(
        width_mult=MN01_WIDTHS[name],
        pretrained_url=MN01_RELEASE_URL + MN01_AS_ASSETS[name],
    )
