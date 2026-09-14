import pytest
import torch

from src.config import STgramMFNConfig
from src.modules.model import STgramMFN
from src.modules.tgramnet import TgramNet

B, NUM_CLASSES = 3, 5
SR, N_FFT, HOP, WIN = 16000, 1024, 512, 1024
SECS = 1.0
N_FRAMES = 1 + int(SECS * SR) // HOP


def make_cfg(use_arcface: bool = True, **overrides) -> STgramMFNConfig:
    base = dict(
        num_classes=NUM_CLASSES,
        n_frames=N_FRAMES,
        secs=SECS,
        sample_rate=SR,
        n_fft=N_FFT,
        n_mels=128,
        win_length=WIN,
        hop_length=HOP,
        win_len=WIN,
        hop_len=HOP,
        spatial_size=(8, 2),
        use_arcface=use_arcface,
    )
    base.update(overrides)
    return STgramMFNConfig(**base)


def make_batch(cfg):
    x_wav = torch.randn(B, int(SECS * SR))
    x_mel = torch.randn(B, cfg.n_mels, N_FRAMES)
    labels = torch.randint(0, NUM_CLASSES, (B,))
    return x_wav, x_mel, labels


def test_forward_shapes_with_arcface():
    cfg = make_cfg(use_arcface=True)
    model = STgramMFN(cfg).eval()
    x_wav, x_mel, labels = make_batch(cfg)
    with torch.no_grad():
        logits, feature = model(x_wav, x_mel, labels)
    assert logits.shape == (B, NUM_CLASSES)
    assert feature.shape == (B, cfg.embed_dim)
    assert torch.isfinite(logits).all()


def test_forward_shapes_without_arcface():
    cfg = make_cfg(use_arcface=False)
    model = STgramMFN(cfg).eval()
    x_wav, x_mel, _ = make_batch(cfg)
    with torch.no_grad():
        logits, feature = model(x_wav, x_mel, None)
    assert logits.shape == (B, NUM_CLASSES)
    assert feature.shape == (B, cfg.embed_dim)


def test_arcface_requires_label():
    cfg = make_cfg(use_arcface=True)
    model = STgramMFN(cfg).eval()
    x_wav, x_mel, _ = make_batch(cfg)
    with pytest.raises(ValueError):
        model(x_wav, x_mel, None)


def test_tgram_shares_mel_time_axis():
    cfg = make_cfg()
    tgram = TgramNet(cfg).eval()
    x_wav = torch.randn(B, int(SECS * SR))
    with torch.no_grad():
        t = tgram(x_wav.unsqueeze(1))
    assert t.shape == (B, cfg.c_dim, N_FRAMES)


def test_get_tgram_accepts_raw_waveform():
    cfg = make_cfg()
    model = STgramMFN(cfg).eval()
    x_wav = torch.randn(B, int(SECS * SR))
    with torch.no_grad():
        t = model.get_tgram(x_wav)
    assert t.shape == (B, cfg.c_dim, N_FRAMES)


def test_config_rejects_mismatched_mels():
    with pytest.raises(ValueError):
        make_cfg(n_mels=64, c_dim=128)
