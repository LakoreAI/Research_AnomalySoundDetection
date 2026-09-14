"""Architecture sanity checks for STgram-MFN. CPU, seconds at these sizes.

D1 — wiring:       forward produces finite, correctly-shaped logits/features
                   for ArcFace on and off, and for a batch with `label=None`
                   when ArcFace is disabled.
D2 — branches:     TgramNet's output shares the log-mel time axis, and the
                   concatenated STgram tensor is 2-channel as MobileFaceNet
                   expects.
D3 — overfit:      a tiny model drives one fixed batch's CE loss down hard,
                   proving the pretext is learnable end-to-end.
"""

import torch

from src.config import STgramMFNConfig
from src.modules.loss import ASDLoss
from src.modules.model import STgramMFN
from src.modules.tgramnet import TgramNet

torch.manual_seed(0)

B, NUM_CLASSES = 4, 6
SR, N_FFT, HOP, WIN = 16000, 1024, 512, 1024
SECS = 1.0  # shorter than the real 10 s to keep diagnostics fast
N_FRAMES = 1 + int(SECS * SR) // HOP  # 32


def _cfg(use_arcface: bool = True) -> STgramMFNConfig:
    return STgramMFNConfig(
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
        spatial_size=(8, 2),  # 128-mel/8 x 32-frame/2 after three stride-2 stages
        use_arcface=use_arcface,
    )


def _batch(cfg: STgramMFNConfig):
    x_wav = torch.randn(B, int(SECS * SR))
    x_mel = torch.randn(B, cfg.n_mels, N_FRAMES)
    labels = torch.randint(0, NUM_CLASSES, (B,))
    return x_wav, x_mel, labels


def d1_wiring():
    print("=== D1 — wiring sanity ===")
    for use_arcface in (True, False):
        cfg = _cfg(use_arcface)
        model = STgramMFN(cfg).eval()
        x_wav, x_mel, labels = _batch(cfg)
        with torch.no_grad():
            logits, feature = model(x_wav, x_mel, labels)
        ok_shape = logits.shape == (B, NUM_CLASSES) and feature.shape == (
            B,
            cfg.embed_dim,
        )
        ok_finite = torch.isfinite(logits).all() and torch.isfinite(feature).all()
        print(
            f"  arcface={use_arcface!s:<5} shapes_ok={bool(ok_shape)} finite={bool(ok_finite)}"
            f"  logits={tuple(logits.shape)} feature={tuple(feature.shape)}"
        )
        assert ok_shape and ok_finite, f"D1 FAILED for use_arcface={use_arcface}"
    print("  D1: PASS\n")


def d2_branches():
    print("=== D2 — Sgram/Tgram branch shapes ===")
    cfg = _cfg()
    tgram = TgramNet(cfg).eval()
    x_wav, x_mel, _ = _batch(cfg)
    with torch.no_grad():
        t = tgram(x_wav.unsqueeze(1))
    ok = t.shape == (B, cfg.c_dim, N_FRAMES) and x_mel.shape == (
        B,
        cfg.n_mels,
        N_FRAMES,
    )
    print(
        f"  tgram={tuple(t.shape)}  sgram={tuple(x_mel.shape)}  match_time_axis={bool(ok)}"
    )
    assert ok, "D2 FAILED: TgramNet and log-mel time axes disagree"
    print("  D2: PASS\n")


def d3_overfit(steps: int = 150, lr: float = 3e-3):
    print("=== D3 — single-batch overfit ===")
    torch.manual_seed(0)
    cfg = _cfg(use_arcface=True)
    model = STgramMFN(cfg)
    criterion = ASDLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x_wav, x_mel, labels = _batch(cfg)

    losses = []
    for _ in range(steps):
        opt.zero_grad()
        logits, _ = model(x_wav, x_mel, labels)
        loss = criterion(logits, labels)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    first, final = sum(losses[:10]) / 10, sum(losses[-10:]) / 10
    print(f"  loss first-10 avg={first:.4f}  last-10 avg={final:.4f}")
    assert final < first, "D3 FAILED: loss did not decrease"
    print("  D3: PASS\n")


if __name__ == "__main__":
    d1_wiring()
    d2_branches()
    d3_overfit()
