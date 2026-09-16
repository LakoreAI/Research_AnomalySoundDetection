"""mn01 (EfficientAT width-0.1) wrapper tests.

The golden values below were produced by the real EfficientAT `mn01_as` model
on a fixed 1 s / 32 kHz input (`torch.manual_seed(0)`), so the parity test only
needs the checkpoint — no reference code or fixture file. Weights are loaded
from `checkpoints/mn01_as_mAP_298.pt` (or `$MN01_CHECKPOINT`); tests that need
them are skipped when the checkpoint is absent, and never hit the network.
"""

import os
from pathlib import Path

import pytest
import torch

from src.config import MN01Config
from src.modules.mn01 import MN01Embedder, build_backbone, load_mn01

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "mn01_as_mAP_298.pt"

MN01_PARAMS = 123_783
MN01_EMBED_DIM = 96

# Reference EfficientAT `mn01_as` output for `torch.manual_seed(0)` +
# `torch.randn(2, 32000)`: mel statistics and sample 0's 96-d embedding.
MEL_MEAN = 1.8049986362457275
MEL_STD = 0.7126471400260925
FEAT0 = [
    -0.195939,
    0.061876,
    0.090965,
    -0.048717,
    0.19724,
    -0.076393,
    -0.129614,
    0.695934,
    0.0398,
    -0.099228,
    0.4813,
    0.89461,
    -0.202361,
    0.88517,
    0.198834,
    0.861885,
    -0.116811,
    1.112522,
    1.541922,
    -0.022776,
    0.657143,
    -0.035858,
    -0.145949,
    0.646247,
    -0.083008,
    1.075039,
    1.20771,
    -0.00821,
    0.386599,
    0.420712,
    0.030433,
    -0.12415,
    -0.090633,
    -0.083422,
    -0.089888,
    1.864883,
    -0.09094,
    0.208876,
    0.393841,
    -0.009687,
    0.579364,
    -0.001794,
    0.465678,
    -0.137434,
    0.083671,
    -0.172686,
    -0.215291,
    0.081243,
    1.282777,
    0.334702,
    0.018369,
    0.912784,
    0.024072,
    0.15401,
    0.202035,
    -0.109728,
    1.127341,
    0.253807,
    3.12419,
    -0.111482,
    -0.086845,
    -0.187038,
    -0.047864,
    1.426477,
    -0.046007,
    0.131841,
    0.193348,
    0.932874,
    1.483676,
    0.064249,
    0.406061,
    -0.185242,
    0.18273,
    -0.06944,
    0.875849,
    1.50561,
    0.174425,
    0.677249,
    0.032101,
    -0.061349,
    1.665674,
    0.162508,
    1.731019,
    -0.042498,
    -0.002616,
    1.211146,
    1.367855,
    0.358141,
    -0.083916,
    -0.032171,
    0.697003,
    1.155195,
    0.298026,
    0.059457,
    -0.000114,
    0.428615,
]


def _checkpoint() -> str | None:
    for candidate in (os.environ.get("MN01_CHECKPOINT"), str(DEFAULT_CHECKPOINT)):
        if candidate and Path(candidate).exists():
            return candidate
    return None


@pytest.fixture(scope="module")
def pretrained() -> MN01Embedder:
    checkpoint = _checkpoint()
    if checkpoint is None:
        pytest.skip(
            "mn01 checkpoint not found; set $MN01_CHECKPOINT or add "
            "checkpoints/mn01_as_mAP_298.pt"
        )
    # Not wrapped in try/except: a real load failure must fail, not skip.
    return load_mn01(MN01Config(), checkpoint=checkpoint)


def test_param_count_matches_reference():
    model = MN01Embedder(MN01Config())
    assert sum(p.numel() for p in model.parameters()) == MN01_PARAMS


def test_embedding_dim_is_96():
    assert build_backbone(MN01Config()).embed_dim == MN01_EMBED_DIM


def test_frontend_and_embedding_shapes():
    model = MN01Embedder(MN01Config()).eval()
    waveform = torch.randn(2, 32000)
    with torch.no_grad():
        mel = model.frontend(waveform)
        emb = model(waveform)
    assert mel.shape == (2, MN01Config().n_mels, 100)
    assert emb.shape == (2, MN01_EMBED_DIM)
    assert torch.isfinite(emb).all()


def test_pretrained_weights_load_strictly(pretrained):
    # load_mn01 uses strict=True, so reaching here means every tensor matched:
    # the reimplemented module names/indices are compatible with the release.
    assert pretrained.embed_dim == MN01_EMBED_DIM


def test_parity_with_reference_golden(pretrained):
    torch.manual_seed(0)
    waveform = torch.randn(2, 32000)
    with torch.no_grad():
        mel = pretrained.frontend(waveform)
        emb = pretrained(waveform)
    assert mel.shape == (2, 128, 100)
    assert mel.mean().item() == pytest.approx(MEL_MEAN, abs=1e-4)
    assert mel.std().item() == pytest.approx(MEL_STD, abs=1e-4)
    assert torch.allclose(emb[0], torch.tensor(FEAT0), atol=1e-5)
