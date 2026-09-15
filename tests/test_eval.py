from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from src.config import STgramMFNConfig
from src.modules.frontend import FeatureExtractor
from src.modules.model import STgramMFN
from src.pipelines.eval import evaluate, score_file, score_files
from src.utils.audio_utils import metadata_to_label

SR = 16000
SECS = 1.0
N_FRAMES = 1 + int(SECS * SR) // 512
MACHINES = ["fan", "pump"]


def make_cfg() -> STgramMFNConfig:
    return STgramMFNConfig(
        num_classes=2,
        secs=SECS,
        n_frames=N_FRAMES,
        sample_rate=SR,
        n_fft=1024,
        n_mels=128,
        win_length=1024,
        hop_length=512,
        win_len=1024,
        hop_len=512,
        spatial_size=(8, 2),
    )


def write_wav(path: Path, freq: float, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(SR * SECS)) / SR
    rng = np.random.default_rng(seed)
    wav = 0.2 * np.sin(2 * np.pi * freq * t) + 0.01 * rng.standard_normal(t.shape)
    sf.write(str(path), wav.astype(np.float32), SR)


def make_tree(root: Path) -> None:
    for mi, machine in enumerate(MACHINES):
        for split in ("train", "test"):
            write_wav(
                root / machine / split / "normal_id_00_000.wav", 220 + 40 * mi, 10 + mi
            )
        write_wav(
            root / machine / "test" / "anomaly_id_00_000.wav", 500 + 80 * mi, 20 + mi
        )


def test_score_files_matches_score_file(tmp_path):
    make_tree(tmp_path)
    cfg = make_cfg()
    torch.manual_seed(0)
    model = STgramMFN(cfg).eval()
    extractor = FeatureExtractor(cfg)

    files = sorted(str(p) for p in (tmp_path / "fan" / "test").glob("*.wav"))
    batched = score_files(
        model, extractor, files, label=0, device=torch.device("cpu"), batch_size=2
    )
    singles = [score_file(model, extractor, f, 0, torch.device("cpu")) for f in files]
    assert np.allclose(batched, singles, atol=1e-5)


def test_evaluate_returns_finite_auc(tmp_path):
    make_tree(tmp_path)
    cfg = make_cfg()
    torch.manual_seed(0)
    model = STgramMFN(cfg).eval()
    train_dirs = [str(tmp_path / m / "train") for m in MACHINES]
    meta2label, _ = metadata_to_label(train_dirs)
    test_dirs = [str(tmp_path / m / "test") for m in MACHINES]

    result = evaluate(model, cfg, meta2label, test_dirs, torch.device("cpu"))
    assert np.isfinite(result["auc"])
    assert np.isfinite(result["pauc"])
    assert np.isfinite(result["mauc"])
    assert set(result["per_machine"]) == set(MACHINES)
    # mAUC is the worst-case id AUC, so it can never exceed the machine mean.
    for m in result["per_machine"].values():
        assert m["mauc"] <= m["auc"] + 1e-9
