"""End-to-end smoke test on synthetic audio.

Builds a tiny DCASE-style tree of sine-wave clips (2 machines x 1 id, normal
+ anomaly), runs one training epoch, evaluates, and checks that the full
pipeline produces finite AUC. No dataset download required.

Usage:
    uv run python scripts/training/smoke_test.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.pipelines.config import TrainingConfig  # noqa: E402
from src.pipelines.eval import evaluate, format_report  # noqa: E402
from src.pipelines.infer import load_model  # noqa: E402
from src.pipelines.train import train  # noqa: E402
from src.utils.model_utils import detect_device  # noqa: E402

SR = 16000
SECS = 10
MACHINES = ["fan", "pump"]


def write_wav(path: Path, freq: float, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(SR * SECS)) / SR
    rng = np.random.default_rng(seed)
    wav = 0.2 * np.sin(2 * np.pi * freq * t) + 0.01 * rng.standard_normal(t.shape)
    sf.write(str(path), wav.astype(np.float32), SR)


def build_tree(root: Path) -> None:
    for mi, machine in enumerate(MACHINES):
        for split in ("train", "test"):
            write_wav(
                root / machine / split / "normal_id_00_00000000.wav",
                220 + 40 * mi,
                10 + mi,
            )
        write_wav(
            root / machine / "test" / "anomaly_id_00_00000000.wav",
            500 + 80 * mi,
            20 + mi,
        )


def main() -> None:
    torch.manual_seed(0)
    tmp = Path(tempfile.mkdtemp(prefix="stgram_smoke_"))
    try:
        build_tree(tmp)
        cfg = TrainingConfig(
            data_root=str(tmp),
            machines=MACHINES,
            ckpt_dir=str(tmp / "checkpoints"),
            result_dir=str(tmp / "results"),
            epochs=1,
            batch_size=2,
            eval_every=1,
            ckpt_every=1,
            log_every=1,
            lr=1e-3,
            lr_scheduler={"type": "none"},
            save_best=True,
            best_metric="auc",
            best_mode="max",
        )
        train(cfg)

        best = tmp / "checkpoints"
        run_dirs = list(best.glob("*/best.pt"))
        if not run_dirs:
            raise RuntimeError("no best.pt produced")
        model, model_cfg, meta2label = load_model(run_dirs[0], detect_device())
        test_dirs = [str(tmp / m / "test") for m in MACHINES]
        result = evaluate(model, model_cfg, meta2label, test_dirs, detect_device())
        print(format_report(result))
        assert np.isfinite(result["auc"]), "AUC is not finite"
        print("\nsmoke test: PASS")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
