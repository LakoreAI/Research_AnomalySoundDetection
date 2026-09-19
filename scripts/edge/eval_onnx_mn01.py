"""Evaluate an exported mn01-swap ONNX scorer with the DCASE AUC protocol.

mn01 consumes raw 32 kHz audio (its frontend is inside the graph), so this is
the mn01 counterpart to `eval_onnx.py`. Used to measure the INT8 delta for the
mn01 swap (B1 vs B2).

Usage:
    uv run python scripts/edge/eval_onnx_mn01.py --onnx export/lean_B1ft.onnx \
        --data_root data/raw --add_root data/raw_eval --out export/eval_B1ft.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnxruntime as ort
import torch
from sklearn.metrics import roc_auc_score

# Cap threads to the pod's real vCPU count (host may report far more).
_THREADS = int(os.environ.get("EVAL_ONNX_THREADS", "8"))
torch.set_num_threads(_THREADS)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.config import MN01Config  # noqa: E402
from src.data import load_waveform  # noqa: E402
from src.modules.mn01 import Mn01MelFrontend  # noqa: E402
from src.utils.audio_utils import (  # noqa: E402
    create_test_file_list,
    get_machine_id_list,
    metadata_to_label,
)

MAX_FPR = 0.1


def _clip32k(path: str, cfg: MN01Config) -> np.ndarray:
    n = int(cfg.secs * cfg.sample_rate)
    wav = load_waveform(path, cfg.sample_rate)
    if wav.shape[0] < n:
        wav = torch.nn.functional.pad(wav, (0, n - wav.shape[0]))
    return wav[:n].numpy().astype(np.float32)


def _batched(items: List[str], n: int):
    for i in range(0, len(items), n):
        yield items[i : i + n]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument(
        "--add_root", type=Path, default=REPO_ROOT / "data" / "raw_eval"
    )
    parser.add_argument(
        "--machines",
        nargs="*",
        default=["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"],
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--providers", nargs="*", default=["CPUExecutionProvider"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    train_dirs = [str(args.data_root / m / "train") for m in args.machines]
    if args.add_root:
        train_dirs += [str(args.add_root / m / "train") for m in args.machines]
    meta2label, _ = metadata_to_label(train_dirs)
    test_dirs = [str(args.data_root / m / "test") for m in args.machines]

    cfg = MN01Config()
    frontend = Mn01MelFrontend(cfg).eval()
    so = ort.SessionOptions()
    so.intra_op_num_threads = _THREADS
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(
        str(args.onnx), sess_options=so, providers=args.providers
    )

    per_id: Dict[str, Dict[str, float]] = {}
    per_machine: Dict[str, Dict[str, float]] = {}
    aucs_by_m: Dict[str, List[float]] = {}
    paucs_by_m: Dict[str, List[float]] = {}

    for target_dir in sorted(test_dirs):
        machine_type = Path(target_dir).parent.name
        for id_str in get_machine_id_list(target_dir):
            label = meta2label[f"{machine_type}-{id_str}"]
            files, y_true = create_test_file_list(target_dir, id_str)
            if len(np.unique(y_true)) < 2:
                continue
            files = list(files)
            if args.limit:
                files = files[: args.limit]
                y_true = y_true[: args.limit]
            y_pred: List[float] = []
            for chunk in _batched(files, args.batch_size):
                with torch.no_grad():
                    # each frontend call -> (1, n_mels, n_frames); graph wants
                    # (B, 1, n_mels, n_frames)
                    mels = [
                        frontend(
                            torch.from_numpy(_clip32k(f, cfg)).unsqueeze(0)
                        ).numpy()[:, None, :, :]
                        for f in chunk
                    ]
                feed = {
                    "x_mel": np.concatenate(mels, axis=0).astype(np.float32),
                    "label": np.full((len(chunk),), label, dtype=np.int64),
                }
                score = sess.run(["score"], feed)[0].reshape(-1)
                y_pred.extend(score.tolist())
            auc = roc_auc_score(y_true, y_pred)
            pauc = roc_auc_score(y_true, y_pred, max_fpr=MAX_FPR)
            per_id[f"{machine_type}-{id_str}"] = {"auc": auc, "pauc": pauc}
            aucs_by_m.setdefault(machine_type, []).append(auc)
            paucs_by_m.setdefault(machine_type, []).append(pauc)

    aucs_m, paucs_m, maucs_m = [], [], []
    for machine_type, aucs in aucs_by_m.items():
        per_machine[machine_type] = {
            "auc": float(np.mean(aucs)),
            "pauc": float(np.mean(paucs_by_m[machine_type])),
            "mauc": float(np.min(aucs)),
        }
        aucs_m.append(float(np.mean(aucs)))
        paucs_m.append(float(np.mean(paucs_by_m[machine_type])))
        maucs_m.append(float(np.min(aucs)))

    result = {
        "onnx": str(args.onnx),
        "auc": float(np.mean(aucs_m)) if aucs_m else float("nan"),
        "pauc": float(np.mean(paucs_m)) if paucs_m else float("nan"),
        "mauc": float(np.mean(maucs_m)) if maucs_m else float("nan"),
        "per_machine": per_machine,
        "per_id": per_id,
    }
    print(
        f"{args.onnx.name:30s} AUC={result['auc'] * 100:6.3f}  "
        f"pAUC={result['pauc'] * 100:6.3f}  mAUC={result['mauc'] * 100:6.3f}"
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
