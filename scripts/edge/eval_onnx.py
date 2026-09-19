"""Evaluate an exported ONNX scorer (fp32 or INT8) with the DCASE AUC protocol.

Runs the ONNX graph's `score` output over `<data_root>/<machine>/test`, using
the same label mapping and AUC/pAUC/mAUC definitions as
`src.pipelines.eval.evaluate`. Used to measure the INT8 accuracy delta
(A1 vs A2, B1 vs B2).

Usage:
    uv run python scripts/edge/eval_onnx.py --onnx export/stgram_mfn.onnx \
        --data_root data/raw --add_root data/raw_eval --out export/eval_fp32.json

Requires `onnxruntime` (uv sync --extra edge).
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

# Keep CPU threading proportional to the container's real vCPU quota. Without
# this, onnxruntime spawns one thread per HOST core (e.g. 96) even when the pod
# has ~9 vCPU, which oversubscribes and pegs the CPU.
_THREADS = int(os.environ.get("EVAL_ONNX_THREADS", "8"))
torch.set_num_threads(_THREADS)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.config import STgramMFNConfig  # noqa: E402
from src.data import load_waveform  # noqa: E402
from src.modules.frontend import FeatureExtractor  # noqa: E402
from src.utils.audio_utils import (  # noqa: E402
    create_test_file_list,
    get_machine_id_list,
    metadata_to_label,
)

MAX_FPR = 0.1


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

    cfg = STgramMFNConfig(num_classes=len(meta2label))
    extractor = FeatureExtractor(cfg)
    so = ort.SessionOptions()
    so.intra_op_num_threads = _THREADS
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(
        str(args.onnx), sess_options=so, providers=args.providers
    )
    label_dtype = next(i.type for i in sess.get_inputs() if i.name == "label")

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
                wavs, mels = [], []
                for f in chunk:
                    wav = load_waveform(f, cfg.sample_rate)
                    x_wav, x_mel = extractor(wav)
                    wavs.append(x_wav.numpy())
                    mels.append(x_mel.numpy())
                feed = {
                    "x_wav": np.stack(wavs).astype(np.float32),
                    "x_mel": np.stack(mels).astype(np.float32),
                    "label": np.full((len(chunk),), label, dtype=np.int64),
                }
                if "int32" in label_dtype:
                    feed["label"] = feed["label"].astype(np.int32)
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
        f"{args.onnx.name:34s} AUC={result['auc'] * 100:6.3f}  "
        f"pAUC={result['pauc'] * 100:6.3f}  mAUC={result['mauc'] * 100:6.3f}"
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
