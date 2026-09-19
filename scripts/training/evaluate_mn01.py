"""Evaluate a trained mn01-swap checkpoint on an arbitrary test split.

The counterpart to `evaluate.py` for backbone B. The label map is read from the
checkpoint, so this can score any DCASE-layout directory (e.g. the held-out
evaluation campaign with `--data_root data/raw_eval`).

Usage:
    uv run python scripts/training/evaluate_mn01.py \
        --ckpt checkpoints/lean_B1ft_mn01/best.pt --data_root data/raw_eval
"""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.training.train_mn01 import evaluate_mn01_ft  # noqa: E402
from src.config import mn01_config  # noqa: E402
from src.modules.mn01_head import MN01ArcFace  # noqa: E402
from src.utils.io_utils import save_json  # noqa: E402
from src.utils.model_utils import detect_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument(
        "--machines",
        nargs="*",
        default=["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"],
    )
    parser.add_argument("--test_subdir", type=str, default="test")
    parser.add_argument("--mn01_name", type=str, default="mn01")
    parser.add_argument("--mn01_ckpt", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    device = detect_device()
    raw = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = raw.get("extra", {}).get("cfg", {})
    meta2label = cfg["meta2label"]
    model = MN01ArcFace(mn01_config(args.mn01_name), len(meta2label), args.mn01_ckpt)
    model.load_state_dict(raw["model"])
    model.to(device)

    test_dirs = [str(args.data_root / m / args.test_subdir) for m in args.machines]
    test_dirs = [d for d in test_dirs if Path(d).exists()]
    if not test_dirs:
        raise FileNotFoundError(f"no test dirs under {args.data_root}")

    res = evaluate_mn01_ft(
        model, meta2label, test_dirs, device, num_workers=args.num_workers
    )
    print(
        f"{args.ckpt.parent.name}: AUC={res['auc'] * 100:.3f} "
        f"pAUC={res['pauc'] * 100:.3f} mAUC={res['mauc'] * 100:.3f}"
    )
    for machine, m in sorted(res["per_machine"].items()):
        print(f"  {machine:14s} AUC={m['auc'] * 100:6.3f}  mAUC={m['mauc'] * 100:6.3f}")
    if args.json:
        save_json(res, args.json)
        print(f"saved: {args.json}")


if __name__ == "__main__":
    main()
