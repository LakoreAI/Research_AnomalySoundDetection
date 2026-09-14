"""Evaluate a trained STgram-MFN checkpoint on the DCASE test split.

Reports AUC / pAUC per machine type and overall, optionally writing the
per-file anomaly scores and a JSON summary.

Usage:
    uv run python scripts/evaluate.py --ckpt checkpoints/<run>/best.pt \
        --data_root data/raw --save_dir results/<run>
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.pipelines.eval import evaluate, format_report  # noqa: E402
from src.pipelines.infer import load_model  # noqa: E402
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
    parser.add_argument("--save_dir", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    device = detect_device()
    model, cfg, meta2label = load_model(args.ckpt, device)
    test_dirs = [str(args.data_root / m / args.test_subdir) for m in args.machines]
    test_dirs = [d for d in test_dirs if Path(d).exists()]
    if not test_dirs:
        raise FileNotFoundError(f"no test dirs under {args.data_root}")

    result = evaluate(model, cfg, meta2label, test_dirs, device, save_dir=args.save_dir)
    print(format_report(result))

    if args.json:
        save_json(result, args.json)
        print(f"saved: {args.json}")


if __name__ == "__main__":
    main()
