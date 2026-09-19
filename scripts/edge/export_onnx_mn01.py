"""Export a trained mn01-swap checkpoint to ONNX for edge deployment.

Exports `MN01MelScorer` (mn01 conv backbone + ArcFace + anomaly score): inputs
`(x_mel, label)`, outputs `(feature, score)`. The 32 kHz mel frontend is kept
OUTSIDE the graph — its complex STFT is not ONNX-exportable — and must be
reproduced on-device (same situation as STgram-MFN's Sgram), or applied with
`Mn01MelFrontend` for evaluation.

Usage:
    uv run python scripts/edge/export_onnx_mn01.py \
        --ckpt checkpoints/lean_B1ft_mn01/best.pt --out export/lean_B1ft.onnx
"""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.config import mn01_config  # noqa: E402
from src.modules.mn01 import Mn01MelFrontend  # noqa: E402
from src.modules.mn01_head import MN01ArcFace, MN01MelScorer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "export" / "mn01.onnx")
    parser.add_argument("--mn01_name", type=str, default="mn01")
    parser.add_argument("--mn01_ckpt", type=str, default=None)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    raw = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = raw.get("extra", {}).get("cfg", {})
    num_classes = len(cfg.get("meta2label", {}))
    if num_classes == 0:
        raise ValueError(f"checkpoint {args.ckpt} has no meta2label in extra.cfg")

    mn_cfg = mn01_config(args.mn01_name)
    model = MN01ArcFace(mn_cfg, num_classes, checkpoint=args.mn01_ckpt)
    model.load_state_dict(raw["model"])
    scorer = MN01MelScorer(model).eval()

    # Use the frontend's ACTUAL output shape: mn01's pre-emphasis shortens the
    # signal by one sample, so frames = 1000 for a 10 s/32 kHz clip, not the
    # config property's 1001.
    n = int(mn_cfg.secs * mn_cfg.sample_rate)
    with torch.no_grad():
        x_mel = Mn01MelFrontend(mn_cfg).eval()(torch.randn(1, n)).unsqueeze(1)
    label = torch.zeros(1, dtype=torch.long)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        scorer,
        (x_mel, label),
        str(args.out),
        input_names=["x_mel", "label"],
        output_names=["feature", "score"],
        dynamic_axes={"x_mel": {0: "batch"}, "label": {0: "batch"}},
        opset_version=args.opset,
        dynamo=False,
    )
    print(f"exported: {args.out}  ({args.out.stat().st_size / 1e6:.2f} MB)")

    if args.verify:
        import onnx

        onnx.checker.check_model(str(args.out))
        print("onnx.checker: OK")


if __name__ == "__main__":
    main()
