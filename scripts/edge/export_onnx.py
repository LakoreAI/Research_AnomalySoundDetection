"""Export a trained STgram-MFN checkpoint to ONNX for edge deployment.

Exports the `STgramMFNScorer` graph: inputs `(x_wav, x_mel, label)`, outputs
`(feature, score)`. The time axis is FIXED (TgramNet's LayerNorm is sized to
`cfg.n_frames`); only the batch axis is dynamic.

Usage:
    uv run python scripts/edge/export_onnx.py --ckpt checkpoints/<run>/best.pt \
        --out export/stgram_mfn.onnx

Requires the `onnx` package (install the edge extra):
    uv sync --extra edge
"""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.modules.edge import STgramMFNScorer  # noqa: E402
from src.pipelines.infer import load_model  # noqa: E402
from src.utils.model_utils import detect_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "export" / "stgram_mfn.onnx"
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--verify", action="store_true", help="run onnx.checker if available"
    )
    args = parser.parse_args()

    device = detect_device()
    model, cfg, _ = load_model(args.ckpt, device)
    scorer = STgramMFNScorer(model).eval().to(device)

    clip_samples = int(cfg.secs * cfg.sample_rate)
    x_wav = torch.randn(1, clip_samples, device=device)
    x_mel = torch.randn(1, cfg.n_mels, cfg.n_frames, device=device)
    label = torch.zeros(1, dtype=torch.long, device=device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        scorer,
        (x_wav, x_mel, label),
        str(args.out),
        input_names=["x_wav", "x_mel", "label"],
        output_names=["feature", "score"],
        dynamic_axes={
            "x_wav": {0: "batch"},
            "x_mel": {0: "batch"},
            "label": {0: "batch"},
        },
        opset_version=args.opset,
        # Legacy TorchScript exporter: avoids the onnxscript dependency the
        # dynamo exporter needs. Equivalent output for this feed-forward graph.
        dynamo=False,
    )
    print(f"exported: {args.out}  ({args.out.stat().st_size / 1e6:.2f} MB)")

    if args.verify:
        try:
            import onnx

            onnx.checker.check_model(str(args.out))
            print("onnx.checker: OK")
        except ImportError:
            print("onnx not installed — skipping verification (uv pip install onnx)")


if __name__ == "__main__":
    main()
