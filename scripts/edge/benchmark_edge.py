"""Edge-readiness benchmark for STgram-MFN.

Reports, for a checkpoint and (optionally) its ONNX fp32/INT8 exports:
  * parameter count and fp32 weight size
  * CPU latency (mean/std, warmup + N runs) at batch 1
  * a rough activation-memory estimate (sum of unique forward tensor outputs)
    as an early proxy for the TFLite Micro arena

Latency is measured with `intra_op_num_threads=1` for ONNX (single-core, closer
to MCU conditions). Real arena sizing still needs the TFLite Micro
`--print_arena` path — see scripts/edge/README.md.

Usage:
    uv run python scripts/edge/benchmark_edge.py --ckpt checkpoints/<run>/best.pt
    uv run python scripts/edge/benchmark_edge.py --ckpt ... \
        --onnx export/stgram_mfn.onnx --onnx_int8 export/stgram_mfn_int8.onnx
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.modules.edge import STgramMFNScorer  # noqa: E402
from src.pipelines.infer import load_model  # noqa: E402


def activation_bytes(model: torch.nn.Module, inputs) -> int:
    """Sum of unique forward-tensor output sizes (bytes, fp32). A loose upper
    bound on peak activation memory — not the TFLite Micro arena number.
    """
    seen = set()
    total = 0

    def hook(_module, _inp, out):
        nonlocal total
        tensors = out if isinstance(out, (tuple, list)) else (out,)
        for t in tensors:
            if isinstance(t, torch.Tensor) and id(t) not in seen:
                seen.add(id(t))
                total += t.numel() * 4

    handles = [m.register_forward_hook(hook) for m in model.modules()]
    with torch.no_grad():
        model(*inputs)
    for h in handles:
        h.remove()
    return total


def torch_latency(model, inputs, runs: int, warmup: int = 5):
    with torch.no_grad():
        for _ in range(warmup):
            model(*inputs)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(*inputs)
            times.append((time.perf_counter() - t0) * 1e3)
    arr = np.array(times)
    return arr.mean(), arr.std()


def onnx_latency(onnx_path: Path, feeds, runs: int, warmup: int = 5):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    sess = ort.InferenceSession(
        str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
    )
    for _ in range(warmup):
        sess.run(None, feeds)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, feeds)
        times.append((time.perf_counter() - t0) * 1e3)
    arr = np.array(times)
    return arr.mean(), arr.std()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, default=None)
    parser.add_argument("--onnx_int8", type=Path, default=None)
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cpu")
    model, cfg, _ = load_model(args.ckpt, device)
    scorer = STgramMFNScorer(model).eval()

    clip = int(cfg.secs * cfg.sample_rate)
    x_wav = torch.randn(1, clip)
    x_mel = torch.randn(1, cfg.n_mels, cfg.n_frames)
    label = torch.zeros(1, dtype=torch.long)
    inputs = (x_wav, x_mel, label)

    n_params = sum(p.numel() for p in scorer.parameters())
    print("=== STgram-MFN edge benchmark ===")
    print(f"params: {n_params:,}  fp32 weights: {n_params * 4 / 1e6:.2f} MB")

    act = activation_bytes(scorer, inputs)
    print(f"activation estimate (sum of forward outputs, fp32): {act / 1e6:.2f} MB")

    mean, std = torch_latency(scorer, inputs, args.runs)
    print(f"torch CPU latency (batch 1): {mean:.2f} +/- {std:.2f} ms")

    if args.onnx is not None:
        feeds = {
            "x_wav": x_wav.numpy(),
            "x_mel": x_mel.numpy(),
            "label": label.numpy(),
        }
        try:
            mean, std = onnx_latency(args.onnx, feeds, args.runs)
            size = args.onnx.stat().st_size / 1e6
            print(f"onnx fp32 ({size:.2f} MB) latency: {mean:.2f} +/- {std:.2f} ms")
        except ImportError:
            print("onnxruntime not installed — skipping ONNX benchmark")

    if args.onnx_int8 is not None:
        feeds = {
            "x_wav": x_wav.numpy(),
            "x_mel": x_mel.numpy(),
            "label": label.numpy(),
        }
        try:
            mean, std = onnx_latency(args.onnx_int8, feeds, args.runs)
            size = args.onnx_int8.stat().st_size / 1e6
            print(f"onnx int8 ({size:.2f} MB) latency: {mean:.2f} +/- {std:.2f} ms")
        except ImportError:
            print("onnxruntime not installed — skipping INT8 benchmark")


if __name__ == "__main__":
    main()
