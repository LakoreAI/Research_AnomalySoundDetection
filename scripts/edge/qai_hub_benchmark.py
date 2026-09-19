"""Benchmark an exported STgram-MFN ONNX model on Qualcomm AI Hub devices.

Compiles + profiles the model on every available Snapdragon device (mobile,
IoT, compute, automotive) and writes a latency/memory table to JSON. This is
the Qualcomm-NPU counterpart to the TFLite-Micro arena check in
``scripts/edge/README.md``.

Prerequisites (local, not the training venv):
    pip install qai-hub
    # token from .env (QAI_HUB_API_TOKEN) or: qai-hub configure --api_token ...

Usage:
    uv run python scripts/edge/export_onnx.py --ckpt checkpoints/<run>/best.pt \
        --out export/stgram_mfn.onnx
    uv run python scripts/edge/qai_hub_benchmark.py --onnx export/stgram_mfn.onnx \
        --runtime tflite --out export/qai_hub_results.json

Notes:
    * AI Hub requires fixed input shapes (no dynamic batch), so the exported
      graph is profiled at batch 1.
    * ``label`` is an int64 input in the ONNX graph; if a target runtime rejects
      it, split the scorer or cast the input before re-export.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.config import MN01Config, STgramMFNConfig  # noqa: E402
from src.modules.mn01 import Mn01MelFrontend  # noqa: E402
from src.utils.io_utils import load_env  # noqa: E402


def _input_specs(backbone: str = "stgram") -> dict:
    """AI Hub requires explicit (shape, dtype) and concrete (non-dynamic)
    dims; `label` is int64 in the exported graphs."""
    if backbone == "mn01":
        # mn01 mel scorer takes x_mel (B, 1, n_mels, n_frames). Frames come from
        # the frontend itself (pre-emphasis drops one sample: 1000, not 1001).
        cfg = MN01Config()
        with torch.no_grad():
            mel = Mn01MelFrontend(cfg).eval()(
                torch.randn(1, int(cfg.secs * cfg.sample_rate))
            )
        return {
            "x_mel": ((1, 1, mel.shape[1], mel.shape[2]), "float32"),
            "label": ((1,), "int64"),
        }
    cfg = STgramMFNConfig()
    clip_samples = int(cfg.secs * cfg.sample_rate)
    return {
        "x_wav": ((1, clip_samples), "float32"),
        "x_mel": ((1, cfg.n_mels, cfg.n_frames), "float32"),
        "label": ((1,), "int64"),
    }


def _client():
    try:
        import qai_hub as hub
    except ImportError as e:  # pragma: no cover - optional dependency
        raise SystemExit("pip install qai-hub") from e

    import os

    token = os.getenv("QAI_HUB_API_TOKEN")
    if token:
        return hub, hub.Client(hub.ClientConfig(api_token=token))
    return hub, hub.Client()


def _all_devices(client, hub):
    for attr in ("get_devices", "devices"):
        fn = getattr(client, attr, None)
        if callable(fn):
            return list(fn())
        if fn is not None:
            return list(fn)
    fn = getattr(hub, "get_devices", None)
    if callable(fn):
        return list(fn())
    raise SystemExit("could not enumerate AI Hub devices (update qai-hub)")


def _profile_summary(profile: dict) -> dict:
    """Pull the headline numbers out of an AI Hub profile JSON, tolerating
    schema drift across client versions."""
    if not isinstance(profile, dict):
        return {}
    summary = profile.get("execution_summary") or profile
    out = {}
    for key in (
        "estimated_inference_time",
        "estimated_inference_time_unit",
        "estimated_inference_peak_memory",
        "estimated_inference_peak_memory_unit",
    ):
        if key in summary:
            out[key] = summary[key]
    return out


def _require_success(job, label: str) -> None:
    """Block until the job finishes; raise with the Hub's reason on failure."""
    job.wait()
    status = job.get_status()
    code = getattr(status, "code", None)
    if code == "SUCCESS" or getattr(status, "success", False):
        return
    reason = getattr(status, "message", None) or getattr(status, "failure", None)
    raise RuntimeError(f"{label} {code}: {reason or repr(status)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument(
        "--runtime",
        default="tflite",
        choices=["tflite", "onnx", "qnn_dlc", "qnn_context_binary"],
    )
    parser.add_argument(
        "--devices",
        nargs="*",
        default=None,
        help="device names; default = every device the account can access",
    )
    parser.add_argument(
        "--backbone",
        choices=["stgram", "mn01"],
        default="stgram",
        help="input contract of the ONNX graph",
    )
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "export" / "qai_hub_results.json"
    )
    args = parser.parse_args()

    load_env(REPO_ROOT / ".env")
    if not args.onnx.exists():
        raise FileNotFoundError(args.onnx)

    hub, client = _client()
    specs = _input_specs(args.backbone)

    if args.devices:
        devices = [hub.Device(name) for name in args.devices]
    else:
        devices = _all_devices(client, hub)
    print(
        f"profiling {args.onnx.name} on {len(devices)} device(s), runtime={args.runtime}"
    )

    results = []
    for device in devices:
        name = getattr(device, "name", str(device))
        row = {"device": name, "runtime": args.runtime, "status": "ok"}
        try:
            options = [f"--target_runtime {args.runtime}"]
            if args.runtime == "tflite":
                # The graph's `label` input is int64; TFLite requires explicit
                # permission to narrow 64-bit IO.
                options.append("--truncate_64bit_io")
            compile_job = client.submit_compile_job(
                model=args.onnx,
                device=device,
                input_specs=specs,
                options=" ".join(options),
            )
            _require_success(compile_job, "compile")
            target_model = compile_job.get_target_model()
            profile_job = client.submit_profile_job(model=target_model, device=device)
            _require_success(profile_job, "profile")
            row.update(_profile_summary(profile_job.download_profile()))
            row["compile_url"] = getattr(compile_job, "url", None)
            row["profile_url"] = getattr(profile_job, "url", None)
            print(
                f"  ok   {name}: {row.get('estimated_inference_time')} "
                f"{row.get('estimated_inference_time_unit', '')}"
            )
        except Exception as e:  # noqa: BLE001 - one bad device must not abort the sweep
            row["status"] = "error"
            row["error"] = str(e)[:300]
            print(f"  FAIL {name}: {row['error']}")
        results.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\nwrote {args.out}  ({ok}/{len(results)} devices profiled)")


if __name__ == "__main__":
    main()
