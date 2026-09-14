"""ONNX -> TensorFlow SavedModel -> TFLite (float and INT8) for edge deployment.

This is the bridge from the PyTorch/ONNX artifact to the TFLite Micro target.
It is OPTIONAL and depends on heavy packages that are not in the core
environment:

    uv pip install tensorflow onnx2tf onnx onnxruntime

Pipeline:
    onnx  --(onnx2tf)-->  saved_model  --(tf.lite converter)-->  .tflite
                                       (--int8 uses a representative dataset)

Usage:
    uv run python scripts/export_tflite.py --onnx export/stgram_mfn.onnx \
        --out_dir export/tflite
    uv run python scripts/export_tflite.py --onnx export/stgram_mfn.onnx \
        --out_dir export/tflite --int8 --calib_root data/raw --calib_samples 200

For TFLite Micro (ESP32 / Nano 33 BLE) you then convert the .tflite to a C
byte array and measure the arena — see scripts/edge/README.md.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import STgramMFNConfig  # noqa: E402
from src.data import load_waveform  # noqa: E402
from src.modules.frontend import FeatureExtractor  # noqa: E402
from src.utils.audio_utils import (  # noqa: E402
    build_train_file_list,
    machine_id_of_file,
    machine_of_file,
    metadata_to_label,
)


def check_deps() -> None:
    missing = []
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        missing.append("tensorflow")
    if shutil.which("onnx2tf") is None:
        missing.append("onnx2tf")
    if missing:
        raise SystemExit(
            "missing optional dependency(ies): "
            + ", ".join(missing)
            + "\ninstall with: uv pip install "
            + " ".join(missing)
        )


def onnx_to_saved_model(onnx_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["onnx2tf", "-i", str(onnx_path), "-o", str(out_dir), "-osd"]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    saved = out_dir / "saved_model"
    if not saved.exists():
        # onnx2tf names the subdir after the model stem in some versions
        candidates = list(out_dir.glob("saved_model*"))
        if not candidates:
            raise FileNotFoundError(f"no saved_model produced under {out_dir}")
        saved = candidates[0]
    return saved


def make_representative_dataset(calib_root: Path, machines, cfg, max_samples: int):
    """Yields [x_wav, x_mel, label] lists in the input order of the TFLite model
    (onnx2tf preserves the ONNX input order: x_wav, x_mel, label).
    """
    train_dirs = [
        str(calib_root / m / "train")
        for m in machines
        if (calib_root / m / "train").is_dir()
    ]
    meta2label, _ = metadata_to_label(train_dirs)
    extractor = FeatureExtractor(cfg)
    files = build_train_file_list(train_dirs)[:max_samples]

    def gen():
        for f in files:
            machine = machine_of_file(f)
            label = meta2label.get(f"{machine}-{machine_id_of_file(f)}", 0)
            wav = load_waveform(f, cfg.sample_rate)
            x_wav, x_mel = extractor(wav)
            yield [
                x_wav.unsqueeze(0).numpy().astype(np.float32),
                x_mel.unsqueeze(0).numpy().astype(np.float32),
                np.array([[label]], dtype=np.int64),
            ]

    return gen


def convert(saved_model: Path, out_path: Path, int8: bool, rep_gen=None) -> None:
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model))
    if int8:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if rep_gen is not None:
            converter.representative_dataset = rep_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    out_path.write_bytes(tflite_model)
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=REPO_ROOT / "export" / "tflite")
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--calib_root", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--calib_samples", type=int, default=200)
    parser.add_argument(
        "--machines",
        nargs="*",
        default=["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"],
    )
    args = parser.parse_args()

    check_deps()
    saved_model = onnx_to_saved_model(args.onnx, args.out_dir)

    rep_gen = None
    if args.int8:
        rep_gen = make_representative_dataset(
            args.calib_root, args.machines, STgramMFNConfig(), args.calib_samples
        )
    out_path = args.out_dir / (
        "stgram_mfn_int8.tflite" if args.int8 else "stgram_mfn.tflite"
    )
    convert(saved_model, out_path, args.int8, rep_gen)
    print("next: scripts/edge/README.md (TFLite Micro arena measurement)")


if __name__ == "__main__":
    main()
