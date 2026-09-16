"""INT8 post-training quantization (PTQ) of the exported ONNX scorer.

Two modes:
  dynamic (default) — weights quantized to INT8, activations dynamic. No
                      calibration data needed; quick baseline.
  static            — full INT8 with per-tensor scales calibrated on real
                      audio. Needs `--calib_root` pointing at the DCASE layout.

Both use ONNX Runtime's quantization toolkit:
    uv pip install onnx onnxruntime

Usage:
    uv run python scripts/edge/quantize_onnx.py --onnx export/stgram_mfn.onnx \
        --out export/stgram_mfn_int8.onnx --mode dynamic
    uv run python scripts/edge/quantize_onnx.py --onnx export/stgram_mfn.onnx \
        --out export/stgram_mfn_int8_static.onnx --mode static \
        --calib_root data/raw --calib_samples 200
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
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


def _cfg_from_onnx_dir() -> STgramMFNConfig:
    """Calibration uses the default architecture (the exported graph already
    fixes the shapes; this only needs matching audio params)."""
    return STgramMFNConfig()


class WavCalibrationReader:
    """Yields the three model inputs (x_wav, x_mel, label) for a sample of real
    normal clips, in the exact preprocessing the model was trained with.
    """

    def __init__(self, files, meta2label, cfg, max_samples: int):
        self.cfg = cfg
        self.extractor = FeatureExtractor(cfg)
        self.files = files[:max_samples]
        self.meta2label = meta2label
        self._iter = iter(self.files)

    def get_next(self):
        try:
            f = next(self._iter)
        except StopIteration:
            return None
        machine = machine_of_file(f)
        label = self.meta2label.get(f"{machine}-{machine_id_of_file(f)}", 0)
        wav = load_waveform(f, self.cfg.sample_rate)
        x_wav, x_mel = self.extractor(wav)
        return {
            "x_wav": x_wav.unsqueeze(0).numpy().astype(np.float32),
            "x_mel": x_mel.unsqueeze(0).numpy().astype(np.float32),
            "label": np.array([label], dtype=np.int64),
        }

    def rewind(self):
        self._iter = iter(self.files)


def dynamic_quantize(src: Path, dst: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)


def static_quantize(src: Path, dst: Path, reader) -> None:
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    quantize_static(
        str(src),
        str(dst),
        reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=["dynamic", "static"], default="dynamic")
    parser.add_argument("--calib_root", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--calib_samples", type=int, default=200)
    parser.add_argument(
        "--machines",
        nargs="*",
        default=["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"],
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "dynamic":
        dynamic_quantize(args.onnx, args.out)
    else:
        train_dirs = [
            str(args.calib_root / m / "train")
            for m in args.machines
            if (args.calib_root / m / "train").is_dir()
        ]
        if not train_dirs:
            raise FileNotFoundError(f"no calibration dirs under {args.calib_root}")
        meta2label, _ = metadata_to_label(train_dirs)
        cfg = _cfg_from_onnx_dir()
        reader = WavCalibrationReader(
            build_train_file_list(train_dirs), meta2label, cfg, args.calib_samples
        )
        static_quantize(args.onnx, args.out, reader)

    fp32 = args.onnx.stat().st_size / 1e6
    int8 = args.out.stat().st_size / 1e6
    print(f"quantized: {args.out}")
    print(f"  fp32: {fp32:.2f} MB -> int8: {int8:.2f} MB ({int8 / fp32 * 100:.1f}%)")


if __name__ == "__main__":
    main()
