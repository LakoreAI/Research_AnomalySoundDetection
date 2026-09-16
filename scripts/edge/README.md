# Edge deployment path (ESP32 / Arduino Nano 33 BLE)

STgram-MFN is small (~1.16M params, ~4.6 MB fp32), but the Tgram branch's
large-kernel 1-D conv and the 128x313 spectrogram are heavy for MCU-class
hardware. This directory documents the conversion and measurement path; the
core repo only depends on PyTorch, so the conversion tools below are optional.

## 1. PyTorch -> ONNX

```bash
uv run python scripts/edge/export_onnx.py --ckpt checkpoints/<run>/best.pt \
    --out export/stgram_mfn.onnx --verify
```

Exports `STgramMFNScorer`: inputs `(x_wav, x_mel, label)`, outputs
`(feature, score)`. The time axis is fixed; only batch is dynamic.

## 2. INT8 PTQ (ONNX Runtime)

```bash
uv pip install onnx onnxruntime
# dynamic (weights INT8, no calibration)
uv run python scripts/edge/quantize_onnx.py --onnx export/stgram_mfn.onnx \
    --out export/stgram_mfn_int8.onnx --mode dynamic
# static (full INT8, calibrated on real clips)
uv run python scripts/edge/quantize_onnx.py --onnx export/stgram_mfn.onnx \
    --out export/stgram_mfn_int8_static.onnx --mode static \
    --calib_root data/raw --calib_samples 200
```

This is the primary INT8 PTQ path. If accuracy loss is unacceptable, the
fallback is QAT (train with fake-quant observers) — not yet implemented here.

## 3. ONNX -> TFLite

```bash
uv pip install tensorflow onnx2tf
uv run python scripts/edge/export_tflite.py --onnx export/stgram_mfn.onnx \
    --out_dir export/tflite --int8 --calib_root data/raw --calib_samples 200
```

Produces `export/tflite/stgram_mfn.tflite` and, with `--int8`, a full
integer-quantized `stgram_mfn_int8.tflite` using a representative dataset of
real normal clips. TFLite Micro requires **full-integer** quantization — the
dynamic-weight INT8 ONNX file is not directly convertible.

## 4. TFLite -> C array

```bash
xxd -i export/tflite/stgram_mfn_int8.tflite > model_data.cc
```

## 5. Measure the arena (do this early, on x86)

The project's stated policy is to validate the memory footprint before touching
hardware. Two options:

- **tflite-micro Python interpreter** (`pip install tflite-micro`), which
  reports the arena size it needs:
  ```python
  from tflite_micro import runtime
  interp = runtime.Interpreter.from_bytes(open("model_int8.tflite","rb").read())
  print(interp.arena_size)  # bytes
  ```
- **C++ `MicroInterpreter`** with the `--print_arena` / `GetTensorArena` path,
  or the `tflite-micro` benchmark binary built for the target.

`scripts/edge/benchmark_edge.py` gives a PyTorch/ONNX-side estimate (parameter size,
CPU latency, activation-output sum) as a first screen, but the interpreter's
reported arena is the number that must fit the ESP32's SRAM.

## 6. Target notes

- **ESP32**: ~520 KB SRAM (no PSRAM). Arena must fit alongside the interpreter
  and audio buffers. If it does not, shrink `n_mels` / `n_frames` / `c_dim` in
  `STgramMFNConfig`, or replace the Tgram branch with a fixed STFT front-end.
- **Arduino Nano 33 BLE Sense**: nRF52840, 256 KB RAM, 1 MB flash — the
  tighter target; expect to need a reduced STgram-MFN variant.
- The MelSpectrogram/AmplitudeToDB preprocessing must be reimplemented in C
  (or baked into the graph) on-device — it is not part of the exported model.
