# Report 001 — Qualcomm AI Hub edge profiling (both backbones)

- **Date:** 2026-09-19
- **Milestone:** M4 (Domain Shift + Edge)
- **Status:** partial — NPU latency measured; TFLite-Micro arena blocked
- **Task:** reports / edge

## What ran

The INT8 ONNX scorers for both backbones were compiled to TFLite and profiled
through Qualcomm AI Hub on five Snapdragon devices (`scripts/edge/
qai_hub_benchmark.py`, `--backbone {stgram,mn01}`). Estimated single-inference
latency, microseconds converted to milliseconds:

| Device (SoC) | STgram-MFN (INT8) | mn01 (INT8) | ratio |
|---|---|---|---|
| Galaxy S21 (SM8350) | 7.76 ms | 7.92 ms | 1.0× |
| Galaxy S23 (SM8550) | 9.85 ms | **49.83 ms** | 5.1× |
| Galaxy S24 (SM8650) | 7.00 ms | **36.77 ms** | 5.3× |
| Galaxy S25 (SM8750) | 5.75 ms | **33.47 ms** | 5.8× |
| QCS8550 (IoT)       | 9.87 ms | **50.08 ms** | 5.1× |

Artifacts: `export/qai_A1.json`, `export/qai_B1ft.json`.

## Finding

**The smaller model is much slower on the NPU.** mn01 is roughly nine times
smaller in parameters, yet on four of five devices its quantized TFLite graph
runs three to five times slower than STgram-MFN's. The most likely cause is
partial CPU fallback: the mn01 graph (128 × 1000 mel input, twenty-plus
depthwise stages) has lower delegate coverage than the STgram graph, so part of
it runs on the CPU cores. The one device where they tie (S21) is the oldest
SoC, where neither reaches the fastest path.

This is the edge-deployment mirror of the PTQ result: on this task, parameter
count predicts neither quantization robustness nor NPU latency. Both have to be
measured.

## TFLite-Micro arena — blocked

The ESP32 / Nano 33 BLE arena measurement did not complete. Two concrete
blockers, recorded so the next attempt starts from here:

1. **ArcFace head is not TFLite-friendly.** `onnx2tf` fails on the full scorer
   (`wa/model/arcface/Expand` and the int64 label indexing), because the
   dynamic scatter/gather does not map onto a Keras functional output.
2. **1-D conv transpose mismatch.** Exporting the embedding only (via
   `onnx.utils.extract_model` on the `feature` output) gets past the head but
   then fails in the custom TgramNet 1-D convolutions
   (`Layer weight shape (1,1,64) not compatible with (64,1,1)`).

The `tflite-micro` Python interpreter (`0.dev2026…`) also fails to import on
Python 3.12 (`undefined symbol: PyObject_ClearManagedDict`); a Python 3.11
environment or the C++ benchmark binary is needed for the arena number.

**Recommendation:** export an **embedding-only** graph and run scoring on the
host, then convert with a TF-native exporter (or LiteRT) rather than `onnx2tf`,
and measure the arena under Python 3.11. That is a self-contained follow-up.

## Not GPU-bound

This whole report is CPU/cloud only; it ran locally after the GPU pod was
released.
