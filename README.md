# Embedded Acoustic Anomaly Detection

Backbone-swap tradeoff study for factory equipment sound monitoring on
microcontroller-class hardware: **STgram-MFN vs. mn01 under INT8 quantization**.

See [`docs/RESEARCH.md`](docs/RESEARCH.md) for the research question, scope, and
comparison matrix; [`docs/REFERENCE.md`](docs/REFERENCE.md) for models, datasets,
and metrics; [`docs/NOTES.md`](docs/NOTES.md) for decisions; and
[`docs/TASKS.md`](docs/TASKS.md) for the sprint roadmap.

## Status

M1 (Baseline Reproduction) — **STgram-MFN reference backbone implemented**.
mn01 swap, INT8 PTQ, and the comparison matrix are upcoming.

## Setup

Requires Python >= 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync              # core: torch/torchaudio (CUDA 12.6 on this machine)
uv sync --extra edge # optional: onnx / onnxruntime for export + quantization
uv run pytest        # tests
```

Platform-aware torch builds resolve from `pyproject.toml` (Linux + NVIDIA uses
cu126 — the local Quadro P2000/Pascal needs it; macOS resolves CPU/MPS wheels).

## Layout

```
src/
├── config.py            # STgramMFNConfig — model + audio architecture
├── data.py              # ASDDataset, waveform loading
├── diagnostics.py       # D1/D2/D3 architecture sanity checks
├── modules/
│   ├── frontend.py      # Wave2Mel, FeatureExtractor (Sgram branch)
│   ├── tgramnet.py      # learned time-domain Tgram branch
│   ├── mobilefacenet.py # MobileFaceNet backbone
│   ├── arcface.py       # ArcMarginProduct head
│   ├── loss.py          # ASDLoss + anomaly score
│   ├── model.py         # STgramMFN
│   └── edge.py          # STgramMFNScorer (deployment wrapper)
├── pipelines/           # config.py (TrainingConfig), train.py, eval.py, infer.py
├── callbacks/           # base, checkpoint, early_stopping, lr_scheduler, wandb
└── utils/               # io_utils, model_utils, audio_utils
configs/train.yaml       # training-loop config
scripts/                 # data download/prep, train, evaluate, edge export/quantize
tests/                   # pytest suite
```

## Train

```bash
uv run python scripts/download_data.py --dataset dcase2020 --dest data/raw
uv run python scripts/download_data.py --dataset dcase2020-eval --dest data/raw_eval
uv run python scripts/prepare_data.py --root data/raw --check

uv run python scripts/train.py --config configs/train.yaml --add_root data/raw_eval
uv run python scripts/evaluate.py --ckpt checkpoints/<run>/best.pt --data_root data/raw
```

Or without downloading anything:

```bash
uv run python scripts/smoke_test.py     # synthetic-audio end-to-end
```

## Edge / INT8

```bash
uv run python scripts/export_onnx.py --ckpt checkpoints/<run>/best.pt --out export/stgram_mfn.onnx
uv run python scripts/quantize_onnx.py --onnx export/stgram_mfn.onnx \
    --out export/stgram_mfn_int8.onnx --mode static --calib_root data/raw
uv run python scripts/benchmark_edge.py --ckpt checkpoints/<run>/best.pt \
    --onnx export/stgram_mfn.onnx --onnx_int8 export/stgram_mfn_int8.onnx
```

See [`scripts/edge/README.md`](scripts/edge/README.md) for the full
ONNX → TFLite → TFLite Micro path and arena measurement on ESP32 / Nano 33 BLE.
