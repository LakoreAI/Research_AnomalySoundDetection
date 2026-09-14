# scripts/

Utility scripts for the anomaly-sound-detection study. Run them from the repo
root with `uv run python scripts/<name>.py ...`.

## Data

| Script | Purpose |
|---|---|
| `download_data.py` | Download MIMII / DCASE 2020 / DCASE 2022 data from Zenodo (URLs resolved live from the API) |
| `prepare_data.py` | Validate the `<machine>/<split>/` layout; convert raw MIMII to DCASE-style filenames; build a small symlink subset |
| `push_dataset_to_hf.py` | Upload a prepared dataset directory to Hugging Face Hub (optional `huggingface_hub`) |

## Train / evaluate

| Script | Purpose |
|---|---|
| `train.py` | Wrapper for `src.pipelines.train` (YAML + CLI overrides) |
| `evaluate.py` | AUC / pAUC on the DCASE test split for a checkpoint |
| `smoke_test.py` | Synthetic-audio end-to-end test (no dataset needed) |

## Edge / quantization

| Script | Purpose |
|---|---|
| `export_onnx.py` | Export the scorer graph to ONNX |
| `quantize_onnx.py` | INT8 PTQ (dynamic or calibrated static) via ONNX Runtime |
| `export_tflite.py` | ONNX -> TF SavedModel -> TFLite (float + full-INT8) |
| `benchmark_edge.py` | Params, model size, CPU latency, activation-memory estimate |
| `edge/README.md` | Full TFLite Micro / ESP32 / Nano 33 BLE deployment path |
| `push_model_to_hf.py` | Upload a checkpoint + model card to Hugging Face Hub |

## Quick start

```bash
# 1) data
uv run python scripts/download_data.py --dataset dcase2020 --dest data/raw
uv run python scripts/download_data.py --dataset dcase2020-eval --dest data/raw_eval
uv run python scripts/prepare_data.py --root data/raw --check

# 2) train + evaluate
uv run python scripts/train.py --config configs/train.yaml --add_root data/raw_eval
uv run python scripts/evaluate.py --ckpt checkpoints/<run>/best.pt --data_root data/raw

# 3) edge
uv run python scripts/export_onnx.py --ckpt checkpoints/<run>/best.pt --out export/stgram_mfn.onnx
uv run python scripts/quantize_onnx.py --onnx export/stgram_mfn.onnx \
    --out export/stgram_mfn_int8.onnx --mode dynamic
uv run python scripts/benchmark_edge.py --ckpt checkpoints/<run>/best.pt \
    --onnx export/stgram_mfn.onnx --onnx_int8 export/stgram_mfn_int8.onnx
```

Optional dependencies for the edge/upload scripts are installed on demand
(`onnx`, `onnxruntime`, `tensorflow`, `onnx2tf`, `huggingface_hub`) and are not
part of the core environment.

## Colab (GPU training)

Drives a real Colab GPU runtime from this machine via the `colab` CLI (ported
from the sibling `bicafe` project; see `.agents/skills/colab-*`). The local
P2000 is compute-bound at ~60 clips/s — a Colab T4 is faster.

```bash
scripts/colab_setup.sh                       # one-time: gcloud + colab CLI + ADC login
scripts/colab_new_session.sh asd T4          # create/reconnect the session
scripts/colab_push_repo.sh                   # push committed code + pull DCASE data from HF
scripts/colab_check_training.sh              # tail checkpoints + train_log.json
scripts/colab_pull_results.sh                # fetch checkpoints/results back
scripts/colab_stop.sh                        # stop when done
```

- `colab_push_repo.sh` ships only committed files (`git archive HEAD`) and
  refuses a dirty tree; commit first.
- The dataset is pulled from `LakoreAI/stgram-mfn-dcase2020-{dev,eval}` on HF
  rather than uploaded (see `.agents/skills/hf-dataset-fast-resume`).
- On the VM, run entrypoints with cwd `/content/asd` (no `pip install -e .`):
  `cd /content/asd && python -m src.pipelines.train --config configs/train.yaml --add_root data/raw_eval`.
