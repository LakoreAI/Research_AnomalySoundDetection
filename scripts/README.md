# scripts/

Utility entrypoints for the anomaly-sound-detection study, grouped by category.
Run them from the repo root, e.g.
`uv run python scripts/<category>/<name>.py ...`.

```
scripts/
├── data/       download + prepare datasets
├── training/   train / evaluate / smoke-test / detached launcher
├── edge/       ONNX / INT8 / TFLite export + benchmarking
├── publish/    push datasets and checkpoints to Hugging Face
└── colab/      drive a Colab GPU session via the `colab` CLI
```

## data/

| Script | Purpose |
|---|---|
| `data/download_data.py` | Download MIMII / DCASE 2020 / DCASE 2022 data from Zenodo (URLs resolved live from the API) |
| `data/prepare_data.py` | Validate the `<machine>/<split>/` layout; convert raw MIMII to DCASE-style filenames; build a small symlink subset |

## training/

| Script | Purpose |
|---|---|
| `training/train.py` | Wrapper for `src.pipelines.train` (YAML + CLI overrides) |
| `training/run_detached.sh` | Launch training in a detached tmux/nohup session (survives SSH drops) |
| `training/evaluate.py` | AUC / pAUC / mAUC on the DCASE test split for a checkpoint |
| `training/smoke_test.py` | Synthetic-audio end-to-end test (no dataset needed) |

## edge/

| Script | Purpose |
|---|---|
| `edge/export_onnx.py` | Export the scorer graph to ONNX |
| `edge/quantize_onnx.py` | INT8 PTQ (dynamic or calibrated static) via ONNX Runtime |
| `edge/export_tflite.py` | ONNX -> TF SavedModel -> TFLite (float + full-INT8) |
| `edge/benchmark_edge.py` | Params, model size, CPU latency, activation-memory estimate |
| `edge/README.md` | Full TFLite Micro / ESP32 / Nano 33 BLE deployment path |

## publish/

| Script | Purpose |
|---|---|
| `publish/push_dataset_to_hf.py` | Upload a prepared dataset directory to Hugging Face Hub |
| `publish/push_model_to_hf.py` | Upload a checkpoint + model card to Hugging Face Hub |

## Quick start

```bash
# 1) data
uv run python scripts/data/download_data.py --dataset dcase2020 --dest data/raw
uv run python scripts/data/download_data.py --dataset dcase2020-eval --dest data/raw_eval
uv run python scripts/data/prepare_data.py --root data/raw --check

# 2) train + evaluate
uv run python scripts/training/train.py --config configs/train.yaml --add_root data/raw_eval
uv run python scripts/training/evaluate.py --ckpt checkpoints/<run>/best.pt --data_root data/raw

# 3) edge
uv run python scripts/edge/export_onnx.py --ckpt checkpoints/<run>/best.pt --out export/stgram_mfn.onnx
uv run python scripts/edge/quantize_onnx.py --onnx export/stgram_mfn.onnx \
    --out export/stgram_mfn_int8.onnx --mode static --calib_root data/raw
uv run python scripts/edge/benchmark_edge.py --ckpt checkpoints/<run>/best.pt \
    --onnx export/stgram_mfn.onnx --onnx_int8 export/stgram_mfn_int8.onnx
```

Optional dependencies for the edge/publish scripts are installed on demand
(`onnx`, `onnxruntime`, `tensorflow`, `onnx2tf`, `huggingface_hub`) and are not
part of the core environment.

## Detached training (any VM)

`training/run_detached.sh` runs training inside a tmux session (nohup fallback)
so an SSH/Colab disconnect can't kill it; pair it with the config's `hf_push`
for auto-resume. See [`../docs/HARDWARE.md`](../docs/HARDWARE.md).

```bash
scripts/training/run_detached.sh start --config configs/train_t4_long.yaml
scripts/training/run_detached.sh status     # log tail + latest checkpoints
scripts/training/run_detached.sh attach
scripts/training/run_detached.sh stop
```

## Colab (GPU training)

Drives a real Colab GPU runtime from this machine via the `colab` CLI (ported
from the sibling `bicafe` project; see `.agents/skills/colab-*`). The local
P2000 is compute-bound at ~60 clips/s — a rented GPU is faster (see
[`../docs/HARDWARE.md`](../docs/HARDWARE.md)).

```bash
scripts/colab/colab_setup.sh                  # one-time: gcloud + colab CLI + ADC login
scripts/colab/colab_new_session.sh asd T4     # create/reconnect + start keep-alive
scripts/colab/colab_push_repo.sh              # push committed code + pull DCASE data from HF
scripts/colab/colab_check_training.sh         # tail checkpoints + train_log.json
scripts/colab/colab_pull_results.sh           # fetch checkpoints/results back
scripts/colab/colab_stop.sh                   # stop when done
```

- `colab_new_session.sh` starts `colab_keepalive.sh` (background
  `colab keep-alive`, 24 h cap) to prevent the idle reclaim that killed a run.
- Training configs with `hf_push: {enabled, repo_id}` push checkpoints to HF
  every `ckpt_every` epochs and **auto-resume** from the latest HF epoch on
  startup — a reclaimed VM costs minutes, not the run.
- `colab_push_repo.sh` ships only committed files (`git archive HEAD`) and
  refuses a dirty tree; commit first.
- The dataset is pulled from `LakoreAI/stgram-mfn-dcase2020-{dev,eval}` on HF
  rather than uploaded (see `.agents/skills/hf-dataset-fast-resume`).
- On the VM, run entrypoints with cwd `/content/asd` (no `pip install -e .`):
  `cd /content/asd && python -m src.pipelines.train --config configs/train.yaml --add_root data/raw_eval`.
