# Embedded Acoustic Anomaly Detection: STgram-MFN vs. mn01 under INT8 Quantization

Anomalous sound detection (ASD) for factory machine condition monitoring on
microcontroller-class hardware. This repository is a controlled tradeoff
study: it swaps STgram-MFN's hand-designed spectral-temporal frontend for a
general pretrained AudioSet embedding (**EfficientAT `mn01`**), then carries
both models through INT8 post-training quantization and on-device profiling
rather than stopping at floating point.

**Question.** Does a small, pretrained audio embedding beat a task-specific
spectral model once quantization and edge deployment are taken into account —
and what is the accuracy/latency/size tradeoff at each step?

**Short answer.** No. A fine-tuned `mn01` is about nine times smaller but eight
AUC points weaker, and it collapses under INT8 (91.45 → 91.22 for STgram-MFN
versus 83.18 → 57.61 for `mn01`). Quantization robustness does not follow
parameter count.

Keywords: anomalous sound detection, acoustic anomaly detection, machine
condition monitoring, DCASE 2020 Task 2, MIMII, STgram-MFN, EfficientAT mn01,
ArcFace, INT8 post-training quantization, ONNX Runtime, TFLite Micro, ESP32,
Qualcomm AI Hub, edge AI, embedded machine learning.

## Contents

- [Key results](#key-results)
- [Models and data](#models-and-data)
- [Quickstart](#quickstart)
- [Training](#training)
- [INT8 quantization and edge deployment](#int8-quantization-and-edge-deployment)
- [Repository layout](#repository-layout)
- [Documentation](#documentation)
- [FAQ](#faq)
- [Citation](#citation)

## Key results

All numbers use one protocol on DCASE 2020 Task 2 (36,283 training clips, 41
machine-identity classes, 100 epochs, batch 128, AMP, seed 42). AUC / mAUC in
percent; mAUC is the worst-case AUC across machine units.

| Model | Params | AUC | mAUC | INT8 AUC | ΔAUC |
|---|---|---|---|---|---|
| STgram-MFN (reference) | 1,161,279 | **91.45** | **83.76** | **91.22** | **−0.23** |
| mn01 fine-tuned | 127,719 | 83.16 | 75.22 | 57.61 | −25.6 |
| mn01 frozen + linear head | 127,719 | 64.36 | 54.71 | — | — |
| STgram-MFN, small tier | 309,247 | 90.04 | — | — | — |
| STgram-MFN, large tier | 2,801,087 | 91.63 | — | — | — |

Edge latency (INT8 TFLite, Qualcomm AI Hub, single inference, ms):

| Device | STgram-MFN | mn01 |
|---|---|---|
| Galaxy S25 | **5.75** | 33.47 |
| Galaxy S24 | 7.00 | 36.77 |
| Galaxy S21 | 7.76 | 7.92 |
| Galaxy S23 | 9.85 | 49.83 |
| QCS8550 (IoT) | 9.87 | 50.08 |

The smaller `mn01` is **3–5× slower on four of five devices** — parameter count
predicts neither robustness nor NPU latency.

Three findings worth remembering:

1. **Fine-tuning is mandatory** for the pretrained embedding. Frozen, it peaks
   at 64.36 AUC and then degrades; fine-tuned, it reaches 83.16.
2. **The smaller model quantizes worst.** INT8 leaves STgram-MFN almost
   untouched and destroys `mn01`.
3. **Capacity saturates early.** Tripling parameters from the small tier buys
   1.6 AUC points; going from 1.16 M to 2.80 M buys 0.18.

Full write-up: [`docs/paper/main.pdf`](docs/paper/main.pdf). Per-run logs:
[`docs/reports/`](docs/reports), analyses: [`docs/analysis/`](docs/analysis).

## Models and data

- **STgram-MFN** — a learned time-domain branch (TgramNet) fused with a log-mel
  spectrogram, then MobileFaceNet and an ArcFace head. Reference:
  Liu *et al.*, ICASSP 2022 ([arXiv:2201.05510](https://arxiv.org/abs/2201.05510)).
- **mn01** — EfficientAT MobileNetV3, width 0.1, 123,783 parameters, 96-d
  pooled output, pretrained on AudioSet. Reference:
  Schmid *et al.*, ICASSP 2023 ([arXiv:2211.04772](https://arxiv.org/abs/2211.04772)).
- **Data** — DCASE 2020 Task 2 development set (MIMII + ToyADMOS), six machine
  types, normal-only training audio and mixed normal/anomaly test audio
  ([arXiv:2006.05822](https://arxiv.org/abs/2006.05822),
  [MIMII, arXiv:1909.09347](https://arxiv.org/abs/1909.09347)).
- **Metrics** — AUC, partial AUC (max FPR 0.1), and worst-case mAUC.

Dataset mirrors in the exact directory layout the code expects:
`LakoreAI/stgram-mfn-dcase2020-dev` and `LakoreAI/stgram-mfn-dcase2020-eval`
on the Hugging Face Hub.

## Quickstart

Requires Python ≥ 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/LakoreAI/Research_AnomalySoundDetection
cd Research_AnomalySoundDetection

uv sync                # core: torch/torchaudio
uv sync --extra edge   # optional: onnx / onnxruntime for export + quantization
uv run pytest          # test suite

# end-to-end on synthetic audio, no dataset download
uv run python scripts/training/smoke_test.py
```

Platform-aware torch builds resolve from `pyproject.toml`: Linux + NVIDIA uses
the cu126 wheels, macOS resolves CPU/MPS wheels.

## Training

```bash
# 1. data (or pull the HF mirrors above)
uv run python scripts/data/download_data.py --dataset dcase2020      --dest data/raw
uv run python scripts/data/download_data.py --dataset dcase2020-eval --dest data/raw_eval
uv run python scripts/data/prepare_data.py --root data/raw --check

# 2. STgram-MFN (the reference backbone)
uv run python scripts/training/train.py \
    --config configs/train_lean.yaml --add_root data/raw_eval

# 3. mn01 swap (fine-tuned)
uv run python scripts/training/train_mn01.py \
    --config configs/train_lean.yaml --finetune --ft_batch 128

# 4. evaluate
uv run python scripts/training/evaluate.py \
    --ckpt checkpoints/<run>/best.pt --data_root data/raw
```

`configs/` also holds the capacity-ladder configs (`tier_small`,
`tier_large`, `tier_xlarge`) and the 300-epoch reference schedule. Runs log to
Weights & Biases and push `best.pt` to a Hugging Face model repo.

## INT8 quantization and edge deployment

```bash
# export the deployed scorer to ONNX
uv run python scripts/edge/export_onnx.py \
    --ckpt checkpoints/<run>/best.pt --out export/stgram_mfn.onnx

# static QDQ post-training quantization, calibrated on real clips
uv run python scripts/edge/quantize_onnx.py \
    --onnx export/stgram_mfn.onnx --out export/stgram_mfn_int8.onnx \
    --mode static --calib_root data/raw

# accuracy delta
uv run python scripts/edge/eval_onnx.py --onnx export/stgram_mfn_int8.onnx

# latency / memory
uv run python scripts/edge/benchmark_edge.py --ckpt checkpoints/<run>/best.pt \
    --onnx export/stgram_mfn.onnx --onnx_int8 export/stgram_mfn_int8.onnx
```

For the mn01 swap, use `export_onnx_mn01.py` / `eval_onnx_mn01.py` (the 32 kHz
mel frontend stays outside the graph). For Snapdragon NPUs,
`scripts/edge/qai_hub_benchmark.py` compiles and profiles the model across
Qualcomm AI Hub devices. See [`scripts/edge/README.md`](scripts/edge/README.md)
for the ONNX → TFLite → TFLite Micro path and the ESP32 / Nano 33 BLE arena
measurement.

## Repository layout

```
src/
├── config.py            # STgramMFNConfig + MN01Config (model + audio architecture)
├── data.py              # ASDDataset, waveform loading
├── diagnostics.py       # architecture sanity checks
├── modules/
│   ├── frontend.py      # Wave2Mel, FeatureExtractor (Sgram branch)
│   ├── tgramnet.py      # learned time-domain Tgram branch
│   ├── mobilefacenet.py # MobileFaceNet backbone
│   ├── arcface.py       # ArcMarginProduct head
│   ├── loss.py          # ASDLoss + anomaly score
│   ├── model.py         # STgramMFN
│   ├── mn01.py          # EfficientAT mn01 embedder + 32 kHz frontend
│   ├── mn01_head.py     # mn01 + ArcFace (frozen / fine-tuned)
│   ├── augment.py       # optional training augmentation (off by default)
│   └── edge.py          # STgramMFNScorer (deployment wrapper)
├── pipelines/           # TrainingConfig, train.py, eval.py, infer.py
├── callbacks/           # checkpoint, early_stopping, lr_scheduler, wandb
└── utils/               # io, model, audio helpers
configs/                 # training configs (lean, tiers, reference)
scripts/
├── data/                # download_data, prepare_data
├── training/            # train, train_mn01, train_dg, evaluate, smoke_test
├── edge/                # export, quantize, benchmark, Qualcomm AI Hub
├── publish/             # push_dataset_to_hf, push_model_to_hf
└── colab/               # session scripts
tests/                   # pytest suite
docs/                    # research docs, reports, analysis, paper
```

## Documentation

| Document | Contents |
|---|---|
| [`docs/RESEARCH.md`](docs/RESEARCH.md) | research question, scope, comparison matrix |
| [`docs/REFERENCE.md`](docs/REFERENCE.md) | models, datasets, metrics |
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | gated experiment runbook |
| [`docs/NOTES.md`](docs/NOTES.md) | decisions log |
| [`docs/TASKS.md`](docs/TASKS.md) | sprint roadmap |
| [`docs/HARDWARE.md`](docs/HARDWARE.md) | compute plan |
| [`docs/paper/main.pdf`](docs/paper/main.pdf) | IEEE-style report |
| [`docs/reports/`](docs/reports) · [`docs/analysis/`](docs/analysis) | per-run reports and analyses |

## FAQ

**What is anomalous sound detection?**
Detecting abnormal machine sounds having seen only normal recordings during
training. The model learns what "normal" sounds like and flags deviations. This
is the DCASE Task 2 formulation.

**Why compare STgram-MFN with the mn01 AudioSet embedding?**
They represent two philosophies: a frontend designed for the task versus a
general audio embedding reused for it. The study measures which one survives
quantization and edge deployment.

**Does the pretrained embedding need fine-tuning?**
Yes. Frozen, `mn01` plus a linear head reaches 64.36 AUC and then declines;
fine-tuned end-to-end it reaches 83.16.

**Which model should I deploy on a microcontroller?**
Based on this study, STgram-MFN — it keeps its accuracy under INT8 while the
smaller `mn01` does not. Start from the `small` tier (309 K parameters, 90.04
AUC) if footprint matters.

**Why is the mn01 model worse after quantization?**
Its activations are narrow, so per-tensor INT8 scales cannot represent their
dynamic range, and the error compounds through the network. The larger
STgram-MFN has wider activations and a bounded spectrogram input.

**Can I reproduce the results?**
Yes. Configure `uv`, run the commands above, and the seeds, configs, and logs
in `docs/` regenerate every table. Checkpoints are on the Hugging Face Hub.

## Citation

```bibtex
@misc{leducminh2026asd,
  title  = {Quantization Robustness of Spectral versus Pretrained Audio
            Features for Embedded Anomalous Sound Detection},
  author = {Le Duc Minh},
  year   = {2026},
  note   = {https://github.com/LakoreAI/Research_AnomalySoundDetection}
}
```

## License

Code released under the repository [LICENSE](LICENSE). The DCASE 2020 Task 2
data is redistributed under CC BY-NC-SA 4.0, inherited from the original
Zenodo records; check those terms before any commercial use.
