# Embedded Acoustic Anomaly Detection: STgram-MFN vs. mn01 Under INT8 Quantization

Backbone-swap tradeoff study for factory equipment sound monitoring on microcontroller-class hardware.

## Research question

Does swapping STgram-MFN's hand-designed spectral-temporal features for a general pretrained AudioSet embedding (mn01, from EfficientAT) hold up in accuracy once INT8 quantization and domain shift (MIMII Domain Generalization, DG) are both in play — and what's the actual accuracy/latency/memory tradeoff at each step?

The novelty is the tradeoff characterization, not a new architecture. Both backbones are proven components; this doesn't propose a new one.

## Scope

- **In scope:** STgram-MFN (TgramNet + spectrogram branch + MobileFaceNet + Additive Angular Margin (ArcFace) loss) as reference backbone; mn01 as a frontend+backbone swap sharing only the ArcFace head (pinned to EfficientAT `mn01_as`, 32 kHz own frontend — see `docs/REFERENCE.md`); INT8 Post-Training Quantization (PTQ) with Quantization-Aware Training (QAT) as fallback; TensorFlow Lite (TFLite) Micro deployment target.
- **Benchmark:** DCASE Task 2, 2020–2022 formulation (not the 2023+ "first-shot" format).
- **Target hardware:** ESP32, Arduino Nano 33 BLE Sense.
- **Out of scope:** drone-deployable track (optional appendix only, not primary).

## Status

22-task sprint roadmap across 5 milestones (M0 Research & Framing → M5 Write-up), each with Go/No-Go metrics and fallback actions. Currently wrapping M0 documentation (methodology, experiment log template, reading list) before Sprint 1 (M1 Baseline Reproduction). ESP32 hardware acquisition agreed but not yet in hand.

Output target: scholarship application (4–6 month timeline).

## Repo structure

Mirrors the layout of the sibling `bicafe` codebase: architecture config in
`src/config.py`, training config + loop under `src/pipelines/`, reusable
callbacks under `src/callbacks/`, model pieces under `src/modules/`.

```
src/
├── config.py          # STgramMFNConfig (model + audio)
├── data.py            # ASDDataset, waveform loading
├── diagnostics.py     # D1 wiring / D2 branches / D3 overfit
├── modules/           # frontend, tgramnet, mobilefacenet, arcface, loss, model, edge
├── pipelines/         # config.py, train.py, eval.py, infer.py
├── callbacks/         # base, checkpoint, early_stopping, lr_scheduler, wandb
└── utils/             # io_utils, model_utils, audio_utils
configs/train.yaml
scripts/               # download_data, prepare_data, train, evaluate, edge export/quantize
tests/
docs/
```

## Docs

- [`docs/REFERENCE.md`](docs/REFERENCE.md) — models, datasets, hardware, methodology, metrics
- [`docs/NOTES.md`](docs/NOTES.md) — decisions log, open questions, non-goals

## Key references

- Harada et al. (EUSIPCO 2023) — current DCASE Task 2 baseline (dense autoencoder + selective Mahalanobis scoring)
- EfficientAT paper — mn01
- DCASE Task 2 challenge papers (2022, 2023, 2026)
