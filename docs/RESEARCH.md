# Embedded Acoustic Anomaly Detection: STgram-MFN vs. mn01 Under INT8 Quantization

Backbone-swap tradeoff study for factory equipment sound monitoring on microcontroller-class hardware.

## Research question

Does swapping STgram-MFN's hand-designed spectral-temporal features for a general pretrained AudioSet embedding (mn01, from EfficientAT) hold up in accuracy once INT8 quantization and domain shift (MIMII Domain Generalization, DG) are both in play — and what's the actual accuracy/latency/memory tradeoff at each step?

The novelty is the tradeoff characterization, not a new architecture. Both backbones are proven components; this doesn't propose a new one.

## Scope

- **In scope:** STgram-MFN (TgramNet + spectrogram branch + MobileFaceNet + Additive Angular Margin (ArcFace) loss) as reference backbone; mn01 as an embeddings-in swap to the same MobileFaceNet+ArcFace head; INT8 Post-Training Quantization (PTQ) with Quantization-Aware Training (QAT) as fallback; TensorFlow Lite (TFLite) Micro deployment target.
- **Benchmark:** DCASE Task 2, 2020–2022 formulation (not the 2023+ "first-shot" format).
- **Target hardware:** ESP32, Arduino Nano 33 BLE Sense.
- **Out of scope:** drone-deployable track (optional appendix only, not primary).

## Status

22-task sprint roadmap across 5 milestones (M0 Research & Framing → M5 Write-up), each with Go/No-Go metrics and fallback actions. Currently wrapping M0 documentation (methodology, experiment log template, reading list) before Sprint 1 (M1 Baseline Reproduction). ESP32 hardware acquisition agreed but not yet in hand.

Output target: scholarship application (4–6 month timeline).

## Repo structure

> TBD — fill in once code exists. Suggested starting layout:

```
.
├── configs/          # experiment configs per condition (backbone × quant × dataset)
├── data/              # dataset prep scripts (MIMII, DCASE T2, MIMII DG, ToyADMOS2)
├── src/            # STgram-MFN, mn01 wrapper, MobileFaceNet+ArcFace head
├── ├── eval/              # AUC/pAUC scoring, comparison matrix aggregation
├── quantization/       # PTQ/QAT + TFLite Micro conversion
├── docs/
│   ├── REFERENCE.md
│   └── NOTES.md
└── results/           # per-run logs, summary tables
```

## Docs

- [`docs/REFERENCE.md`](docs/REFERENCE.md) — models, datasets, hardware, methodology, metrics
- [`docs/NOTES.md`](docs/NOTES.md) — decisions log, open questions, non-goals

## Key references

- Harada et al. (EUSIPCO 2023) — current DCASE Task 2 baseline (dense autoencoder + selective Mahalanobis scoring)
- EfficientAT paper — mn01
- DCASE Task 2 challenge papers (2022, 2023, 2026)