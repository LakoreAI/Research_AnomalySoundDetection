# Technical Reference

## Backbones

**STgram-MFN** (reference)
- TgramNet (learned time-domain features) + spectrogram branch → concatenated
- MobileFaceNet (MFN) embedding extractor
- Additive Angular Margin (ArcFace) loss

**mn01** (swap candidate)
- Pretrained AudioSet embedding from EfficientAT
- Swapped in as embeddings-in to the *same* MobileFaceNet+ArcFace head — this is not a full backbone replacement, just a front-end swap

## Quantization

- Primary: INT8 Post-Training Quantization (PTQ)
- Fallback: Quantization-Aware Training (QAT), if PTQ accuracy loss is unacceptable
- Deployment format: TensorFlow Lite (TFLite) Micro
- Memory footprint validated early via x86 TFLite Micro arena size checks — not deferred to on-device deployment

## Target hardware

- ESP32
- Arduino Nano 33 BLE Sense

## Datasets & benchmarks

| Dataset | Role |
|---|---|
| MIMII | primary training/eval |
| DCASE Task 2 (2020–2022 formulation) | benchmark — **not** the 2023+ first-shot format |
| MIMII Domain Generalization (DG) | domain-shift held-out eval |
| ToyADMOS2 | supplementary |

## Comparison matrix

2×3 minimum: {STgram-MFN, mn01} × {FP32, INT8, domain shift}. Each cell logged separately (see experiment log format) and assembled into summary tables for the paper.

## Metrics

- Primary: AUC, partial AUC (pAUC), per machine type/section where applicable
- Efficiency: latency (ms), TFLite Micro arena size, parameter count — only where measured
- Reported as tables/tradeoff plots, descriptive (mean ± std across seeds) — not p-values, unless a real statistical test is actually run

## Current DCASE baseline

Harada et al. (EUSIPCO 2023): dense autoencoder + selective Mahalanobis scoring. This supersedes the older MobileNetV2 baseline — don't cite that as current.

## Related work (citation only, not model candidates)

Transformer architectures survey: DPTrans, SSDPT, MSANet, Audio Spectrogram Transformer (AST), Patchout faSt Spectrogram Transformer (PaSST), BEATs, EAT. These are for related-work framing only — explicitly not candidates for a model swap in this study.
