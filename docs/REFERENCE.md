# Technical Reference

## Backbones

**STgram-MFN** (reference)
- TgramNet (learned time-domain features) + spectrogram branch → concatenated
- MobileFaceNet (MFN) embedding extractor
- Additive Angular Margin (ArcFace) loss

**mn01** (swap candidate) — pinned 2026-09-16
- EfficientAT MobileNetV3, `width_mult=0.1` (`mn01` in EfficientAT's `helpers/utils.py:NAME_TO_WIDTH`).
- Weights: `mn01_as_mAP_298.pt` (AudioSet, mAP 29.8). **Not** `mn01_im.pt` — that one is ImageNet-only.
- **123,783 parameters**; pooled output is **96-d** (global-average-pool of the final 1×1 conv), not 1280. The MLP classifier is 96→128→527 and is discarded.
- Own frontend (required by the weights, not interchangeable with STgram's): 32 kHz, pre-emphasis `[-0.97, 1]`, STFT `n_fft=1024` / `win_length=800` / `hop=320`, Kaldi mel banks (128 mels, vtln), `log(x + 1e-5)`, `(x + 4.5) / 5` → 1000 frames per 10 s.
- Swap semantics: mn01 replaces the **entire STgram frontend + backbone** (Sgram + Tgram + MobileFaceNet); only the **ArcFace** metric-learning head is shared. "Embeddings-in to MobileFaceNet" is not shape-compatible — mn01 emits a 1-D 96-vector, while MobileFaceNet expects a `(2, n_mels, n_frames)` image.

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
