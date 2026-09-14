# Task Roadmap

22 tasks across 5 milestones (M0 Research & Framing → M5 Write-up). Each
milestone has a Go/No-Go metric and a fallback. Status is updated as work lands.

Legend: `[x]` done · `[~]` in progress · `[ ]` not started

## M0 — Research & Framing

| # | Task | Status |
|---|---|---|
| 1 | Lock research question, scope, non-goals | [x] |
| 2 | Backbone decisions (STgram-MFN reference, mn01 swap definition) | [x] |
| 3 | Dataset/benchmark scope (DCASE T2 2020–2022, MIMII, MIMII DG, ToyADMOS2) | [x] |
| 4 | Metrics + reporting conventions (AUC/pAUC, descriptive) | [x] |
| 5 | Experiment-log template + decisions log | [x] |
| 6 | Codebase scaffolding (mirror bicafe under `src/`) | [x] |

**Go/No-Go:** scope is executable in the 4–6 month window with proven
components only. **Go.** Fallback: drop the domain-shift row to a single
dataset.

## M1 — Baseline Reproduction

| # | Task | Status |
|---|---|---|
| 7 | STgram-MFN model (TgramNet + Sgram + MobileFaceNet + ArcFace) | [x] |
| 8 | Data pipeline (MIMII/DCASE layout, download + prepare scripts) | [x] |
| 9 | Train/eval/infer pipelines + callbacks | [x] |
| 10 | Download DCASE 2020 dev + eval data | [ ] |
| 11 | Reproduce published STgram-MFN AUC/pAUC (DCASE 2020) | [ ] |
| 12 | Baselines: LogMel-MFN / Tgram-MFN ablations | [ ] |

**Go/No-Go:** reproduced average AUC within ~2 points of the published 92.36
(AUC) / 84.86 (mAUC) on DCASE 2020. Fallback: document the gap and treat the
in-house reproduction as the reference for all later comparisons (no
backbone-swap claim until it is credible).

## M2 — mn01 Swap

| # | Task | Status |
|---|---|---|
| 13 | mn01 (EfficientAT) embeddings-in wrapper; resolve 16k/32k + mel mismatch | [ ] |
| 14 | Same MobileFaceNet+ArcFace head on mn01 embeddings | [ ] |
| 15 | FP32 head-to-head: STgram-MFN vs mn01 (accuracy + params) | [ ] |

**Go/No-Go:** mn01 within a reportable margin of STgram-MFN in FP32, or a clear
accuracy/efficiency tradeoff. Fallback: mn01 as a negative result (still a
publishable tradeoff datapoint).

## M3 — INT8 Quantization

| # | Task | Status |
|---|---|---|
| 16 | PTQ pipeline (ONNX Runtime; static QDQ preferred over dynamic) | [~] |
| 17 | INT8 accuracy delta for both backbones | [ ] |
| 18 | QAT fallback if PTQ loss is unacceptable | [ ] |

**Go/No-Go:** INT8 delta small enough to be reportable (< ~3 AUC points) for at
least one backbone. Fallback: QAT.

## M4 — Domain Shift + Edge

| # | Task | Status |
|---|---|---|
| 19 | MIMII DG / domain-shift held-out eval, both backbones | [ ] |
| 20 | TFLite Micro conversion + arena size (x86 first, then device) | [~] |
| 21 | Latency/arena/param tradeoff table + plots | [~] |

**Go/No-Go:** TFLite Micro arena fits the target SRAM at INT8. Fallback:
reduced STgram-MFN variant (smaller `n_mels`/`n_frames`) or fixed STFT front-end.

## M5 — Write-up

| # | Task | Status |
|---|---|---|
| 22 | Paper draft + comparison matrix + related work synthesis | [ ] |

Supporting artifacts (not in the 22): Research Paper Outline, Scholarship
Statement of Purpose, Related Work synthesis — tracked in `docs/NOTES.md` under
open questions.
