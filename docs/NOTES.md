# Decisions & Notes

## Resolved decisions

- DCASE Task 2 changed formulation in 2023 — every writeup must state 2020–2022 scope explicitly to avoid ambiguity.
- mn01 swap is a **frontend+backbone replacement with only the ArcFace head shared** (see the pinned definition below). "Embeddings-in to the MobileFaceNet+ArcFace head" was the original framing but is not shape-compatible: mn01 emits a 1-D 96-vector, MobileFaceNet expects a `(2, n_mels, n_frames)` image.
- Hardware memory footprint validated early (x86 TFLite Micro arena checks), not deferred to deployment.
- Current DCASE baseline is Harada et al. (EUSIPCO 2023), not MobileNetV2.
- Transformer survey architectures are related-work citations only — not swap candidates.
- Drone-deployable track deprioritized to optional appendix; factory track is the sole primary focus. Scope deliberately held to proven components to stay executable in the timeline.
- Codebase layout mirrors the sibling `bicafe` project, directly under `src/` (no nested package): architecture config in `src/config.py`, training config/loop in `src/pipelines/`, callbacks in `src/callbacks/`, model pieces in `src/modules/`.
- STgram-MFN is a faithful re-expression of the reference `net.py` (github.com/liuyoude/STgram-MFN); the reference's hardcoded `313` LayerNorm width, `(8,20)` collapse kernel, and `128` embedding width are lifted into `STgramMFNConfig` rather than left implicit.
- Audio is loaded with `soundfile`, not `torchaudio.load` — torchaudio >= 2.11 requires the separate TorchCodec package for loading, which is avoided.
- Best-epoch selection uses the official DCASE test set (reference behavior) via `BestCheckpoint(monitor="auc")`. This is test-set selection and is flagged as a methodological caveat; a held-out normal `val_fraction` path exists for a leak-free `val_loss` if the protocol is tightened later.
- INT8 PTQ is done through ONNX Runtime (`scripts/edge/quantize_onnx.py`): dynamic-weight quantization by default, calibrated static (QDQ) preferred for accuracy. TFLite Micro requires full-integer quantization, so the dynamic ONNX INT8 file is not a direct TFLite input.
- First measured INT8 result: dynamic-weight ONNX INT8 is smaller (1.23 MB vs 4.56 MB) but ~3.5x SLOWER than fp32 on CPU (149 ms vs 42 ms) — expected for dynamically-quantized convnets; this motivates the static QDQ path, not a conclusion about the architecture.
- **Compute decided (2026-09-18):** rent **Ckey.vn GPU3 listing `106600`** — 1× RTX 3090 24 GB, 8c/16t, 64 GB RAM, 378 GB NVMe, 452↓·386↑ Mbps, ~6,563 VND/h (~$0.25/h). Chosen as the cheapest machine clearing the `HARDWARE.md` minimums (Ampere `sm_86` runs the `cu126` pin; 24 GB holds reference batch 128). Order settings + full setup/train recipe in [`HARDWARE.md`](HARDWARE.md#ckeyvn--chosen-rental-2026-09-18). First run scheduled 2026-09-19 (60-epoch trajectory, `--amp --batch_size 128 --accum_steps 1`).
- **Lean A-vs-B result (2026-09-19):** on DCASE 2020 dev (41 classes, 36,283 train clips), 100 epochs, batch 128, seed 42 — **A1 STgram-MFN AUC 91.45 / mAUC 83.76** (M1 gate passed: ≥90.4 / ≥82.9), **B1 mn01 frozen 64.36**, **B1-ft mn01 fine-tuned 83.16 / 75.22**. mn01 (~9× smaller: 0.13 M vs 1.16 M params) is a clear accuracy ↔ size tradeoff. **mn01 must be fine-tuned** — frozen AudioSet embeddings do not transfer to machine identity, and the frozen head's AUC declines after epoch 10. Report: [`reports/2026-09-19/lean/report_001.md`](reports/2026-09-19/lean/report_001.md); analyses: `analysis/2026-09-19/{mn01,training}/`.
- **Compute pivot (2026-09-19):** Ckey.vn went fully down (backend DB unreachable; every route errors), so training moves to **RunPod, 1× RTX A5000 24 GB (~$0.27/h Secure)**; fallbacks A4500 (20 GB, ~$0.19) / 3090 (24 GB, ~$0.22). Pod config + setup in [`HARDWARE.md`](HARDWARE.md#runpod--chosen-rental-2026-09-19).
- **mn01 is pinned (2026-09-16)** to EfficientAT's `mn01_as`: MobileNetV3 `width_mult=0.1`, weights `mn01_as_mAP_298.pt` (AudioSet, mAP 29.8), **123,783 params**, pooled output **96-d**. `mn01_im.pt` is ImageNet-only (unused). The 16k/32k mismatch is resolved: mn01 brings its own 32 kHz frontend (pre-emphasis; STFT `n_fft=1024`/`win 800`/`hop 320`; Kaldi mel banks; `log(x+1e-5)`; `(x+4.5)/5`), so MIMII's 16 kHz audio must be resampled up to 32 kHz for the mn01 row. Verified empirically by loading the real EfficientAT model (not just reading config).

## Reporting conventions

Empirical ML systems idiom — experimental conditions, ablations, descriptive metrics. Not social-science conventions (hypothesis tables, p-values, validity taxonomies) unless the work involves human subjects or survey data.

## Open questions / horizon

- Research Paper Outline — still generic, needs filling with actual argument structure once results exist
- Scholarship Statement of Purpose — not started
- Related Work synthesis — not started
- ESP32 hardware — acquisition agreed, not yet in hand
- Sprint 1 (M1 Baseline Reproduction) — in progress: STgram-MFN reference backbone implemented (model, data, train/eval/infer, edge export/quantize scripts). Rented RTX 3090 (`106600`) booked for 2026-09-19; next: download DCASE 2020 dev + eval, then the 60-epoch trajectory run (see `HARDWARE.md`) to reproduce published AUC/pAUC.

## Non-goals

- Drone-deployable variant (appendix only)
- DCASE 2023+ first-shot formulation
- Novel architecture — this is a tradeoff study, not an architecture paper

## Parking lot (unrelated to this project)

Alternatives to backpropagation (equilibrium propagation, Mono-Forward, predictive coding, MeZO-style zeroth-order fine-tuning, forward-gradient hybrids) — separate curiosity, not tied to ASD.
