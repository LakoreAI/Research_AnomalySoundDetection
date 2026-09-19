# Report 001 — Lean A-vs-B: STgram-MFN vs mn01 (RunPod RTX A5000)

- **Date:** 2026-09-19
- **Milestone:** M1 (Baseline Reproduction) + M2 (mn01 swap, FP32)
- **Status:** passed — M1 gate met; B is a clear size/accuracy tradeoff
- **Task:** reports / lean

## Setup

- **Runtime:** RunPod **RTX A5000 24 GB**, driver 580 (CUDA 13.0). Local
  P2000/T4 numbers are not comparable (single runtime for the matrix).
- **Protocol (frozen, identical for A and B):** DCASE 2020 dev (`data/raw`,
  30,987 clips) + eval-normal as `add_root` (`data/raw_eval`, 23,267 clips) →
  **36,283 train clips**, **41 machine-id classes**; 100 epochs, batch 128,
  AMP, Adam lr 1e-4, cosine (t_max=100, warm start epoch 5); best epoch on the
  official test AUC (reference behavior — test-set selection caveat,
  `docs/NOTES.md`).
- **Seed:** 42 (single seed).

## Results

| Cell | Model | Params | Best epoch | AUC | pAUC | mAUC | W&B |
|---|---|---|---|---|---|---|---|
| **A1** | STgram-MFN (FP32) | 1,161,279 | 100 | **91.45** | 84.93 | **83.76** | `r2vc9mcm` |
| **B1** | mn01 **frozen** + ArcFace | 127,719 (3,936 trainable) | 10 | 64.36 | 57.11 | 54.71 | `gy8lds0d` |
| **B1-ft** | mn01 **fine-tuned** + ArcFace | 127,719 | 100 | 83.16 | 75.58 | 75.22 | `at5143fu` |

Checkpoints/artifacts: `LakoreAI/stgram-mfn-lean` (`lean_A1_stgram/`,
`lean_B1ft_mn01/best.pt`; B1 frozen best also pushed). One W&B best artifact
per run.

## Gate assessment (EXPERIMENTS.md)

- **Stage 1 (A1) — PASS.** AUC 91.45 ≥ 90.4 and mAUC 83.76 ≥ 82.9
  (within ~2 of published 92.36 / 84.86).
- **Stage 3 (B1) — tradeoff, not parity.** mn01 is **~9× smaller**
  (127,719 vs 1,161,279 params) but **~8.3 AUC points lower** (83.16 vs 91.45).
  A clear accuracy ↔ size tradeoff; the negative result is reportable.

## Findings

1. **Fine-tuning is mandatory for the swap.** Frozen mn01 embeds general
   AudioSet content, not machine identity: its ArcFace head plateaued and the
   test AUC *declined* after epoch 10 (64 → 58). Fine-tuning the backbone
   lifts AUC to 83 (see `analysis/2026-09-19/mn01/analysis_001.md`).
2. **The tradeoff is real and large.** Even fine-tuned, the 0.13 M-param mn01
   does not reach the 1.16 M-param STgram-MFN; the gap is stable across epochs
   (B1-ft improves monotonically 78 → 83, A1 90 → 91.45).
3. **A1 is well-behaved:** AUC 89.9 → 91.45, mAUC 79.1 → 83.76, monotone-ish.

## Caveats

- Single seed; no error bars.
- Best epoch selected on the test set (inherited reference protocol).
- Test set is dev `test/` only (no held-out eval-test scoring yet — Stage 5).

## Next steps

1. **A2/B2 INT8 PTQ** on the A1 and B1-ft `best.pt` (static QDQ), re-evaluated
   (Stage 4).
2. **Edge:** ONNX → Qualcomm AI Hub sweep across all chips for A1 and B1-ft.
3. Cross-backbone params/latency tradeoff table (Stage 7).
