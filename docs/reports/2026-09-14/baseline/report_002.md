# Report 002 — ToyCar real-data canary

- **Date:** 2026-09-14
- **Milestone:** M1 (Baseline Reproduction)
- **Status:** passed — training/eval pipeline works on real DCASE data
- **Task:** reports / baseline

## Setup

First real-data run, scoped to one machine (`ToyCar`) while the rest downloads.

| Parameter | Value |
|---|---|
| Data | `data/raw/ToyCar` — 4000 train / 2459 test clips, 4 machine ids |
| Epochs | 3 |
| Batch size | 32 |
| LR | 1e-3 (no scheduler) |
| Params | 1,151,770 (4 classes) |
| Checkpoint | `checkpoints/canary_toycar/best.pt` |

Deliberately NOT reference hyperparameters (which are 300 epochs, batch 128,
lr 1e-4, cosine from epoch 20) — this is a bring-up canary, not a result.

## Results

| Epoch | Train loss | AUC | pAUC |
|---|---|---|---|
| 1 | 5.099 | **89.444** | **84.338** |
| 2 | 0.891 | 85.779 | 85.257 |
| 3 | 0.572 | 88.345 | 83.986 |

Best: **AUC 89.444 / pAUC 84.338** (epoch 1).

## Interpretation

- The full path works on real data: DCASE filename parsing, machine-id label
  mapping (4 ids), 10 s feature extraction, ArcFace training, and AUC/pAUC
  scoring all produce finite, sensible numbers.
- Train loss collapses within ~1 epoch (4-class ID classification with ArcFace
  is easy), so AUC is non-monotonic across these 3 epochs — expected, not a
  concern.
- **Not comparable to published numbers** (STgram-MFN ArcFace ToyCar: AUC 94.44,
  mAUC 83.07). This run uses 3 epochs, 10x the reference LR, no LR schedule,
  batch 32, a single machine, and test-set best-epoch selection.

## Next steps

1. Finish the DCASE 2020 dev + eval download (~3 h remaining).
2. Run the full 6-machine, 300-epoch reference configuration.
3. Compare against published 92.36 AUC / 84.86 mAUC.
