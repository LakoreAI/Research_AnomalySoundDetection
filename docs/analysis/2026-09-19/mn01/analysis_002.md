# Analysis 002 — augmentation hurt the fine-tuned mn01 (negative result)

- **Date:** 2026-09-19
- **Topic:** mn01
- **Question:** the fine-tuned mn01 trails STgram-MFN by ~8 AUC points. Does
  training-time augmentation close the gap?

## Finding

**No — the augmentation we tried made it substantially worse.** The augmented
fine-tune reached 69.2 AUC at epoch 30 and 68.0 at epoch 40, against 83.16 for
the un-augmented run. Training loss stalled near 13.5, close to the frozen
run's plateau (13.9) and far from the un-augmented run's 3.7.

| Run | Augmentation | AUC (ep 40) | Final/plateau loss |
|---|---|---|---|
| `lean_B1ft_mn01` | none | ~79.2 | 3.7 |
| `B1ft_mn01_aug` | noise + gain + shift + SpecAugment | 68.0 | 13.5 |

## Configuration tried

`configs/train_lean_mn01_aug.yaml`: waveform noise std 0.005, gain ±6 dB, time
shift ±10 %, and SpecAugment with 2 frequency masks (max 27 bins) and 2 time
masks (max 25 frames), applied identically to both backbones via
`src/modules/augment.py`.

## Interpretation

The perturbation budget is almost certainly too large for a 124 k-parameter
backbone trained on 36 k clips. The loss signature — flat and high — says the
model cannot fit the augmented distribution, so it stops improving on the
clean one too. This is the opposite failure mode from overfitting, which
augmentation is meant to fix.

Two plausible explanations, not mutually exclusive: (i) the additive noise and
gain destroy the low-energy cues that carry machine identity; (ii) two
frequency and two time masks per clip is aggressive for a 128 × 1000 input
with no repeated passes over the data.

## Dropped mid-training

The run was **stopped at epoch 57** (AUC 70.92) rather than carried to epoch
100. The loss had been flat since roughly epoch 15 and the AUC was not
recovering, so further epochs only burn GPU time. Decision 2026-09-19: drop the
augmentation branch entirely for this study.

## Not reported in the paper

Per the project decision (2026-09-19), a negative augmentation result is kept
in `docs/` only and left out of `docs/paper/main.tex`.

## Recommended follow-up

If revisited, start from a tenth of these magnitudes (noise std 0.001, gain
±2 dB, one mask of 10 bins/frames) and raise only if the loss still tracks the
un-augmented curve. Also worth trying augmentation on the **larger** STgram-MFN
tiers, where the model has the capacity to absorb it.
