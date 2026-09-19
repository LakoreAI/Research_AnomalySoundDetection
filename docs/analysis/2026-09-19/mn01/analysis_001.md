# Analysis 001 — frozen mn01 embeddings do not transfer; fine-tuning is required

- **Date:** 2026-09-19
- **Topic:** mn01
- **Question:** the mn01 swap shares "only the ArcFace head" — should the
  pretrained mn01 backbone stay frozen, or be fine-tuned, for a fair A-vs-B
  comparison?

## Finding

**Frozen mn01 cannot learn machine identity; fine-tuning can (partially).**

| Eval (same 100-epoch protocol) | AUC | pAUC | mAUC |
|---|---|---|---|
| mn01 frozen + ArcFace (best, ~ep10) | 64.36 | 57.11 | 54.71 |
| mn01 frozen + ArcFace (final, ep100) | 58.52 | 59.01 | 46.45 |
| mn01 **fine-tuned** + ArcFace (best, ep100) | **83.16** | 75.58 | 75.22 |
| A1 STgram-MFN (reference) | 91.45 | 84.93 | 83.76 |

Two signatures in the frozen run:
- `train_loss` crashed from 25 → 13.9 within ~10 epochs then **flat**, while the
  test AUC **declined** after epoch 10 (64 → 58). The ArcFace head overfits the
  machine-ID pretext without any transferable structure in the fixed features.
- The frozen head is only 3,936 trainable params (41 classes × 96-d); it cannot
  reshape representations.

Fine-tuning changes the loss shape: 23.2 → 3.68 (ep100) with a **monotone** AUC
climb 78.1 → 83.16 every 10 epochs.

## Why

mn01 is trained on AudioSet *acoustic events*; its global-average-pooled 96-d
feature encodes "what sound", not "which physical machine id". The STgram-MFN
pretext (self-supervised machine-ID classification) needs the representation
to encode fine machine identity, which only exists after adapting the backbone.

## Conclusion

- The swap must **fine-tune mn01** (all params trainable) to match how A trains
  its whole backbone; otherwise the comparison is one-sided and trivially lost.
- Even fine-tuned, mn01 (0.13 M) trails STgram-MFN (1.16 M) by ~8 AUC points —
  the accuracy ↔ size tradeoff the study set out to characterize.
- Note the ArcFace input width is width-specific (mn01 = 96-d, mn02 = 192-d);
  the head must derive it from the loaded embedder (`train_mn01.py`).

## Reproduction

```bash
# frozen (negative control)
uv run python scripts/training/train_mn01.py --config configs/train_lean.yaml \
    --run_name lean_B1_mn01
# fine-tuned (the comparable cell)
uv run python scripts/training/train_mn01.py --config configs/train_lean.yaml \
    --finetune --ft_batch 128 --run_name lean_B1ft_mn01
```
