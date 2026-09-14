# Analysis 002 — batch size does not improve throughput (GPU compute-bound)

- **Date:** 2026-09-14
- **Topic:** training
- **Question:** the GPU showed 39% at one point — will a larger batch raise
  utilization and speed up training?

## Finding

The GPU is **not** under-utilized; the 39% reading was transient (startup /
eval / data-loading gap). During training it sits at **86–100%** (typically
~94%).

Measured on the real dataset with the real DataLoader (4 workers, 30 s windows,
expandable segments):

| Batch | Throughput | ms/step | Peak GPU mem |
|---|---|---|---|
| 16 | 60.8 clips/s | 263 | 1.55 GB |
| 32 | 61.8 clips/s | 518 | 3.07 GB |

Batch 32 buys **+1.6%** throughput for **2× the memory** (peak 3.06 GB on a
3.94 GB card → OOM risk). Batch 48+ OOMs outright.

## Conclusion

The Quadro P2000 is **compute-bound at ~60 clips/s** for this model/input
(10 s × 128 × 313). Batch size is not the lever:

- Keep `batch_size: 16` (equal speed, half the memory, no OOM risk).
- More DataLoader workers do not help either (2 vs 4 workers: same ~60 clips/s),
  so the input pipeline is not the bottleneck.
- To finish sooner, reduce **epochs** or use a **data subset** — not batch size.

## Implication for the schedule

36,283 train clips / 60 clips/s ≈ **10 min/epoch** → the 30-epoch trajectory is
**~5 h** (plus ~2 min per eval × 6). The earlier ~4 h estimate assumed the
optimistic 65.8 clips/s from a warm-cache micro-benchmark.
