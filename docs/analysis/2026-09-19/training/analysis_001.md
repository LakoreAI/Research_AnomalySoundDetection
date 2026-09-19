# Analysis 001 — eval path stalled the GPU; dataloader, not batch, is the lever

- **Date:** 2026-09-19
- **Topic:** training
- **Question:** B1 fine-tune showed ~20–60% GPU utilization and long stalls —
  is batch size the problem, or is the pipeline mis-configured?

## Finding

Three separate issues, none of them batch size.

**1. The eval loop loaded test clips serially in the main process.**
`evaluate()` (STgram) and `evaluate_mn01_ft()` (mn01) iterated the test files
one at a time, running `soundfile` + STFT in the training process. On the
**10,868 test clips** this pegged the main process at ~292% CPU while the GPU
sat at **0%** for minutes per eval (every 10 epochs). This blocked training and
is what produced the "GPU not optimized" reading.

Fix: load test clips through a worker `DataLoader` (`num_workers` from config)
and batch the forward. After the fix the STgram eval is byte-identical
(`num_workers=0` vs `2` produce the same AUC/pAUC/mAUC — verified), and the GPU
stays busy.

**2. Fine-tune batch was 32 vs A1's 128 — a protocol mismatch, not a speed
lever.** The mn01 fine-tune initially ran at `--ft_batch 32`; VRAM was only
927 MB / 24 GB, so it was not memory-limited. Raising it to **128** (matching
A1) lifted utilization from ~22–58% to ~60–97% and cut epoch time to ~50 s
(36,283 clips → ~725 clips/s). GPU compute is the limit; batch only mattered
because it matched the reference effective batch.

**3. Worker CPU was low (~7% each), so adding workers alone did not help.**
The bottleneck was host-side orchestration around many small ops for a tiny
model on 32 kHz (320k-sample) inputs, plus the serial eval. Parallelizing eval
addressed the dominant term.

## Conclusion

- For this workload the lever is the **eval/data path**, not VRAM or batch:
  bigger batches fit easily but do not raise throughput once the GPU is
  saturated (consistent with `2026-09-14/training/analysis_002.md`).
- Keep `--ft_batch 128` for the A/B protocol, `num_workers` ≥ 8, and always
  parallelize test loading.

## Changes shipped

- `src/pipelines/eval.py`: `evaluate(..., num_workers=0)` — worker-DataLoader
  scoring; default preserves old behavior; `train.py` passes
  `train_cfg.num_workers`.
- `scripts/training/train_mn01.py`: same for the fine-tune eval.
- `-u` (unbuffered) logging so progress is visible in real time.
