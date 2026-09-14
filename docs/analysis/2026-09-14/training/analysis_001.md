# Analysis 001 — GPU memory footprint on the local Quadro P2000

- **Date:** 2026-09-14
- **Topic:** training
- **Question:** can STgram-MFN train on this machine's GPU, and at what batch size?

## Hardware

Quadro P2000, `sm_61` (Pascal), **4096 MiB total (~3.94 GiB usable)**, driver
575.57.08. No fp16 tensor cores.

## Method

Per-process forward + backward + Adam step, fp32, at the reference input shape
(10 s @ 16 kHz, 128 mels, 313 frames, 24 classes), measuring
`torch.cuda.max_memory_allocated()`. Each batch size ran in a **fresh process**
to avoid cross-trial allocator fragmentation.

## Results

| Batch | Peak allocated | Peak reserved | Outcome |
|---|---|---|---|
| 16 | 1.56 GB | 1.77 GB | fits |
| 32 | 3.08 GB | 3.54 GB | fits only with headroom |
| 48 | — | — | **OOM** |
| 64 | — | — | **OOM** |
| 128 (reference) | — | — | **OOM** |

Memory scales ~linearly at **~96 MB/sample**. A real training run at batch 32
OOM'd (a 628 MiB PReLU allocation failed with only ~550 MiB free) even though
the isolated test fit — i.e. batch 32 has no margin once the CUDA context and
transient buffers are included.

## Conclusion

- **Yes, it trains on this GPU**, but not at the reference batch size 128.
- Use **batch 16 × `accum_steps=8` = effective batch 128** (the reference's
  effective batch). `configs/train.yaml` is set to this.
- The 1.16M parameters are negligible; the memory driver is the **activation
  footprint of the 10 s × 128 × 313 inputs** through MobileFaceNet.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set by `train()` to
  reduce fragmentation on this tight card.
- AMP is **not** recommended here: Pascal (GP107) has no fp16 tensor cores and
  runs fp16 at 1/64 rate — it would slow training down, not just fail to help.
- BatchNorm statistics are computed at batch 16, not 128; that is an accepted
  deviation from the reference and is noted for the reproduction comparison.
