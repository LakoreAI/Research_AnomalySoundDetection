# Analysis 001 — DCASE 2020 dataset download throughput

- **Date:** 2026-09-14
- **Topic:** dataset
- **Question:** how long will the DCASE 2020 dev + eval download take, and is the
  bottleneck the client or the source?

## Method

- Measured the live download started by `scripts/download_data.py`
  (Zenodo record 3678171, `dev_data_ToyCar.zip`, 1.82 GB).
- Measured a clean single-connection `curl` range request (12 s window).
- Measured 4 parallel `curl` range requests (12 s window) to test per-connection
  vs aggregate limits.
- Spot-checked a non-Zenodo host (GitHub) as a network baseline.

## Findings

| Measurement | Result |
|---|---|
| Live `urllib` download (60 s window) | ~0.98 MB/s |
| Clean single `curl` range | ~0.36 MB/s |
| 4 parallel `curl` connections | ~0.48 MB/s **aggregate** (no gain) |
| GitHub (unrelated) | ~0.14 MB/s (small page) |

- Parallel connections do **not** increase aggregate throughput → not a
  per-connection Zenodo throttle.
- A non-Zenodo host is also slow → the bottleneck is the **local network link**,
  not the source.
- Effective sustained rate for the live download: **~0.9–1.0 MB/s**.

## Estimate

Total payload: dev ≈ 7.5 GB + eval (additional train + test) ≈ 6.1 GB ≈ **13.6 GB**.
At ~1 MB/s → **~3.5–4 h**.

## Implication

- Let the download run unattended in the background; do not attempt to speed it
  up with multi-connection tooling (measured to not help).
- The real-data canary should trigger on the first extracted machine rather than
  waiting for the full set (watcher armed on `ToyCar`).
- If the link degrades, prioritise the 6 dev machines over the eval set — dev is
  sufficient for the primary DCASE 2020 baseline comparison.

## Status at write time

ToyCar at 36% (634 MB / 1.82 GB) after ~12 min. Log: `/tmp/asd_download.log`.
