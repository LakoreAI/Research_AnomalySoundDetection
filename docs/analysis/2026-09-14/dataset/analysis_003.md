# Analysis 003 — Zenodo per-connection throttling (download is not link-bound)

- **Date:** 2026-09-14
- **Topic:** dataset
- **Question:** why is a ~1 GB download taking so long? Is the connection bad?

## Finding

The local link is **fast**; Zenodo throttles each connection.

| Target | Throughput |
|---|---|
| `dl.google.com` (link baseline) | **21.15 MB/s** |
| `speed.cloudflare.com` | 403 (blocked, N/A) |
| Zenodo, 1 connection | ~0.15–0.26 MB/s |

WiFi (`wlp59s0`, 192.168.19.16/24, gateway .1), signal −53 dBm, no proxy. IPv6
unavailable. A browser User-Agent did not change Zenodo's rate.

## Parallel scaling (Zenodo, same file, range requests)

| Connections | Aggregate |
|---|---|
| 1 | ~0.25 MB/s |
| 8 | 1.75 MB/s |
| 16 | 5.76 MB/s |
| 32 | **7.94 MB/s** |

Zenodo's limit is **per connection**, not aggregate — throughput scales with
concurrency up to at least 32 connections.

## Fix

`scripts/download_data.py` now does a **multi-connection range download** by
default (`--connections`, default 16; the live run uses 32): split the file by
byte range, fetch parts in a thread pool, concatenate, verify the size, and
fall back to a single stream if Range is unsupported or the size mismatches.
Zenodo's own `size` field is passed in, so no extra HEAD request is needed.

Observed in the real run: **~4.5 MB/s** (vs ~1 MB/s single-stream). Full
dev+eval (~14 GB) drops from ~3.5 h to roughly ~1 h.

## Gotcha encountered

A `rm -f data/raw/_archives/*.part* ...` aborted under zsh because the unmatched
glob raises `no matches found`, leaving a stale partial archive that the
"skip if exists" check then tried to extract (`BadZipFile`). Deleting the file
explicitly (no glob) resolved it. Worth using `setopt NULL_GLOB` or explicit
paths in resume scripts.
