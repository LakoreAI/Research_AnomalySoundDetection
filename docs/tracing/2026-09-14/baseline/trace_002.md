# Trace 002 — session paused for machine shutdown

- **Date:** 2026-09-14
- **Topic:** baseline
- **Scope:** end of session; all background jobs stopped, work committed

## Actions

- Stopped the background DCASE download (was downloading `dev_data_pump.zip`).
- Confirmed no training / canary / wandb processes remain.
- Deleted the partial `data/raw/_archives/dev_data_pump.zip` so the downloader's
  "skip if exists" check does not resume onto a corrupt archive.
- Committed all working changes.

## State at shutdown

**Data (`data/raw/`)** — DCASE 2020 dev, 2 of 6 machines extracted:
- `ToyCar` ✅, `ToyConveyor` ✅
- `pump`, `valve`, `fan`, `slider` — not downloaded
- `data/raw_eval/` — not started (DCASE 2020 eval additional + test)

**Checkpoints** (gitignored): `checkpoints/canary_toycar/`, `checkpoints/wandb_smoke/`.

## Resume instructions

```bash
# 1) finish DCASE 2020 dev (skips ToyCar/ToyConveyor already extracted)
uv run python scripts/download_data.py --dataset dcase2020 --dest data/raw
# 2) DCASE 2020 eval (additional train + test)
uv run python scripts/download_data.py --dataset dcase2020-eval --dest data/raw_eval
# 3) full 6-machine run, batch 16 x accum 8, logging to W&B project stgram-mfn
uv run python scripts/train.py --config configs/train.yaml --add_root data/raw_eval
```

Note: always pass `--config configs/train.yaml`; without it the dataclass
default `batch_size=128` is used and OOMs on the local GPU.
