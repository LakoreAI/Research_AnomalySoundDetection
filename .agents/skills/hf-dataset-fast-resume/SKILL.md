---
name: hf-dataset-fast-resume
description: >-
  Procedure for fast-resuming on Hugging Face-hosted datasets for this project.
  Bypasses non-essential metadata/index downloads when the required audio files
  are already present on local disk, and pulls the DCASE 2020 data from HF
  instead of re-hitting Zenodo. Use when setting up a new environment (e.g.
  Colab) or resuming a run.
---

# Hugging Face Dataset Fast-Resume Protocol

Ported from the sibling `bicafe` project, adapted to this project's data.

## Repos

| Local dir | HF dataset repo |
|---|---|
| `data/raw` (DCASE 2020 dev) | `LakoreAI/stgram-mfn-dcase2020-dev` |
| `data/raw_eval` (DCASE 2020 eval) | `LakoreAI/stgram-mfn-dcase2020-eval` |

Both are private; export `HF_TOKEN` (or `HF_API_KEY`) first.

## Problem

`snapshot_download()` blocks until **all** files in the repo are fetched, even
when the required `.wav` files are already on disk. The DCASE sets are ~14 GB
across ~37k files, so a naive re-download is very expensive.

## Solution Pattern: readiness check before download

Check that every machine/split directory already holds audio before calling
`snapshot_download()`:

```python
import pathlib
from huggingface_hub import snapshot_download

MACHINES = ["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"]

def complete(root: pathlib.Path) -> bool:
    root = pathlib.Path(root)
    for m in MACHINES:
        for split in ("train", "test"):
            d = root / m / split
            if not d.is_dir() or not any(d.glob("*.wav")):
                return False
    return True

pairs = [
    ("data/raw", "LakoreAI/stgram-mfn-dcase2020-dev"),
    ("data/raw_eval", "LakoreAI/stgram-mfn-dcase2020-eval"),
]
for local, repo in pairs:
    if complete(local):
        print(f"✓ {local} already complete — bypassing HF download")
    else:
        print(f"pulling {repo} -> {local} ...")
        snapshot_download(
            repo_id=repo, repo_type="dataset",
            local_dir=local, max_workers=16,
        )
```

Note: `data/raw_eval` is the *additional training* + *evaluation* set; only
`<machine>/train` is needed as `add_root` for training, but pulling `test` too
matches the downloader's layout.

## Benefits

- Instant resume across execution restarts / new Colab sessions.
- Avoids Zenodo's per-connection throttle (see
  `docs/analysis/2026-09-14/dataset/analysis_003.md`).
