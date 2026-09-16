---
name: colab-safe-sync
description: >-
  Safe code-repository synchronization protocol for remote VM environments
  (Colab). Prevents accidental deletion of downloaded datasets, checkpoints,
  results, and logs when re-pushing code. Use before syncing code to Colab.
---

# Colab Safe Code Sync Protocol

Ported from the sibling `bicafe` project, adapted for this repo.

## Problem

Using destructive directory removal (`rm -rf /content/asd` or
`shutil.rmtree('/content/asd')`) before unpacking an updated code archive
permanently wipes hours of downloaded datasets (`data/`, `data/raw_eval/`),
training checkpoints, results, and logs.

---

## Solution Pattern: In-Place Archive Overlay Extraction

Always sync code updates using `tarfile.extractall()` overlay extraction,
explicitly preserving protected data directories:

```python
import tarfile, pathlib

archive_path = pathlib.Path("/content/asd_code.tar.gz")
target_dir = pathlib.Path("/content/asd")
target_dir.mkdir(parents=True, exist_ok=True)

PROTECTED = ("data/", "checkpoints/", "results/", "export/", "wandb/")
with tarfile.open(archive_path, "r:gz") as tar:
    for member in tar.getmembers():
        if any(member.name.startswith(p) for p in PROTECTED):
            continue
        tar.extract(member, path=target_dir)

print("✓ Overlay code sync complete. Data directories preserved!")
```

---

## Shell Helper (`scripts/colab/colab_push_repo.sh`)

In bash, export code archives with `--exclude`:

```bash
tar --exclude='data' \
    --exclude='checkpoints' \
    --exclude='results' \
    --exclude='export' \
    --exclude='wandb' \
    --exclude='.venv' \
    --exclude='.git' \
    -czf /tmp/asd_code.tar.gz .
```

`scripts/colab/colab_push_repo.sh` uses `git archive HEAD` (tracked, committed files
only) and refuses a dirty working tree by default — an uncommitted config would
silently not reach the VM otherwise.
