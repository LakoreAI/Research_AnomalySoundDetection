#!/usr/bin/env bash
# Push this repo's committed code onto a running Colab session, pull the DCASE
# dataset from Hugging Face, and install the runtime deps.
#
# Usage: scripts/colab_push_repo.sh [session_name] [--skip-data] [--allow-dirty]
#
# `git archive HEAD` ships only TRACKED, COMMITTED files. An uncommitted
# config/script silently won't reach the VM. By default this script refuses to
# push over a dirty working tree; pass --allow-dirty only if you know exactly
# what you are omitting.
#
# Why chunked upload: `colab upload`'s content API rejects requests above
# ~64MB with an opaque `SSLEOFError`. Code tarballs are tiny here, but the
# split/reattach path is kept for safety.
#
# Why not `uv sync` on the VM: pyproject.toml pins torch/torchaudio to the
# `pytorch-cu126` index for the LOCAL machine's Pascal GPU (cu128+ dropped
# sm_61 kernels). Colab's VM already ships a CUDA-matched torch; `uv sync`
# would discard it. This script installs only the non-torch runtime deps and
# reuses whatever torch is preinstalled.
#
# Data: instead of pushing ~14 GB, the VM pulls the DCASE sets from Hugging
# Face (`LakoreAI/stgram-mfn-dcase2020-{dev,eval}`) using HF_TOKEN from the
# local .env. See .agents/skills/hf-dataset-fast-resume.
set -euo pipefail

SESSION="asd"
SKIP_DATA=0
ALLOW_DIRTY=0
for arg in "$@"; do
  case "$arg" in
    --skip-data) SKIP_DATA=1 ;;
    --allow-dirty) ALLOW_DIRTY=1 ;;
    *) SESSION="$arg" ;;
  esac
done

export PATH="$HOME/google-cloud-sdk/bin:$PATH"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="/content/asd"
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT

echo "[colab_push_repo] Packaging tracked code..."
cd "$REPO_ROOT"
if [ "$ALLOW_DIRTY" -eq 0 ] && { [ -n "$(git status --porcelain)" ] || [ -n "$(git ls-files --others --exclude-standard)" ]; }; then
  echo "[colab_push_repo] ERROR: working tree is dirty — git archive ships only committed files."
  echo "[colab_push_repo] Commit or stash first, or pass --allow-dirty to push anyway."
  exit 1
fi
git archive --format=tar.gz -o "$SCRATCH/asd_code.tar.gz" HEAD
echo "[colab_push_repo] Code tarball: $(du -h "$SCRATCH/asd_code.tar.gz" | cut -f1)"

echo "[colab_push_repo] Uploading code..."
colab --auth=adc upload -s "$SESSION" "$SCRATCH/asd_code.tar.gz" /content/asd_code.tar.gz

echo "import pathlib, tarfile
p = pathlib.Path('$REMOTE'); p.mkdir(parents=True, exist_ok=True)
PROTECTED = ('data/', 'checkpoints/', 'results/', 'export/', 'wandb/')
with tarfile.open('/content/asd_code.tar.gz', 'r:gz') as tar:
    for m in tar.getmembers():
        if any(m.name.startswith(x) for x in PROTECTED):
            continue
        tar.extract(m, path=str(p))
print('✓ overlay-extracted code to', p, '(data preserved)')" | colab --auth=adc exec -s "$SESSION"

if [ "$SKIP_DATA" -eq 0 ]; then
  HF_TOKEN_LOCAL="$(grep -E '^HF_(TOKEN|API_KEY)=' "$REPO_ROOT/.env" 2>/dev/null | head -1 | cut -d= -f2- | tr -d '\"'"'"'')"
  if [ -z "$HF_TOKEN_LOCAL" ]; then
    echo "[colab_push_repo] No HF_TOKEN/HF_API_KEY in .env — skipping dataset pull."
    echo "[colab_push_repo] Set it and re-run, or pass --skip-data to silence this."
  else
    echo "[colab_push_repo] Pulling DCASE 2020 dev + eval from Hugging Face on the VM..."
    echo "import os, pathlib
os.environ['HF_TOKEN'] = '''$HF_TOKEN_LOCAL'''
from huggingface_hub import snapshot_download
MACHINES = ['fan', 'pump', 'slider', 'valve', 'ToyCar', 'ToyConveyor']
def complete(root):
    root = pathlib.Path(root)
    return all((root / m / s).is_dir() and any((root / m / s).glob('*.wav'))
               for m in MACHINES for s in ('train', 'test'))
for local, repo in [('$REMOTE/data/raw', 'LakoreAI/stgram-mfn-dcase2020-dev'),
                    ('$REMOTE/data/raw_eval', 'LakoreAI/stgram-mfn-dcase2020-eval')]:
    if complete(local):
        print(f'✓ {local} already complete — bypassing HF download')
    else:
        print(f'pulling {repo} -> {local} ...')
        snapshot_download(repo_id=repo, repo_type='dataset', local_dir=local, max_workers=16)
print('data ready')" | colab --auth=adc exec -s "$SESSION"
  fi
fi

echo "[colab_push_repo] Installing runtime deps (reusing preinstalled torch if CUDA works)..."
echo "import subprocess, sys
try:
    import torch
    has_cuda = torch.cuda.is_available()
except ImportError:
    has_cuda = False
print(f'preinstalled torch usable: {has_cuda}')
deps = ['numpy>=1.26', 'pyyaml>=6.0.3', 'scikit-learn>=1.4', 'soundfile>=0.13.1', 'huggingface_hub>=0.24']
if not has_cuda:
    deps = ['torch', 'torchaudio'] + deps
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q'] + deps)
print('deps installed')" | colab --auth=adc exec -s "$SESSION"

# Deliberately NOT `pip install -e .`: with data/ at the repo root, setuptools'
# flat-layout auto-discovery sees multiple top-level dirs and refuses to guess
# the package. Not needed anyway — run entrypoints with cwd=$REMOTE so `python
# -m src.pipelines.train` resolves `import src` via cwd-on-sys.path.
echo "[colab_push_repo] Done. Repo is live at $REMOTE on session '$SESSION'."
echo "[colab_push_repo] Example: colab --auth=adc exec -s $SESSION <<< 'cd $REMOTE && python -m src.pipelines.train --config configs/train.yaml --add_root data/raw_eval'"
