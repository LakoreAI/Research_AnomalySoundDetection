#!/usr/bin/env bash
# Download checkpoints + results from the Colab VM back to this machine, under
# checkpoints/colab_pull_<timestamp>/ (data/ and checkpoints/ are gitignored,
# so this is safe to run repeatedly without polluting git status).
#
# Checkpoints are packaged and chunked defensively: `colab upload` rejects
# >~64MB with an opaque SSLEOFError (see scripts/colab_push_repo.sh); whether
# download has the same limit is unverified, so split rather than assume.
#
# Usage: scripts/colab_pull_results.sh [session_name]
set -euo pipefail

SESSION="${1:-asd}"
REMOTE="/content/asd"
export PATH="$HOME/google-cloud-sdk/bin:$PATH"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
DEST="$REPO_ROOT/checkpoints/colab_pull_${STAMP}"
mkdir -p "$DEST"

echo "[colab_pull_results] Pulling results/ (csv + json, small)..."
echo "import pathlib
for p in sorted(pathlib.Path('$REMOTE/results').rglob('*')):
    if p.is_file():
        print(p.relative_to('$REMOTE'))" | colab --auth=adc exec -s "$SESSION" | while read -r rel; do
  [ -z "$rel" ] && continue
  mkdir -p "$DEST/$(dirname "$rel")"
  colab --auth=adc download -s "$SESSION" "$REMOTE/$rel" "$DEST/$rel" 2>&1 || true
done

echo "[colab_pull_results] Packaging checkpoints on the VM..."
echo "import shutil, pathlib
ck = pathlib.Path('$REMOTE/checkpoints')
if ck.exists() and any(ck.iterdir()):
    shutil.make_archive('/content/asd_checkpoints', 'gztar', ck)
    print('packaged')
else:
    print('no checkpoints found')" | colab --auth=adc exec -s "$SESSION"

echo "import pathlib, subprocess
p = pathlib.Path('/content/asd_checkpoints.tar.gz')
if p.exists():
    subprocess.run(['split', '-b', '45m', str(p), '/content/ckpt_chunks_'])
    print('chunked')
else:
    print('nothing to chunk')" | colab --auth=adc exec -s "$SESSION"

echo "import pathlib
chunks = sorted(pathlib.Path('/content').glob('ckpt_chunks_*'))
print('\\n'.join(c.name for c in chunks))" | colab --auth=adc exec -s "$SESSION" | while read -r name; do
  [ -z "$name" ] && continue
  echo "[colab_pull_results] downloading $name..."
  colab --auth=adc download -s "$SESSION" "/content/$name" "$DEST/$name"
done

if ls "$DEST"/ckpt_chunks_* >/dev/null 2>&1; then
  cat "$DEST"/ckpt_chunks_* > "$DEST/checkpoints.tar.gz"
  rm -f "$DEST"/ckpt_chunks_*
  tar -xzf "$DEST/checkpoints.tar.gz" -C "$DEST"
  rm -f "$DEST/checkpoints.tar.gz"
  echo "[colab_pull_results] Checkpoints extracted to $DEST"
else
  echo "[colab_pull_results] No checkpoint chunks downloaded (none existed on VM yet)."
fi

echo "[colab_pull_results] Logs + results in: $DEST"
