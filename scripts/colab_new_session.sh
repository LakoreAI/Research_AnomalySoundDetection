#!/usr/bin/env bash
# Provision (or reconnect to) the Colab GPU session this project runs on.
#
# Usage: scripts/colab_new_session.sh [session_name] [gpu]
#   session_name defaults to "asd"
#   gpu defaults to "T4" (free-tier-eligible; A100/H100 need a paid
#   entitlement and will 400 without one)
set -euo pipefail

SESSION="${1:-asd}"
GPU="${2:-T4}"

export PATH="$HOME/google-cloud-sdk/bin:$PATH"

if colab --auth=adc sessions 2>&1 | grep -q "$SESSION"; then
  echo "[colab_new_session] Session '$SESSION' already exists:"
  colab --auth=adc status -s "$SESSION"
else
  echo "[colab_new_session] Creating session '$SESSION' (--gpu $GPU)..."
  colab --auth=adc new -s "$SESSION" --gpu "$GPU"
fi

echo "[colab_new_session] Hardware check:"
echo "import subprocess; print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)" \
  | colab --auth=adc exec -s "$SESSION"
