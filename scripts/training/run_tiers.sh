#!/usr/bin/env bash
# Stage 8 tier ladder — train small -> large -> x-large sequentially on the GPU.
# The "normal" tier is A1 (configs/train_lean.yaml, 1.16 M, already trained).
# Each run pushes best.pt to HF and logs one best W&B artifact.
set -u
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
LOG=/workspace
say() { echo "[tiers] $* $(date)" | tee -a "$LOG/tiers.log"; }

for cfg in tier_small tier_large tier_xlarge; do
    say "start $cfg"
    uv run python scripts/training/train.py --config "configs/$cfg.yaml" > "$LOG/train_$cfg.log" 2>&1
    say "$cfg exit=$?"
done
say "tiers done"
