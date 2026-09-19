#!/usr/bin/env bash
# Lean experiment matrix driver (RunPod A5000).
#
# Waits for the running A1 to finish, then runs B1 with the SAME protocol as
# A1 (configs/train_lean.yaml, cosine LR) so the A-vs-B comparison is valid.
# A1const (constant-LR ablation, configs/train_lean_constlr.yaml) was dropped
# 2026-09-19: A1 already clears the gate and the traj30 evidence was
# unverifiable. Each run pushes best.pt to HF and logs to W&B; survives SSH
# disconnects (nohup).
set -u
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
LOG=/workspace
say() { echo "[driver] $* $(date)" | tee -a "$LOG/driver.log"; }

# 1. wait for A1 (bracket trick avoids self-match)
while pgrep -f "lean_A1_stgr[a]m" >/dev/null 2>&1; do sleep 60; done
say "A1 finished"

say "start B1 (mn01), same protocol as A1"
uv run python scripts/training/train_mn01.py \
    --config configs/train_lean.yaml \
    --run_name lean_B1_mn01 \
    --hf_push_repo LakoreAI/stgram-mfn-lean \
    --mn01_ckpt checkpoints/mn01_as_mAP_298.pt > "$LOG/train_B1.log" 2>&1
say "B1 exit=$?"

say "matrix training done"
