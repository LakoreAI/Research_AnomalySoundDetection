#!/usr/bin/env bash
# Launch STgram-MFN training in a detached session so an SSH / Colab disconnect
# never kills the run.
#
# The training loop already pushes checkpoints to HF when the config sets
# `hf_push` and auto-resumes from the latest epoch on startup, so a VM reclaim
# costs at most one `eval_every` interval. This script only adds the "keep it
# running while my client is gone" layer (tmux, or nohup as a fallback).
#
# Usage:
#   scripts/training/run_detached.sh [start|attach|status|stop] [options] [-- <train args>]
#
# Options:
#   -c, --config <path>   training YAML (default: configs/train.yaml)
#   -n, --name <name>     session name (default: asd-train)
#   -l, --log  <path>     log file (default: logs/<name>.log)
#   -h, --help            show this help
#
# Examples:
#   scripts/training/run_detached.sh start --config configs/train_t4_long.yaml
#   scripts/training/run_detached.sh start --config configs/train.yaml -- --machines ToyCar --epochs 3
#   scripts/training/run_detached.sh status
#   scripts/training/run_detached.sh attach
#   scripts/training/run_detached.sh stop
#
# Train args after `--` are passed straight to `python -m src.pipelines.train`.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

ACTION="start"
CONFIG="configs/train.yaml"
SESSION="asd-train"
LOG=""
EXTRA=()

usage() {
  sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

# --- parse args (options before `--`, train args after) ---
while [ $# -gt 0 ]; do
  case "$1" in
    start|attach|status|stop) ACTION="$1" ;;
    -c|--config) CONFIG="${2:?--config needs a path}"; shift ;;
    -n|--name)   SESSION="${2:?--name needs a value}"; shift ;;
    -l|--log)    LOG="${2:?--log needs a path}"; shift ;;
    -h|--help)   usage 0 ;;
    --)          shift; EXTRA=("$@"); break ;;
    *) echo "[run_detached] unknown argument: $1" >&2; usage 1 ;;
  esac
  shift
done

[ -n "$LOG" ] || LOG="logs/${SESSION}.log"
PIDFILE="logs/${SESSION}.pid"

# Prefer uv (repo convention); fall back to the venv, then plain python.
if command -v uv >/dev/null 2>&1 && [ -f pyproject.toml ]; then
  LAUNCH=(uv run python)
elif [ -x .venv/bin/python ]; then
  LAUNCH=(.venv/bin/python)
else
  LAUNCH=(python)
fi

TRAIN=("${LAUNCH[@]}" -m src.pipelines.train --config "$CONFIG")
[ ${#EXTRA[@]} -gt 0 ] && TRAIN+=("${EXTRA[@]}")

have_tmux() { command -v tmux >/dev/null 2>&1; }

session_running() {
  have_tmux && tmux has-session -t "$SESSION" 2>/dev/null
}

do_start() {
  if session_running; then
    echo "[run_detached] session '$SESSION' is already running — attach:"
    echo "  scripts/training/run_detached.sh attach   (or stop first)"
    exit 1
  fi
  if [ ! -f "$CONFIG" ]; then
    echo "[run_detached] config not found: $CONFIG" >&2
    exit 1
  fi

  mkdir -p logs

  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[run_detached] GPU:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true
  else
    echo "[run_detached] WARNING: nvidia-smi not found — is this a GPU box?"
  fi

  # Inner command: cd, run, stream to the log.
  local run_str inner
  printf -v run_str '%q ' "${TRAIN[@]}"
  inner="cd $(printf '%q' "$REPO_ROOT") && ${run_str}2>&1 | tee -a $(printf '%q' "$LOG")"

  echo "[run_detached] config : $CONFIG"
  echo "[run_detached] log    : $LOG"
  echo "[run_detached] cmd    : ${TRAIN[*]}"

  if have_tmux; then
    tmux new-session -d -s "$SESSION" "bash -lc $(printf '%q' "$inner")"
    sleep 1
    if session_running; then
      echo "[run_detached] started detached tmux session '$SESSION'."
      echo "[run_detached] attach: scripts/training/run_detached.sh attach"
    else
      echo "[run_detached] tmux session died immediately — last log lines:" >&2
      tail -n 20 "$LOG" 2>/dev/null || true
      exit 1
    fi
  else
    echo "[run_detached] tmux not found — falling back to nohup."
    nohup bash -c "$inner" >/dev/null 2>&1 &
    echo $! > "$PIDFILE"
    echo "[run_detached] started pid $(cat "$PIDFILE") (stop with: kill \$(cat $PIDFILE))"
  fi
  echo "[run_detached] status: scripts/training/run_detached.sh status"
}

do_attach() {
  if ! session_running; then
    echo "[run_detached] no running tmux session '$SESSION'." >&2
    exit 1
  fi
  exec tmux attach -t "$SESSION"
}

do_status() {
  echo "=== session ==="
  if session_running; then
    echo "tmux '$SESSION' running"
  elif [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "nohup pid $(cat "$PIDFILE") running"
  else
    echo "'$SESSION' not running"
  fi

  echo "=== log tail ($LOG) ==="
  if [ -f "$LOG" ]; then tail -n 15 "$LOG"; else echo "(no log yet)"; fi

  echo "=== checkpoints ==="
  local cks
  cks=$(ls -1t checkpoints/*/best.pt 2>/dev/null | head -5 || true)
  [ -n "$cks" ] && echo "$cks" || echo "(none yet)"

  echo "=== latest train_log.json ==="
  "${LAUNCH[@]}" - <<'PY' || echo "(no train_log.json yet)"
import json, pathlib
logs = sorted(pathlib.Path("results").glob("*/train_log.json")) or sorted(
    pathlib.Path("checkpoints").glob("*/train_log.json")
)
if not logs:
    raise SystemExit(0)
data = json.loads(logs[-1].read_text())
print(f"{logs[-1]}  ({len(data)} epochs)")
for rec in data[-5:]:
    print(" ", rec)
PY
}

do_stop() {
  if session_running; then
    tmux kill-session -t "$SESSION"
    echo "[run_detached] killed tmux session '$SESSION'."
  elif [ -f "$PIDFILE" ]; then
    kill "$(cat "$PIDFILE")" 2>/dev/null || true
    rm -f "$PIDFILE"
    echo "[run_detached] stopped nohup pid."
  else
    echo "[run_detached] nothing to stop for '$SESSION'."
  fi
}

case "$ACTION" in
  start)  do_start ;;
  attach) do_attach ;;
  status) do_status ;;
  stop)   do_stop ;;
  *)      usage 1 ;;
esac
