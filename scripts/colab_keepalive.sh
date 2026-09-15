#!/usr/bin/env bash
# Keep an active Colab session alive so an idle GPU isn't reclaimed mid-run.
#
# Runs a background `colab keep-alive <endpoint> <session>` loop locally (the
# CLI's own cap is 24 h). Start it right after `colab_new_session.sh`; a lost
# session (404/401) is what killed a training run before this existed.
#
# Usage: scripts/colab_keepalive.sh [session_name]
set -euo pipefail

SESSION="${1:-asd}"
export PATH="$HOME/google-cloud-sdk/bin:$PATH"

ENDPOINT="$(colab --auth=adc sessions 2>/dev/null | grep "\[$SESSION\]" | awk '{print $2}')"
if [ -z "${ENDPOINT:-}" ]; then
  echo "[colab_keepalive] no active session '$SESSION' (run colab_new_session.sh first)"
  exit 1
fi

if pgrep -f "keep-alive $ENDPOINT" >/dev/null 2>&1; then
  echo "[colab_keepalive] keep-alive already running for $SESSION"
  exit 0
fi

LOG="/tmp/colab_keepalive_${SESSION}.log"
nohup colab --auth=adc keep-alive "$ENDPOINT" "$SESSION" >"$LOG" 2>&1 &
echo "[colab_keepalive] started for $SESSION ($ENDPOINT), pid $!, log $LOG"
