#!/usr/bin/env bash
# Stop a Colab session. Idle GPU sessions burn compute units with nothing to
# show for it -- always stop when done (a 24 h keep-alive cap is the only
# automatic reclaim).
#
# Usage: scripts/colab/colab_stop.sh [session_name]
set -euo pipefail

SESSION="${1:-asd}"
export PATH="$HOME/google-cloud-sdk/bin:$PATH"

colab --auth=adc stop -s "$SESSION"
echo "[colab_stop] Session '$SESSION' stopped."
