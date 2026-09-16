#!/usr/bin/env bash
# Poll a training run on the Colab VM: checkpoints, the latest epochs from the
# run's train_log.json, and running python processes.
#
# Usage: scripts/colab/colab_check_training.sh [session_name]
set -euo pipefail

SESSION="${1:-asd}"
REMOTE="/content/asd"
export PATH="$HOME/google-cloud-sdk/bin:$PATH"

echo "import pathlib, json, subprocess
base = pathlib.Path('$REMOTE')
print('=== checkpoints ===')
cks = sorted(base.glob('checkpoints/*/best.pt'))
if cks:
    for p in cks:
        print(' ', p.relative_to(base), p.stat().st_size // 1024, 'KB')
else:
    print(' (none yet)')
print('=== latest train_log.json ===')
logs = sorted(base.glob('results/*/train_log.json')) or sorted(base.glob('checkpoints/*/train_log.json'))
if logs:
    data = json.loads(logs[-1].read_text())
    print(' file:', logs[-1].relative_to(base), '| epochs logged:', len(data))
    for rec in data[-5:]:
        print(' ', rec)
else:
    print(' (no train_log.json yet)')
print('=== running python processes ===')
ps = subprocess.run(['ps', 'aux'], capture_output=True, text=True).stdout
for line in ps.splitlines():
    if 'python' in line and 'grep' not in line:
        print(' ', line[:150])
" | colab --auth=adc exec -s "$SESSION"
