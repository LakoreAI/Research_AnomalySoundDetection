---
name: colab-session-management
description: >-
  Rules and procedures for managing Google Colab sessions reliably for this
  project. Prevents accidental VM deletion, enforces session re-attachment, and
  bans unnecessary creation of new Colab instances when active assignments
  exist. Use when driving Colab GPU runs via the `colab` CLI.
---

# Colab Session Management & Anti-Recreation Guide

Ported from the sibling `bicafe` project. Session name for this project is
**`asd`**; the remote repo lives at **`/content/asd`**.

## Core Directives

1. **NEVER call `colab new` automatically** when an active assignment exists for
   the workspace.
2. **Always re-attach** to existing assignments in the local CLI state before
   attempting any operational commands.
3. **Keep-Alive Daemon**: maintain a persistent background `colab keep-alive`
   process for active GPU jobs (a 24 h keep-alive cap is the only automatic
   reclaim).

---

## Session Re-attachment Recipe

If `colab exec` throws `404/401 Session 'asd' appears to be lost`, **do NOT
create a new session**. Run this Python snippet to re-bind the local state store
to the active cloud endpoint assignment:

```python
from colab_cli.common import state
from colab_cli.state import SessionState

assignments = state.client.list_assignments()
if assignments:
    a = assignments[0]
    s = SessionState(
        name="asd",
        token=a.runtime_proxy_info.token,
        url=a.runtime_proxy_info.url,
        endpoint=a.endpoint,
        variant=a.variant.name,
        accelerator=a.accelerator.value,
    )
    state.store.add(s)
    print(f"✓ Reattached session 'asd' to endpoint {a.endpoint}")
```

---

## Check Active Session Status

```bash
export PATH="$HOME/google-cloud-sdk/bin:$PATH"
echo "import pathlib, subprocess
ps = subprocess.run(['ps', 'aux'], capture_output=True, text=True).stdout
print('=== Active GPU Processes ===')
for line in ps.splitlines():
    if 'python' in line or 'train' in line:
        print(line)
" | colab --auth=adc exec -s asd
```

---

## Provisioning / stopping

Use the project scripts (they wrap the `colab` CLI and already pass
`--auth=adc`):

- `scripts/colab_setup.sh` — one-time gcloud + `colab` CLI install/login (ADC).
- `scripts/colab_new_session.sh [session] [gpu]` — create or reconnect (`asd`, `T4`); auto-starts keep-alive.
- `scripts/colab_keepalive.sh [session]` — background `colab keep-alive` (24 h cap), prevents idle reclaim.
- `scripts/colab_push_repo.sh` — push committed code (and pull the dataset).
- `scripts/colab_pull_results.sh` — fetch checkpoints/results back.
- `scripts/colab_stop.sh` — stop the session when done.

For long jobs, training configs with `hf_push` push checkpoints to HF every
`ckpt_every` epochs and auto-resume from the latest on startup, so a reclaim
costs minutes rather than the whole run.
