# Trace 003 — resumed after reboot; full run auto-armed

- **Date:** 2026-09-14
- **Topic:** baseline
- **Scope:** resume after machine shutdown

## Actions

- Confirmed environment intact after reboot (uptime 1 min; venv, git clean at
  `af8fb42`).
- **Fixed** `scripts/download_data.py`: added `already_extracted()` so a resumed
  run skips machines already on disk instead of re-downloading multi-GB
  archives. Verified: ToyCar + ToyConveyor skipped, `pump` download resumed.
- Relaunched the DCASE 2020 dev + eval download (background).
- Armed `/tmp/asd_full_launcher.sh`: waits until all six dev machines and the
  eval-train splits exist, then launches the full 6-machine 300-epoch run
  (`--config configs/train.yaml --add_root data/raw_eval`), logging to W&B
  project `stgram-mfn`. Times out after 6 h of waiting.

## Logs

- Download: `/tmp/asd_download.log`
- Full run: `/tmp/asd_full_train.log`

## Notes

- The full run uses batch 16 × accum 8 = effective 128 (see
  `docs/analysis/2026-09-14/training/analysis_001.md`).
- If the launcher is not wanted, kill it: `pkill -f asd_full_launcher`.
