# Report 001 — Reliability + metric upgrades (mAUC, HF checkpoint/resume)

- **Date:** 2026-09-15
- **Milestone:** M1 (Baseline Reproduction)
- **Status:** implemented and unit-tested
- **Task:** reports / baseline

## Motivation

The 2026-09-14 T4 run was reclaimed by Colab ~1 h in, losing the checkpoint, and
its evaluation used AUC/pAUC only — not the reference's headline **mAUC**
(minimum AUC across machine ids). Both are fixed here.

## Changes

### 1. mAUC metric (`src/pipelines/eval.py`)
- `evaluate` now returns `mauc` (per machine: minimum id AUC; overall: mean
  across machine types) alongside `auc`/`pauc`, plus per-machine `mauc`.
- `format_report` prints all three; the training loop logs `mauc` to W&B and
  `train_log.json`.

### 2. Incremental HF checkpointing + auto-resume
- `src/utils/hub_utils.py`: `push_file`, `list_run_files`, `pull_run`,
  `latest_epoch_checkpoint` (optional `huggingface_hub`; fails soft at call
  sites).
- `TrainingConfig.hf_push = {enabled, repo_id}` and CLI `--hf_push_repo`.
- `train.py`: after each periodic checkpoint, push it to
  `<repo>/<run_name>/epoch_N.pt`; after each validation, push `best.pt`. On
  startup, if `resume_from` is unset and `hf_push` is enabled, pull the run's
  checkpoints from HF and resume from the latest epoch.
- Enabled in `configs/train_t4_long.yaml` (repo `LakoreAI/stgram-mfn-t4-long60`)
  and `configs/train_trajectory.yaml` (`LakoreAI/stgram-mfn-t4-traj30`).

### 3. Colab keep-alive
- `scripts/colab_keepalive.sh` runs a background `colab keep-alive <endpoint>
  <session>` (24 h cap); `colab_new_session.sh` now calls it automatically.

## Verification

- `uv run pytest -q` → **25 passed** (added mAUC assertions and
  `tests/test_hub_utils.py`).
- HF push/pull smoke against a throwaway repo: push `epoch_5.pt` + `best.pt`,
  list, pull, resolve latest, delete — all OK.

## Notes

- `pull_run` writes flat filenames into the local checkpoint dir, so a resumed
  run finds `epoch_N.pt` exactly where the loop expects it.
- Pushing `best.pt` every validation is ~14 MB; at Colab upload speed that is
  seconds, and it bounds worst-case loss to one eval interval.
