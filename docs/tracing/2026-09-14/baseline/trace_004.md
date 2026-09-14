# Trace 004 — pivot to 30-epoch trajectory run + HuggingFace pushes

- **Date:** 2026-09-14
- **Topic:** baseline
- **Scope:** scheduling decision + HF hand-off for cross-environment use

## Decision

Option 3 from the earlier discussion: run a **30-epoch trajectory check** first
instead of the full 300-epoch reference schedule.

Rationale: measured training throughput is **65.8 clips/s** on the local Quadro
P2000 (batch 16), so the full schedule is **~26 h**. A 30-epoch run is ~2.5 h
and tells us whether the AUC trajectory justifies the full run.

## Changes

- `configs/train_trajectory.yaml` (new): 30 epochs, constant LR 1e-4 (the
  300-epoch cosine `t_max` would be wrong at this length), `eval_every: 5`,
  `add_root: data/raw_eval`, W&B group `trajectory`.
- `scripts/push_dataset_to_hf.py`: added `--large` (resumable
  `upload_large_folder`) for multi-GB datasets.
- Trained model pushed to `LakoreAI/stgram-mfn-traj30` (private).
- Dataset pushed to `LakoreAI/stgram-mfn-dcase2020-dev` and
  `LakoreAI/stgram-mfn-dcase2020-eval` (private) for use on other environments.

## Armed background jobs

| Job | Trigger | Action | Log |
|---|---|---|---|
| `/tmp/asd_traj_launcher.sh` | all 6 dev + eval-train present | train 30 epochs → push model to HF | `/tmp/asd_traj_train.log` |
| `/tmp/asd_dataset_push.sh` | download process exits + data complete | `upload_large_folder` dev + eval | `/tmp/asd_dataset_push.log` |

## Notes

- HF upload measured at **~1.6 MB/s** (asymmetric link), so the ~14 GB dataset
  upload is ~2.4 h. It runs concurrently with GPU-bound training.
- HF namespace: `LakoreAI` (authenticated user `minhleduc` is org admin).
- No artifacts are logged to W&B (`log_artifacts: false`); HF is the model/data
  hand-off path.
