# Report 003 — W&B logging + machine-appropriate batch settings

- **Date:** 2026-09-14
- **Milestone:** M1 (Baseline Reproduction)
- **Status:** passed — W&B logging live, no artifacts; GPU batch sizing resolved
- **Task:** reports / baseline

## Changes

- `configs/train.yaml`: `wandb.enabled: true`, `log_artifacts: false`; batch
  settings set to `batch_size: 16`, `accum_steps: 8` (effective 128),
  `num_workers: 2`.
- `src/pipelines/train.py`: `train()` now calls `load_env(.env)` (so
  `WANDB_API_KEY` / `HF_API_KEY` are picked up automatically) and sets
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- `pyproject.toml`: added `hub` extra (`huggingface_hub`); `wandb` extra already
  present.

## W&B smoke run

`scripts/train.py --config configs/train.yaml --data_root data/raw --machines ToyCar --epochs 1 --run_name wandb_smoke`

- Run: https://wandb.ai/octoopt/stgram-mfn/runs/61zkyhya (project `stgram-mfn`)
- Logged: `train/loss`, `train/lr`, `eval/auc`, `eval/pauc`, `eval/val_loss`
- Artifacts uploaded: **0** (as required)
- Result (not meaningful — 32 optimizer steps): ToyCar AUC 68.97 / pAUC 58.97

## Notes

- The first smoke attempt OOM'd because it was launched **without `--config`**,
  so the dataclass default `batch_size=128` was used. Correct invocation always
  passes `--config configs/train.yaml` (as in `scripts/README.md`).
- Batch sizing rationale: `docs/analysis/2026-09-14/training/analysis_001.md`.

## Next steps

1. Full DCASE 2020 dev + eval download (~2.5 h remaining).
2. Full 6-machine, 300-epoch run at batch 16 × accum 8, logging to
   `stgram-mfn` on W&B.
