# Trace 005 — Colab T4 pivot

- **Date:** 2026-09-14
- **Topic:** baseline
- **Scope:** moving training from the local P2000 to a Colab T4

## Why

Local P2000 is compute-bound at ~60 clips/s (see
`docs/analysis/2026-09-14/training/analysis_002.md`). Colab T4 measured at
**~230 clips/s** (batch 64, 6.1 GB; batch 128, 12.2 GB; 256 OOMs) — ~3.8×.

## Setup

- Ported the Colab workflow from `bicafe`: `.agents/skills/{colab-session-management,
  colab-safe-sync,hf-dataset-fast-resume,hf-model-publishing}` and
  `scripts/colab_{setup,new_session,push_repo,pull_results,stop,check_training}.sh`
  (session `asd`, remote `/content/asd`). Commit `d420981`.
- Session `asd` created (T4, 15 GB). Code pushed via `git archive` overlay.
- Dataset pulled on the VM from Zenodo using the multipart downloader
  (32 connections) — dev + eval done in ~13 min.
- Local P2000 training stopped (redundant).
- Local HF dataset upload (`LakoreAI/stgram-mfn-dcase2020-{dev,eval}`) continues
  in the background for future environments.

## Incidents

- **Kernel blocked:** backgrounding a download with `subprocess.run` captured the
  kernel's output pipe, hanging every later `colab exec`. Fixed with
  `colab restart-kernel` and launching jobs via `subprocess.Popen(...,
  start_new_session=True, stdout=log)` so they fully detach.
- **Training OOM on T4:** a GPU benchmark run *inside the Jupyter kernel* left
  ~14 GB allocated (its model/tensors persist in the kernel namespace), starving
  the training process. Fixed by restarting the kernel before training. Lesson:
  never benchmark in the kernel that will host/adjacent-run training; use a
  detached subprocess.

## Running now

`/content/train_driver.sh` (detached) → 30-epoch trajectory on T4:
`batch_size=64, accum_steps=2` (effective 128), `num_workers=8`, W&B group
`trajectory`, run `stgram_mfn_t4_traj30`. Rate ~77 opt-steps/min → ~3.7 min/epoch
→ finishes ~01:45 local. On completion it pushes the model to
`LakoreAI/stgram-mfn-t4-traj30`.

Logs on VM: `/content/train.log`, `/content/driver.log`.
