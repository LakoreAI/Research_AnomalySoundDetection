# Trace 001 — RunPod A5000 session: lean A/B matrix

- **Date:** 2026-09-19
- **Task:** tracing / runpod
- **Runtime:** RunPod RTX A5000 24 GB (driver 580 / CUDA 13.0), 40 GB disk.

## Timeline

1. **Ckey.vn went down** (backend DB unreachable — every route returned
   `Không thể kết nối đến cơ sở dữ liệu`), so the run moved to RunPod
   (`docs/HARDWARE.md`). Picked A5000 (Ampere `sm_86`, runs the `cu126` pin).
2. **SSH key friction.** Pasted the key *fingerprint* (`SHA256:…`) → RunPod
   rejected it (`Key must begin with ssh-…`). Correct full public key is
   `~/.ssh/id_ed25519.pub`; appended it to the pod via the proxy web shell to
   unblock exposed-TCP SSH.
3. **Provisioned:** `uv` install, repo synced from local (private GitHub →
   rsync over the exposed TCP tunnel), `uv sync --no-cache` (40 GB disk budget).
4. **Data pull (dev 30,988 + eval 23,268 = 54k tiny wav files).** HF served
   ~8 files/s (~20 Mbps) despite the pod doing 47 MB/s on large files —
   per-file overhead, not bandwidth. `hf_transfer` + 16 workers did not fix the
   small-file rate; total ~2 h. **Incident:** one restart command had a
   malformed path expression that created a literal `"` directory holding
   `ToyCar/`; removed (208 MB). Correct data was always under `data/raw`.
5. **A1 (STgram-MFN, 100 ep, batch 128, AMP):** AUC 89.9 → **91.45**, mAUC
   83.76. M1 gate passed. Pushed `best.pt` to `LakoreAI/stgram-mfn-lean`.
6. **Decided to skip A1const** (constant-LR ablation): A1 already cleared the
   gate and the `traj30` evidence was unverifiable (those W&B runs are crashed
   with zero synced history).
7. **B1 frozen mn01 + ArcFace:** implemented `train_mn01.py`; AUC peaked 64.36
   (ep10) then fell to 58.5 — frozen embeddings do not transfer.
8. **B1-ft (fine-tuned mn01 + ArcFace):** implemented `--finetune`; batch 128,
   parallel eval. AUC 78.1 → **83.16**. See
   `docs/analysis/2026-09-19/mn01/analysis_001.md`.
9. **Pipeline fixes:** parallel test-set eval in both paths; `-u` logging
   (`docs/analysis/2026-09-19/training/analysis_001.md`).
10. **Stage-8 readiness:** mn01 width/checkpoint factory (`--mn01_name`) and
    STgram `arch` override + derived `spatial_size`.
11. **Git:** commit `285a7b8` pushed to `origin/main`.

## Incidents / lessons

- Ckey DB outage → provider pivot mid-plan.
- RunPod needs the *public key*, not the fingerprint.
- `pkill -f "<pattern>"` in a one-shot SSH command can match and kill its own
  shell — use a bracketed pattern or kill by PID (hit twice).
- Don't assert storage savings for precomputed features: STgram needs the raw
  waveform too, so `.npy` features are larger than 16-bit wav.

## Artifacts

- HF model repo: `LakoreAI/stgram-mfn-lean` (runs `lean_A1_stgram`,
  `lean_B1_mn01`, `lean_B1ft_mn01`).
- W&B project `stgram-mfn`: `r2vc9mcm` (A1), `gy8lds0d` (B1 frozen),
  `at5143fu` (B1-ft).
