# Experiment Runbook

A gated checklist for walking the comparison matrix
`{STgram-MFN, mn01} × {FP32, INT8, domain shift}`, one cell at a time. Each
stage runs only after the previous gate passes; a failed gate names its
fallback. This is the operational companion to [`TASKS.md`](TASKS.md) (roadmap)
and [`REFERENCE.md`](REFERENCE.md) (specs).

Legend: **`A` = STgram-MFN**, **`B` = mn01**. A cell is backbone × condition
(`A1` = A FP32, `A2` = A INT8, `A3` = A domain shift).

## Ground rules

- **One protocol, both backbones.** Same runtime, config, seed, and
  best-epoch rule for A and B. The current best-epoch rule selects on the
  official test AUC — a known caveat (see [`NOTES.md`](NOTES.md)) — but it
  must be *identical* across cells or the comparison is meaningless.
- **Record every cell:** `run_name`, config, seed, git SHA, best epoch,
  AUC / pAUC / mAUC, parameter count, and (later) size / latency. Persist to
  `results/<run>/train_log.json` and W&B (project `stgram-mfn`).
- **Compare only within a runtime.** Colab T4 and local P2000 numbers are not
  interchangeable; pick one for the whole matrix.

## Flow

```
A1 FP32 ─gate─► B1 FP32 ─gate─► A2 + B2 INT8 ─gate─► A3 + B3 domain shift ─► edge ─► write-up
```

## Stage 0 — Pre-flight (once)

- [ ] `uv sync && uv run pytest -q` → green (30 tests).
- [ ] Data present: `data/raw` (DCASE 2020 dev) + `data/raw_eval` (eval), all 6
      machines, `train/` + `test/`.
- [ ] `.env` holds `WANDB_API_KEY` and `HF_API_KEY`.
- [ ] Runtime chosen and fixed for the matrix: Colab T4
      (`configs/train_t4_long.yaml`, ~4 h, HF checkpoint push/auto-resume),
      local P2000 (`configs/train_trajectory.yaml`, 30 epochs ≈ 5 h), or a
      rented GPU — see [`HARDWARE.md`](HARDWARE.md) for the config and shortlist.
- [ ] Launch it detached so a disconnect can't kill it:
      `scripts/training/run_detached.sh start --config <yaml>`.
- [ ] Protocol frozen: config, seed, and best-epoch rule.

## Stage 1 — `A1` STgram-MFN FP32 baseline (M1 · task 11)

- [ ] Train:
      `uv run python scripts/training/train.py --config configs/train_t4_long.yaml --add_root data/raw_eval`
- [ ] Evaluate:
      `uv run python scripts/training/evaluate.py --ckpt checkpoints/<run>/best.pt --data_root data/raw`
- [ ] Record AUC / pAUC / mAUC, params (1,161,536 @ 42 classes), best epoch, W&B link.

**Gate A → B:** avg AUC ≥ **90.4** (within ~2 of published 92.36) **and** mAUC
≥ **82.9** (within ~2 of 84.86).
- Pass → Stage 2.
- Fail → fallback (from `TASKS.md`): document the gap, adopt the in-house run as
  the reference for later comparisons, and make **no** swap claims until it is
  credible.

## Stage 2 — mn01 training path (M2 prereq · tasks 13–14)

- [ ] `src/modules/mn01.py` (parity-tested) exists. **Still missing:** a
      training entry point that maps `mn01(waveform) → 96-d → ArcFace`:
      upsample MIMII 16 kHz → 32 kHz, keep only the ArcFace head.
- [ ] Canary it: overfit a single batch, then a short `ToyCar` run.

**Gate:** finite loss that decreases and non-trivial AUC on the canary.
- Pass → Stage 3.
- Fail → fix the adapter before spending GPU hours on a full run.

## Stage 3 — `B1` mn01 FP32 (M2 · task 15)

- [ ] Run with the **same** protocol as `A1`.

**Gate A vs B:** `B1` within ~2 AUC points of `A1`, **or** a clear
accuracy ↔ params/latency tradeoff (B is 123,783 params vs A's 1.16 M). A
negative result still proceeds — it is a publishable datapoint.

## Stage 4 — INT8 PTQ `A2` + `B2` (M3 · tasks 16–18)

- [ ] Export: `uv run python scripts/edge/export_onnx.py --ckpt <best.pt> --out export/<cell>.onnx`
- [ ] Static QDQ (preferred):
      `uv run python scripts/edge/quantize_onnx.py --onnx export/<cell>.onnx --out export/<cell>_int8.onnx --mode static --calib_root data/raw`
      (dynamic INT8 was ~3.5× *slower* on CPU — do not use it for the result.)
- [ ] Re-evaluate the INT8 graph; record `ΔAUC_A = A1 − A2` and `ΔAUC_B = B1 − B2`.

**Gate:** `ΔAUC < ~3` points for **at least one** backbone.
- Pass → Stage 5.
- Fail both → QAT fallback (task 18).

## Stage 5 — Domain shift `A3` + `B3` (M4 · task 19)

- [ ] Acquire the MIMII Domain Generalization held-out set (not yet under
      `data/`; check the datasets in `scripts/data/download_data.py`).
- [ ] Score both backbones (FP32 and INT8) on it; report descriptively.

**Gate:** the backbone ranking is stable from in-domain to domain shift, and the
INT8 delta stays comparable. A flip is a finding, not a failure.

## Stage 6 — Edge (M4 · tasks 20–21)

- [ ] `uv run python scripts/edge/export_tflite.py ...` → float + full-INT8.
- [ ] `uv run python scripts/edge/benchmark_edge.py ...` for latency / params.
- [ ] TFLite Micro arena size on **x86 first**, then on device.

**Gate:** INT8 arena fits target SRAM (ESP32 ~320 KB, Nano 33 BLE ~256 KB).
- Pass → Stage 7.
- Fail → reduced variant (smaller `n_mels` / `n_frames`) or a fixed STFT frontend.

## Stage 7 — Assemble (M5 · task 22)

- [ ] Fill the 2×3 matrix (one row per cell; mean ± std across seeds).
- [ ] Tradeoff plots (AUC vs params / latency / arena) and related-work synthesis.

## Stage 8 — Winner + model tiers + release (post-matrix, run ONLY after everything above)

Requested 2026-09-19. Gate: do nothing here until the lean matrix (A1 /
A1const / B1 / A2 / edge) has finished and a winner is chosen.

1. **Pick the winning architecture** from the matrix by AUC / mAUC, then
   params, latency, and edge arena/latency as tie-breakers. Record the decision
   in [`NOTES.md`](NOTES.md).
2. **Train a size ladder** of the winner — `small`, `normal`, `large`,
   `x-large`. Scaling **confirmed 2026-09-19**:
   - *If STgram-MFN wins* — scale `c_dim`/`n_mels`, `embed_dim`, and
     `bottleneck_setting` repeats:
     `small` ~0.3 M · `normal` = reference 1.16 M · `large` ~3 M · `x-large` ~8 M.
   - *If mn01 wins* — EfficientAT MobileNetV3 width multipliers
     (`NAME_TO_WIDTH`): `small` mn02 · `normal` mn04 · `large` mn10 ·
     `x-large` mn20 (each has its own AudioSet checkpoint; ArcFace head only).
   - Train each tier with the frozen protocol from Stage 7 (same data, seed,
     best-epoch rule).
   - Mechanics: STgram tiers set `arch: {c_dim, n_mels, embed_dim, ...}` in the
     training YAML (`spatial_size` now derives automatically in
     `src/config.py`); mn01 tiers pass `--mn01_name mn02|mn04|mn10|mn20` to
     `scripts/training/train_mn01.py`, which resolves width + AudioSet
     checkpoint via `src.config.mn01_config`. The ArcFace input width is read
     from the loaded embedder (mn02 = 192-d, not 96).
3. **Push every tier to HF** as its own model repo with a model card
   (params, AUC/pAUC/mAUC, size, latency, license), via
   `scripts/publish/push_model_to_hf.py`.
4. **Edge test every tier**: ONNX/INT8 (`scripts/edge/quantize_onnx.py`) +
   Qualcomm AI Hub sweep across all available chips
   (`scripts/edge/qai_hub_benchmark.py`) + TFLite-Micro arena. Record
   latency/arena per tier in the tradeoff table.
