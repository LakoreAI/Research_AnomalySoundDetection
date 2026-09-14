# Report 001 — M1 STgram-MFN baseline implementation

- **Date:** 2026-09-14
- **Milestone:** M1 (Baseline Reproduction)
- **Status:** implementation complete, verification passed, training not yet run
  (awaiting dataset download)
- **Task:** reports / baseline

## Summary

The STgram-MFN reference backbone is implemented on this codebase, mirroring the
sibling `bicafe` layout directly under `src/`. All architecture sanity checks and
the test suite pass; ONNX export and INT8 quantization paths are functional. The
DCASE 2020 dev + eval download is in progress; no real-data training result exists
yet.

## What was built

| Area | Artifact |
|---|---|
| Architecture | `src/config.py`, `src/modules/{frontend,tgramnet,mobilefacenet,arcface,loss,model}.py` |
| Deployment | `src/modules/edge.py` (`STgramMFNScorer`) |
| Data | `src/data.py`, `src/utils/audio_utils.py` |
| Training | `src/pipelines/{config,train,eval,infer}.py`, `src/callbacks/*` |
| Sanity | `src/diagnostics.py`, `tests/` (21 tests) |
| Tooling | `pyproject.toml` (uv, cu126), `configs/train.yaml` |
| Scripts | `scripts/{download_data,prepare_data,train,evaluate,smoke_test,export_onnx,quantize_onnx,export_tflite,benchmark_edge,push_*}.py` |

Fidelity notes: the reference's hardcoded `313` LayerNorm width, `(8,20)`
collapse kernel, and `128` embedding width are lifted into `STgramMFNConfig`.
Audio is loaded with `soundfile` (torchaudio ≥ 2.11 requires TorchCodec for
loading).

## Verification

- `uv run pytest -q` → **21 passed**.
- `uv run python -m src.diagnostics` → D1 wiring, D2 branch shapes, D3 overfit all
  PASS.
- Real 10 s forward pass (config defaults, 313 frames, `(8,20)` spatial):
  logits/feature shapes correct; **1,161,536 params** at 42 classes.
- `scripts/smoke_test.py` (synthetic audio, 1 epoch, train→eval): PASS.
- ONNX export: 4.56 MB fp32. Dynamic INT8: 1.23 MB, but **~3.5× slower** on CPU
  (149 ms vs 42 ms) — expected for dynamically-quantized convnets; static QDQ is
  the path to use.

## Decisions / caveats

- Best-epoch selection currently monitors the official DCASE test-set AUC
  (reference behavior). This is test-set selection; a leak-free held-out
  `val_fraction` path exists if the protocol is tightened. See `docs/NOTES.md`.
- `--machines` CLI flag added to scope partial runs (canary).

## Next steps

1. Finish DCASE 2020 dev + eval download (in progress).
2. Real-data canary on `ToyCar` (3 epochs) — auto-armed.
3. Full 300-epoch run (decision pending: full vs shorter trajectory check).
4. Compare reproduced AUC/pAUC against published 92.36 / 84.86.

## References

- Liu et al., *Anomalous Sound Detection using Spectral-Temporal Information
  Fusion*, ICASSP 2022 (arXiv:2201.05510)
- Reference implementation: github.com/liuyoude/STgram-MFN
- `docs/RESEARCH.md`, `docs/TASKS.md`
