# Trace 001 — M1 session (read → scaffold → implement → push)

- **Date:** 2026-09-14
- **Topic:** baseline
- **Scope:** work session from docs review through M1 implementation and push

## Chronology

1. **Docs review.** Read `docs/{RESEARCH,REFERENCE,NOTES,TASKS}.md`; identified
   M1 (Baseline Reproduction) as the selected next task.
2. **Prior-art research.** Verified STgram-MFN (arXiv:2201.05510, ICASSP 2022;
   github.com/liuyoude/STgram-MFN) and mn01 (EfficientAT / EfficientAT_HEAR,
   ~120k-param MobileNetV3 AudioSet GPAEE). Flagged a 16 kHz vs 32 kHz front-end
   mismatch for the mn01 swap.
3. **Reference codebase study.** Read `../bicafe` (config/data/modules/pipelines/
   callbacks/utils, `CLAUDE.md`, scripts). Adopted its structure and conventions.
4. **Fetched reference implementation.** Pulled `net.py`, `loss.py`,
   `dataset.py`, `trainer.py`, `config.yaml` from the STgram-MFN repo.
5. **Scaffolded + implemented** `src/` (mirror bicafe, directly under `src/`),
   `configs/`, `tests/`, and `scripts/`.
6. **Environment.** `uv sync` (torch 2.14+cu126, torchaudio 2.11); ruff pinned to
   `0.15.15` to match bicafe's default rule set; 21 tests pass.
7. **Fixed during bring-up:**
   - `metadata_to_label` used the split dir basename instead of the machine
     parent dir (`.../fan/train` → `fan`).
   - torchaudio ≥ 2.11 `load` needs TorchCodec → switched to `soundfile`.
   - Eval moved a shared CPU `FeatureExtractor` to CUDA, breaking the dataset
     extractor → eval now builds/owns its device extractor.
8. **Verified** diagnostics, real 10 s forward (1.16M params), ONNX export +
   dynamic INT8 + latency benchmark.
9. **Git.** Committed `f545db4` and pushed to `origin/main`.
10. **M1 run started.** Launched DCASE 2020 dev + eval download (background);
    armed a `ToyCar` canary watcher.

## Open threads

- Download in progress (~3.5 h ETA); canary pending first machine.
- Full 300-epoch run vs shorter trajectory check — decision pending.
- `--machines` CLI flag added after the push (uncommitted).
- mn01 swap (M2) not started.
