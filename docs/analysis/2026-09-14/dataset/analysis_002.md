# Analysis 002 — HuggingFace availability of the required datasets

- **Date:** 2026-09-14
- **Topic:** dataset
- **Question:** can the DCASE 2020 Task 2 / MIMII data be sourced from HuggingFace
  instead of Zenodo?

## Method

Queried the HF datasets API (`/api/datasets?search=...&author=...`) for `mimii`,
`dcase`, `dcase2020`, `ToyADMOS`, `valve`, `slider`, `anomalous sound`, and listed
all datasets by the `stdt1` author (the main MIMII uploader). Inspected the one
plausible candidate's file tree.

## Findings

| Candidate | What it is | Usable? |
|---|---|---|
| `stdt1/mimii_pump_datasets` (+ 16 kHz/RMS/1s/10s variants) | **pump only**, preprocessed (RMS segments), not the full 4-machine MIMII | No — incomplete |
| `renumics/dcase23-task2-enriched` | DCASE **2023** Task 2 | No — 2023+ first-shot, explicitly out of scope |
| `HTill/dcase2025_task2_dev` | DCASE **2025** Task 2 dev (parquet) | No — out of scope |
| `Fhrozen/dcase22_task3`, `wangxu26/dcase-task4`, … | unrelated tasks | No |

There is **no** HuggingFace mirror of DCASE 2020 Task 2 (dev + eval, six machine
types) or of the full MIMII dataset (fan/pump/slider/valve at the DCASE-style
layout this project expects).

## Implication

- Zenodo remains the canonical source; `scripts/download_data.py` is unchanged.
- No speed-up is available from HF: even if a mirror existed, the local network
  link is the bottleneck (~1 MB/s, see analysis_001), not the host.
- A pump-only processed set could seed a *pump-only* canary, but it is not a
  faithful baseline and would not reproduce the published numbers — not worth
  the layout-conversion risk.

## Status at write time

ToyCar at 56% (973 MB / 1.82 GB).
