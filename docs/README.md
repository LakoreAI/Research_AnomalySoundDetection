# docs/

Project documentation. Stable reference material lives at the top level;
time-stamped working output lives under the three task trees below.

## Reporting convention

Agent/working reports follow:

```
docs/<task>/<yyyy-mm-dd>/<category>/<filename>.md
```

- **`<task>`** — one of:
  - `tracing/` — chronological session/work traces (what happened, when)
  - `analysis/` — investigations and measurements (questions → findings)
  - `reports/` — experiment/result and milestone reports
- **`<yyyy-mm-dd>`** — date folder.
- **`<category>`** — topic slug, e.g. `baseline`, `mn01`, `int8`, `edge`, `dataset`.
- **`<filename>`** — numbered per date + category, resetting per date:
  `trace_001.md`, `analysis_001.md`, `report_001.md`.

Example: `docs/reports/2026-09-14/baseline/report_001.md`.

## Stable references

- [`RESEARCH.md`](RESEARCH.md) — research question, scope, status
- [`REFERENCE.md`](REFERENCE.md) — models, datasets, hardware, methodology, metrics
- [`NOTES.md`](NOTES.md) — decisions log, open questions, non-goals
- [`TASKS.md`](TASKS.md) — sprint/milestone roadmap
