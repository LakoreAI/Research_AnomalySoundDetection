# Decisions & Notes

## Resolved decisions

- DCASE Task 2 changed formulation in 2023 — every writeup must state 2020–2022 scope explicitly to avoid ambiguity.
- mn01 swap is embeddings-in to the MobileFaceNet+ArcFace head, not a full backbone replacement.
- Hardware memory footprint validated early (x86 TFLite Micro arena checks), not deferred to deployment.
- Current DCASE baseline is Harada et al. (EUSIPCO 2023), not MobileNetV2.
- Transformer survey architectures are related-work citations only — not swap candidates.
- Drone-deployable track deprioritized to optional appendix; factory track is the sole primary focus. Scope deliberately held to proven components to stay executable in the timeline.

## Reporting conventions

Empirical ML systems idiom — experimental conditions, ablations, descriptive metrics. Not social-science conventions (hypothesis tables, p-values, validity taxonomies) unless the work involves human subjects or survey data.

## Open questions / horizon

- Research Paper Outline — still generic, needs filling with actual argument structure once results exist
- Scholarship Statement of Purpose — not started
- Related Work synthesis — not started
- ESP32 hardware — acquisition agreed, not yet in hand
- Sprint 1 (M1 Baseline Reproduction) — not yet started

## Non-goals

- Drone-deployable variant (appendix only)
- DCASE 2023+ first-shot formulation
- Novel architecture — this is a tradeoff study, not an architecture paper

## Parking lot (unrelated to this project)

Alternatives to backpropagation (equilibrium propagation, Mono-Forward, predictive coding, MeZO-style zeroth-order fine-tuning, forward-gradient hybrids) — separate curiosity, not tied to ASD.