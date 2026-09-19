# Analysis 001 — domain-shift evaluation is blocked by missing labels, not compute

- **Date:** 2026-09-19
- **Topic:** domain_shift
- **Question:** Stage 5 asks for a held-out domain-shift evaluation of both
  backbones. What is actually runnable today, and what has to be built?

## Two candidate routes, both blocked

**Route A — cross-campaign eval on the DCASE 2020 evaluation set.**
`data/raw_eval/<machine>/test` holds 1,342 files for `fan` alone, but they are
named `id_01_00000000.wav`: the normal/anomaly label is not in the filename and
no ground-truth file was shipped with the download. Running the standard
evaluator returns `NaN` because `create_test_file_list` finds neither a
`normal_` nor an `anomaly_` prefix. Evaluation is therefore impossible without
the challenge ground-truth CSV.

**Route B — MIMII Domain Generalization (DCASE 2022 Task 2), already on disk.**
`data/raw_dg` (7.7 GB) is the anonymized formulation. Recall the layout:
`<machine>/<train,test>/section_<00,01,02>_<source,target>_<train,test>_<normal,anomaly>_<idx>_<attributes>.wav`,
where the trailing token is a masked attribute such as `m-n_W`, `f-n_A`, or
`n-lv_L1`. Only `gearbox/section_02` carries real `id_XX` tokens, and `gearbox`
is not one of our trained machines. For every overlapping machine
(`fan`, `slider`, `valve`, `ToyCar`) there are **no machine IDs**, so the
identity-supervised score `-log softmax(z)_y` cannot be computed.

## Consequence

We cannot report a number from the current models and the current data under the
in-domain scoring rule. Producing a domain-shift result requires the DCASE 2022
attribute-based pipeline, which is a build task, not a download.

## Plan for tomorrow (data in hand, GPU needed)

1. Parse the DG tree into `(machine, section, domain, normal/anomaly, attributes)`.
2. Train each backbone as a classifier over the **source-domain attribute
   tokens** present in `<machine>/train` (the identity is the masked attribute,
   not the machine).
3. Score test clips ID-agnostically, e.g. `s = -max_c log softmax(z)_c` over the
   source attribute classes.
4. Report AUC per section, split source vs target, and the source/target
   harmonic mean that the 2022 challenge uses, for both backbones.
5. Flag clearly that the scoring rule differs from the in-domain protocol, so
   the number is descriptive rather than directly comparable to Table II.

Estimated effort: roughly half a day of implementation plus a short training run
per backbone (the DG training set is comparable in size to DCASE 2020 dev).

## What this means for the paper

The limitation paragraph in `docs/paper/main.tex` is updated to state both
concrete blockers: eval labels withheld (Route A) and anonymized identities
(Route B), rather than the vaguer "task-formulation mismatch".
