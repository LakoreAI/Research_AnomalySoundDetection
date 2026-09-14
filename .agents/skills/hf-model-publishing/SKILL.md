---
name: hf-model-publishing
description: >-
  Standard operating procedure for publishing trained STgram-MFN checkpoints,
  configs, and model cards to the Hugging Face Hub. Use after a training run to
  hand the model off to another environment.
---

# Hugging Face Model Publishing Protocol

Ported from the sibling `bicafe` project, adapted to this project.

## Checklist

Before pushing a checkpoint to Hugging Face Hub:

1. **Verify checkpoint file**: `checkpoints/<run>/best.pt` exists, is non-zero,
   and contains the `{"model", "optimizer", "step", "extra"}` schema (see
   `src/utils/io_utils.py`).
2. **Verify config**: ship the exact training YAML (`configs/*.yaml`) used.
3. **Model card (`README.md`)**:
   - YAML header (`license: mit`, `tags: [audio, anomaly-detection, dcase, stgram-mfn, edge, pytorch]`).
   - Input spec: 16 kHz mono, 10 s; log-mel Sgram (128 mels, n_fft 1024, hop 512) + learned Tgram.
   - Head: MobileFaceNet (128-d) + ArcFace; anomaly score `-log_softmax(logits)[machine_id]`.
   - Metrics: AUC / pAUC (and the machine-id class count).
   - Instantiation snippet.

## Script

`scripts/push_model_to_hf.py` implements this:

```bash
uv run --extra hub python scripts/push_model_to_hf.py \
    --ckpt checkpoints/stgram_mfn_traj30/best.pt \
    --repo_id LakoreAI/stgram-mfn-traj30 \
    --config configs/train_trajectory.yaml
```

## Policy

Model repos are created **private** by default (flip with `--public`). Verify
visibility in the API response (`private` field) rather than trusting the call
to have succeeded. No artifacts are uploaded to W&B (`log_artifacts: false`);
HF is the model hand-off path.
