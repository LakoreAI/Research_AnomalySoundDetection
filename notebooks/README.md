Notebooks for experiments and analysis.

- `train_stgram_mfn_colab.ipynb` — thin Colab T4 training run: clone repo →
  install deps → pull DCASE 2020 from Hugging Face → train
  `configs/train_t4_long.yaml` with HF checkpoint push/auto-resume. Every long
  step is resumable; re-run after a disconnect.
- `stgram_mfn_from_scratch.ipynb` — the full implementation in one notebook:
  every model block (log-mel frontend, TgramNet, MobileFaceNet, ArcFace, loss,
  STgramMFN fusion), the dataset/loader, the AUC/pAUC/mAUC evaluation, and the
  training loop — no `src/` imports. Verified to match `src/` exactly
  (params `1,152,284`; feature `(B, 128)`, logits `(B, num_classes)`).
