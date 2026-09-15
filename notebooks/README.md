Notebooks for experiments and analysis.

- `train_stgram_mfn_colab.ipynb` — self-contained Colab T4 training run:
  clone repo → install deps → pull DCASE 2020 from Hugging Face → train
  `configs/train_t4_long.yaml` with HF checkpoint push/auto-resume. Every long
  step is resumable; re-run after a disconnect.
