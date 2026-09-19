---
license: cc-by-nc-sa-4.0
pretty_name: DCASE 2020 Task 2 Evaluation + Additional Training — STgram-MFN
task_categories:
- audio-classification
tags:
- audio
- anomaly-detection
- anomalous-sound-detection
- machine-condition-monitoring
- dcase
- dcase-2020
- mimii
- toyadmos
- stgram-mfn
language:
- en
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files:
  - split: train
    path: "*/train/*.wav"
  - split: test
    path: "*/test/*.wav"
---

# DCASE 2020 Task 2 — Evaluation + Additional Training (STgram-MFN redistribution)

Redistribution of the **DCASE 2020 Challenge Task 2 Additional Training Dataset**
(Zenodo [3727685](https://zenodo.org/records/3727685)) and **Evaluation Dataset**
(Zenodo [3841772](https://zenodo.org/records/3841772)), merged into the same
on-disk layout as the development set.

**Original authors:** Yuma Koizumi, Yohei Kawaguchi, Keisuke Imoto.
**License:** [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
(**non-commercial**) — inherited from the source; attribute the original authors
and keep any redistribution under the same terms.

## Layout

```
<machine>/{train,test}/<label>_id_<XX>_<index>.wav
```

- `<machine>` ∈ `fan, pump, slider, valve, ToyCar, ToyConveyor`
- `train/` — additional **normal** clips (used as extra training data)
- `test/` — normal + anomaly (held-out evaluation)
- audio: mono, 16 kHz, 10 s (`.wav`, 16-bit PCM)

## Labels

No separate label file — encoded in the path:

| Field | Source | Example |
|---|---|---|
| machine type | parent directory | `fan` |
| machine id (0–7) | `id_XX` token | `id_00` |
| normal / anomaly | filename prefix | `anomaly` → 1 |

## Statistics

- 6 machine types, 23,267 clips, ~7.3 GB
- `train`: additional normal · `test`: normal + anomaly

## Intended use

- `train/` clips are concatenated onto the dev `train/` set as the reference's
  `add_dirs` (extra normal-only data) — config key `add_root`, e.g.
  `--add_root data/raw_eval`.
- `test/` is the held-out evaluation set for the backbone-swap / INT8
  quantization study.

## Usage

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="LakoreAI/stgram-mfn-dcase2020-eval",
    repo_type="dataset", local_dir="data/raw_eval",
)
```

```bash
uv run python scripts/training/train.py --config configs/train_lean.yaml \
    --add_root data/raw_eval
```

## Related

- Development set: `LakoreAI/stgram-mfn-dcase2020-dev`
- Sources: https://zenodo.org/records/3727685 · https://zenodo.org/records/3841772

## Citation

```bibtex
@inproceedings{koizumi2020dcase,
  title     = {Description and Discussion on DCASE2020 Challenge Task 2:
               Unsupervised Anomalous Sound Detection for Machine Condition Monitoring},
  author    = {Koizumi, Yuma and Kawaguchi, Yohei and Imoto, Keisuke and others},
  booktitle = {DCASE Workshop},
  year      = {2020}
}
```
