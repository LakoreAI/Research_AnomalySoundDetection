"""Anomaly-detection evaluation: AUC / pAUC per machine type and machine id.

Follows the DCASE Task 2 / reference protocol:
  * score every file in `<machine>/test` by `-log_softmax(logits)[own_id]`
  * AUC and partial-AUC (max FPR = 0.1) per machine id, averaged within a
    machine type, then averaged equally across machine types.

This is the metric the training loop's BestCheckpoint monitors (the reference
selects its best epoch the same way — see docs/NOTES.md for the
test-set-selection caveat that this inherits).
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from src.config import STgramMFNConfig
from src.modules.frontend import FeatureExtractor
from src.modules.loss import anomaly_score
from src.modules.model import STgramMFN
from src.utils.audio_utils import (
    create_test_file_list,
    get_machine_id_list,
)
from src.data import load_waveform

MAX_FPR = 0.1


@torch.no_grad()
def score_files(
    model: STgramMFN,
    extractor: FeatureExtractor,
    file_paths: List[str],
    label: int,
    device: torch.device,
    batch_size: int = 32,
) -> List[float]:
    """Anomaly scores for many files that share one machine-ID label.

    Clips are cropped/padded to a fixed length by FeatureExtractor, so their
    tensors stack directly and the forward pass can be batched — much faster
    than one file at a time (feature extraction stays per-file, forward is
    batched). All files here share `label`, so one label vector is reused.
    """
    model.eval()
    scores: List[float] = []
    for start in range(0, len(file_paths), batch_size):
        chunk = file_paths[start : start + batch_size]
        wavs, mels = [], []
        for f in chunk:
            waveform = load_waveform(f, extractor.cfg.sample_rate).to(device)
            x_wav, x_mel = extractor(waveform)
            wavs.append(x_wav)
            mels.append(x_mel)
        x_wav = torch.stack(wavs)
        x_mel = torch.stack(mels)
        labels = torch.full((len(chunk),), label, dtype=torch.long, device=device)
        logits, _ = model(x_wav, x_mel, labels)
        scores.extend(anomaly_score(logits, labels).tolist())
    return scores


@torch.no_grad()
def score_file(
    model: STgramMFN,
    extractor: FeatureExtractor,
    file_path: str,
    label: int,
    device: torch.device,
) -> float:
    """Anomaly score for a single file relative to its own machine-ID class."""
    return score_files(model, extractor, [file_path], label, device, batch_size=1)[0]


class _ScoreDataset(Dataset):
    """Test clips -> (x_wav, x_mel, label) for the parallel scorer."""

    def __init__(self, files: List[str], labels: List[int], cfg: STgramMFNConfig):
        self.files = files
        self.labels = labels
        self.extractor = FeatureExtractor(cfg)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        waveform = load_waveform(self.files[i], self.extractor.cfg.sample_rate)
        x_wav, x_mel = self.extractor(waveform)
        return x_wav, x_mel, int(self.labels[i])


def evaluate(
    model: STgramMFN,
    cfg: STgramMFNConfig,
    meta2label: Dict[str, int],
    test_dirs: List[str],
    device: torch.device,
    extractor: Optional[FeatureExtractor] = None,
    save_dir: Optional[Path] = None,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Dict[str, object]:
    """Returns {"auc", "pauc", "per_machine", "per_id"} (auc/pauc as
    fractions in [0, 1], matching the reference's raw sklearn values).

    Test clips are loaded through a `DataLoader`; pass `num_workers > 0` to
    keep the GPU busy (the previous serial path stalled it for minutes on the
    ~10.9k test clips). `num_workers=0` preserves the old serial behaviour.
    """
    model.eval()

    files_all: List[str] = []
    labels_all: List[int] = []
    keys: List[tuple] = []
    ys: List[int] = []
    for target_dir in sorted(test_dirs):
        machine_type = Path(target_dir).parent.name
        for id_str in get_machine_id_list(target_dir):
            label = meta2label[f"{machine_type}-{id_str}"]
            files, y_true = create_test_file_list(target_dir, id_str)
            if len(np.unique(y_true)) < 2:
                continue
            for f, y in zip(files, y_true):
                files_all.append(f)
                labels_all.append(label)
                keys.append((machine_type, id_str))
                ys.append(int(y))

    scores: List[float] = []
    if files_all:
        loader = DataLoader(
            _ScoreDataset(files_all, labels_all, cfg),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
        with torch.no_grad():
            for x_wav, x_mel, lab in loader:
                x_wav = x_wav.to(device)
                x_mel = x_mel.to(device)
                lab = lab.to(device)
                logits, _ = model(x_wav, x_mel, lab)
                scores.extend(anomaly_score(logits, lab).cpu().tolist())

    y_by_key: Dict[tuple, List[int]] = {}
    s_by_key: Dict[tuple, List[float]] = {}
    f_by_key: Dict[tuple, List[str]] = {}
    for f, key, y, s in zip(files_all, keys, ys, scores):
        y_by_key.setdefault(key, []).append(y)
        s_by_key.setdefault(key, []).append(s)
        f_by_key.setdefault(key, []).append(f)

    per_machine: Dict[str, Dict[str, float]] = {}
    per_id: Dict[str, Dict[str, float]] = {}
    aucs_by_m: Dict[str, List[float]] = {}
    paucs_by_m: Dict[str, List[float]] = {}
    for (machine_type, id_str), y_true in y_by_key.items():
        y_pred = s_by_key[(machine_type, id_str)]
        auc = roc_auc_score(y_true, y_pred)
        pauc = roc_auc_score(y_true, y_pred, max_fpr=MAX_FPR)
        per_id[f"{machine_type}-{id_str}"] = {"auc": auc, "pauc": pauc}
        aucs_by_m.setdefault(machine_type, []).append(auc)
        paucs_by_m.setdefault(machine_type, []).append(pauc)
        if save_dir is not None:
            _write_score_csv(
                Path(save_dir) / f"anomaly_score_{machine_type}_{id_str}.csv",
                f_by_key[(machine_type, id_str)],
                y_pred,
            )

    machine_aucs, machine_paucs, machine_maucs = [], [], []
    for machine_type, aucs in aucs_by_m.items():
        per_machine[machine_type] = {
            "auc": float(np.mean(aucs)),
            "pauc": float(np.mean(paucs_by_m[machine_type])),
            # mAUC: worst-case (minimum) AUC among this machine type's ids —
            # the reference's stability metric (Table 3, avg 84.86).
            "mauc": float(np.min(aucs)),
        }
        machine_aucs.append(float(np.mean(aucs)))
        machine_paucs.append(float(np.mean(paucs_by_m[machine_type])))
        machine_maucs.append(float(np.min(aucs)))

    avg_auc = float(np.mean(machine_aucs)) if machine_aucs else float("nan")
    avg_pauc = float(np.mean(machine_paucs)) if machine_paucs else float("nan")
    avg_mauc = float(np.mean(machine_maucs)) if machine_maucs else float("nan")
    return {
        "auc": avg_auc,
        "pauc": avg_pauc,
        "mauc": avg_mauc,
        "per_machine": per_machine,
        "per_id": per_id,
    }


def _write_score_csv(path: Path, files, scores) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        for file_path, score in zip(files, scores):
            writer.writerow([Path(file_path).name, score])


def format_report(result: Dict[str, object]) -> str:
    lines = []
    for machine, m in sorted(result["per_machine"].items()):
        lines.append(
            f"{machine:14s} AUC={m['auc'] * 100:6.3f}  "
            f"pAUC={m['pauc'] * 100:6.3f}  mAUC={m['mauc'] * 100:6.3f}"
        )
    lines.append(
        f"{'TOTAL':14s} AUC={result['auc'] * 100:6.3f}  "
        f"pAUC={result['pauc'] * 100:6.3f}  mAUC={result['mauc'] * 100:6.3f}"
    )
    return "\n".join(lines)
