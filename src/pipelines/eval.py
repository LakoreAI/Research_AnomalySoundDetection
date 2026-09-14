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
def score_file(
    model: STgramMFN,
    extractor: FeatureExtractor,
    file_path: str,
    label: int,
    device: torch.device,
) -> float:
    """Anomaly score for one file relative to its own machine-ID class."""
    waveform = load_waveform(file_path, extractor.cfg.sample_rate).to(device)
    x_wav, x_mel = extractor(waveform)
    labels = torch.tensor([label], device=device)
    logits, _ = model(
        x_wav.unsqueeze(0).to(device), x_mel.unsqueeze(0).to(device), labels
    )
    return anomaly_score(logits, labels).item()


def evaluate(
    model: STgramMFN,
    cfg: STgramMFNConfig,
    meta2label: Dict[str, int],
    test_dirs: List[str],
    device: torch.device,
    extractor: Optional[FeatureExtractor] = None,
    save_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """Returns {"auc", "pauc", "per_machine", "per_id"} (auc/pauc as
    fractions in [0, 1], matching the reference's raw sklearn values).
    """
    model.eval()
    extractor = (extractor or FeatureExtractor(cfg)).to(device)

    per_machine: Dict[str, Dict[str, float]] = {}
    per_id: Dict[str, Dict[str, float]] = {}
    machine_aucs, machine_paucs = [], []

    for target_dir in sorted(test_dirs):
        machine_type = Path(target_dir).parent.name
        id_list = get_machine_id_list(target_dir)
        aucs, paucs = [], []
        for id_str in id_list:
            meta = f"{machine_type}-{id_str}"
            label = meta2label[meta]
            test_files, y_true = create_test_file_list(target_dir, id_str)
            if len(np.unique(y_true)) < 2:
                continue
            y_pred = [
                score_file(model, extractor, f, label, device) for f in test_files
            ]
            auc = roc_auc_score(y_true, y_pred)
            pauc = roc_auc_score(y_true, y_pred, max_fpr=MAX_FPR)
            aucs.append(auc)
            paucs.append(pauc)
            per_id[f"{machine_type}-{id_str}"] = {"auc": auc, "pauc": pauc}
            if save_dir is not None:
                _write_score_csv(
                    Path(save_dir) / f"anomaly_score_{machine_type}_{id_str}.csv",
                    test_files,
                    y_pred,
                )
        if aucs:
            per_machine[machine_type] = {
                "auc": float(np.mean(aucs)),
                "pauc": float(np.mean(paucs)),
            }
            machine_aucs.append(float(np.mean(aucs)))
            machine_paucs.append(float(np.mean(paucs)))

    avg_auc = float(np.mean(machine_aucs)) if machine_aucs else float("nan")
    avg_pauc = float(np.mean(machine_paucs)) if machine_paucs else float("nan")
    return {
        "auc": avg_auc,
        "pauc": avg_pauc,
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
            f"{machine:14s} AUC={m['auc'] * 100:6.3f}  pAUC={m['pauc'] * 100:6.3f}"
        )
    lines.append(
        f"{'TOTAL':14s} AUC={result['auc'] * 100:6.3f}  pAUC={result['pauc'] * 100:6.3f}"
    )
    return "\n".join(lines)
