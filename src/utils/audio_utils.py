"""Dataset-layout helpers for MIMII / DCASE Task 2.

The benchmark ships as one directory per machine type (fan, pump, slider,
valve, ToyCar, ToyConveyor), each with `train/` (normal-only) and `test/`
(normal + anomaly) subdirectories. Filenames encode the machine id:
`normal_id_00_00000000.wav` / `anomaly_id_00_00000000.wav`. Machine type is
recovered from the path (`<root>/<machine>/<split>/<file>.wav`), machine id
from the `id_XX` token in the filename.

These mirror the reference implementation's `utils.py` (github.com/liuyoude/
STgram-MFN) so labels line up exactly with the published DCASE numbers.
"""

import glob
import os
import re
from typing import Dict, List, Tuple

import numpy as np


def get_filename_list(dir_path: str, pattern: str = "*", ext: str = "*") -> List[str]:
    """All files under `dir_path` matching `<pattern>.<ext>` (recursive),
    sorted for deterministic label/file ordering.
    """
    filename_list: List[str] = []
    for root, _, _ in os.walk(dir_path):
        file_path_pattern = os.path.join(root, f"{pattern}.{ext}")
        filename_list += sorted(glob.glob(file_path_pattern))
    return filename_list


def get_machine_id_list(data_dir: str) -> List[str]:
    """Sorted unique `id_XX` tokens present anywhere under `data_dir`."""
    machine_id_list = sorted(
        set(
            _id
            for path in get_filename_list(data_dir)
            for _id in re.findall(r"id_[0-9][0-9]", path)
        )
    )
    return machine_id_list


def metadata_to_label(data_dirs: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Assign one class label per (machine type, machine id) pair.

    STgram-MFN's pretext task is self-supervised classification by machine ID:
    every normal recording of one physical machine is one class. Labels are
    assigned by iterating the (sorted) machine dirs and ids, so the mapping is
    deterministic given the same directory set.
    """
    meta2label: Dict[str, int] = {}
    label2meta: Dict[int, str] = {}
    label = 0
    for data_dir in data_dirs:
        # data_dir is `<root>/<machine>/train`, so the machine type is the
        # PARENT directory name (the reference uses split('/')[-2]).
        machine = os.path.basename(os.path.dirname(os.path.normpath(data_dir)))
        for id_str in get_machine_id_list(data_dir):
            meta = f"{machine}-{id_str}"
            meta2label[meta] = label
            label2meta[label] = meta
            label += 1
    return meta2label, label2meta


def machine_of_file(file_path: str) -> str:
    """Machine type for a `<root>/<machine>/<split>/<file>.wav` path."""
    return file_path.split(os.sep)[-3]


def machine_id_of_file(file_path: str) -> str:
    """The `id_XX` token in a filename (raises if absent)."""
    matches = re.findall(r"id_[0-9][0-9]", file_path)
    if not matches:
        raise ValueError(f"no id_XX token in {file_path!r}")
    return matches[0]


def create_test_file_list(
    target_dir: str,
    id_name: str,
    prefix_normal: str = "normal",
    prefix_anomaly: str = "anomaly",
    ext: str = "wav",
) -> Tuple[np.ndarray, np.ndarray]:
    """(files, labels) for one machine id: normal=0, anomaly=1.

    Kept deliberately identical in ordering to the reference so AUC/pAUC are
    directly comparable to the published STgram-MFN numbers.
    """
    normal_files = sorted(
        glob.glob(os.path.join(target_dir, f"{prefix_normal}_{id_name}*.{ext}"))
    )
    anomaly_files = sorted(
        glob.glob(os.path.join(target_dir, f"{prefix_anomaly}_{id_name}*.{ext}"))
    )
    files = np.concatenate((normal_files, anomaly_files), axis=0)
    labels = np.concatenate(
        (np.zeros(len(normal_files)), np.ones(len(anomaly_files))), axis=0
    )
    return files, labels


def build_train_file_list(train_dirs: List[str]) -> List[str]:
    """Concatenate every training file across machine dirs (train + add)."""
    file_list: List[str] = []
    for train_dir in train_dirs:
        file_list.extend(get_filename_list(train_dir))
    return file_list
