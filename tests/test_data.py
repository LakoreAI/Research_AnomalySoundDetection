from pathlib import Path

import numpy as np

from src.utils.audio_utils import (
    build_train_file_list,
    create_test_file_list,
    get_filename_list,
    get_machine_id_list,
    machine_id_of_file,
    machine_of_file,
    metadata_to_label,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def make_tree(root: Path) -> None:
    for name in [
        "fan/train/normal_id_00_00000000.wav",
        "fan/train/normal_id_01_00000000.wav",
        "fan/test/normal_id_00_00000000.wav",
        "fan/test/anomaly_id_00_00000000.wav",
        "pump/train/normal_id_00_00000000.wav",
        "pump/test/normal_id_00_00000000.wav",
    ]:
        _touch(root / name)


def test_machine_and_id_parsing(tmp_path):
    make_tree(tmp_path)
    f = str(tmp_path / "fan" / "train" / "normal_id_00_00000000.wav")
    assert machine_of_file(f) == "fan"
    assert machine_id_of_file(f) == "id_00"


def test_get_filename_list_recursive(tmp_path):
    make_tree(tmp_path)
    files = get_filename_list(str(tmp_path / "fan"))
    assert len(files) == 4
    assert all(f.endswith(".wav") for f in files)


def test_get_machine_id_list(tmp_path):
    make_tree(tmp_path)
    ids = get_machine_id_list(str(tmp_path / "fan"))
    assert ids == ["id_00", "id_01"]


def test_metadata_to_label_deterministic(tmp_path):
    make_tree(tmp_path)
    dirs = [str(tmp_path / "fan" / "train"), str(tmp_path / "pump" / "train")]
    meta2label, label2meta = metadata_to_label(dirs)
    assert meta2label == {"fan-id_00": 0, "fan-id_01": 1, "pump-id_00": 2}
    assert label2meta[2] == "pump-id_00"


def test_create_test_file_list_labels(tmp_path):
    make_tree(tmp_path)
    files, labels = create_test_file_list(str(tmp_path / "fan" / "test"), "id_00")
    assert len(files) == 2
    assert labels.tolist() == [0.0, 1.0]
    assert np.all([Path(f).name.startswith(("normal", "anomaly")) for f in files])


def test_build_train_file_list(tmp_path):
    make_tree(tmp_path)
    files = build_train_file_list([str(tmp_path / "fan" / "train")])
    assert len(files) == 2
