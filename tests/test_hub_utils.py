from src.utils.hub_utils import latest_epoch_checkpoint


def test_latest_epoch_checkpoint(tmp_path):
    for name in ["epoch_5.pt", "epoch_10.pt", "epoch_2.pt", "best.pt"]:
        (tmp_path / name).write_bytes(b"x")
    latest = latest_epoch_checkpoint(tmp_path)
    assert latest is not None
    assert latest.endswith("epoch_10.pt")


def test_latest_epoch_checkpoint_none(tmp_path):
    (tmp_path / "best.pt").write_bytes(b"x")
    assert latest_epoch_checkpoint(tmp_path) is None
