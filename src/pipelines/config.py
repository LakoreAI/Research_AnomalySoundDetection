"""Training-loop hyperparameters — separate from `src.config.STgramMFNConfig`
(model/audio architecture only). Controllable via a YAML file (see
`configs/train.yaml`) with individual fields overridable from the CLI.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Union

from src.utils.io_utils import read_yaml

DEFAULT_MACHINES = ["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"]


@dataclass
class TrainingConfig:
    # --- data ---
    # `data_root/<machine>/<train_subdir>` (normal only) and
    # `data_root/<machine>/<test_subdir>` (normal + anomaly).
    data_root: str = "data/raw"
    # Optional second root whose `<machine>/train` clips are added to training
    # (the reference's `add_dirs`, i.e. DCASE `eval_dataset`). Normal-only.
    add_root: Optional[str] = None
    machines: List[str] = field(default_factory=lambda: list(DEFAULT_MACHINES))
    train_subdir: str = "train"
    test_subdir: str = "test"
    load_in_memory: bool = False

    # --- output ---
    ckpt_dir: str = "checkpoints"
    result_dir: str = "results"
    run_name: Optional[str] = None

    # --- optimization ---
    epochs: int = 300
    batch_size: int = 128
    accum_steps: int = 1
    num_workers: int = 0
    pin_memory: bool = False
    amp: bool = False
    lr: float = 1e-4
    weight_decay: float = 0.0
    log_every: int = 10
    eval_every: int = 10  # reference `valid_every_epochs`
    ckpt_every: int = 50

    # Fraction of TRAIN normal clips held out for a self-supervised val_loss.
    # 0 disables it (the reference has no held-out split and selects its best
    # epoch on the official test set — see docs/NOTES.md).
    val_fraction: float = 0.0
    seed: int = 42

    # --- callbacks ---
    # None (or {"type": "none"}) disables LR scheduling. See
    # src/callbacks/lr_scheduler.py.
    lr_scheduler: Optional[dict] = None
    early_stopping: Optional[dict] = None
    save_best: bool = True
    # Which validation metric BestCheckpoint/early stopping monitor.
    # "auc"/"pauc" are maximized; "val_loss" is minimized.
    best_metric: str = "auc"
    best_mode: str = "max"

    # None (or {"enabled": false}) disables W&B logging.
    wandb: Optional[dict] = None

    # Optional incremental checkpoint push to Hugging Face Hub (requires the
    # `hub` extra). Fails soft — a push error never kills training. Enables
    # auto-resume after a Colab VM reclaim:
    #   {"enabled": true, "repo_id": "LakoreAI/stgram-mfn-t4", "every_epochs": 5}
    # On startup, if `resume_from` is unset and this is enabled, the run's
    # checkpoints are pulled from the repo and the latest epoch is resumed.
    hf_push: Optional[dict] = None

    resume_from: Optional[str] = None


def load_training_config(
    config_path: Optional[Union[str, Path]] = None, **cli_overrides
) -> TrainingConfig:
    """Build a TrainingConfig from defaults, a YAML file, then CLI overrides
    (highest precedence, but only applied for keys the caller actually passed —
    argparse defaults of None are treated as "not set").
    """
    cfg = TrainingConfig()

    if config_path is not None:
        yaml_data = read_yaml(config_path)
        known_fields = set(asdict(cfg).keys())
        for key, value in yaml_data.items():
            if key not in known_fields:
                raise ValueError(
                    f"unknown training config key {key!r} in {config_path}"
                )
            setattr(cfg, key, value)

    for key, value in cli_overrides.items():
        if value is not None:
            setattr(cfg, key, value)

    return cfg
