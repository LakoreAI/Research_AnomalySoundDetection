"""STgram-MFN training entrypoint.

Self-supervised classification by machine ID (the reference's pretext): every
normal recording of one physical machine is one class, trained with ArcFace +
cross-entropy. Validation is the DCASE test-set AUC/pAUC (the reference selects
its best epoch this way); an optional held-out normal split can supply a
self-supervised `val_loss` for early stopping instead.

Controlled via a YAML config (see configs/train.yaml) with individual CLI
overrides. Callbacks (LR schedule, early stopping, best checkpoint, W&B) are
wired from that file, mirroring the reference codebase's training loop.

Usage:
    uv run python -m src.pipelines.train --config configs/train.yaml
    uv run python -m src.pipelines.train --config configs/train.yaml --epochs 80 --lr 5e-4
"""

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader

from src.callbacks import (
    BestCheckpoint,
    EarlyStopping,
    TrainContext,
    TrainerState,
    build_lr_scheduler,
)
from src.callbacks.wandb_callback import WandbCallback
from src.config import STgramMFNConfig
from src.data import ASDDataset, make_extractor
from src.modules.loss import ASDLoss
from src.modules.model import STgramMFN
from src.pipelines.config import TrainingConfig, load_training_config
from src.pipelines.eval import evaluate, format_report
from src.utils.audio_utils import build_train_file_list, metadata_to_label
from src.utils.io_utils import save_checkpoint, save_json
from src.utils.model_utils import count_parameters, detect_device, get_run_name

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_train_dirs(train_cfg: TrainingConfig) -> List[str]:
    dirs = [
        str(Path(train_cfg.data_root) / m / train_cfg.train_subdir)
        for m in train_cfg.machines
    ]
    if train_cfg.add_root:
        dirs += [
            str(Path(train_cfg.add_root) / m / train_cfg.train_subdir)
            for m in train_cfg.machines
        ]
    return dirs


def build_test_dirs(train_cfg: TrainingConfig) -> List[str]:
    return [
        str(Path(train_cfg.data_root) / m / train_cfg.test_subdir)
        for m in train_cfg.machines
    ]


def build_model(model_cfg: STgramMFNConfig, device: torch.device) -> STgramMFN:
    return STgramMFN(model_cfg).to(device)


def build_callbacks(
    train_cfg: TrainingConfig,
    optimizer: torch.optim.Optimizer,
    run_name: str,
    wandb_config: dict,
):
    callbacks = []

    lr_cb = build_lr_scheduler(optimizer, train_cfg.lr_scheduler)
    if lr_cb is not None:
        callbacks.append(lr_cb)

    if train_cfg.early_stopping and train_cfg.early_stopping.get("enabled", True):
        es_kwargs = {
            k: v for k, v in train_cfg.early_stopping.items() if k != "enabled"
        }
        callbacks.append(EarlyStopping(**es_kwargs))

    if train_cfg.save_best:
        callbacks.append(
            BestCheckpoint(monitor=train_cfg.best_metric, mode=train_cfg.best_mode)
        )

    # after BestCheckpoint: relies on <ckpt_dir>/best.pt already existing
    if train_cfg.wandb and train_cfg.wandb.get("enabled", False):
        wb = train_cfg.wandb
        callbacks.append(
            WandbCallback(
                project_name=wb.get("project", "stgram-mfn"),
                run_name=run_name,
                config=wandb_config,
                entity=wb.get("entity"),
                monitor=wb.get("monitor", train_cfg.best_metric),
                mode=wb.get("mode", train_cfg.best_mode),
                log_artifacts=wb.get("log_artifacts", True),
                group=wb.get("group"),
            )
        )

    return callbacks


def split_val(
    file_list: List[str], val_fraction: float, seed: int
) -> Tuple[List[str], List[str]]:
    """Random clip-level split of the (normal-only) training list. Used only
    for an optional self-supervised val_loss; the AUC path uses the official
    test dirs.
    """
    if val_fraction <= 0 or len(file_list) < 2:
        return file_list, []
    n_val = max(int(round(len(file_list) * val_fraction)), 1)
    n_val = min(n_val, len(file_list) - 1)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(file_list), generator=g).tolist()
    val_idx = set(perm[:n_val])
    train = [f for i, f in enumerate(file_list) if i not in val_idx]
    val = [f for i, f in enumerate(file_list) if i in val_idx]
    return train, val


def _optimizer_step(
    optimizer, scaler, callbacks, ctx, pending_losses, step, epoch, log_every
):
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    step += 1
    avg_loss = sum(pending_losses) / len(pending_losses)
    state = TrainerState(step=step, train_loss=avg_loss, epoch=epoch)
    for cb in callbacks:
        cb.on_step_end(ctx, state)
    if step % log_every == 0 or step == 1:
        print(f"  epoch {epoch:3d}  step {step:5d}  loss={avg_loss:.4f}")
    return step, avg_loss


def train_per_epoch(
    model,
    criterion,
    loader,
    optimizer,
    scaler,
    accum_steps,
    device,
    callbacks,
    ctx,
    step,
    epoch,
    log_every,
):
    model.train()
    epoch_losses, pending = [], []
    optimizer.zero_grad()

    for x_wavs, x_mels, labels in loader:
        x_wavs = x_wavs.float().to(device)
        x_mels = x_mels.float().to(device)
        labels = labels.reshape(-1).long().to(device)

        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            logits, _ = model(x_wavs, x_mels, labels)
            loss = criterion(logits, labels)

        scaler.scale(loss / accum_steps).backward()
        pending.append(loss.item())

        if len(pending) == accum_steps:
            step, avg_loss = _optimizer_step(
                optimizer, scaler, callbacks, ctx, pending, step, epoch, log_every
            )
            epoch_losses.append(avg_loss)
            pending = []

    if pending:
        step, avg_loss = _optimizer_step(
            optimizer, scaler, callbacks, ctx, pending, step, epoch, log_every
        )
        epoch_losses.append(avg_loss)

    avg_epoch_loss = (
        sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("nan")
    )
    return avg_epoch_loss, step


@torch.no_grad()
def val_loss_per_epoch(model, criterion, loader, device) -> float:
    """Self-supervised CE on a held-out NORMAL split (same pretext, no
    anomalies needed). Always full precision.
    """
    if loader is None:
        return float("nan")
    model.eval()
    losses = []
    for x_wavs, x_mels, labels in loader:
        x_wavs = x_wavs.float().to(device)
        x_mels = x_mels.float().to(device)
        labels = labels.reshape(-1).long().to(device)
        logits, _ = model(x_wavs, x_mels, labels)
        losses.append(criterion(logits, labels).item())
    model.train()
    return sum(losses) / len(losses) if losses else float("nan")


def train(train_cfg: TrainingConfig):
    device = detect_device()
    print(f"device: {device}")

    torch.manual_seed(train_cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(train_cfg.seed)
    elif device.type == "mps":
        torch.mps.manual_seed(train_cfg.seed)

    pin_memory = train_cfg.pin_memory and device.type == "cuda"

    train_dirs = build_train_dirs(train_cfg)
    test_dirs = build_test_dirs(train_cfg)
    meta2label, label2meta = metadata_to_label(train_dirs)
    print(f"classes (machine-ids): {len(meta2label)}")

    model_cfg = STgramMFNConfig(num_classes=len(meta2label))
    extractor = make_extractor(model_cfg)

    all_files = build_train_file_list(train_dirs)
    train_files, val_files = split_val(
        all_files, train_cfg.val_fraction, train_cfg.seed
    )

    train_ds = ASDDataset(
        train_files, meta2label, extractor, load_in_memory=train_cfg.load_in_memory
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=train_cfg.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = None
    if val_files:
        val_ds = ASDDataset(
            val_files, meta2label, extractor, load_in_memory=train_cfg.load_in_memory
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=min(train_cfg.batch_size, len(val_ds)),
            shuffle=False,
            num_workers=train_cfg.num_workers,
            pin_memory=pin_memory,
        )

    model = build_model(model_cfg, device)
    count_parameters(model)
    criterion = ASDLoss().to(device)

    params = list(model.parameters())
    optimizer = torch.optim.Adam(
        params, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    scaler = torch.amp.GradScaler(
        "cuda" if device.type == "cuda" else "cpu",
        enabled=(train_cfg.amp and device.type == "cuda"),
    )

    run_name = train_cfg.run_name or get_run_name(
        "stgram-mfn",
        Path(train_cfg.data_root).name,
        train_cfg.lr,
        train_cfg.batch_size,
    )
    ckpt_dir = Path(train_cfg.ckpt_dir) / run_name
    result_dir = Path(train_cfg.result_dir) / run_name
    print(f"run: {run_name}")
    print(
        f"train/val clips: {len(train_ds)}/{len(val_files)}  batch_size: {train_cfg.batch_size}"
        f"  accum_steps: {train_cfg.accum_steps}  epochs: {train_cfg.epochs}"
    )

    cfg_for_ckpt = {**asdict(model_cfg), "meta2label": meta2label}
    callbacks = build_callbacks(
        train_cfg, optimizer, run_name, {**asdict(model_cfg), **asdict(train_cfg)}
    )
    early_stoppers = [cb for cb in callbacks if isinstance(cb, EarlyStopping)]

    wandb_run_id = None
    for cb in callbacks:
        if isinstance(cb, WandbCallback) and cb.enabled and cb.run is not None:
            wandb_run_id = cb.run.id

    ctx = TrainContext(
        model=model,
        optimizer=optimizer,
        ckpt_dir=ckpt_dir,
        cfg_dict=cfg_for_ckpt,
        wandb_run_id=wandb_run_id,
    )
    for cb in callbacks:
        cb.on_train_start(ctx)

    history = []
    step = 0
    stopped_early = False
    start_epoch = 1
    if train_cfg.resume_from:
        raw = torch.load(train_cfg.resume_from, map_location=str(device))
        model.load_state_dict(raw["model"])
        if "optimizer" in raw:
            optimizer.load_state_dict(raw["optimizer"])
        step = int(raw.get("step", 0))
        start_epoch = int(raw.get("extra", {}).get("epoch", 0)) + 1
        print(f"resumed at step {step}, continuing from epoch {start_epoch}")

    epoch = 0
    for epoch in range(start_epoch, train_cfg.epochs + 1):
        train_loss, step = train_per_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            scaler,
            train_cfg.accum_steps,
            device,
            callbacks,
            ctx,
            step,
            epoch,
            train_cfg.log_every,
        )
        record = {"epoch": epoch, "step": step, "loss": train_loss}
        print(f"epoch {epoch:3d}/{train_cfg.epochs}  train_loss={train_loss:.4f}")

        should_validate = epoch % train_cfg.eval_every == 0 or epoch == train_cfg.epochs
        if should_validate:
            val_loss = val_loss_per_epoch(model, criterion, val_loader, device)
            extra = {}
            if test_dirs and all(Path(d).exists() for d in test_dirs):
                result = evaluate(model, model_cfg, meta2label, test_dirs, device)
                extra = {"auc": result["auc"], "pauc": result["pauc"]}
                record.update(extra)
                print(format_report(result))
            state = TrainerState(
                step=step,
                train_loss=train_loss,
                epoch=epoch,
                val_loss=val_loss if val_loader is not None else None,
                extra=extra,
            )
            record["val_loss"] = state.val_loss
            for cb in callbacks:
                cb.on_validation_end(ctx, state)
            if any(es.should_stop for es in early_stoppers):
                stopped_early = True

        history.append(record)

        if epoch % train_cfg.ckpt_every == 0 or epoch == train_cfg.epochs:
            ckpt_path = ckpt_dir / f"epoch_{epoch}.pt"
            save_checkpoint(
                model,
                optimizer,
                step,
                ckpt_path,
                extra={
                    "cfg": cfg_for_ckpt,
                    "epoch": epoch,
                    "wandb_run_id": ctx.wandb_run_id,
                },
            )
            print(f"  saved checkpoint: {ckpt_path}")

        if stopped_early:
            break

    for cb in callbacks:
        cb.on_train_end(ctx)

    save_json(history, ckpt_dir / "train_log.json")
    save_json(history, result_dir / "train_log.json")
    print(
        f"\n{'stopped early' if stopped_early else 'finished'} at epoch {epoch} (step {step})"
    )
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=None, help="YAML, see configs/train.yaml"
    )
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--add_root", type=str, default=None)
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--result_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--accum_steps", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--log_every", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--ckpt_every", type=int, default=None)
    parser.add_argument("--val_fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--amp", action="store_true", default=None)
    parser.add_argument("--pin_memory", action="store_true", default=None)
    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    train_cfg = load_training_config(args.config, **overrides)
    train(train_cfg)
