"""B1 — mn01 swap: frozen EfficientAT `mn01_as` (96-d) + ArcFace head.

The swap replaces STgram-MFN's frontend+backbone with the pretrained AudioSet
`mn01` embedder and shares only the ArcFace head (see docs/NOTES.md). mn01 is
FROZEN, so its 96-d embeddings are computed once and cached; only the ArcFace
head is trained. Protocol mirrors `train_lean.yaml` (same data, epochs, LR) so
A and B are comparable.

Usage:
    uv run python scripts/training/train_mn01.py --config configs/train_lean_constlr.yaml \
        --run_name lean_B1_mn01 --hf_push_repo LakoreAI/stgram-mfn-lean
"""

import argparse
import glob
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.callbacks import BestCheckpoint, TrainContext, build_lr_scheduler  # noqa: E402
from src.callbacks.base import TrainerState  # noqa: E402
from src.callbacks.wandb_callback import WandbCallback  # noqa: E402
from src.config import (  # noqa: E402
    MN01Config,
    STgramMFNConfig,
    mn01_config,
)
from src.data import load_waveform  # noqa: E402
from src.modules.arcface import ArcMarginProduct  # noqa: E402
from src.modules.loss import ASDLoss, anomaly_score  # noqa: E402
from src.modules.mn01 import MN01Embedder, load_mn01  # noqa: E402
from src.pipelines.config import load_training_config  # noqa: E402
from src.pipelines.train import (  # noqa: E402
    build_test_dirs,
    build_train_dirs,
    format_report,
)
from src.utils.audio_utils import (  # noqa: E402
    build_train_file_list,
    create_test_file_list,
    get_machine_id_list,
    machine_id_of_file,
    machine_of_file,
    metadata_to_label,
)
from src.utils.io_utils import load_env, save_checkpoint, save_json  # noqa: E402
from src.utils.model_utils import detect_device  # noqa: E402

MAX_FPR = 0.1


def _clip32k(path: str, cfg: MN01Config) -> torch.Tensor:
    n = int(cfg.secs * cfg.sample_rate)
    wav = load_waveform(path, cfg.sample_rate)
    if wav.shape[0] < n:
        wav = torch.nn.functional.pad(wav, (0, n - wav.shape[0]))
    return wav[:n]


def _labels_for(files: List[str], meta2label: Dict[str, int]) -> np.ndarray:
    return np.array(
        [meta2label[f"{machine_of_file(f)}-{machine_id_of_file(f)}"] for f in files],
        dtype=np.int64,
    )


@torch.no_grad()
def embed_files(
    embedder: MN01Embedder,
    files: List[str],
    cfg: MN01Config,
    device,
    batch_size: int = 64,
) -> np.ndarray:
    """96-d embedding per file, in input order."""
    embedder.eval()
    out: List[np.ndarray] = []
    for start in range(0, len(files), batch_size):
        chunk = files[start : start + batch_size]
        wavs = torch.stack([_clip32k(f, cfg) for f in chunk]).to(device)
        out.append(embedder(wavs).float().cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 96), np.float32)


def _cached_embeddings(
    embedder: MN01Embedder,
    files: List[str],
    cfg: MN01Config,
    device,
    cache_dir: Path,
    tag: str,
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{tag}.npy"
    if path.exists():
        emb = np.load(path)
        if emb.shape[0] == len(files):
            print(f"  [cache] {path.name} ({emb.shape})")
            return emb
    print(f"  [embed] {tag}: {len(files)} files ...")
    emb = embed_files(embedder, files, cfg, device)
    np.save(path, emb)
    return emb


@torch.no_grad()
def evaluate_mn01(
    arcface: ArcMarginProduct,
    meta2label: Dict[str, int],
    test_dirs: List[str],
    emb_index: Dict[str, int],
    test_emb: np.ndarray,
    device,
) -> Dict[str, object]:
    """Same protocol as src.pipelines.eval.evaluate, over cached embeddings."""
    arcface.eval()
    per_machine, per_id = {}, {}
    aucs_m, paucs_m, maucs_m = [], [], []
    for target_dir in sorted(test_dirs):
        machine_type = Path(target_dir).parent.name
        aucs, paucs = [], []
        for id_str in get_machine_id_list(target_dir):
            label = meta2label[f"{machine_type}-{id_str}"]
            files, y_true = create_test_file_list(target_dir, id_str)
            keep = [i for i, f in enumerate(files) if f in emb_index]
            if not keep:
                continue
            files = [files[i] for i in keep]
            y_true = y_true[keep]
            idx = [emb_index[f] for f in files]
            emb = torch.from_numpy(test_emb[idx]).to(device)
            labels = torch.full((len(idx),), label, dtype=torch.long, device=device)
            logits = arcface(emb, labels)
            y_pred = anomaly_score(logits, labels).cpu().numpy()
            if len(np.unique(y_true)) < 2:
                continue
            auc = roc_auc_score(y_true, y_pred)
            pauc = roc_auc_score(y_true, y_pred, max_fpr=MAX_FPR)
            aucs.append(auc)
            paucs.append(pauc)
            per_id[f"{machine_type}-{id_str}"] = {"auc": auc, "pauc": pauc}
        if aucs:
            per_machine[machine_type] = {
                "auc": float(np.mean(aucs)),
                "pauc": float(np.mean(paucs)),
                "mauc": float(np.min(aucs)),
            }
            aucs_m.append(float(np.mean(aucs)))
            paucs_m.append(float(np.mean(paucs)))
            maucs_m.append(float(np.min(aucs)))
    return {
        "auc": float(np.mean(aucs_m)) if aucs_m else float("nan"),
        "pauc": float(np.mean(paucs_m)) if paucs_m else float("nan"),
        "mauc": float(np.mean(maucs_m)) if maucs_m else float("nan"),
        "per_machine": per_machine,
        "per_id": per_id,
    }


class MN01WaveDataset(Dataset):
    """32 kHz clip + machine-ID label, for fine-tuning mn01 end to end."""

    def __init__(self, files: List[str], labels: np.ndarray, cfg: MN01Config):
        self.files = files
        self.labels = labels
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        return _clip32k(self.files[i], self.cfg), int(self.labels[i])


class MN01ArcFace(nn.Module):
    """mn01 embedder + ArcFace head (both trainable when fine-tuning).

    The ArcFace input width is taken from the loaded embedder, so this works
    for any EfficientAT width (mn01..mn40), not just the 96-d mn01.
    """

    def __init__(self, mn_cfg: MN01Config, num_classes: int, ckpt):
        super().__init__()
        self.cfg_mn = mn_cfg
        self.embedder = load_mn01(mn_cfg, checkpoint=ckpt, map_location="cpu")
        head_cfg = STgramMFNConfig(
            num_classes=num_classes,
            embed_dim=self.embedder.embed_dim,
            arcface_m=0.7,
            arcface_s=30.0,
        )
        self.arcface = ArcMarginProduct(head_cfg)
        self.embed_dim = self.embedder.embed_dim

    def forward(self, wav: torch.Tensor, label: torch.Tensor):
        emb = self.embedder(wav)
        return self.arcface(emb, label), emb


@torch.no_grad()
def evaluate_mn01_ft(
    model: MN01ArcFace,
    meta2label,
    test_dirs,
    device,
    batch_size: int = 128,
    num_workers: int = 8,
):
    """Same metric as `src.pipelines.eval`, but test clips are loaded through a
    worker DataLoader so evaluation doesn't stall the GPU (the serial version
    pegged the main process for minutes on the 10.9k test clips).
    """
    files_all, score_labels, keys, ys = [], [], [], []
    for target_dir in sorted(test_dirs):
        machine_type = Path(target_dir).parent.name
        for id_str in get_machine_id_list(target_dir):
            files, y_true = create_test_file_list(target_dir, id_str)
            if len(np.unique(y_true)) < 2:
                continue
            label = meta2label[f"{machine_type}-{id_str}"]
            for f, y in zip(files, y_true):
                files_all.append(f)
                score_labels.append(label)
                keys.append((machine_type, id_str))
                ys.append(int(y))

    scores: List[float] = []
    if files_all:
        model.eval()
        loader = DataLoader(
            MN01WaveDataset(files_all, np.asarray(score_labels), model.cfg_mn),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        for wav, lab in loader:
            wav, lab = wav.to(device), lab.to(device)
            logits, _ = model(wav, lab)
            scores.extend(anomaly_score(logits, lab).cpu().tolist())

    y_by_key: Dict[tuple, List[int]] = {}
    s_by_key: Dict[tuple, List[float]] = {}
    for key, y, s in zip(keys, ys, scores):
        y_by_key.setdefault(key, []).append(y)
        s_by_key.setdefault(key, []).append(s)

    per_machine, per_id = {}, {}
    aucs_by_m: Dict[str, List[float]] = {}
    paucs_by_m: Dict[str, List[float]] = {}
    for (machine_type, id_str), y_true in y_by_key.items():
        y_pred = s_by_key[(machine_type, id_str)]
        if len(set(y_true)) < 2:
            continue
        auc = roc_auc_score(y_true, y_pred)
        pauc = roc_auc_score(y_true, y_pred, max_fpr=MAX_FPR)
        per_id[f"{machine_type}-{id_str}"] = {"auc": auc, "pauc": pauc}
        aucs_by_m.setdefault(machine_type, []).append(auc)
        paucs_by_m.setdefault(machine_type, []).append(pauc)

    aucs_m, paucs_m, maucs_m = [], [], []
    for machine_type, aucs in aucs_by_m.items():
        per_machine[machine_type] = {
            "auc": float(np.mean(aucs)),
            "pauc": float(np.mean(paucs_by_m[machine_type])),
            "mauc": float(np.min(aucs)),
        }
        aucs_m.append(float(np.mean(aucs)))
        paucs_m.append(float(np.mean(paucs_by_m[machine_type])))
        maucs_m.append(float(np.min(aucs)))

    return {
        "auc": float(np.mean(aucs_m)) if aucs_m else float("nan"),
        "pauc": float(np.mean(paucs_m)) if paucs_m else float("nan"),
        "mauc": float(np.mean(maucs_m)) if maucs_m else float("nan"),
        "per_machine": per_machine,
        "per_id": per_id,
    }


def train_finetune(
    train_cfg,
    args,
    mn_cfg,
    device,
    meta2label,
    num_classes,
    train_files,
    train_lab,
    test_dirs,
) -> None:
    run_name = args.run_name or train_cfg.run_name
    ckpt_dir = Path(train_cfg.ckpt_dir) / run_name
    result_dir = Path(train_cfg.result_dir) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model = MN01ArcFace(mn_cfg, num_classes, args.mn01_ckpt).to(device)
    print(
        f"mn01+arcface params (trainable): {sum(p.numel() for p in model.parameters()):,}"
        f"  embed_dim: {model.embed_dim}"
    )
    opt = torch.optim.Adam(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    crit = ASDLoss().to(device)
    use_amp = train_cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loader = DataLoader(
        MN01WaveDataset(train_files, train_lab, mn_cfg),
        batch_size=args.ft_batch,
        shuffle=True,
        drop_last=True,
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory,
    )
    callbacks = [
        BestCheckpoint(monitor=train_cfg.best_metric, mode=train_cfg.best_mode)
    ]
    lr_cb = build_lr_scheduler(opt, train_cfg.lr_scheduler)
    if lr_cb is not None:
        callbacks.insert(0, lr_cb)
    if train_cfg.wandb and train_cfg.wandb.get("enabled", False):
        wb = train_cfg.wandb
        callbacks.append(
            WandbCallback(
                project_name=wb.get("project", "stgram-mfn"),
                run_name=run_name,
                config={
                    "backbone": "mn01",
                    "freeze": False,
                    "embed_dim": model.embed_dim,
                },
                entity=wb.get("entity"),
                monitor=wb.get("monitor", "auc"),
                mode=wb.get("mode", "max"),
                log_artifacts=wb.get("log_artifacts", True),
                group=wb.get("group"),
            )
        )

    def push(path):
        if not (train_cfg.hf_push and train_cfg.hf_push.get("enabled")):
            return
        try:
            from src.utils.hub_utils import push_file

            print(
                "  [hf_push]",
                push_file(
                    path,
                    train_cfg.hf_push["repo_id"],
                    path_in_repo=f"{run_name}/{Path(path).name}",
                ),
            )
        except Exception as e:  # noqa: BLE001
            print(f"  [hf_push] failed: {e}")

    cfg_for_ckpt = {
        "backbone": "mn01",
        "freeze": False,
        "embed_dim": model.embed_dim,
        "meta2label": meta2label,
        "arcface_m": 0.7,
        "arcface_s": 30.0,
    }
    wandb_run_id = next(
        (
            cb.run.id
            for cb in callbacks
            if isinstance(cb, WandbCallback) and cb.enabled and cb.run
        ),
        None,
    )
    ctx = TrainContext(
        model=model,
        optimizer=opt,
        ckpt_dir=ckpt_dir,
        cfg_dict=cfg_for_ckpt,
        wandb_run_id=wandb_run_id,
    )
    for cb in callbacks:
        cb.on_train_start(ctx)

    history, step, pushed_best = [], 0, None
    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        tot = 0.0
        for wav, lab in loader:
            wav, lab = wav.to(device), lab.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, _ = model(wav, lab)
                loss = crit(logits, lab)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            step += 1
            tot += loss.item()
            for cb in callbacks:
                cb.on_step_end(
                    ctx, TrainerState(step=step, train_loss=loss.item(), epoch=epoch)
                )
        train_loss = tot / max(len(loader), 1)
        rec = {"epoch": epoch, "step": step, "loss": train_loss}
        print(f"epoch {epoch:3d}/{train_cfg.epochs}  train_loss={train_loss:.4f}")

        if epoch % train_cfg.eval_every == 0 or epoch == train_cfg.epochs:
            res = evaluate_mn01_ft(
                model,
                meta2label,
                test_dirs,
                device,
                num_workers=train_cfg.num_workers,
            )
            rec.update({"auc": res["auc"], "pauc": res["pauc"], "mauc": res["mauc"]})
            print(format_report(res))
            state = TrainerState(
                step=step,
                train_loss=train_loss,
                epoch=epoch,
                val_loss=None,
                extra={"auc": res["auc"], "pauc": res["pauc"], "mauc": res["mauc"]},
            )
            for cb in callbacks:
                cb.on_validation_end(ctx, state)
            v = rec.get(train_cfg.best_metric)
            if v is not None and (pushed_best is None or v > pushed_best):
                push(ckpt_dir / "best.pt")
                pushed_best = v
        history.append(rec)

        if epoch % train_cfg.ckpt_every == 0 or epoch == train_cfg.epochs:
            save_checkpoint(
                model,
                opt,
                step,
                ckpt_dir / f"epoch_{epoch}.pt",
                extra={
                    "cfg": cfg_for_ckpt,
                    "epoch": epoch,
                    "wandb_run_id": wandb_run_id,
                },
            )
            print(f"  saved checkpoint: {ckpt_dir / f'epoch_{epoch}.pt'}")

    push(ckpt_dir / "best.pt")
    for cb in callbacks:
        cb.on_train_end(ctx)
    save_json(history, ckpt_dir / "train_log.json")
    save_json(history, result_dir / "train_log.json")
    print(f"\nfinished {run_name} at epoch {epoch}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs" / "train_lean_constlr.yaml"
    )
    parser.add_argument("--run_name", type=str, default="lean_B1_mn01")
    parser.add_argument("--hf_push_repo", type=str, default=None)
    parser.add_argument(
        "--mn01_ckpt", type=str, default=None, help="local mn01_as weights"
    )
    parser.add_argument(
        "--mn01_name",
        type=str,
        default="mn01",
        help="EfficientAT width name (mn01/mn02/mn04/mn05/mn10/mn20/mn30/mn40)",
    )
    parser.add_argument(
        "--cache_dir", type=Path, default=REPO_ROOT / "features" / "mn01"
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--limit", type=int, default=None, help="cap files (smoke test)"
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--no_hf", action="store_true")
    parser.add_argument("--machines", nargs="*", default=None)
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="train mn01 end-to-end (all params) instead of frozen + linear head",
    )
    parser.add_argument("--ft_batch", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=None)
    args = parser.parse_args()

    load_env(REPO_ROOT / ".env")
    train_cfg = load_training_config(args.config)
    if args.epochs is not None:
        train_cfg.epochs = args.epochs
    if args.machines:
        train_cfg.machines = args.machines
    if args.num_workers is not None:
        train_cfg.num_workers = args.num_workers
    if args.no_wandb:
        train_cfg.wandb = {"enabled": False}
    if args.no_hf:
        train_cfg.hf_push = {"enabled": False}
    if args.mn01_ckpt is None and args.mn01_name == "mn01":
        local = REPO_ROOT / "checkpoints" / "mn01_as_mAP_298.pt"
        if local.exists():
            args.mn01_ckpt = str(local)
    if args.hf_push_repo:
        train_cfg.hf_push = {"enabled": True, "repo_id": args.hf_push_repo}
    device = detect_device()
    print(f"device: {device}")

    mn_cfg = mn01_config(args.mn01_name)
    print(f"mn01 variant: {args.mn01_name} (width_mult={mn_cfg.width_mult})")

    train_dirs = build_train_dirs(train_cfg)
    test_dirs = build_test_dirs(train_cfg)
    meta2label, _ = metadata_to_label(train_dirs)
    num_classes = len(meta2label)
    print(f"classes: {num_classes}")

    train_files = build_train_file_list(train_dirs)
    test_files = [
        f for d in test_dirs for f in sorted(glob.glob(os.path.join(d, "*.wav")))
    ]
    if args.limit:
        train_files = train_files[: args.limit]
        test_files = test_files[: args.limit]
    train_lab = _labels_for(train_files, meta2label)

    if args.finetune:
        train_finetune(
            train_cfg,
            args,
            mn_cfg,
            device,
            meta2label,
            num_classes,
            train_files,
            train_lab,
            test_dirs,
        )
        return

    embedder = load_mn01(mn_cfg, checkpoint=args.mn01_ckpt, map_location="cpu").to(
        device
    )
    for p in embedder.parameters():
        p.requires_grad_(False)
    print(f"mn01 params (frozen): {sum(p.numel() for p in embedder.parameters()):,}")

    train_emb = _cached_embeddings(
        embedder, train_files, mn_cfg, device, args.cache_dir, "train"
    )
    test_emb = _cached_embeddings(
        embedder, test_files, mn_cfg, device, args.cache_dir, "test"
    )
    emb_index = {f: i for i, f in enumerate(test_files)}
    print(f"train emb {train_emb.shape}  test emb {test_emb.shape}")

    # ArcFace head over 96-d embeddings (same geometry/config as A).
    head_cfg = STgramMFNConfig(
        num_classes=num_classes,
        embed_dim=int(train_emb.shape[1]),
        arcface_m=0.7,
        arcface_s=30.0,
    )
    arcface = ArcMarginProduct(head_cfg).to(device)
    optimizer = torch.optim.Adam(
        arcface.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    criterion = ASDLoss().to(device)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_emb), torch.from_numpy(train_lab)),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    run_name = args.run_name or train_cfg.run_name
    ckpt_dir = Path(train_cfg.ckpt_dir) / run_name
    result_dir = Path(train_cfg.result_dir) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        BestCheckpoint(monitor=train_cfg.best_metric, mode=train_cfg.best_mode)
    ]
    lr_cb = build_lr_scheduler(optimizer, train_cfg.lr_scheduler)
    if lr_cb is not None:
        callbacks.insert(0, lr_cb)
    if train_cfg.wandb and train_cfg.wandb.get("enabled", False):
        wb = train_cfg.wandb
        callbacks.append(
            WandbCallback(
                project_name=wb.get("project", "stgram-mfn"),
                run_name=run_name,
                config={"backbone": "mn01", "freeze": True, **asdict(head_cfg)},
                entity=wb.get("entity"),
                monitor=wb.get("monitor", "auc"),
                mode=wb.get("mode", "max"),
                log_artifacts=wb.get("log_artifacts", True),
                group=wb.get("group"),
            )
        )

    cfg_for_ckpt = {
        "backbone": "mn01",
        "embed_dim": int(train_emb.shape[1]),
        "meta2label": meta2label,
        "arcface_m": 0.7,
        "arcface_s": 30.0,
    }
    wandb_run_id = next(
        (
            cb.run.id
            for cb in callbacks
            if isinstance(cb, WandbCallback) and cb.enabled and cb.run
        ),
        None,
    )
    ctx = TrainContext(
        model=arcface,
        optimizer=optimizer,
        ckpt_dir=ckpt_dir,
        cfg_dict=cfg_for_ckpt,
        wandb_run_id=wandb_run_id,
    )
    for cb in callbacks:
        cb.on_train_start(ctx)

    def push(path):
        if not (train_cfg.hf_push and train_cfg.hf_push.get("enabled")):
            return
        try:
            from src.utils.hub_utils import push_file

            print(
                "  [hf_push]",
                push_file(
                    path,
                    train_cfg.hf_push["repo_id"],
                    path_in_repo=f"{run_name}/{Path(path).name}",
                ),
            )
        except Exception as e:  # noqa: BLE001
            print(f"  [hf_push] failed: {e}")

    history, step, pushed_best = [], 0, None
    for epoch in range(1, train_cfg.epochs + 1):
        arcface.train()
        tot = 0.0
        for emb, lab in loader:
            emb, lab = emb.to(device), lab.to(device)
            logits = arcface(emb, lab)
            loss = criterion(logits, lab)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            tot += loss.item()
            for cb in callbacks:
                cb.on_step_end(
                    ctx, TrainerState(step=step, train_loss=loss.item(), epoch=epoch)
                )
        train_loss = tot / max(len(loader), 1)
        rec = {"epoch": epoch, "step": step, "loss": train_loss}
        print(f"epoch {epoch:3d}/{train_cfg.epochs}  train_loss={train_loss:.4f}")

        if epoch % train_cfg.eval_every == 0 or epoch == train_cfg.epochs:
            res = evaluate_mn01(
                arcface, meta2label, test_dirs, emb_index, test_emb, device
            )
            rec.update({"auc": res["auc"], "pauc": res["pauc"], "mauc": res["mauc"]})
            print(format_report(res))
            state = TrainerState(
                step=step,
                train_loss=train_loss,
                epoch=epoch,
                val_loss=None,
                extra={"auc": res["auc"], "pauc": res["pauc"], "mauc": res["mauc"]},
            )
            for cb in callbacks:
                cb.on_validation_end(ctx, state)
            v = rec.get(train_cfg.best_metric)
            if v is not None and (pushed_best is None or v > pushed_best):
                push(ckpt_dir / "best.pt")
                pushed_best = v
        history.append(rec)

        if epoch % train_cfg.ckpt_every == 0 or epoch == train_cfg.epochs:
            save_checkpoint(
                arcface,
                optimizer,
                step,
                ckpt_dir / f"epoch_{epoch}.pt",
                extra={
                    "cfg": cfg_for_ckpt,
                    "epoch": epoch,
                    "wandb_run_id": wandb_run_id,
                },
            )
            print(f"  saved checkpoint: {ckpt_dir / f'epoch_{epoch}.pt'}")

    push(ckpt_dir / "best.pt")
    for cb in callbacks:
        cb.on_train_end(ctx)
    save_json(history, ckpt_dir / "train_log.json")
    save_json(history, result_dir / "train_log.json")
    print(f"\nfinished {run_name} at epoch {epoch}")


if __name__ == "__main__":
    main()
