"""DCASE 2022 Task 2 domain-generalization (MIMII-DG) pipeline.

The 2022 set anonymizes machine identity: clips carry masked attribute tokens
(`m-n_W`, `f-n_A`, ...) instead of `id_XX`, and each machine has three sections
evaluated in a source and a target domain. The identity-supervised score used
in the main study is therefore undefined. We instead:

  1. train each backbone as a classifier over (machine, attribute) classes
     present in `<machine>/train` (source normal only);
  2. score test clips ID-agnostically,
     s(x) = -max_c log softmax(z(x))_c  over that machine's source classes;
  3. report AUC per machine / section / domain, then average source and target.

This is the attribute-based adaptation of our protocol, not the reference 2020
rule; numbers are descriptive and are not comparable to the in-domain table.

Usage:
    uv run python scripts/training/train_dg.py --config configs/train_lean.yaml \
        --backbone stgram --run_name dg_stgram --epochs 40
    uv run python scripts/training/train_dg.py --config configs/train_lean.yaml \
        --backbone mn01 --mn01_ckpt checkpoints/mn01_as_mAP_298.pt \
        --run_name dg_mn01 --epochs 40
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.config import STgramMFNConfig, mn01_config  # noqa: E402
from src.data import load_waveform  # noqa: E402
from src.modules.frontend import FeatureExtractor  # noqa: E402
from src.modules.loss import ASDLoss  # noqa: E402
from src.modules.mn01_head import MN01ArcFace  # noqa: E402
from src.modules.model import STgramMFN  # noqa: E402
from src.pipelines.config import load_training_config  # noqa: E402
from src.utils.io_utils import load_env, save_json  # noqa: E402
from src.utils.model_utils import detect_device  # noqa: E402

MACHINES = ["fan", "slider", "valve", "ToyCar"]
MAX_FPR = 0.1


def _attr(fname: str) -> str:
    return "_".join(Path(fname).stem.split("_")[6:])


def _split(fname: str) -> Tuple[str, str, str]:
    p = Path(fname).stem.split("_")
    return p[1], p[2], p[4]  # section, source/target, normal/anomaly


def _label(machine: str, fname: str) -> str:
    return f"{machine}|{_attr(fname)}"


def _train_files(dg_root: Path) -> Tuple[List[str], List[str]]:
    files, labels = [], []
    for m in MACHINES:
        for f in sorted((dg_root / m / "train").glob("*.wav")):
            files.append(str(f))
            labels.append(_label(m, str(f)))
    return files, labels


class DGDataset(Dataset):
    def __init__(self, files, label_map, backbone, stgram_cfg=None, mn_cfg=None):
        self.files = files
        self.label_map = label_map
        self.backbone = backbone
        self.extractor = FeatureExtractor(stgram_cfg) if backbone == "stgram" else None
        self.mn_cfg = mn_cfg

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        machine = Path(f).parents[1].name
        y = self.label_map[f"{machine}|{_attr(f)}"]
        if self.backbone == "stgram":
            wav = load_waveform(f, self.extractor.cfg.sample_rate)
            x_wav, x_mel = self.extractor(wav)
            return x_wav, x_mel, y
        cfg = self.mn_cfg
        n = int(cfg.secs * cfg.sample_rate)
        wav = load_waveform(f, cfg.sample_rate)
        if wav.shape[0] < n:
            wav = torch.nn.functional.pad(wav, (0, n - wav.shape[0]))
        return wav[:n], y


def train_backbone(
    backbone,
    num_classes,
    train_files,
    label_map,
    cfg,
    device,
    epochs,
    batch_size=128,
    num_workers=8,
    mn01_ckpt=None,
    run=None,
):
    if backbone == "stgram":
        net: nn.Module = STgramMFN(STgramMFNConfig(num_classes=num_classes)).to(device)
        mn_cfg = None
    else:
        mn_cfg = mn01_config("mn01")
        net = MN01ArcFace(mn_cfg, num_classes, mn01_ckpt).to(device)

    ds = DGDataset(
        train_files,
        label_map,
        backbone,
        STgramMFNConfig(num_classes=num_classes) if backbone == "stgram" else None,
        mn_cfg,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    crit = ASDLoss().to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    print(f"[dg] training {backbone}: {len(ds)} clips, {num_classes} classes")
    for epoch in range(1, epochs + 1):
        net.train()
        tot, n = 0.0, 0
        for batch in loader:
            if backbone == "stgram":
                x_wav, x_mel, lab = batch
                x_wav, x_mel, lab = x_wav.to(device), x_mel.to(device), lab.to(device)
                with torch.amp.autocast("cuda", enabled=cfg.amp):
                    logits, _ = net(x_wav, x_mel, lab)
                    loss = crit(logits, lab)
            else:
                wav, lab = batch
                wav, lab = wav.to(device), lab.to(device)
                with torch.amp.autocast("cuda", enabled=cfg.amp):
                    logits, _ = net(wav, lab)
                    loss = crit(logits, lab)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
            n += 1
        sched.step()
        avg = tot / max(n, 1)
        if run is not None:
            run.log(
                {"train/loss": avg, "train/lr": sched.get_last_lr()[0], "epoch": epoch}
            )
        if epoch % 10 == 0 or epoch == epochs:
            print(f"  epoch {epoch:3d}/{epochs}  loss={avg:.4f}")
    return net, mn_cfg


@torch.no_grad()
def eval_dg(net, backbone, label_map, mn_cfg, dg_root, device, batch_size=128):
    """AUC per (machine, section, domain), ID-agnostic scoring."""
    class_names = sorted(label_map.keys())
    machine_classes = {
        m: torch.tensor(
            [label_map[c] for c in class_names if c.startswith(f"{m}|")],
            device=device,
        )
        for m in MACHINES
    }
    extractor = FeatureExtractor(STgramMFNConfig()) if backbone == "stgram" else None
    results: Dict[str, Dict[str, float]] = {}
    net.eval()

    for m in MACHINES:
        files = sorted((dg_root / m / "test").glob("*.wav"))
        scores, ys, groups = [], [], []
        for start in range(0, len(files), batch_size):
            chunk = files[start : start + batch_size]
            labs = [label_map.get(f"{m}|{_attr(str(f))}", 0) for f in chunk]
            lab_t = torch.tensor(labs, device=device)
            if backbone == "stgram":
                xs, ms = [], []
                for f in chunk:
                    wav = load_waveform(str(f), 16000)
                    x_wav, x_mel = extractor(wav)
                    xs.append(x_wav.numpy())
                    ms.append(x_mel.numpy())
                x_wav = torch.from_numpy(np.stack(xs)).to(device)
                x_mel = torch.from_numpy(np.stack(ms)).to(device)
                logits, _ = net(x_wav, x_mel, lab_t)
            else:
                n = int(mn_cfg.secs * mn_cfg.sample_rate)
                ws = []
                for f in chunk:
                    wav = load_waveform(str(f), mn_cfg.sample_rate)
                    if wav.shape[0] < n:
                        wav = torch.nn.functional.pad(wav, (0, n - wav.shape[0]))
                    ws.append(wav[:n].numpy())
                logits, _ = net(torch.from_numpy(np.stack(ws)).to(device), lab_t)
            logp = torch.log_softmax(logits.float(), dim=1)
            mc = machine_classes[m]
            s = (
                -logp[:, mc].max(dim=1).values
            )  # higher = less like any source attribute
            scores.extend(s.cpu().tolist())
            for f in chunk:
                sec, dom, cl = _split(str(f))
                groups.append((sec, dom))
                ys.append(1 if cl == "anomaly" else 0)
        for sec in ("00", "01", "02"):
            for dom in ("source", "target"):
                sel = [i for i, g in enumerate(groups) if g == (sec, dom)]
                if len(set(ys[i] for i in sel)) < 2:
                    continue
                auc = roc_auc_score([ys[i] for i in sel], [scores[i] for i in sel])
                results[f"{m}-sec{sec}-{dom}"] = float(auc)
    return results


def summarize(res: Dict[str, float]) -> Dict[str, float]:
    def mean(keys):
        vals = [res[k] for k in keys if k in res]
        return float(np.mean(vals)) if vals else float("nan")

    src = mean([k for k in res if k.endswith("source")])
    tgt = mean([k for k in res if k.endswith("target")])
    allv = mean(list(res))
    return {"source": src * 100, "target": tgt * 100, "all": allv * 100}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs" / "train_lean.yaml"
    )
    parser.add_argument("--backbone", choices=["stgram", "mn01"], default="stgram")
    parser.add_argument("--dg_root", type=Path, default=REPO_ROOT / "data" / "raw_dg")
    parser.add_argument("--run_name", type=str, default="dg_stgram")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--mn01_ckpt", type=str, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    load_env(REPO_ROOT / ".env")
    device = detect_device()
    cfg = load_training_config(args.config)
    train_files, labels = _train_files(args.dg_root)
    class_names = sorted(set(labels))
    label_map = {c: i for i, c in enumerate(class_names)}
    print(f"[dg] {len(train_files)} train clips, {len(class_names)} classes")

    run = None
    wb = cfg.wandb or {}
    if wb.get("enabled", False):
        import wandb

        run = wandb.init(
            project=wb.get("project", "stgram-mfn"),
            name=args.run_name,
            group="dg",
            config={
                "backbone": args.backbone,
                "epochs": args.epochs,
                "classes": len(class_names),
                "protocol": "attribute-id-agnostic",
            },
        )

    net, mn_cfg = train_backbone(
        args.backbone,
        len(class_names),
        train_files,
        label_map,
        cfg,
        device,
        args.epochs,
        args.batch_size,
        args.num_workers,
        args.mn01_ckpt,
        run,
    )

    ckpt_dir = Path(cfg.ckpt_dir) / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": net.state_dict(), "backbone": args.backbone, "label_map": label_map},
        ckpt_dir / "best.pt",
    )
    print(f"[dg] saved {ckpt_dir / 'best.pt'}")

    res = eval_dg(net, args.backbone, label_map, mn_cfg, args.dg_root, device)
    s = summarize(res)
    print(
        f"\n[dg] {args.backbone}: source AUC {s['source']:.2f}  "
        f"target AUC {s['target']:.2f}  overall {s['all']:.2f}"
    )
    for k in sorted(res):
        print(f"   {k:22s} {res[k] * 100:6.2f}")

    if run is not None:
        run.log(
            {
                "eval/source_auc": s["source"],
                "eval/target_auc": s["target"],
                "eval/auc": s["all"],
            }
        )
        for k, v in res.items():
            run.log({f"eval/{k}": v * 100})
        run.finish()

    out = {
        "backbone": args.backbone,
        "run_name": args.run_name,
        **s,
        "per_section": res,
    }
    if args.json:
        save_json(out, args.json)
        print(f"saved: {args.json}")


if __name__ == "__main__":
    main()
