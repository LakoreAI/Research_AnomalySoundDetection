"""Single-file STgram-MFN inference: audio -> embedding / anomaly score.

Loads a checkpoint, extracts the Sgram/Tgram pair, runs the encoder, and
reports the 128-d embedding. If the file's machine type and id can be resolved
(from the path and the checkpoint's saved `meta2label`), the anomaly score
`-log_softmax(logits)[own_id]` is reported too.

Usage:
    uv run python -m src.pipelines.infer \\
        --ckpt checkpoints/<run>/best.pt --audio path/to/clip.wav
"""

import argparse
from dataclasses import fields
from pathlib import Path
from typing import Dict, Optional

import torch

from src.config import STgramMFNConfig
from src.data import load_waveform
from src.modules.frontend import FeatureExtractor
from src.modules.loss import anomaly_score
from src.modules.model import STgramMFN
from src.utils.audio_utils import machine_id_of_file, machine_of_file
from src.utils.model_utils import detect_device


def config_from_checkpoint(ckpt: dict) -> STgramMFNConfig:
    """Rebuild STgramMFNConfig from a checkpoint's saved `extra["cfg"]`,
    dropping non-architecture keys (e.g. meta2label).
    """
    saved = ckpt.get("extra", {}).get("cfg", {})
    known = {f.name for f in fields(STgramMFNConfig)}
    return STgramMFNConfig(**{k: v for k, v in saved.items() if k in known})


def load_model(ckpt_path: Path, device: torch.device):
    raw = torch.load(ckpt_path, map_location=str(device))
    cfg = config_from_checkpoint(raw)
    model = STgramMFN(cfg).to(device)
    model.load_state_dict(raw["model"])
    model.eval()
    meta2label: Dict[str, int] = (
        raw.get("extra", {}).get("cfg", {}).get("meta2label", {})
    )
    return model, cfg, meta2label


@torch.no_grad()
def infer(
    ckpt_path: Path,
    audio_path: Path,
    out_path: Optional[Path] = None,
    label: Optional[int] = None,
):
    device = detect_device()
    model, cfg, meta2label = load_model(ckpt_path, device)
    extractor = FeatureExtractor(cfg).to(device)

    waveform = load_waveform(str(audio_path), cfg.sample_rate).to(device)
    x_wav, x_mel = extractor(waveform)

    if label is None:
        meta = None
        try:
            meta = f"{machine_of_file(str(audio_path))}-{machine_id_of_file(str(audio_path))}"
        except ValueError:
            meta = None
        if meta is not None and meta in meta2label:
            label = meta2label[meta]

    if label is not None:
        labels = torch.tensor([label], device=device)
        logits, feature = model(x_wav.unsqueeze(0), x_mel.unsqueeze(0), labels)
        score = anomaly_score(logits, labels).item()
    else:
        logits, feature = model(x_wav.unsqueeze(0), x_mel.unsqueeze(0), None)
        score = None

    print(f"audio: {audio_path}")
    print(f"feature embedding: {tuple(feature.shape)}")
    if score is not None:
        print(f"anomaly score (higher = more anomalous): {score:.4f}")
    else:
        print("anomaly score: unavailable (no machine-id label resolved)")

    result = {"feature": feature.squeeze(0).cpu(), "source": str(audio_path)}
    if score is not None:
        result["anomaly_score"] = score
    if out_path is not None:
        torch.save(result, out_path)
        print(f"saved: {out_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--label", type=int, default=None, help="override the machine-ID class label"
    )
    args = parser.parse_args()
    infer(args.ckpt, args.audio, args.out, args.label)
