"""Upload a trained STgram-MFN checkpoint (+ model card) to the Hugging Face Hub.

Reads HF_TOKEN (or HF_API_KEY) from .env / the environment. Repositories are
created private by default (flip with --public).

Usage:
    uv run python scripts/push_model_to_hf.py --ckpt checkpoints/<run>/best.pt \
        --repo_id <user>/stgram-mfn-dcase2020
"""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.io_utils import load_env  # noqa: E402

MODEL_CARD = """---
license: mit
tags:
- audio
- anomaly-detection
- anomalous-sound-detection
- dcase
- mimii
- stgram-mfn
- edge
- int8
- pytorch
library_name: pytorch
---

# STgram-MFN — Embedded Acoustic Anomaly Detection

STgram-MFN (spectral-temporal fusion + MobileFaceNet + ArcFace) trained for
DCASE Task 2 (2020–2022 formulation) machine condition monitoring, as the
reference backbone of a backbone-swap / INT8-quantization tradeoff study.

- Input: 16 kHz mono, 10 s clips
- Frontend: log-mel Sgram (128 mels, n_fft 1024, hop 512) + learned Tgram
- Head: MobileFaceNet (128-d embedding) + ArcFace
- Anomaly score: `-log_softmax(logits)[machine_id]`

See `config.yaml` for the exact training configuration and the repository
`docs/` for the experiment protocol.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs" / "train.yaml"
    )
    parser.add_argument("--repo_id", type=str, default=None)
    parser.add_argument("--public", action="store_true")
    args = parser.parse_args()

    load_env(REPO_ROOT / ".env")
    token = os.getenv("HF_TOKEN") or os.getenv("HF_API_KEY")
    if not token:
        raise ValueError("set HF_TOKEN (or HF_API_KEY) in .env or the environment")
    if not args.ckpt.exists():
        raise FileNotFoundError(args.ckpt)

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError as e:
        raise SystemExit("pip install huggingface_hub") from e

    user = os.getenv("HF_USER", "LakoreAI")
    repo_id = args.repo_id or f"{user}/stgram-mfn-dcase2020"
    create_repo(
        repo_id=repo_id,
        repo_type="model",
        token=token,
        exist_ok=True,
        private=not args.public,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "README.md").write_text(MODEL_CARD)
        shutil.copy(args.ckpt, tmp_path / "best.pt")
        if args.config.exists():
            shutil.copy(args.config, tmp_path / "config.yaml")
        HfApi(token=token).upload_folder(
            folder_path=str(tmp_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload STgram-MFN checkpoint and model card",
        )
    print(f"done: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
