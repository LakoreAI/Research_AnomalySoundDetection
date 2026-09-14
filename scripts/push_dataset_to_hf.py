"""Upload a prepared dataset directory to the Hugging Face Hub.

Reads HF_TOKEN (or HF_API_KEY) from .env / the environment.

Usage:
    uv run python scripts/push_dataset_to_hf.py --folder data/raw \
        --repo_id <user>/stgram-mfn-dcase2020
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.io_utils import load_env  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--repo_id", type=str, default=None)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    load_env(REPO_ROOT / ".env")
    token = os.getenv("HF_TOKEN") or os.getenv("HF_API_KEY")
    if not token:
        raise ValueError("set HF_TOKEN (or HF_API_KEY) in .env or the environment")
    if not args.folder.exists():
        raise FileNotFoundError(args.folder)

    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise SystemExit("pip install huggingface_hub") from e

    user = os.getenv("HF_USER", "LakoreAI")
    repo_id = args.repo_id or f"{user}/stgram-mfn-dcase2020"
    print(f"uploading {args.folder} -> dataset {repo_id}")
    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id, repo_type="dataset", private=args.private, exist_ok=True
    )
    api.upload_folder(
        folder_path=str(args.folder),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload prepared anomalous-sound-detection dataset",
    )
    print(f"done: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
