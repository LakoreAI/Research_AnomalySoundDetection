"""Hugging Face Hub helpers for incremental checkpoint push / resume.

Optional dependency: `huggingface_hub` (`uv sync --extra hub`). Every function
raises only on a missing dependency; network/permission failures are left to the
caller so the training loop can log-and-continue (checkpointing must never kill
a run).
"""

import os
from pathlib import Path
from typing import List, Optional, Union


def _token(token: Optional[str] = None) -> Optional[str]:
    return token or os.getenv("HF_TOKEN") or os.getenv("HF_API_KEY")


def _api():
    try:
        from huggingface_hub import HfApi
    except ImportError as e:  # pragma: no cover - optional dep
        raise RuntimeError(
            "huggingface_hub is not installed (uv sync --extra hub)"
        ) from e
    return HfApi(token=_token())


def push_file(
    path: Union[str, Path],
    repo_id: str,
    path_in_repo: Optional[str] = None,
    repo_type: str = "model",
    private: bool = True,
    token: Optional[str] = None,
) -> str:
    """Upload one file to a HF repo (created private if absent). Returns the
    repo URL. Callers should wrap this in try/except — it can raise on
    auth/network errors.
    """
    from huggingface_hub import create_repo

    path = Path(path)
    token = _token(token)
    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        exist_ok=True,
        private=private,
        token=token,
    )
    _api().upload_file(
        path_or_fileobj=str(path),
        path_in_repo=path_in_repo or path.name,
        repo_id=repo_id,
        repo_type=repo_type,
    )
    return f"https://huggingface.co/{repo_id}"


def list_run_files(repo_id: str, prefix: str, repo_type: str = "model") -> List[str]:
    """Files in `repo_id` under `prefix/` (e.g. `<run_name>/epoch_*.pt`)."""
    files = _api().list_repo_files(repo_id=repo_id, repo_type=repo_type)
    prefix = prefix.rstrip("/") + "/"
    return sorted(f for f in files if f.startswith(prefix))


def pull_run(
    repo_id: str,
    run_name: str,
    dest_dir: Union[str, Path],
    repo_type: str = "model",
) -> int:
    """Download every checkpoint under `<run_name>/` into `dest_dir`
    (flat filenames). Returns the number of files pulled. No-op (0) if the
    repo/prefix does not exist.
    """
    from huggingface_hub import hf_hub_download

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        files = list_run_files(repo_id, run_name, repo_type)
    except Exception:
        return 0
    n = 0
    for rel in files:
        try:
            cached = hf_hub_download(
                repo_id=repo_id,
                filename=rel,
                repo_type=repo_type,
                token=_token(),
            )
            (dest_dir / Path(rel).name).write_bytes(Path(cached).read_bytes())
            n += 1
        except Exception:
            continue
    return n


def latest_epoch_checkpoint(ckpt_dir: Union[str, Path]) -> Optional[str]:
    """Path to the highest-numbered `epoch_N.pt` in `ckpt_dir`, or None."""
    import re

    ckpt_dir = Path(ckpt_dir)
    best_epoch, best_path = -1, None
    for p in ckpt_dir.glob("epoch_*.pt"):
        m = re.search(r"epoch_(\d+)\.pt$", p.name)
        if m and int(m.group(1)) > best_epoch:
            best_epoch, best_path = int(m.group(1)), str(p)
    return best_path
