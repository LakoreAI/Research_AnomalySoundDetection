"""Download the anomaly-detection datasets used by this project.

Sources are resolved live from the Zenodo REST API (record -> file list), so
nothing here hardcodes a stale download URL. Verified records:

    mimii          MIMII raw dataset (fan/pump/slider/valve, per-SNR zips)
    dcase2020      DCASE 2020 Task 2 development set  -> <machine>/{train,test}
    dcase2020-eval DCASE 2020 Task 2 additional train + evaluation set
    dcase2022      DCASE 2022 Task 2 development set (MIMII DG-style, 7 machines)

The DCASE zips already extract into the `<machine>/<split>/` layout the training
code expects. Raw MIMII does NOT — run `scripts/data/prepare_data.py --reorganize`
afterwards to convert it to DCASE-style filenames.

Usage:
    uv run python scripts/data/download_data.py --dataset dcase2020 --dest data/raw
    uv run python scripts/data/download_data.py --dataset dcase2020-eval --dest data/raw_eval
    uv run python scripts/data/download_data.py --dataset mimii --snr 0 --dest data/mimii
    uv run python scripts/data/download_data.py --list --dataset dcase2020
"""

import argparse
import concurrent.futures
import shutil
import sys
import threading
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Zenodo record ids, verified via https://zenodo.org/api/records/<id>.
RECORDS: Dict[str, int] = {
    "mimii": 3384388,
    "dcase2020": 3678171,
    "dcase2020-eval-train": 3727685,
    "dcase2020-eval-test": 3841772,
}
# DCASE 2022 Task 2 development set (MIMII DG + ToyADMOS2; machine types are
# fan, gearbox, bearing, slider, ToyCar, ToyTrain, valve — NOT the 2020 set).
DCASE2022_RECORD = 6355122

ALL_MACHINES = ["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"]


def zenodo_files(record: int) -> List[dict]:
    """Return the file list for a Zenodo record via its JSON API."""
    url = f"https://zenodo.org/api/records/{record}"
    with urllib.request.urlopen(url) as resp:
        import json

        data = json.load(resp)
    return data.get("files", [])


def _content_length(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req) as resp:
        return int(resp.headers.get("Content-Length", 0))


def _fetch_range(
    url: str,
    start: int,
    end: int,
    dest: Path,
    progress: dict,
    lock: threading.Lock,
    total: int,
) -> None:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            with lock:
                progress["done"] += len(chunk)
                pct = min(100, progress["done"] * 100 // total)
            print(f"\r    {pct:3d}%", end="", flush=True)


def download(
    url: str, dest: Path, connections: int = 16, size: int | None = None
) -> None:
    """Download `url` to `dest`.

    Zenodo throttles a single connection to ~0.25 MB/s but scales with
    concurrency (measured ~8 MB/s at 32 connections), so the default is a
    multi-connection range download. Falls back to a single stream when the
    size is unknown, `connections <= 1`, or the server ignores Range.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  exists, skipping: {dest.name}")
        return

    if size is None:
        size = _content_length(url)

    if connections <= 1 or size <= 0:
        print(f"  downloading {dest.name} (single stream) ...")
        urllib.request.urlretrieve(url, dest)
        print()
        return

    print(
        f"  downloading {dest.name} with {connections} connections "
        f"({size / 1e9:.2f} GB) ..."
    )
    n = min(connections, max(1, size // (1 << 20)))  # >=1 MiB per part
    bounds = [(size * i // n, size * (i + 1) // n - 1) for i in range(n)]
    parts = [dest.parent / f".{dest.name}.part{i}" for i in range(n)]
    progress = {"done": 0}
    lock = threading.Lock()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
            futures = [
                ex.submit(_fetch_range, url, s, e, p, progress, lock, size)
                for (s, e), p in zip(bounds, parts)
            ]
            for fut in concurrent.futures.as_completed(futures):
                fut.result()  # re-raise any worker failure
        print()
        with open(dest, "wb") as out:
            for p in parts:
                with open(p, "rb") as f:
                    shutil.copyfileobj(f, out)
    finally:
        for p in parts:
            p.unlink(missing_ok=True)

    if dest.stat().st_size != size:
        print(
            f"  size mismatch ({dest.stat().st_size} != {size}); retrying single stream"
        )
        dest.unlink()
        urllib.request.urlretrieve(url, dest)
        print()


def extract(zip_path: Path, dest: Path) -> None:
    print(f"  extracting {zip_path.name} -> {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)


def select_files(files: List[dict], machines: List[str], snr: str = None) -> List[dict]:
    selected = []
    for f in files:
        key = f["key"]
        if not key.endswith(".zip"):
            continue
        if machines and not any(m in key for m in machines):
            continue
        if snr is not None and f"{snr}_dB" not in key:
            continue
        selected.append(f)
    return selected


def url_for(f: dict) -> str:
    """Zenodo's own download link for a file entry."""
    links = f.get("links", {})
    return links.get("self") or links["content"]


def expected_extracted(dest: Path, key: str) -> List[Path]:
    """Directories a given archive should populate under `dest`.

    Lets a re-run skip machines that are already on disk instead of
    re-downloading multi-GB archives. Covers the DCASE dev/eval naming and the
    raw MIMII `<snr>_dB_<machine>` scheme.
    """
    name = key[:-4] if key.endswith(".zip") else key
    if name.startswith("dev_data_"):
        m = name[len("dev_data_") :]
        return [dest / m / "train", dest / m / "test"]
    if name.startswith("eval_data_train_"):
        m = name[len("eval_data_train_") :]
        return [dest / m / "train"]
    if name.startswith("eval_data_test_"):
        m = name[len("eval_data_test_") :]
        return [dest / m / "test"]
    if name.startswith("eval_data_"):
        return [dest / name[len("eval_data_") :]]
    if "_dB_" in name:
        return [dest / name.split("_dB_", 1)[1]]
    return [dest / name]


def already_extracted(dest: Path, key: str) -> bool:
    """True if every directory this archive populates already holds .wav files."""
    for path in expected_extracted(dest, key):
        if not path.is_dir() or not any(path.rglob("*.wav")):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["mimii", "dcase2020", "dcase2020-eval", "dcase2022"],
        default="dcase2020",
    )
    parser.add_argument("--dest", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--machines", nargs="*", default=None)
    parser.add_argument(
        "--snr", choices=["6", "0", "-6"], default=None, help="MIMII SNR level"
    )
    parser.add_argument(
        "--list", action="store_true", help="list files, download nothing"
    )
    parser.add_argument(
        "--keep-archives", action="store_true", help="keep downloaded .zip files"
    )
    parser.add_argument(
        "--connections",
        type=int,
        default=16,
        help="parallel range connections per file (Zenodo throttles single streams)",
    )
    args = parser.parse_args()

    machines = args.machines or []
    if args.dataset == "dcase2020-eval":
        records = [
            (RECORDS["dcase2020-eval-train"], "train"),
            (RECORDS["dcase2020-eval-test"], "test"),
        ]
    elif args.dataset == "dcase2022":
        records = [(DCASE2022_RECORD, None)]
    else:
        records = [(RECORDS[args.dataset], None)]

    archives_dir = args.dest / "_archives"
    for record, _split in records:
        files = zenodo_files(record)
        selected = select_files(files, machines, args.snr)
        print(f"record {record}: {len(selected)} file(s) selected")
        for f in selected:
            print(f"  {f['key']}  ({f['size'] / 1e9:.2f} GB)")
        if args.list:
            continue
        for f in selected:
            if already_extracted(args.dest, f["key"]):
                print(f"  already extracted, skipping: {f['key']}")
                continue
            zip_path = archives_dir / f["key"]
            download(
                url_for(f),
                zip_path,
                connections=args.connections,
                size=f.get("size"),
            )
            extract(zip_path, args.dest)
            if not args.keep_archives:
                zip_path.unlink()

    print(f"\ndone. data under {args.dest}")
    print(
        "next: uv run python scripts/data/prepare_data.py --root", args.dest, "--check"
    )


if __name__ == "__main__":
    main()
