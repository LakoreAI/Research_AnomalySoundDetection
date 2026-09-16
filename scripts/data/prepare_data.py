"""Validate / normalize the on-disk dataset layout.

Expected (DCASE-style) layout after `download_data.py`:

    <root>/<machine>/train/normal_id_XX_*.wav
    <root>/<machine>/test/normal_id_XX_*.wav
    <root>/<machine>/test/anomaly_id_XX_*.wav

Subcommands:
  --check       report per-machine / per-id normal+anomaly counts, flag problems
  --reorganize  convert RAW MIMII (`<machine>/id_XX/normal|abnormal/*.wav`)
                into the DCASE-style layout above (symlinks by default)
  --make-subset N --out DIR
                build a symlink tree with up to N clips per machine/split for
                fast smoke tests / method canaries

Usage:
    uv run python scripts/data/prepare_data.py --root data/raw --check
    uv run python scripts/data/prepare_data.py --root data/mimii --reorganize
    uv run python scripts/data/prepare_data.py --root data/raw --make-subset 20 --out data/mini
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.audio_utils import (  # noqa: E402
    get_filename_list,
    get_machine_id_list,
)


def check(root: Path) -> int:
    machines = sorted([p.name for p in root.iterdir() if p.is_dir()])
    if not machines:
        print(f"no machine directories under {root}")
        return 1
    problems = 0
    for machine in machines:
        mdir = root / machine
        for split in ("train", "test"):
            sdir = mdir / split
            if not sdir.is_dir():
                continue
            ids = get_machine_id_list(str(sdir))
            files = get_filename_list(str(sdir))
            if not files:
                print(f"  WARNING: {machine}/{split} has no .wav files")
                problems += 1
            if split == "train":
                n_anom = len([f for f in files if Path(f).name.startswith("anomaly")])
                if n_anom:
                    print(f"  WARNING: {machine}/train contains {n_anom} anomaly files")
                    problems += 1
            print(
                f"  {machine}/{split}: {len(files):5d} files, {len(ids):2d} ids {ids}"
            )
    print(f"\n{'OK' if problems == 0 else f'{problems} problem(s) found'}")
    return 0 if problems == 0 else 1


def _link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if copy:
        import shutil

        shutil.copyfile(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def reorganize(root: Path, copy: bool = False, train_ratio: float = 0.5) -> None:
    """Raw MIMII `<machine>/id_XX/normal|abnormal/*.wav` -> DCASE-style.

    Normal clips are split `train_ratio`/rest between train and test (test needs
    normals to be scoreable); every abnormal clip goes to test. Deterministic:
    files are sorted before splitting.
    """
    machines = sorted([p.name for p in root.iterdir() if p.is_dir()])
    for machine in machines:
        mdir = root / machine
        if not (mdir / "train").is_dir() and not any(mdir.glob("id_*")):
            continue
        id_dirs = sorted(
            [p for p in mdir.iterdir() if p.is_dir() and p.name.startswith("id_")]
        )
        for id_dir in id_dirs:
            id_str = id_dir.name
            normals = sorted((id_dir / "normal").glob("*.wav"))
            abnormals = sorted((id_dir / "abnormal").glob("*.wav"))
            n_train = int(len(normals) * train_ratio)
            for i, f in enumerate(normals):
                split = "train" if i < n_train else "test"
                dst = mdir / split / f"normal_{id_str}_{f.name}"
                _link_or_copy(f, dst, copy)
            for f in abnormals:
                dst = mdir / "test" / f"anomaly_{id_str}_{f.name}"
                _link_or_copy(f, dst, copy)
        print(f"  reorganized {machine}: {len(id_dirs)} ids")


def make_subset(root: Path, out: Path, n: int) -> None:
    machines = sorted([p.name for p in root.iterdir() if p.is_dir()])
    for machine in machines:
        for split in ("train", "test"):
            sdir = root / machine / split
            if not sdir.is_dir():
                continue
            files = get_filename_list(str(sdir))
            chosen: List[str] = []
            per_id: Dict[str, int] = {}
            for f in files:
                from src.utils.audio_utils import machine_id_of_file

                _id = machine_id_of_file(f)
                if per_id.get(_id, 0) >= n:
                    continue
                per_id[_id] = per_id.get(_id, 0) + 1
                chosen.append(f)
            for f in chosen:
                _link_or_copy(Path(f), out / machine / split / Path(f).name, copy=False)
        print(f"  {machine}: subset written")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--reorganize", action="store_true")
    parser.add_argument("--copy", action="store_true", help="copy instead of symlink")
    parser.add_argument("--train-ratio", type=float, default=0.5)
    parser.add_argument("--make-subset", type=int, default=None, metavar="N")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if not args.root.exists():
        raise FileNotFoundError(args.root)

    if args.reorganize:
        reorganize(args.root, copy=args.copy, train_ratio=args.train_ratio)
    if args.make_subset is not None:
        if args.out is None:
            raise ValueError("--make-subset requires --out")
        make_subset(args.root, args.out, args.make_subset)
    if args.check or not (args.reorganize or args.make_subset):
        sys.exit(check(args.root))


if __name__ == "__main__":
    main()
