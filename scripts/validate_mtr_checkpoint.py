#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tailrisk_mp.checkpoints import validate_checkpoint
from tailrisk_mp.json_utils import dump_json, dumps_json
from tailrisk_mp.runtime import ensure_repo_pythonpath

DEFAULT_AV2_TRAIN = Path("/fs/nexus-projects/pc_driving/datasets/argoverse2_sn/train")
DEFAULT_AV2_VAL = Path("/fs/nexus-projects/pc_driving/datasets/argoverse2_sn/val")
DEFAULT_AV2_CACHE = Path("/fs/nexus-projects/pc_driving/datasets/argoverse_cache")


def resolve_path(path: Path, repo_root: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def resolve_dataset_path(path: Path, default_path: Path, repo_root: Path) -> tuple[Path, str | None]:
    requested = resolve_path(path, repo_root)
    if requested.exists():
        return requested, None
    return default_path.resolve(), f"requested path missing, fell back from {requested} to {default_path.resolve()}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate MTR checkpoints against a small AV2 batch.")
    parser.add_argument("--repo-root", default=REPO_ROOT, type=Path)
    parser.add_argument("--checkpoint", action="append", required=True, type=Path)
    parser.add_argument("--av2-train", default=DEFAULT_AV2_TRAIN, type=Path)
    parser.add_argument("--av2-val", default=DEFAULT_AV2_VAL, type=Path)
    parser.add_argument("--cache-root", default=DEFAULT_AV2_CACHE, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--report-path", type=Path, default=None)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    ensure_repo_pythonpath(repo_root)
    av2_train, train_note = resolve_dataset_path(args.av2_train, DEFAULT_AV2_TRAIN, repo_root)
    av2_val, val_note = resolve_dataset_path(args.av2_val, DEFAULT_AV2_VAL, repo_root)
    cache_root = resolve_path(args.cache_root, repo_root)

    reports = [
        validate_checkpoint(
            repo_root,
            checkpoint_path=checkpoint,
            av2_train_path=av2_train,
            av2_val_path=av2_val,
            cache_root=cache_root,
            device=args.device,
            batch_size=args.batch_size,
        )
        for checkpoint in args.checkpoint
    ]

    payload = {
        "requested_av2_train": str(args.av2_train),
        "requested_av2_val": str(args.av2_val),
        "av2_train": str(av2_train),
        "av2_val": str(av2_val),
        "cache_root": str(cache_root),
        "notes": [note for note in [train_note, val_note] if note is not None],
        "reports": reports,
    }
    if args.report_path is not None:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_path, "w", encoding="utf-8") as f:
            dump_json(payload, f)
    print(dumps_json(payload))


if __name__ == "__main__":
    main()
