#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tailrisk_mp.runtime import ensure_repo_pythonpath
from tailrisk_mp.subsets import build_analysis_subset, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic ScenarioNet-compatible analysis subsets.")
    parser.add_argument("--repo-root", default=REPO_ROOT, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--dest-root", required=True, type=Path)
    parser.add_argument("--train-split", default="training")
    parser.add_argument("--val-split", default="validation")
    parser.add_argument("--train-limit", type=int, default=2000)
    parser.add_argument("--val-limit", type=int, default=500)
    parser.add_argument("--dest-train-split", default=None)
    parser.add_argument("--dest-val-split", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--manifest-path", type=Path, default=None)
    args = parser.parse_args()

    ensure_repo_pythonpath(args.repo_root)

    manifest = build_analysis_subset(
        args.source_root,
        args.dest_root,
        split_limits={args.train_split: args.train_limit, args.val_split: args.val_limit},
        split_aliases={
            args.train_split: args.dest_train_split or args.train_split,
            args.val_split: args.dest_val_split or args.val_split,
        },
        seed=args.seed,
        clear=args.clear,
    )

    manifest_path = args.manifest_path or (args.dest_root / "manifest.json")
    write_manifest(manifest_path, manifest)
    print(manifest_path)


if __name__ == "__main__":
    main()
