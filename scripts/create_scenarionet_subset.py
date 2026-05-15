#!/usr/bin/env python3
import argparse
import pickle
import shutil
from pathlib import Path


def parse_split_arg(value: str) -> tuple[str, str, list[str]]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"Invalid split spec '{value}'. Expected split=shard, split=shard1,shard2, or dest:source=shard."
        )
    split_spec, shard_csv = value.split("=", 1)
    shards = [part.strip() for part in shard_csv.split(",") if part.strip()]
    if ":" in split_spec:
        split_name, source_split_name = [part.strip() for part in split_spec.split(":", 1)]
    else:
        split_name = split_spec.strip()
        source_split_name = split_name
    if not split_name or not source_split_name or not shards:
        raise argparse.ArgumentTypeError(f"Invalid split spec '{value}'.")
    return split_name, source_split_name, shards


def build_subset(
    source_root: Path,
    dest_root: Path,
    split_name: str,
    source_split_name: str,
    shards: list[str],
    limit: int | None,
) -> dict:
    source_split = source_root / source_split_name
    dest_split = dest_root / split_name

    with open(source_split / "dataset_mapping.pkl", "rb") as f:
        mapping = pickle.load(f)
    with open(source_split / "dataset_summary.pkl", "rb") as f:
        summary = pickle.load(f)

    selected_files = sorted([key for key, shard in mapping.items() if shard in shards])
    if limit is not None:
        selected_files = selected_files[:limit]

    subset_mapping = {key: mapping[key] for key in selected_files}
    subset_summary = {key: summary[key] for key in selected_files}

    dest_split.mkdir(parents=True, exist_ok=True)
    for shard in shards:
        source_shard = source_split / shard
        dest_shard = dest_split / shard
        if dest_shard.exists() or dest_shard.is_symlink():
            dest_shard.unlink() if dest_shard.is_symlink() else shutil.rmtree(dest_shard)
        dest_shard.symlink_to(source_shard, target_is_directory=True)

    with open(dest_split / "dataset_mapping.pkl", "wb") as f:
        pickle.dump(subset_mapping, f)
    with open(dest_split / "dataset_summary.pkl", "wb") as f:
        pickle.dump(subset_summary, f)

    return {
        "split": split_name,
        "source_split": source_split_name,
        "shards": shards,
        "selected_files": len(selected_files),
        "output_dir": str(dest_split),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny ScenarioNet-compatible subset with shard symlinks.")
    parser.add_argument("--source-root", required=True, help="Root directory of the processed ScenarioNet dataset.")
    parser.add_argument("--dest-root", required=True, help="Destination root for the smoke-test subset.")
    parser.add_argument(
        "--split",
        required=True,
        action="append",
        type=parse_split_arg,
        help="Subset spec in the form split=shard or split=shard1,shard2. Can be repeated.",
    )
    parser.add_argument("--limit", type=int, default=64, help="Max number of scenario files per split.")
    parser.add_argument("--clear", action="store_true", help="Delete the destination root before rebuilding.")
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    dest_root = Path(args.dest_root).resolve()

    if args.clear and dest_root.exists():
        shutil.rmtree(dest_root)

    results = []
    for split_name, source_split_name, shards in args.split:
        results.append(build_subset(source_root, dest_root, split_name, source_split_name, shards, args.limit))

    for result in results:
        print(
            f"{result['split']}: {result['selected_files']} files "
            f"from {result['source_split']}:{','.join(result['shards'])} -> {result['output_dir']}"
        )


if __name__ == "__main__":
    main()
