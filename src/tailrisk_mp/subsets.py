from __future__ import annotations

import json
import pickle
import random
import shutil
from pathlib import Path

from tailrisk_mp.json_utils import dump_json


def build_subset_split(
    source_root: Path,
    dest_root: Path,
    *,
    source_split: str,
    dest_split: str,
    limit: int | None,
    seed: int,
) -> dict:
    source_split_dir = source_root / source_split
    dest_split_dir = dest_root / dest_split

    with open(source_split_dir / "dataset_mapping.pkl", "rb") as f:
        mapping = pickle.load(f)
    with open(source_split_dir / "dataset_summary.pkl", "rb") as f:
        summary = pickle.load(f)

    scenario_names = sorted(mapping.keys())
    rng = random.Random(seed)
    rng.shuffle(scenario_names)
    if limit is None:
        selected = sorted(scenario_names)
    else:
        selected = sorted(scenario_names[: min(limit, len(scenario_names))])

    subset_mapping = {key: mapping[key] for key in selected}
    subset_summary = {key: summary[key] for key in selected}
    used_shards = sorted(set(subset_mapping.values()))

    if dest_split_dir.exists():
        shutil.rmtree(dest_split_dir)
    dest_split_dir.mkdir(parents=True, exist_ok=True)
    for shard in used_shards:
        source_shard = source_split_dir / shard
        dest_shard = dest_split_dir / shard
        dest_shard.symlink_to(source_shard, target_is_directory=True)

    with open(dest_split_dir / "dataset_mapping.pkl", "wb") as f:
        pickle.dump(subset_mapping, f)
    with open(dest_split_dir / "dataset_summary.pkl", "wb") as f:
        pickle.dump(subset_summary, f)

    return {
        "source_split": source_split,
        "dest_split": dest_split,
        "limit": limit,
        "seed": seed,
        "selected_scenarios": len(selected),
        "used_shards": used_shards,
        "subset_root": str(dest_split_dir),
        "scenario_names": selected,
    }


def build_analysis_subset(
    source_root: Path,
    dest_root: Path,
    *,
    split_limits: dict[str, int],
    split_aliases: dict[str, str] | None = None,
    seed: int = 42,
    clear: bool = False,
) -> dict:
    source_root = source_root.resolve()
    dest_root = dest_root.resolve()
    split_aliases = split_aliases or {}

    if clear and dest_root.exists():
        shutil.rmtree(dest_root)

    dest_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source_root": str(source_root),
        "dest_root": str(dest_root),
        "seed": seed,
        "splits": {},
    }

    for split_name, limit in split_limits.items():
        dest_split = split_aliases.get(split_name, split_name)
        seed_offset = sum((idx + 1) * ord(char) for idx, char in enumerate(split_name))
        limit_offset = 0 if limit is None else limit
        manifest["splits"][dest_split] = build_subset_split(
            source_root,
            dest_root,
            source_split=split_name,
            dest_split=dest_split,
            limit=limit,
            seed=seed + seed_offset + limit_offset,
        )

    return manifest


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        dump_json(manifest, f)
