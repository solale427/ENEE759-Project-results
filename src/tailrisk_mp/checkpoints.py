from __future__ import annotations

from pathlib import Path
import traceback

import torch

from tailrisk_mp.metrics import (
    aggregate_per_sample_metrics,
    build_av2_pred_dicts,
    compare_per_sample_to_official,
    official_av2_metrics_from_pred_dicts,
    prediction_metrics_from_output,
)
from tailrisk_mp.runtime import build_mtr_model, build_unitraj_config, ensure_numpy_pickle_compat, ensure_repo_pythonpath, make_dataloader, move_to_device


PLAIN_PATH_HINTS = ("baseline_exps/mtr_", "mtr_argoverse_base", "mtr_argoverse2")
MODIFIED_PATH_HINTS = ("polysona", "actions", "smart", "balanced", "transfer", "polysmart", "traverse")


def _path_label(path: Path) -> str:
    lowered = str(path).lower()
    if any(token in lowered for token in MODIFIED_PATH_HINTS):
        return "modified MTR"
    if any(token in lowered for token in PLAIN_PATH_HINTS):
        return "plain MTR baseline"
    return "modified MTR"


def _load_checkpoint_model(
    repo_root: Path,
    checkpoint_path: Path,
    *,
    av2_train_path: Path,
    av2_val_path: Path,
    cache_root: Path,
    max_data_num: int | None = None,
) -> tuple[dict, object | None, object | None, bool]:
    repo_root = repo_root.resolve()
    checkpoint_path = checkpoint_path.resolve()
    ensure_repo_pythonpath(repo_root)
    ensure_numpy_pickle_compat()

    result: dict[str, object] = {
        "checkpoint_path": str(checkpoint_path),
        "classification": "unusable",
        "status": "broken",
    }
    if not checkpoint_path.exists():
        result["reason"] = "missing checkpoint"
        return result, None, None, False

    cfg = build_unitraj_config(repo_root, av2_train_path, av2_val_path, cache_root, max_data_num=max_data_num)

    try:
        model = build_mtr_model(cfg)
    except Exception as exc:
        result["reason"] = f"unitraj MTR import failed: {exc!r}"
        result["traceback"] = traceback.format_exc()
        return result, None, None, False

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    result["state_dict_keys"] = len(state_dict)

    try:
        incompatible = model.load_state_dict(state_dict, strict=False)
        result["missing_keys"] = incompatible.missing_keys[:20]
        result["unexpected_keys"] = incompatible.unexpected_keys[:20]
        strict_match = not incompatible.missing_keys and not incompatible.unexpected_keys
        result["strict_match"] = strict_match
        return result, cfg, model, strict_match
    except Exception as exc:
        result["reason"] = f"state dict load failed: {exc!r}"
        result["traceback"] = traceback.format_exc()
        return result, None, None, False


def validate_checkpoint(
    repo_root: Path,
    checkpoint_path: Path,
    *,
    av2_train_path: Path,
    av2_val_path: Path,
    cache_root: Path,
    device: str = "cpu",
    batch_size: int = 4,
    seed: int = 42,
) -> dict:
    result, cfg, model, strict_match = _load_checkpoint_model(
        repo_root,
        checkpoint_path,
        av2_train_path=av2_train_path,
        av2_val_path=av2_val_path,
        cache_root=cache_root,
    )
    if cfg is None or model is None:
        return result

    try:
        _, loader = make_dataloader(cfg, val=True, batch_size=batch_size, num_workers=0, shuffle=False, seed=seed)
        batch = next(iter(loader))
        model = model.to(device)
        model.eval()
        moved_batch = move_to_device(batch, device)
        with torch.no_grad():
            prediction, _ = model(moved_batch)
        metrics = prediction_metrics_from_output(moved_batch, prediction)
        result["sample_rows"] = int(len(metrics))
        result["sample_metrics"] = metrics.head(5).to_dict(orient="records")
        result["status"] = "usable"
        result["classification"] = _path_label(checkpoint_path) if strict_match else "modified MTR"
        return result
    except Exception as exc:
        result["reason"] = f"one-batch inference failed: {exc!r}"
        result["traceback"] = traceback.format_exc()
        return result


def run_checkpoint_split_metrics(
    repo_root: Path,
    checkpoint_path: Path,
    *,
    av2_train_path: Path,
    av2_val_path: Path,
    cache_root: Path,
    split: str,
    device: str = "cpu",
    batch_size: int = 8,
    max_data_num: int | None = None,
    seed: int = 42,
) -> tuple[dict, object]:
    if split not in {"train", "val"}:
        raise ValueError(f"Unsupported split: {split}")

    result, cfg, model, strict_match = _load_checkpoint_model(
        repo_root,
        checkpoint_path,
        av2_train_path=av2_train_path,
        av2_val_path=av2_val_path,
        cache_root=cache_root,
        max_data_num=max_data_num,
    )
    if cfg is None or model is None:
        return result, None

    try:
        _, loader = make_dataloader(
            cfg,
            val=(split == "val"),
            batch_size=batch_size,
            num_workers=0,
            shuffle=False,
            seed=seed,
        )
        model = model.to(device)
        model.eval()

        pred_dicts: list[dict] = []
        metrics_frames = []
        with torch.no_grad():
            for batch in loader:
                moved_batch = move_to_device(batch, device)
                prediction, _ = model(moved_batch)
                pred_dicts.extend(build_av2_pred_dicts(moved_batch, prediction))
                metrics_frames.append(prediction_metrics_from_output(moved_batch, prediction))

        if not metrics_frames:
            result["reason"] = f"{split} loader produced no batches"
            return result, None

        import pandas as pd

        metrics_df = pd.concat(metrics_frames, ignore_index=True)
        result["status"] = "usable"
        result["classification"] = _path_label(checkpoint_path) if strict_match else "modified MTR"
        result["split"] = split
        result["rows"] = int(len(metrics_df))

        if split == "val":
            official_metrics = official_av2_metrics_from_pred_dicts(pred_dicts)
            per_sample_aggregate = aggregate_per_sample_metrics(metrics_df)
            comparison = compare_per_sample_to_official(per_sample_aggregate, official_metrics)
            result["official_metrics"] = official_metrics
            result["per_sample_aggregate"] = per_sample_aggregate
            result["per_sample_vs_official"] = comparison

        return result, metrics_df
    except Exception as exc:
        result["reason"] = f"{split} split inference failed: {exc!r}"
        result["traceback"] = traceback.format_exc()
        return result, None


def audit_checkpoint_metrics(
    repo_root: Path,
    checkpoint_path: Path,
    *,
    av2_train_path: Path,
    av2_val_path: Path,
    cache_root: Path,
    device: str = "cpu",
    batch_size: int = 8,
    seed: int = 42,
) -> tuple[dict, object]:
    result, metrics_df = run_checkpoint_split_metrics(
        repo_root,
        checkpoint_path,
        av2_train_path=av2_train_path,
        av2_val_path=av2_val_path,
        cache_root=cache_root,
        split="val",
        device=device,
        batch_size=batch_size,
        seed=seed,
    )
    if result.get("status") == "usable":
        result["val_rows"] = result.pop("rows", 0)
    return result, metrics_df
