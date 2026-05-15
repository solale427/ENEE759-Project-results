#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tailrisk_mp.json_utils import dump_json
from tailrisk_mp.metrics import aggregate_per_sample_metrics, prediction_metrics_from_output
from tailrisk_mp.runtime import (
    build_mtr_model,
    build_unitraj_config,
    ensure_numpy_pickle_compat,
    ensure_repo_pythonpath,
    make_dataloader,
    move_to_device,
    set_random_seed,
)


DEFAULT_AV2_ROOT = Path("/fs/nexus-projects/pc_driving/datasets/argoverse2_sn")
DEFAULT_AV2_CACHE = Path("/fs/nexus-projects/pc_driving/datasets/argoverse_cache")
DEFAULT_CHECKPOINT = Path(
    "/fs/nexus-projects/pc_driving/baseline_exps/mtr_argoverse_base/epoch=45-best_val/mineADE6=0.00.ckpt"
)


def _normalize_id(value) -> str:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _load_weight_lookup(weights_csv: Path | None, *, weight_column: str) -> dict[tuple[str, str], float]:
    if weights_csv is None:
        return {}
    df = pd.read_csv(weights_csv)
    required = {"scenario_id", "center_objects_id", weight_column}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Weight CSV missing required columns: {sorted(missing)}")
    return {
        (_normalize_id(row.scenario_id), _normalize_id(row.center_objects_id)): float(getattr(row, weight_column))
        for row in df.itertuples(index=False)
    }


def _attach_sample_weights(batch: dict, weight_lookup: dict[tuple[str, str], float], *, default_weight: float) -> dict:
    scenario_ids = batch["input_dict"]["scenario_id"]
    center_ids = batch["input_dict"]["center_objects_id"]
    weights = [
        float(weight_lookup.get((_normalize_id(scenario_id), _normalize_id(center_id)), default_weight))
        for scenario_id, center_id in zip(scenario_ids, center_ids)
    ]
    batch["input_dict"]["sample_weights"] = torch.as_tensor(weights, dtype=torch.float32)
    return batch


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        dump_json(payload, fp)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal weighted fine-tuning driver for MTR smoke tests.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--train-path", type=Path, default=DEFAULT_AV2_ROOT / "train")
    parser.add_argument("--val-path", type=Path, default=DEFAULT_AV2_ROOT / "val")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_AV2_CACHE)
    parser.add_argument("--run-id", default="smoke")
    parser.add_argument("--out-root", type=Path, default=REPO_ROOT / "artifacts" / "difficulty_audit" / "runs")
    parser.add_argument("--weights-csv", type=Path, default=None)
    parser.add_argument("--weight-column", default="sample_weight")
    parser.add_argument("--default-weight", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-batches", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--max-data-num", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_repo_pythonpath(args.repo_root)
    ensure_numpy_pickle_compat()
    set_random_seed(args.seed)

    output_dir = args.out_root / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_unitraj_config(
        args.repo_root,
        args.train_path,
        args.val_path,
        args.cache_root,
        max_data_num=args.max_data_num,
    )
    model = build_mtr_model(cfg)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)

    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError(
            "MTR fine-tuning requires CUDA custom ops in the current UniTraj build. "
            "Run this script on a GPU node, e.g. via scripts/run_on_gpu_node.sh."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Requested CUDA fine-tuning, but torch.cuda.is_available() is False in this shell. "
            "Run on a GPU node with the tailrisk-mp-cu126 environment."
        )
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    train_dataset, train_loader = make_dataloader(
        cfg,
        val=False,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
        seed=args.seed,
    )
    _, val_loader = make_dataloader(
        cfg,
        val=True,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
        seed=args.seed,
    )

    weight_lookup = _load_weight_lookup(args.weights_csv, weight_column=args.weight_column)
    losses: list[float] = []

    model.train()
    train_iter = iter(train_loader)
    for step_idx in range(1, args.steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = _attach_sample_weights(batch, weight_lookup, default_weight=args.default_weight)
        batch = move_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(batch)
        loss.backward()
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        if not args.dry_run:
            optimizer.step()
        losses.append(float(loss.detach().cpu()))

    model.eval()
    val_metrics = None
    with torch.no_grad():
        metrics_frames = []
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= args.val_batches:
                break
            batch = move_to_device(batch, device)
            prediction, _ = model(batch)
            metrics_frames.append(prediction_metrics_from_output(batch, prediction))
        if metrics_frames:
            val_metrics = aggregate_per_sample_metrics(pd.concat(metrics_frames, ignore_index=True))

    ckpt_path = output_dir / f"ckpt_step{args.steps}.pt"
    torch.save({"state_dict": model.state_dict()}, ckpt_path)

    summary = {
        "run_id": args.run_id,
        "checkpoint": str(args.checkpoint),
        "weights_csv": str(args.weights_csv) if args.weights_csv is not None else None,
        "weight_column": args.weight_column,
        "default_weight": args.default_weight,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "device": args.device,
        "seed": args.seed,
        "dry_run": args.dry_run,
        "max_data_num": args.max_data_num,
        "loss_trace": losses,
        "loss_start": losses[0] if losses else None,
        "loss_end": losses[-1] if losses else None,
        "val_metrics": val_metrics,
        "checkpoint_out": str(ckpt_path),
        "missing_keys": incompatible.missing_keys[:20],
        "unexpected_keys": incompatible.unexpected_keys[:20],
        "train_dataset_len": len(train_dataset),
    }
    _save_json(output_dir / "summary.json", summary)

    print(f"Run: {args.run_id}")
    print(f"Checkpoint out: {ckpt_path}")
    if losses:
        print(f"Loss start/end: {losses[0]:.6f} -> {losses[-1]:.6f}")
    if val_metrics:
        print("Val metrics:")
        for key, value in val_metrics.items():
            print(f"  {key}: {value:.6f}")


if __name__ == "__main__":
    main()
