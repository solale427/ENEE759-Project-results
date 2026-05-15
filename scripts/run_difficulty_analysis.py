#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tailrisk_mp.checkpoints import audit_checkpoint_metrics, run_checkpoint_split_metrics, validate_checkpoint
from tailrisk_mp.feature_analysis import candidate_feature_columns, summarize_feature_candidates
from tailrisk_mp.features import extract_batch_rows
from tailrisk_mp.json_utils import dump_json, dumps_json
from tailrisk_mp.runtime import build_unitraj_config, ensure_numpy_pickle_compat, ensure_repo_pythonpath, make_dataloader
from tailrisk_mp.subsets import build_analysis_subset, write_manifest


DEFAULT_WAYMO_ROOT = Path("/fs/nexus-projects/pc_driving/datasets/sn_womd")
DEFAULT_AV2_ROOT = Path("/fs/nexus-projects/pc_driving/datasets/argoverse2_sn")
DEFAULT_WAYMO_CACHE = Path("/fs/nexus-projects/pc_driving/datasets/waymo_cache")
DEFAULT_AV2_CACHE = Path("/fs/nexus-projects/pc_driving/datasets/argoverse_cache")
DEFAULT_CHECKPOINTS = [
    Path("/fs/nexus-projects/pc_driving/baseline_exps/mtr_argoverse_base/epoch=45-best_val/mineADE6=0.00.ckpt"),
    Path("/fs/nexus-projects/pc_driving/yaghoubi/baseline_exps/mtr_argoverse2_20250411_170544/last.ckpt"),
]


def _build_project_decision_note(feature_summary: pd.DataFrame, *, output_path: Path) -> str:
    if feature_summary.empty:
        recommendation = "insufficient evidence"
        lines = [
            "# Project Decision",
            "",
            "Recommendation: **insufficient evidence**",
            "",
            "No feature ranking summary was produced.",
        ]
    else:
        focus = feature_summary[feature_summary["error_column"].isin(["mtr_minfde6", "mtr_brier_minfde6", "cv_fde"])].copy()
        if focus.empty:
            focus = feature_summary.copy()
        best = focus.sort_values("spearman", ascending=False).iloc[0]
        recommendation = "proceed but pivot the core hypothesis"
        if float(best["spearman"]) < 0.15:
            recommendation = "stop this direction"
        elif str(best["feature_name"]) == "kalman_difficulty_6s":
            recommendation = "proceed but pivot the core hypothesis"
        lines = [
            "# Project Decision",
            "",
            f"Recommendation: **{recommendation}**",
            "",
            "## Evidence",
            "",
            f"- Best current feature: `{best['score_name']}` on `{best['error_column']}` "
            f"(spearman={best['spearman']:.3f}, capture={best['top20_error_capture']:.3f}).",
            "",
            "## Interpretation",
            "",
            "- Keep the base CSV as the source of truth.",
            "- Use notebooks to test new feature combinations and clustering ideas.",
            "- Treat driving style as a candidate factor to test, not as a fixed pipeline assumption.",
        ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return recommendation


def _extract_dataset_rows(
    repo_root: Path,
    dataset: str,
    train_path: Path,
    val_path: Path,
    cache_root: Path,
    *,
    batch_size: int,
    max_batches: int | None,
    extract_train: bool = True,
    extract_val: bool = True,
    max_data_num: int | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_repo_pythonpath(repo_root)
    ensure_numpy_pickle_compat()

    cfg = build_unitraj_config(repo_root, train_path, val_path, cache_root, max_data_num=max_data_num)
    train_df = (
        _extract_split_rows(
            cfg,
            split="train",
            val=False,
            batch_size=batch_size,
            max_batches=max_batches,
            dataset=dataset,
            seed=seed,
        )
        if extract_train
        else pd.DataFrame()
    )
    val_df = (
        _extract_split_rows(
            cfg,
            split="val",
            val=True,
            batch_size=batch_size,
            max_batches=max_batches,
            dataset=dataset,
            seed=seed,
        )
        if extract_val
        else pd.DataFrame()
    )
    return train_df, val_df


def _extract_split_rows(
    cfg,
    *,
    split: str,
    val: bool,
    batch_size: int,
    max_batches: int | None,
    dataset: str,
    seed: int,
) -> pd.DataFrame:
    _, loader = make_dataloader(cfg, val=val, batch_size=batch_size, num_workers=0, shuffle=False, seed=seed)
    rows: list[dict] = []
    for batch_idx, batch in enumerate(loader):
        rows.extend(extract_batch_rows(batch, dataset=dataset, split=split))
        if max_batches is not None and batch_idx + 1 >= max_batches:
            break
    return pd.DataFrame(rows)


def _mtr_merge_coverage(
    merged_df: pd.DataFrame,
    *,
    dataset: str,
    split: str,
    error_col: str = "mtr_minfde6",
) -> dict:
    """Report how many feature rows received a non-null MTR metric.

    With shuffle=False and identical UniTraj configs, pass 2 (feature
    extraction) and pass 1/4 (MTR inference) should iterate the same
    (scenario_id, center_objects_id) keys over the same subset dir, so the
    left-merge onto the feature frame is expected to be 100% covered.
    Anything below 100% indicates the two passes diverged (different cfg,
    different subset, truncation, etc.) and Phase 1 analysis on that split
    is compromised.
    """
    rows = int(len(merged_df))
    if rows == 0 or error_col not in merged_df.columns:
        covered = 0
    else:
        covered = int(merged_df[error_col].notna().sum())
    rate = (covered / rows) if rows else 0.0
    print(
        f"[run_difficulty_analysis] MTR merge coverage "
        f"{dataset}/{split}: {covered}/{rows} rows have {error_col} "
        f"({rate:.2%})"
    )
    return {
        "rows": rows,
        "rows_with_mtr": covered,
        "coverage": rate,
        "error_col": error_col,
    }


def _file_md5(path: Path, *, chunk_size: int = 1024 * 1024) -> str | None:
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _write_run_manifest(
    output_root: Path,
    *,
    best_checkpoint: dict | None,
    official_reports: list[dict],
    args_namespace: argparse.Namespace,
    row_counts: dict[str, dict[str, int]],
) -> Path:
    """Freeze checkpoint provenance + run config under ``run_manifest.json``."""
    from tailrisk_mp.json_utils import dump_json

    manifest: dict = {
        "selected_checkpoint": None,
        "all_official_checkpoint_reports": official_reports,
        "cli_args": {
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args_namespace).items()
        },
        "row_counts": row_counts,
    }
    if best_checkpoint is not None:
        ckpt_path = Path(best_checkpoint["checkpoint_path"])
        size_bytes = None
        try:
            size_bytes = ckpt_path.stat().st_size
        except OSError:
            pass
        manifest["selected_checkpoint"] = {
            "path": str(ckpt_path),
            "size_bytes": size_bytes,
            "md5": _file_md5(ckpt_path),
            "official_metrics": best_checkpoint.get("official_metrics", {}),
            "status": best_checkpoint.get("status"),
        }
    manifest_path = output_root / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        dump_json(manifest, f)
    return manifest_path


def _best_checkpoint_report(reports: list[dict]) -> dict | None:
    usable = [report for report in reports if report.get("status") == "usable"]
    if not usable:
        return None
    usable.sort(
        key=lambda report: (
            float(report.get("official_metrics", {}).get("brier_min_FDE", float("inf"))),
            float(report.get("official_metrics", {}).get("min_FDE", float("inf"))),
            report["checkpoint_path"],
        )
    )
    return usable[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract per-sample feature/error tables and raw feature rankings.")
    parser.add_argument("--repo-root", default=REPO_ROOT, type=Path)
    parser.add_argument("--output-root", default=REPO_ROOT / "artifacts" / "day2", type=Path)
    parser.add_argument("--subset-root", default=REPO_ROOT / "data" / "analysis_subsets", type=Path)
    parser.add_argument("--subset-cache-root", default=REPO_ROOT / "data" / "analysis_cache", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, default=2000)
    parser.add_argument("--val-limit", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--datasets", nargs="+", choices=["waymo", "av2"], default=["waymo", "av2"])
    parser.add_argument("--reuse-subsets", action="store_true")
    parser.add_argument("--skip-mtr-eval", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--val-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--mtr-train", action="store_true")
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip the one-batch validate_checkpoint sanity pass. "
             "audit_checkpoint_metrics already confirms every checkpoint works "
             "by running the full val pass, so the validate loop is redundant "
             "and costs one extra full val DataLoader build per candidate.",
    )
    parser.add_argument(
        "--phase1",
        action="store_true",
        help="Phase 1 preset: force MTR merge on train+val, write tables to "
             "artifacts/phase1/, emit run_manifest.json, and imply "
             "--skip-validate (the validate pass is redundant with audit).",
    )
    parser.add_argument("--checkpoint", action="append", type=Path, default=None)
    parser.add_argument("--av2-mtr-train-root", type=Path, default=DEFAULT_AV2_ROOT / "train")
    parser.add_argument("--av2-mtr-val-root", type=Path, default=DEFAULT_AV2_ROOT / "val")
    parser.add_argument("--av2-mtr-cache-root", type=Path, default=DEFAULT_AV2_CACHE)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    if args.phase1:
        if args.output_root == parser.get_default("output_root"):
            args.output_root = repo_root / "artifacts" / "phase1"
        args.mtr_train = True
        args.skip_validate = True
    output_root = args.output_root.resolve()
    subset_root = args.subset_root.resolve()
    subset_cache_root = args.subset_cache_root.resolve()

    ensure_repo_pythonpath(repo_root)
    if args.train_only and args.val_only:
        parser.error("--train-only and --val-only are mutually exclusive")

    extract_train = not args.val_only
    extract_val = not args.train_only

    all_dataset_specs = {
        "waymo": {
            "source_root": DEFAULT_WAYMO_ROOT,
            "train_split": "training",
            "val_split": "validation",
            "cache_root": DEFAULT_WAYMO_CACHE,
        },
        "av2": {
            "source_root": DEFAULT_AV2_ROOT,
            "train_split": "train",
            "val_split": "val",
            "cache_root": DEFAULT_AV2_CACHE,
        },
    }
    dataset_specs = {name: all_dataset_specs[name] for name in args.datasets}

    train_limit = None if args.train_limit is not None and args.train_limit < 0 else args.train_limit
    val_limit = None if args.val_limit is not None and args.val_limit < 0 else args.val_limit

    tables_root = output_root / "tables"
    metrics_root = output_root / "metrics"
    checkpoints_root = output_root / "checkpoints"
    metadata_root = output_root / "metadata"
    for root in [tables_root, metrics_root, checkpoints_root, metadata_root]:
        root.mkdir(parents=True, exist_ok=True)

    checkpoint_candidates = args.checkpoint or DEFAULT_CHECKPOINTS
    av2_train_path = args.av2_mtr_train_root.resolve()
    av2_val_path = args.av2_mtr_val_root.resolve()
    av2_mtr_cache = args.av2_mtr_cache_root.resolve()

    checkpoint_reports = []
    official_checkpoint_reports = []
    official_metrics_frames: dict[str, pd.DataFrame] = {}
    per_sample_vs_official = {}
    validation_skipped = False
    if "av2" in dataset_specs and not args.skip_mtr_eval and args.skip_validate:
        validation_skipped = True
        print(
            "[run_difficulty_analysis] --skip-validate set: skipping the one-batch "
            f"sanity pass for {len(checkpoint_candidates)} checkpoint(s); "
            "audit_checkpoint_metrics will validate them end-to-end."
        )
    if "av2" in dataset_specs and not args.skip_mtr_eval and not args.skip_validate:
        checkpoint_reports = [
            validate_checkpoint(
                repo_root,
                checkpoint_path=checkpoint,
                av2_train_path=av2_train_path,
                av2_val_path=av2_val_path,
                cache_root=av2_mtr_cache,
                device=args.device,
                batch_size=min(args.batch_size, 8),
                seed=args.seed,
            )
            for checkpoint in checkpoint_candidates
        ]
    with open(checkpoints_root / "validation.json", "w", encoding="utf-8") as f:
        dump_json(
            {
                "reports": checkpoint_reports,
                "skipped": validation_skipped,
                "checkpoints_considered": [str(c) for c in checkpoint_candidates],
            },
            f,
        )

    if "av2" in dataset_specs and not args.skip_mtr_eval:
        for checkpoint in checkpoint_candidates:
            report, metrics_df = audit_checkpoint_metrics(
                repo_root,
                checkpoint_path=checkpoint,
                av2_train_path=av2_train_path,
                av2_val_path=av2_val_path,
                cache_root=av2_mtr_cache,
                device=args.device,
                batch_size=min(args.batch_size, 16),
                seed=args.seed,
            )
            official_checkpoint_reports.append(report)
            if report.get("status") == "usable" and metrics_df is not None:
                official_metrics_frames[str(Path(report["checkpoint_path"]).resolve())] = metrics_df

    with open(checkpoints_root / "official_av2_metrics.json", "w", encoding="utf-8") as f:
        dump_json({"reports": official_checkpoint_reports}, f)

    best_checkpoint = _best_checkpoint_report(official_checkpoint_reports)
    best_checkpoint_metrics_df = None
    if best_checkpoint is not None:
        per_sample_vs_official = best_checkpoint.get("per_sample_vs_official", {})
        best_checkpoint_metrics_df = official_metrics_frames.get(str(Path(best_checkpoint["checkpoint_path"]).resolve()))
    with open(checkpoints_root / "per_sample_vs_official_check.json", "w", encoding="utf-8") as f:
        dump_json(per_sample_vs_official, f)

    summary_payload = {
        "seed": args.seed,
        "train_limit": args.train_limit,
        "val_limit": args.val_limit,
        "av2_mtr_train_root": str(av2_train_path),
        "av2_mtr_val_root": str(av2_val_path),
        "av2_mtr_cache_root": str(av2_mtr_cache),
        "checkpoint_reports": checkpoint_reports,
        "official_checkpoint_reports": official_checkpoint_reports,
        "selected_checkpoint": best_checkpoint,
        "per_sample_vs_official_check": per_sample_vs_official,
        "datasets": {},
    }

    if args.audit_only:
        decision_path = output_root / "project_decision.md"
        summary_payload["project_recommendation"] = "audit metrics before analysis"
        summary_payload["project_decision_path"] = str(decision_path)
        decision_path.write_text(
            "# Project Decision\n\nRecommendation: **audit metrics before analysis**\n\n"
            "This run only audited checkpoint metrics on the AV2 validation split.\n",
            encoding="utf-8",
        )
        with open(output_root / "summary.json", "w", encoding="utf-8") as f:
            dump_json(summary_payload, f)
        print(dumps_json(summary_payload))
        return

    subset_manifests = {}
    dataset_paths: dict[str, dict[str, Path | int | None]] = {}
    for dataset, spec in dataset_specs.items():
        dataset_subset_root = subset_root / dataset
        use_full_source_direct = (args.train_only or args.val_only) and not args.reuse_subsets
        if use_full_source_direct:
            manifest = {
                "source_root": str(spec["source_root"]),
                "dest_root": str(spec["source_root"]),
                "seed": args.seed,
                "splits": {},
            }
            if extract_train:
                manifest["splits"][spec["train_split"]] = {
                    "dest_split": spec["train_split"],
                    "subset_root": str(spec["source_root"] / spec["train_split"]),
                    "used_full_source_split": True,
                }
            if extract_val:
                manifest["splits"][spec["val_split"]] = {
                    "dest_split": spec["val_split"],
                    "subset_root": str(spec["source_root"] / spec["val_split"]),
                    "used_full_source_split": True,
                }
            max_data_num = None
            if args.train_only:
                max_data_num = train_limit
            elif args.val_only:
                max_data_num = val_limit
            dataset_paths[dataset] = {
                "train_path": spec["source_root"] / spec["train_split"],
                "val_path": spec["source_root"] / spec["val_split"],
                "cache_root": spec["cache_root"],
                "max_data_num": max_data_num,
            }
            manifest_path = output_root / "subsets" / f"{dataset}_manifest.json"
            write_manifest(manifest_path, manifest)
            subset_manifests[dataset] = manifest
            continue

        train_subset_dir = dataset_subset_root / spec["train_split"]
        val_subset_dir = dataset_subset_root / spec["val_split"]
        train_mapping = train_subset_dir / "dataset_mapping.pkl"
        train_summary = train_subset_dir / "dataset_summary.pkl"
        val_mapping = val_subset_dir / "dataset_mapping.pkl"
        val_summary = val_subset_dir / "dataset_summary.pkl"
        have_reusable_train = train_mapping.exists() and train_summary.exists()
        have_reusable_val = val_mapping.exists() and val_summary.exists()
        if args.reuse_subsets and (not extract_train or have_reusable_train) and (not extract_val or have_reusable_val):
            manifest = {
                "source_root": str(spec["source_root"]),
                "dest_root": str(dataset_subset_root),
                "seed": args.seed,
                "splits": {},
            }
            if extract_train:
                manifest["splits"][spec["train_split"]] = {
                    "dest_split": spec["train_split"],
                    "subset_root": str(train_subset_dir),
                    "reused_existing": True,
                }
            if extract_val:
                manifest["splits"][spec["val_split"]] = {
                    "dest_split": spec["val_split"],
                    "subset_root": str(val_subset_dir),
                    "reused_existing": True,
                }
        else:
            split_limits = {}
            if extract_train:
                split_limits[spec["train_split"]] = train_limit
            if extract_val:
                split_limits[spec["val_split"]] = val_limit
            manifest = build_analysis_subset(spec["source_root"], dataset_subset_root, split_limits=split_limits, seed=args.seed, clear=False)
        manifest_path = output_root / "subsets" / f"{dataset}_manifest.json"
        write_manifest(manifest_path, manifest)
        subset_manifests[dataset] = manifest
        dataset_paths[dataset] = {
            "train_path": (subset_root / dataset / spec["train_split"]) if extract_train else (spec["source_root"] / spec["train_split"]),
            "val_path": (subset_root / dataset / spec["val_split"]) if extract_val else (spec["source_root"] / spec["val_split"]),
            "cache_root": subset_cache_root / dataset,
            "max_data_num": None,
        }
    combined_feature_summaries = []

    for dataset in dataset_specs:
        train_path = Path(dataset_paths[dataset]["train_path"])
        val_path = Path(dataset_paths[dataset]["val_path"])
        cache_root = Path(dataset_paths[dataset]["cache_root"])
        max_data_num = dataset_paths[dataset].get("max_data_num")

        train_df, val_df = _extract_dataset_rows(
            repo_root,
            dataset,
            train_path,
            val_path,
            cache_root,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
            extract_train=extract_train,
            extract_val=extract_val,
            max_data_num=max_data_num,
            seed=args.seed,
        )

        mtr_merge_coverage: dict[str, dict] = {}
        if dataset == "av2" and best_checkpoint is not None and best_checkpoint_metrics_df is not None and not args.skip_mtr_eval and extract_val:
            val_df = val_df.merge(best_checkpoint_metrics_df, on=["scenario_id", "center_objects_id"], how="left")
            mtr_merge_coverage["val"] = _mtr_merge_coverage(val_df, dataset=dataset, split="val")
        # Merge MTR errors onto the train split whenever a usable checkpoint exists
        # and we actually extracted train rows. --mtr-train is kept as a legacy flag
        # but is no longer required for this merge (matches Phase 1 plan §1).
        run_mtr_on_train = (
            dataset == "av2"
            and best_checkpoint is not None
            and not args.skip_mtr_eval
            and extract_train
            and (args.mtr_train or args.phase1)
        )
        if run_mtr_on_train:
            train_mtr_report, train_mtr_df = run_checkpoint_split_metrics(
                repo_root,
                Path(best_checkpoint["checkpoint_path"]),
                av2_train_path=train_path,
                av2_val_path=val_path,
                cache_root=cache_root,
                split="train",
                device=args.device,
                batch_size=min(args.batch_size, 16),
                max_data_num=max_data_num,
                seed=args.seed,
            )
            summary_payload.setdefault("train_split_mtr_report", train_mtr_report)
            if train_mtr_df is not None:
                train_df = train_df.merge(train_mtr_df, on=["scenario_id", "center_objects_id"], how="left")
                mtr_merge_coverage["train"] = _mtr_merge_coverage(train_df, dataset=dataset, split="train")

        train_table_path = tables_root / f"{dataset}_train_feature_table.csv"
        val_table_path = tables_root / f"{dataset}_val_feature_table.csv"
        if extract_train:
            train_df.to_csv(train_table_path, index=False)
        if extract_val:
            val_df.to_csv(val_table_path, index=False)

        feature_source_df = train_df if extract_train and not train_df.empty else val_df
        feature_cols = candidate_feature_columns(feature_source_df)
        metadata = {
            "feature_columns": feature_cols,
            "error_columns": [
                col
                for col in [
                    "cv_ade",
                    "cv_fde",
                    "cv_miss_rate",
                    "mtr_minade6",
                    "mtr_minfde6",
                    "mtr_brier_minade6",
                    "mtr_brier_minfde6",
                    "mtr_miss_rate",
                    "mtr_meanfde6_weighted",
                    "mtr_meanfde6_unweighted",
                ]
                if col in set(train_df.columns) | set(val_df.columns)
            ],
            "id_columns": ["dataset", "split", "scenario_id", "center_objects_id"],
            "row_counts": {
                "train": int(len(train_df)),
                "val": int(len(val_df)),
            },
        }
        with open(metadata_root / f"{dataset}_feature_metadata.json", "w", encoding="utf-8") as f:
            dump_json(metadata, f)

        dataset_payload = {
            "metadata_path": str(metadata_root / f"{dataset}_feature_metadata.json"),
            "summary_paths": {},
            "mtr_merge_coverage": mtr_merge_coverage,
        }
        if extract_train:
            train_summary = summarize_feature_candidates(
                train_df,
                dataset=dataset,
                split="train",
                output_path=metrics_root / f"{dataset}_train_feature_scores.csv",
            )
            combined_feature_summaries.append(train_summary)
            dataset_payload["train_rows"] = int(len(train_df))
            dataset_payload["train_table"] = str(train_table_path)
            dataset_payload["summary_paths"]["train"] = str(metrics_root / f"{dataset}_train_feature_scores.csv")
        if extract_val:
            val_summary = summarize_feature_candidates(
                val_df,
                dataset=dataset,
                split="val",
                output_path=metrics_root / f"{dataset}_val_feature_scores.csv",
            )
            combined_feature_summaries.append(val_summary)
            dataset_payload["val_rows"] = int(len(val_df))
            dataset_payload["val_table"] = str(val_table_path)
            dataset_payload["summary_paths"]["val"] = str(metrics_root / f"{dataset}_val_feature_scores.csv")
        summary_payload["datasets"][dataset] = dataset_payload

    combined = pd.concat(combined_feature_summaries, ignore_index=True) if combined_feature_summaries else pd.DataFrame()
    if not combined.empty:
        combined.to_csv(metrics_root / "combined_feature_summary.csv", index=False)
        go_no_go_rows = combined.sort_values(["error_column", "spearman"], ascending=[True, False]).groupby("error_column").head(15)
        summary_payload["top_feature_view"] = go_no_go_rows.to_dict(orient="records")

    decision_path = output_root / "project_decision.md"
    summary_payload["project_recommendation"] = _build_project_decision_note(combined, output_path=decision_path)
    summary_payload["project_decision_path"] = str(decision_path)

    row_counts = {
        dataset: {
            "train": summary_payload["datasets"].get(dataset, {}).get("train_rows", 0),
            "val": summary_payload["datasets"].get(dataset, {}).get("val_rows", 0),
            "mtr_merge_coverage": summary_payload["datasets"].get(dataset, {}).get("mtr_merge_coverage", {}),
        }
        for dataset in summary_payload.get("datasets", {})
    }
    manifest_path = _write_run_manifest(
        output_root,
        best_checkpoint=best_checkpoint,
        official_reports=official_checkpoint_reports,
        args_namespace=args,
        row_counts=row_counts,
    )
    summary_payload["run_manifest_path"] = str(manifest_path)

    with open(output_root / "summary.json", "w", encoding="utf-8") as f:
        dump_json(summary_payload, f)

    print(dumps_json(summary_payload))


if __name__ == "__main__":
    main()
