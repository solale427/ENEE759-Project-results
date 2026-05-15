from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ID_COLUMNS = {
    "dataset",
    "split",
    "scenario_id",
    "center_objects_id",
}

ERROR_COLUMNS = {
    "cv_ade",
    "cv_fde",
    "cv_miss_rate",
    "mtr_minade6",
    "mtr_minfde6",
    "mtr_miss_rate",
    "mtr_brier_minade6",
    "mtr_brier_minfde6",
    "mtr_meanfde6_weighted",
    "mtr_meanfde6_unweighted",
}

SIGNED_FEATURES = {
    "avg_long_accel",
    "avg_lat_accel",
    "avg_long_jerk",
    "avg_lat_jerk",
    "heading_change_signed",
    "curvature_signed",
    "max_closing_speed",
}

ENDOGENOUS_EXACT_COLUMNS = {
    "mtr_best_mode_prob",
    "mtr_best_fde_mode_idx",
}

ENDOGENOUS_PREFIXES = (
    "mtr_best_",
    "model_",
    "pred_",
    "confidence_",
)

STYLE_LABEL_COLUMNS = {
    "style_label_K2",
    "style_label_K3",
}

LEARNED_DIFFICULTY_PRE_INFERENCE = "learned_difficulty_pre_inference"
LEARNED_DIFFICULTY_FROM_MODEL_CONFIDENCE = "learned_difficulty_from_model_confidence"
LEARNED_DIFFICULTY_FULL = "learned_difficulty_full"

LEARNED_FAMILY_NAMES = (
    LEARNED_DIFFICULTY_PRE_INFERENCE,
    LEARNED_DIFFICULTY_FROM_MODEL_CONFIDENCE,
    LEARNED_DIFFICULTY_FULL,
)


def is_endogenous_feature(column: str) -> bool:
    if column in ENDOGENOUS_EXACT_COLUMNS:
        return True
    return any(column.startswith(prefix) for prefix in ENDOGENOUS_PREFIXES)


def candidate_feature_columns(
    df: pd.DataFrame, *, family: str = LEARNED_DIFFICULTY_FULL
) -> list[str]:
    if family not in LEARNED_FAMILY_NAMES:
        raise ValueError(f"Unsupported feature family: {family}")

    cols: list[str] = []
    for column in df.columns:
        if column in ID_COLUMNS or column in ERROR_COLUMNS or column in STYLE_LABEL_COLUMNS:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().sum() == 0:
            continue
        endogenous = is_endogenous_feature(column)
        if family == LEARNED_DIFFICULTY_PRE_INFERENCE and endogenous:
            continue
        if family == LEARNED_DIFFICULTY_FROM_MODEL_CONFIDENCE and not endogenous:
            continue
        cols.append(column)
    return cols


def candidate_feature_families(df: pd.DataFrame) -> dict[str, list[str]]:
    return {family: candidate_feature_columns(df, family=family) for family in LEARNED_FAMILY_NAMES}


def feature_family_definitions(df: pd.DataFrame) -> dict[str, object]:
    families = candidate_feature_families(df)
    return {
        "family_names": list(LEARNED_FAMILY_NAMES),
        "id_columns": sorted(ID_COLUMNS),
        "error_columns": sorted(ERROR_COLUMNS),
        "endogenous_exact_columns": sorted(ENDOGENOUS_EXACT_COLUMNS),
        "endogenous_prefixes": list(ENDOGENOUS_PREFIXES),
        "style_label_columns": sorted(STYLE_LABEL_COLUMNS),
        "families": families,
    }


def transform_series(series: pd.Series, mode: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).astype(float)
    if values.notna().any():
        fill_value = float(values.dropna().median())
    else:
        fill_value = 0.0
    values = values.fillna(fill_value)

    if mode == "raw":
        return values
    if mode == "log1p":
        return np.log1p(np.clip(values, a_min=0.0, a_max=None))
    if mode == "abs":
        return values.abs()
    if mode == "abs_log1p":
        return np.log1p(values.abs())
    if mode == "signed_log1p":
        return np.sign(values) * np.log1p(np.abs(values))
    if mode == "rank":
        return values.rank(method="average", pct=True)
    raise ValueError(f"Unknown transform mode: {mode}")


def default_feature_variants(feature_name: str) -> list[tuple[str, str]]:
    if feature_name in SIGNED_FEATURES:
        return [
            (feature_name, "raw"),
            (f"{feature_name}__abs", "abs"),
            (f"{feature_name}__abs_log1p", "abs_log1p"),
        ]
    return [
        (feature_name, "raw"),
        (f"{feature_name}__log1p", "log1p"),
    ]


def summarize_feature_candidates(
    df: pd.DataFrame,
    *,
    dataset: str,
    split: str,
    feature_columns: list[str] | None = None,
    family: str = LEARNED_DIFFICULTY_FULL,
    output_path: Path | None = None,
) -> pd.DataFrame:
    feature_columns = feature_columns or candidate_feature_columns(df, family=family)
    error_columns = [
        column
        for column in [
            "mtr_minade6",
            "mtr_minfde6",
            "mtr_brier_minade6",
            "mtr_brier_minfde6",
            "mtr_miss_rate",
            "cv_ade",
            "cv_fde",
            "cv_miss_rate",
        ]
        if column in df.columns
    ]

    rows: list[dict[str, float | str]] = []
    for feature_name in feature_columns:
        for score_name, mode in default_feature_variants(feature_name):
            score_series = transform_series(df[feature_name], mode)
            for error_column in error_columns:
                error_series = pd.to_numeric(df[error_column], errors="coerce")
                valid = score_series.notna() & error_series.notna()
                if valid.sum() < 10:
                    continue
                score_valid = score_series[valid]
                error_valid = error_series[valid]
                hard_cut = score_valid.quantile(0.80)
                easy_cut = score_valid.quantile(0.20)
                error_cut = error_valid.quantile(0.80)
                predicted_hard = score_valid >= hard_cut
                actual_hard = error_valid >= error_cut
                capture = float((predicted_hard & actual_hard).sum() / max(actual_hard.sum(), 1))
                easy_mean = float(error_valid[score_valid <= easy_cut].mean())
                hard_mean = float(error_valid[score_valid >= hard_cut].mean())
                rows.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "feature_family": family,
                        "feature_name": feature_name,
                        "score_name": score_name,
                        "transform_mode": mode,
                        "error_column": error_column,
                        "spearman": float(score_valid.corr(error_valid, method="spearman")),
                        "top20_error_capture": capture,
                        "easy_mean_error": easy_mean,
                        "hard_mean_error": hard_mean,
                        "hard_over_easy_uplift": hard_mean - easy_mean,
                    }
                )

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(["error_column", "spearman"], ascending=[True, False])
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_path, index=False)
    return summary
