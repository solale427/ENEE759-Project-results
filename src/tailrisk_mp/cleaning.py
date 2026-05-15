"""Pre-clustering data cleaning for Phase 1 driving-style validation.

Single entry point: ``filter_for_clustering(df, config=None)``.

Drop reasons tracked in the returned ``drop_report``:

- ``stationary``: ``is_stationary == True`` (parked / idling; no driving style).
- ``short_track``: ``n_valid_future_steps < config.min_valid_future_steps``.
- ``non_vehicle``: ``center_objects_type != 1`` when ``vehicle_only`` is set.
- ``nan_feature``: any non-finite value in the clustering features after transform.
- ``percentile_clip_outlier``: row whose clustering feature falls outside
  the ``[p_lo, p_hi]`` train-split percentile for any clustering feature.
- ``outlier_mahalanobis``: robust Mahalanobis distance (sklearn
  ``EllipticEnvelope``) in the top ``contamination`` fraction.

The percentile clips and the Mahalanobis threshold are *fitted* on the
train split and re-applied to val. Call with ``fit=True`` on train, then
pass the returned ``clean_stats`` back in ``config`` as ``reuse_stats``
when cleaning val.

Outputs ``cleaning_report.json`` when ``report_path`` is provided so that
every Phase-1 figure can be captioned with the exact kept/dropped counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from sklearn.covariance import EllipticEnvelope
except ImportError:  # pragma: no cover - sklearn is a hard dep in this env
    EllipticEnvelope = None  # type: ignore[assignment]

from tailrisk_mp.json_utils import dump_json


CLUSTERING_FEATURES: tuple[str, ...] = (
    "max_abs_accel",
    "var_speed",
    "var_acceleration",
    "gamma",
)

VEHICLE_TYPE_CODE = 1


@dataclass
class CleaningConfig:
    """Configuration for ``filter_for_clustering``.

    Mirrors the pre-registered criteria in ``docs/phase1_decision_criteria.md``.
    """

    clustering_features: tuple[str, ...] = CLUSTERING_FEATURES
    vehicle_only: bool = True
    vehicle_type_code: int = VEHICLE_TYPE_CODE
    min_valid_future_steps: int = 30
    percentile_lo: float = 0.1
    percentile_hi: float = 99.9
    mahalanobis_contamination: float = 0.01
    mahalanobis_random_state: int = 42
    reuse_stats: dict[str, Any] | None = None
    fit_stats: bool = True
    extra_drop_reasons: dict[str, pd.Series] = field(default_factory=dict)


# Keep the order deterministic so the report is comparable across runs.
_DROP_ORDER: tuple[str, ...] = (
    "stationary",
    "short_track",
    "non_vehicle",
    "nan_feature",
    "percentile_clip_outlier",
    "outlier_mahalanobis",
)


def _coerce_bool_mask(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.fillna(False).astype(bool)


def _fit_percentile_clips(
    df: pd.DataFrame, features: Iterable[str], lo: float, hi: float
) -> dict[str, tuple[float, float]]:
    clips: dict[str, tuple[float, float]] = {}
    for feature in features:
        if feature not in df.columns:
            continue
        values = pd.to_numeric(df[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            clips[feature] = (float("-inf"), float("inf"))
            continue
        lo_val = float(np.percentile(values, lo))
        hi_val = float(np.percentile(values, hi))
        clips[feature] = (lo_val, hi_val)
    return clips


def _apply_percentile_clips(
    df: pd.DataFrame, clips: dict[str, tuple[float, float]]
) -> pd.Series:
    """Return a boolean mask: True == row is OUTSIDE the train percentile band on at least one feature."""
    outside = pd.Series(False, index=df.index)
    for feature, (lo_val, hi_val) in clips.items():
        if feature not in df.columns:
            continue
        values = pd.to_numeric(df[feature], errors="coerce")
        outside = outside | (values < lo_val) | (values > hi_val)
    return outside


def _fit_mahalanobis(
    scaled: np.ndarray,
    *,
    contamination: float,
    random_state: int,
) -> tuple[Any | None, float | None]:
    if EllipticEnvelope is None or len(scaled) < 10:
        return None, None
    try:
        envelope = EllipticEnvelope(
            contamination=max(min(contamination, 0.2), 1e-4),
            random_state=random_state,
            support_fraction=None,
        )
        envelope.fit(scaled)
    except Exception:
        return None, None
    distances = envelope.mahalanobis(scaled)
    threshold = float(np.quantile(distances, 1.0 - contamination))
    return envelope, threshold


def _apply_mahalanobis(
    envelope: Any | None,
    threshold: float | None,
    scaled: np.ndarray,
) -> np.ndarray:
    if envelope is None or threshold is None or len(scaled) == 0:
        return np.zeros(len(scaled), dtype=bool)
    distances = envelope.mahalanobis(scaled)
    return distances > threshold


def _standardize(
    df: pd.DataFrame,
    features: Iterable[str],
    stats: dict[str, tuple[float, float]],
) -> np.ndarray:
    cols = []
    for feature in features:
        values = pd.to_numeric(df.get(feature, pd.Series([], dtype=float)), errors="coerce")
        mean, std = stats.get(feature, (0.0, 1.0))
        std = std if std > 1e-6 else 1.0
        cols.append(((values.fillna(mean) - mean) / std).to_numpy())
    if not cols:
        return np.zeros((len(df), 0), dtype=np.float64)
    return np.stack(cols, axis=1)


def _fit_standardize_stats(
    df: pd.DataFrame, features: Iterable[str]
) -> dict[str, tuple[float, float]]:
    stats: dict[str, tuple[float, float]] = {}
    for feature in features:
        values = pd.to_numeric(df.get(feature, pd.Series([], dtype=float)), errors="coerce").dropna()
        if values.empty:
            stats[feature] = (0.0, 1.0)
            continue
        stats[feature] = (float(values.mean()), float(values.std(ddof=0) or 1.0))
    return stats


def filter_for_clustering(
    df: pd.DataFrame,
    config: CleaningConfig | None = None,
    *,
    report_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply Phase-1 cleaning and return ``(cleaned_df, drop_report)``.

    When ``config.fit_stats`` is True (default), percentile clips and the
    Mahalanobis envelope are fit on ``df`` and stored on the report under
    ``"clean_stats"``. Pass that dict back in ``config.reuse_stats`` to
    apply the train-fitted cleaner to val.
    """
    config = config or CleaningConfig()
    features = tuple(config.clustering_features)
    input_rows = int(len(df))
    if df.empty:
        report = {"input_rows": 0, "drop_counts": {}, "kept_rows": 0}
        return df.copy(), report

    df = df.copy()
    keep = pd.Series(True, index=df.index)
    drop_counts: dict[str, int] = {reason: 0 for reason in _DROP_ORDER}

    def apply_reason(reason: str, reason_mask: pd.Series) -> None:
        nonlocal keep
        reason_mask = reason_mask.reindex(df.index).fillna(False).astype(bool)
        to_drop = reason_mask & keep
        drop_counts[reason] = int(to_drop.sum())
        keep = keep & ~reason_mask

    if "is_stationary" in df.columns:
        apply_reason("stationary", _coerce_bool_mask(df["is_stationary"]))

    if "n_valid_future_steps" in df.columns:
        short_mask = pd.to_numeric(df["n_valid_future_steps"], errors="coerce").fillna(0) < config.min_valid_future_steps
        apply_reason("short_track", short_mask)

    if config.vehicle_only and "center_objects_type" in df.columns:
        types = pd.to_numeric(df["center_objects_type"], errors="coerce")
        apply_reason("non_vehicle", types != config.vehicle_type_code)

    # Everything below operates on the clustering features; need a clean numeric view.
    feature_df = df[list(features)].apply(pd.to_numeric, errors="coerce")
    nan_mask = feature_df.replace([np.inf, -np.inf], np.nan).isna().any(axis=1)
    apply_reason("nan_feature", nan_mask)

    # Percentile clips and Mahalanobis are fit on the currently-kept rows
    # so they are not contaminated by stationary / non-vehicle extremes.
    kept_for_fit = df.loc[keep]
    feature_for_fit = feature_df.loc[keep]

    reuse_stats = config.reuse_stats or {}
    if config.fit_stats or "percentile_clips" not in reuse_stats:
        clips = _fit_percentile_clips(
            feature_for_fit, features, config.percentile_lo, config.percentile_hi
        )
    else:
        clips = {k: tuple(v) for k, v in reuse_stats["percentile_clips"].items()}

    clip_outlier_mask = _apply_percentile_clips(feature_df, clips)
    apply_reason("percentile_clip_outlier", clip_outlier_mask)

    feature_for_fit = feature_df.loc[keep]
    if config.fit_stats or "standardize_stats" not in reuse_stats:
        std_stats = _fit_standardize_stats(feature_for_fit, features)
    else:
        std_stats = {k: tuple(v) for k, v in reuse_stats["standardize_stats"].items()}

    scaled_fit = _standardize(feature_for_fit, features, std_stats)
    mahal_info: dict[str, Any] = {"fitted": False, "threshold": None}
    if config.fit_stats or "mahalanobis" not in reuse_stats:
        envelope, threshold = _fit_mahalanobis(
            scaled_fit,
            contamination=config.mahalanobis_contamination,
            random_state=config.mahalanobis_random_state,
        )
        mahal_info = {
            "fitted": envelope is not None,
            "threshold": threshold,
            "contamination": config.mahalanobis_contamination,
        }
    else:
        envelope = reuse_stats["mahalanobis"].get("_envelope")
        threshold = reuse_stats["mahalanobis"].get("threshold")
        mahal_info = dict(reuse_stats["mahalanobis"])

    scaled_all = _standardize(feature_df, features, std_stats)
    mahal_outlier = pd.Series(False, index=df.index)
    if envelope is not None and threshold is not None:
        flags = _apply_mahalanobis(envelope, threshold, scaled_all)
        mahal_outlier = pd.Series(flags, index=df.index)
    apply_reason("outlier_mahalanobis", mahal_outlier)

    for reason, reason_mask in config.extra_drop_reasons.items():
        drop_counts.setdefault(reason, 0)
        apply_reason(reason, reason_mask)

    cleaned = df.loc[keep].copy()
    kept_rows = int(len(cleaned))

    clean_stats: dict[str, Any] = {
        "percentile_clips": {k: list(v) for k, v in clips.items()},
        "standardize_stats": {k: list(v) for k, v in std_stats.items()},
        "mahalanobis": {
            "fitted": mahal_info.get("fitted", False),
            "threshold": mahal_info.get("threshold"),
            "contamination": config.mahalanobis_contamination,
        },
    }
    # Keep envelope out of JSON; expose via a private key for reuse in-process.
    clean_stats["mahalanobis"]["_envelope"] = envelope

    report: dict[str, Any] = {
        "input_rows": input_rows,
        "drop_counts": drop_counts,
        "drop_fractions": {
            k: float(v / max(input_rows, 1)) for k, v in drop_counts.items()
        },
        "kept_rows": kept_rows,
        "kept_fraction": float(kept_rows / max(input_rows, 1)),
        "percentile_clips": clean_stats["percentile_clips"],
        "standardize_stats": clean_stats["standardize_stats"],
        "mahalanobis_threshold": clean_stats["mahalanobis"]["threshold"],
        "mahalanobis_contamination": config.mahalanobis_contamination,
        "config": {
            "clustering_features": list(features),
            "vehicle_only": config.vehicle_only,
            "min_valid_future_steps": config.min_valid_future_steps,
            "percentile_lo": config.percentile_lo,
            "percentile_hi": config.percentile_hi,
        },
        "clean_stats": clean_stats,  # reusable for val split
    }

    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        json_report = {k: v for k, v in report.items() if k != "clean_stats"}
        json_report["clean_stats"] = {
            "percentile_clips": clean_stats["percentile_clips"],
            "standardize_stats": clean_stats["standardize_stats"],
            "mahalanobis": {
                k: v for k, v in clean_stats["mahalanobis"].items() if not k.startswith("_")
            },
        }
        with open(report_path, "w", encoding="utf-8") as f:
            dump_json(json_report, f)

    return cleaned, report


__all__ = ["CleaningConfig", "filter_for_clustering", "CLUSTERING_FEATURES"]
