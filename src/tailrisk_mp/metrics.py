from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


MISS_THRESHOLD_METERS = 2.0
NUM_FRAMES_TO_EVAL = 60


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    try:
        import torch

        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(value)


def _normalize_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def sort_and_normalize_modes(
    pred_scores: np.ndarray,
    pred_trajs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sort_idxs = np.asarray(pred_scores).argsort()[::-1]
    sorted_scores = np.asarray(pred_scores, dtype=np.float64)[sort_idxs]
    sorted_trajs = np.asarray(pred_trajs, dtype=np.float64)[sort_idxs]
    score_sum = float(sorted_scores.sum())
    if score_sum > 0.0:
        sorted_scores = sorted_scores / score_sum
    elif len(sorted_scores):
        sorted_scores = np.full(len(sorted_scores), 1.0 / len(sorted_scores), dtype=np.float64)
    return sorted_scores, sorted_trajs, sort_idxs


def build_av2_pred_dicts(batch: dict[str, Any], prediction: dict[str, Any]) -> list[dict[str, Any]]:
    import unitraj.datasets.common_utils as common_utils

    inputs = batch["input_dict"]
    pred_scores = prediction["predicted_probability"]
    pred_trajs = prediction["predicted_trajectory"]
    center_objects_world = inputs["center_objects_world"].type_as(pred_trajs)

    num_center_objects, num_modes, num_timestamps, num_feat = pred_trajs.shape
    pred_trajs_world = common_utils.rotate_points_along_z_tensor(
        points=pred_trajs.reshape(num_center_objects, num_modes * num_timestamps, num_feat),
        angle=center_objects_world[:, 6].reshape(num_center_objects),
    ).reshape(num_center_objects, num_modes, num_timestamps, num_feat)
    pred_trajs_world[:, :, :, 0:2] += center_objects_world[:, None, None, 0:2] + inputs["map_center"][:, None, None, 0:2]

    pred_dicts: list[dict[str, Any]] = []
    for bs_idx in range(batch["batch_size"]):
        pred_dicts.append(
            {
                "scenario_id": _normalize_string(inputs["scenario_id"][bs_idx]),
                "pred_trajs": pred_trajs_world[bs_idx, :, :, 0:2].detach().cpu().numpy(),
                "pred_scores": pred_scores[bs_idx, :].detach().cpu().numpy(),
                "object_id": _normalize_string(inputs["center_objects_id"][bs_idx]),
                "object_type": int(_to_numpy(inputs["center_objects_type"][bs_idx])),
                "gt_trajs": _to_numpy(inputs["center_gt_trajs_src"][bs_idx]),
                "track_index_to_predict": int(_to_numpy(inputs["track_index_to_predict"][bs_idx])),
            }
        )
    return pred_dicts


def compute_av2_sample_metrics_from_pred_dict(pred_dict: dict[str, Any]) -> dict[str, Any]:
    from av2.datasets.motion_forecasting.eval.metrics import (
        compute_ade,
        compute_brier_ade,
        compute_brier_fde,
        compute_fde,
        compute_is_missed_prediction,
    )

    pred_scores, pred_trajs, _ = sort_and_normalize_modes(pred_dict["pred_scores"], pred_dict["pred_trajs"])
    gt_src = _to_numpy(pred_dict["gt_trajs"])
    gt_positions = gt_src[-NUM_FRAMES_TO_EVAL:, :2]
    valid_mask = gt_src[-NUM_FRAMES_TO_EVAL:, -1].astype(bool)

    base_row = {
        "scenario_id": _normalize_string(pred_dict["scenario_id"]),
        "center_objects_id": _normalize_string(pred_dict["object_id"]),
        "mtr_minade6": np.nan,
        "mtr_minfde6": np.nan,
        "mtr_brier_minade6": np.nan,
        "mtr_brier_minfde6": np.nan,
        "mtr_miss_rate": np.nan,
        "mtr_meanfde6_weighted": np.nan,
        "mtr_meanfde6_unweighted": np.nan,
        "mtr_best_mode_prob": np.nan,
        "mtr_best_fde_mode_idx": np.nan,
    }
    if not np.any(valid_mask):
        return base_row

    pred_valid = pred_trajs[:, valid_mask, :]
    gt_valid = gt_positions[valid_mask]

    ade = compute_ade(pred_valid, gt_valid)
    fde = compute_fde(pred_valid, gt_valid)
    missed = compute_is_missed_prediction(pred_valid, gt_valid, miss_threshold_m=MISS_THRESHOLD_METERS)
    brier_ade = compute_brier_ade(pred_valid, gt_valid, pred_scores)
    brier_fde = compute_brier_fde(pred_valid, gt_valid, pred_scores)

    best_ade_idx = int(np.argmin(ade))
    best_fde_idx = int(np.argmin(fde))

    base_row.update(
        {
            "mtr_minade6": float(ade[best_ade_idx]),
            "mtr_minfde6": float(fde[best_fde_idx]),
            "mtr_brier_minade6": float(brier_ade[best_ade_idx]),
            "mtr_brier_minfde6": float(brier_fde[best_fde_idx]),
            "mtr_miss_rate": float(missed[best_fde_idx]),
            "mtr_meanfde6_weighted": float(np.sum(fde * pred_scores)),
            "mtr_meanfde6_unweighted": float(np.mean(fde)),
            "mtr_best_mode_prob": float(pred_scores[best_fde_idx]),
            "mtr_best_fde_mode_idx": best_fde_idx,
        }
    )
    return base_row


def prediction_metrics_from_pred_dicts(pred_dicts: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [compute_av2_sample_metrics_from_pred_dict(pred_dict) for pred_dict in pred_dicts]
    return pd.DataFrame(rows)


def prediction_metrics_from_output(batch: dict[str, Any], prediction: dict[str, Any]) -> pd.DataFrame:
    return prediction_metrics_from_pred_dicts(build_av2_pred_dicts(batch, prediction))


def aggregate_per_sample_metrics(metrics_df: pd.DataFrame) -> dict[str, float]:
    metric_map = {
        "min_ADE": "mtr_minade6",
        "min_FDE": "mtr_minfde6",
        "brier_min_ADE": "mtr_brier_minade6",
        "brier_min_FDE": "mtr_brier_minfde6",
        "miss_rate": "mtr_miss_rate",
        "mean_FDE_weighted": "mtr_meanfde6_weighted",
        "mean_FDE_unweighted": "mtr_meanfde6_unweighted",
    }
    result: dict[str, float] = {}
    for official_name, column in metric_map.items():
        if column not in metrics_df.columns:
            continue
        values = pd.to_numeric(metrics_df[column], errors="coerce")
        result[official_name] = float(values.mean()) if values.notna().any() else float("nan")
    return result


def official_av2_metrics_from_pred_dicts(pred_dicts: list[dict[str, Any]], *, num_modes_for_eval: int = 6) -> dict[str, float]:
    from unitraj.models.base_model.av2_eval import argoverse2_evaluation

    return argoverse2_evaluation(pred_dicts=pred_dicts, num_modes_for_eval=num_modes_for_eval)


def compare_per_sample_to_official(
    per_sample_aggregate: dict[str, float],
    official_metrics: dict[str, float],
    *,
    matched_tol: float = 1e-5,
    close_tol: float = 1e-3,
) -> dict[str, Any]:
    compared_keys = ["min_ADE", "min_FDE", "brier_min_FDE", "miss_rate"]
    rows = []
    max_abs_diff = 0.0
    for key in compared_keys:
        per_value = float(per_sample_aggregate.get(key, float("nan")))
        off_value = float(official_metrics.get(key, float("nan")))
        abs_diff = abs(per_value - off_value) if np.isfinite(per_value) and np.isfinite(off_value) else float("inf")
        max_abs_diff = max(max_abs_diff, abs_diff if np.isfinite(abs_diff) else max_abs_diff)
        rows.append(
            {
                "metric": key,
                "per_sample_mean": per_value,
                "official": off_value,
                "abs_diff": abs_diff,
            }
        )

    status = "mismatched"
    if rows and all(np.isfinite(row["abs_diff"]) and row["abs_diff"] <= matched_tol for row in rows):
        status = "matched"
    elif rows and all(np.isfinite(row["abs_diff"]) and row["abs_diff"] <= close_tol for row in rows):
        status = "close but investigate"

    return {
        "status": status,
        "matched_tol": matched_tol,
        "close_tol": close_tol,
        "max_abs_diff": max_abs_diff,
        "comparisons": rows,
    }
