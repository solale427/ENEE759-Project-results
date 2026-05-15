from __future__ import annotations

from typing import Any

import numpy as np


TDBM_RAW_CLASSES = {
    0: "Aggressive",
    1: "Reckless",
    2: "Threatening",
    3: "Careful",
    4: "Cautious",
    5: "Timid",
}

# Three-way style taxonomy used by the current analysis workflow.
DRIVING_STYLE_CLASSES = {
    0: "Aggressive",
    1: "Normal",
    2: "Cautious",
}

# Backward-compatible alias for downstream imports that expect the main
# exported style labels to be the user-facing driving-style taxonomy.
TDBM_CLASSES = DRIVING_STYLE_CLASSES


def collapse_tdbm_raw_to_three(raw_style_id: int) -> int:
    if raw_style_id in {0, 1}:
        return 0
    if raw_style_id in {2, 3}:
        return 1
    return 2


def _masked_mean(values: np.ndarray, mask: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    masked_sum = np.sum(values * mask, axis=axis)
    masked_count = np.sum(mask, axis=axis)
    invalid = masked_count == 0
    masked_count = np.where(invalid, 1, masked_count)
    mean = masked_sum / masked_count
    if np.isscalar(mean):
        return 0.0 if invalid else mean
    mean = np.asarray(mean)
    mean[invalid] = 0.0
    return mean


def _linear_clip(values: np.ndarray, lower_percentile: float = 0.05, upper_percentile: float = 0.90) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return values
    low = np.quantile(finite, lower_percentile)
    high = np.quantile(finite, upper_percentile)
    return np.clip(values, low, high)


def compute_behavior_scores(
    s_center: np.ndarray,
    v_nei: np.ndarray,
    s_front: np.ndarray,
    v_avg: np.ndarray,
    j_l: np.ndarray,
) -> np.ndarray:
    features = np.stack([s_center, v_nei, s_front, v_avg, j_l, np.ones_like(v_avg)], axis=-1)
    weight_matrix = np.asarray(
        [
            [1.63, 4.04, -0.46, -0.82, 0.88, -2.58],
            [1.58, 3.08, -0.45, 0.02, -0.10, -1.67],
            [1.35, 4.08, -0.58, -0.43, -0.28, -1.99],
            [-1.51, -3.17, 1.06, 0.51, -0.51, 1.39],
            [-2.47, -2.60, 1.43, 0.98, -0.82, 1.27],
            [-3.59, -2.19, 1.75, 1.73, -0.30, 0.61],
        ],
        dtype=np.float32,
    )
    return features @ weight_matrix.T


def compute_style_metrics_from_scene(
    past_positions: np.ndarray,
    past_mask: np.ndarray,
    future_positions: np.ndarray | None = None,
    future_mask: np.ndarray | None = None,
    *,
    dt: float = 0.1,
) -> dict[str, Any]:
    if future_positions is not None and future_mask is not None:
        trajectories = np.concatenate([past_positions, future_positions], axis=1)
        valid = np.concatenate([past_mask, future_mask], axis=1).astype(bool)
    else:
        trajectories = np.asarray(past_positions)
        valid = np.asarray(past_mask).astype(bool)

    num_agents, num_steps, _ = trajectories.shape
    if num_steps < 4:
        zeros = np.zeros((num_agents,), dtype=np.float32)
        scores = compute_behavior_scores(zeros, zeros, zeros, zeros, zeros)
        return {
            "v_avg": zeros,
            "j_l": zeros,
            "s_center": zeros,
            "v_nei": zeros,
            "s_front": zeros,
            "behavior_scores": scores,
            "behavior_type": scores.argmax(axis=-1),
        }

    velocities = np.diff(trajectories, axis=1) / dt
    speed = np.linalg.norm(velocities, axis=-1)
    v_avg = _masked_mean(speed, valid[:, 1:], axis=1)

    accelerations = np.diff(velocities, axis=1) / dt
    jerks = np.diff(accelerations, axis=1) / dt

    init_velocity = velocities[:, 0, :]
    init_norm = np.linalg.norm(init_velocity, axis=-1, keepdims=True)
    safe_norm = np.where(init_norm > 1e-8, init_norm, 1.0)
    unit_dir = init_velocity / safe_norm
    proj_jerk = np.sum(jerks * unit_dir[:, None, :], axis=-1)
    j_l = _masked_mean(np.abs(proj_jerk), valid[:, 3:], axis=1)

    s_center = np.zeros((num_agents,), dtype=np.float32)

    speed1 = speed[:, None, :]
    speed2 = speed[None, :, :]
    rel_speed = speed1 - speed2
    pairwise_mask = valid[:, None, :] & valid[None, :, :]
    speed_mask = pairwise_mask[:, :, 1:]
    eye = np.eye(num_agents, dtype=bool)
    speed_mask[eye, :] = False
    v_nei = _masked_mean(rel_speed, speed_mask, axis=(1, 2))

    xs = trajectories[..., 0]
    gaps = xs[:, None, :] - xs[None, :, :]
    valid_gaps = (gaps > 0) & pairwise_mask
    gaps = np.where(valid_gaps, gaps, 100.0)
    min_gaps = np.min(gaps, axis=1)
    s_front = _masked_mean(min_gaps, valid, axis=1)

    metrics = {
        "v_avg": _linear_clip(np.asarray(v_avg, dtype=np.float32)),
        "j_l": _linear_clip(np.asarray(j_l, dtype=np.float32)),
        "s_center": _linear_clip(np.asarray(s_center, dtype=np.float32)),
        "v_nei": _linear_clip(np.asarray(v_nei, dtype=np.float32)),
        "s_front": _linear_clip(np.asarray(s_front, dtype=np.float32)),
    }
    behavior_scores = compute_behavior_scores(
        metrics["s_center"], metrics["v_nei"], metrics["s_front"], metrics["v_avg"], metrics["j_l"]
    )
    metrics["behavior_scores"] = behavior_scores
    metrics["behavior_type"] = behavior_scores.argmax(axis=-1)
    return metrics


def compute_center_style_row(
    past_positions: np.ndarray,
    past_mask: np.ndarray,
    future_positions: np.ndarray,
    future_mask: np.ndarray,
    center_index: int,
    *,
    dt: float = 0.1,
) -> dict[str, float | int | str]:
    metrics = compute_style_metrics_from_scene(
        past_positions,
        past_mask,
        future_positions=future_positions,
        future_mask=future_mask,
        dt=dt,
    )
    behavior_scores = np.asarray(metrics["behavior_scores"][center_index], dtype=np.float32)
    raw_behavior_type = int(metrics["behavior_type"][center_index])
    behavior_type = collapse_tdbm_raw_to_three(raw_behavior_type)
    row = {
        "tdbm_style_id": behavior_type,
        "tdbm_style_label": DRIVING_STYLE_CLASSES[behavior_type],
        "tdbm_raw_style_id": raw_behavior_type,
        "tdbm_raw_style_label": TDBM_RAW_CLASSES[raw_behavior_type],
        "tdbm_s_center": float(metrics["s_center"][center_index]),
        "tdbm_v_nei": float(metrics["v_nei"][center_index]),
        "tdbm_s_front": float(metrics["s_front"][center_index]),
        "tdbm_v_avg": float(metrics["v_avg"][center_index]),
        "tdbm_j_l": float(metrics["j_l"][center_index]),
    }
    for idx in range(behavior_scores.shape[0]):
        row[f"tdbm_behavior_score_{idx}"] = float(behavior_scores[idx])
    return row


def compute_center_tdbm_subcluster_features(
    center_past_positions: np.ndarray,
    center_past_mask: np.ndarray,
    center_gt_trajs: np.ndarray,
    center_gt_mask: np.ndarray,
    *,
    dt: float = 0.1,
) -> dict[str, float]:
    future_positions = center_gt_trajs[:, :2]
    future_mask = center_gt_mask.astype(bool)
    positions = np.concatenate([center_past_positions[:, :2], future_positions], axis=0)
    valid = np.concatenate([center_past_mask.astype(bool), future_mask], axis=0)
    valid_positions = positions[valid]
    if len(valid_positions) < 4:
        return {
            "tdbm_cluster_mean_acceleration": 0.0,
            "tdbm_cluster_var_acceleration": 0.0,
            "tdbm_cluster_mean_jerk": 0.0,
            "tdbm_cluster_var_jerk": 0.0,
        }

    velocity = np.linalg.norm(np.diff(valid_positions, axis=0) / dt, axis=-1)
    if len(velocity) < 3:
        acceleration = np.diff(velocity) / dt if len(velocity) >= 2 else np.zeros((0,), dtype=np.float32)
        jerk = np.zeros((0,), dtype=np.float32)
    else:
        acceleration = np.diff(velocity) / dt
        jerk = np.diff(acceleration) / dt if len(acceleration) >= 2 else np.zeros((0,), dtype=np.float32)

    return {
        "tdbm_cluster_mean_acceleration": float(np.mean(acceleration)) if len(acceleration) else 0.0,
        "tdbm_cluster_var_acceleration": float(np.var(acceleration)) if len(acceleration) else 0.0,
        "tdbm_cluster_mean_jerk": float(np.mean(jerk)) if len(jerk) else 0.0,
        "tdbm_cluster_var_jerk": float(np.var(jerk)) if len(jerk) else 0.0,
    }
