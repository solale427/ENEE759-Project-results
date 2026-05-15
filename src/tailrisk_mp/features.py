from __future__ import annotations

from typing import Any

import numpy as np

from tailrisk_mp.metrics import prediction_metrics_from_output


DT = 0.1
MISS_THRESHOLD_METERS = 2.0

HARD_BRAKE_LONG_ACCEL = -3.0
HARD_ACCEL_LONG_ACCEL = 3.0
LATERAL_G_SPIKE_ACCEL = 3.0
HIGH_JERK_MAG = 4.0
CLOSE_ENCOUNTER_RADIUS = 5.0
STATIONARY_PATH_LENGTH = 2.0
STATIONARY_AVG_SPEED = 0.5
STATIONARY_DISPLACEMENT = 1.0
MIN_VALID_FUTURE_STEPS = 30


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


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _last_valid_index(mask: np.ndarray) -> int | None:
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return None
    return int(indices[-1])


def _first_two_valid_positions(positions: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    valid_idx = np.flatnonzero(mask)
    if len(valid_idx) < 2:
        return None
    return positions[valid_idx[-2]], positions[valid_idx[-1]]


def _future_valid_positions(center_gt_trajs: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid_idx = np.flatnonzero(mask)
    if len(valid_idx) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
    return center_gt_trajs[valid_idx, :2], center_gt_trajs[valid_idx, 2:4]


def _aligned_motion_directions(positions: np.ndarray, velocities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dirs = np.zeros_like(velocities, dtype=np.float32)
    valid = np.zeros(len(velocities), dtype=bool)

    velocity_norm = np.linalg.norm(velocities, axis=-1)
    velocity_valid = velocity_norm > 1e-3
    if np.any(velocity_valid):
        dirs[velocity_valid] = velocities[velocity_valid] / velocity_norm[velocity_valid, None]
        valid[velocity_valid] = True

    if len(positions) >= 2:
        pos_steps = np.diff(positions, axis=0)
        pos_norm = np.linalg.norm(pos_steps, axis=-1)
        pos_valid = pos_norm > 1e-3
        if np.any(pos_valid):
            step_dirs = pos_steps[pos_valid] / pos_norm[pos_valid, None]
            padded_dirs = np.zeros_like(velocities, dtype=np.float32)
            padded_valid = np.zeros(len(velocities), dtype=bool)
            padded_dirs[1:][pos_valid] = step_dirs
            padded_valid[1:] = pos_valid

            first_valid = np.flatnonzero(padded_valid)
            if len(first_valid):
                padded_dirs[0] = padded_dirs[first_valid[0]]
                padded_valid[0] = True

            missing = ~valid & padded_valid
            dirs[missing] = padded_dirs[missing]
            valid[missing] = True

    return dirs, valid


def compute_center_kinematics(center_gt_trajs: np.ndarray, center_gt_mask: np.ndarray) -> dict[str, float]:
    positions, velocities = _future_valid_positions(center_gt_trajs, center_gt_mask)
    n_valid_future_steps = int(np.count_nonzero(center_gt_mask))
    track_duration_s = float(n_valid_future_steps * DT)
    if len(positions) == 0:
        return {
            "avg_speed": np.nan,
            "max_speed": np.nan,
            "var_speed": np.nan,
            "avg_accel": np.nan,
            "max_accel": np.nan,
            "max_abs_accel": np.nan,
            "var_acceleration": np.nan,
            "avg_jerk": np.nan,
            "max_abs_jerk": np.nan,
            "avg_long_accel": np.nan,
            "max_abs_long_accel": np.nan,
            "avg_lat_accel": np.nan,
            "max_abs_lat_accel": np.nan,
            "avg_long_jerk": np.nan,
            "max_abs_long_jerk": np.nan,
            "avg_lat_jerk": np.nan,
            "max_abs_lat_jerk": np.nan,
            "heading_change_signed": np.nan,
            "heading_change_abs": np.nan,
            "max_abs_heading_delta": np.nan,
            "max_abs_heading_rate": np.nan,
            "displacement": np.nan,
            "path_length": np.nan,
            "curvature_proxy": np.nan,
            "curvature_signed": np.nan,
            "gamma": np.nan,
            "hard_brake_count": 0,
            "hard_accel_count": 0,
            "lateral_g_spike_count": 0,
            "high_jerk_count": 0,
            "heading_rate_p95": np.nan,
            "n_valid_future_steps": n_valid_future_steps,
            "track_duration_s": track_duration_s,
            "is_stationary": False,
        }

    speeds = np.linalg.norm(velocities, axis=-1)
    accel = np.diff(velocities, axis=0) / DT if len(velocities) >= 2 else np.zeros((0, 2), dtype=np.float32)
    accel_mag = np.linalg.norm(accel, axis=-1) if len(accel) else np.zeros((0,), dtype=np.float32)
    jerk = np.diff(accel, axis=0) / DT if len(accel) >= 2 else np.zeros((0, 2), dtype=np.float32)
    jerk_mag = np.linalg.norm(jerk, axis=-1) if len(jerk) else np.zeros((0,), dtype=np.float32)

    dirs, valid_dir = _aligned_motion_directions(positions, velocities)
    headings = np.arctan2(dirs[:, 1], dirs[:, 0]) if len(dirs) else np.zeros((0,))
    valid_heading = headings[valid_dir]
    heading_delta = _wrap_angle(np.diff(valid_heading)) if len(valid_heading) >= 2 else np.zeros((0,), dtype=np.float32)
    heading_rate = heading_delta / DT if len(heading_delta) else np.zeros((0,), dtype=np.float32)
    heading_change_signed = float(np.sum(heading_delta)) if len(heading_delta) else 0.0
    heading_change_abs = float(np.sum(np.abs(_wrap_angle(np.diff(valid_heading))))) if len(valid_heading) >= 2 else 0.0

    path_length = float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=-1))) if len(positions) >= 2 else 0.0
    displacement = float(np.linalg.norm(positions[-1] - positions[0])) if len(positions) >= 1 else 0.0

    if len(accel):
        accel_dirs = dirs[1 : 1 + len(accel)]
        accel_dir_valid = valid_dir[1 : 1 + len(accel)]
        accel_perp_dirs = np.stack([-accel_dirs[:, 1], accel_dirs[:, 0]], axis=-1)
        long_accel = np.sum(accel * accel_dirs, axis=-1)
        lat_accel = np.sum(accel * accel_perp_dirs, axis=-1)
        long_accel = np.where(accel_dir_valid, long_accel, 0.0)
        lat_accel = np.where(accel_dir_valid, lat_accel, 0.0)
    else:
        long_accel = np.zeros((0,), dtype=np.float32)
        lat_accel = np.zeros((0,), dtype=np.float32)

    if len(jerk):
        jerk_dirs = dirs[2 : 2 + len(jerk)]
        jerk_dir_valid = valid_dir[2 : 2 + len(jerk)]
        jerk_perp_dirs = np.stack([-jerk_dirs[:, 1], jerk_dirs[:, 0]], axis=-1)
        long_jerk = np.sum(jerk * jerk_dirs, axis=-1)
        lat_jerk = np.sum(jerk * jerk_perp_dirs, axis=-1)
        long_jerk = np.where(jerk_dir_valid, long_jerk, 0.0)
        lat_jerk = np.where(jerk_dir_valid, lat_jerk, 0.0)
    else:
        long_jerk = np.zeros((0,), dtype=np.float32)
        lat_jerk = np.zeros((0,), dtype=np.float32)

    mean_jerk = float(np.mean(jerk_mag)) if len(jerk_mag) else 0.0
    var_jerk = float(np.var(jerk_mag)) if len(jerk_mag) else 0.0
    gamma = 0.0 if abs(mean_jerk) < 1e-6 else var_jerk / mean_jerk

    hard_brake_count = int(np.sum(long_accel < HARD_BRAKE_LONG_ACCEL)) if len(long_accel) else 0
    hard_accel_count = int(np.sum(long_accel > HARD_ACCEL_LONG_ACCEL)) if len(long_accel) else 0
    lateral_g_spike_count = int(np.sum(np.abs(lat_accel) > LATERAL_G_SPIKE_ACCEL)) if len(lat_accel) else 0
    high_jerk_count = int(np.sum(jerk_mag > HIGH_JERK_MAG)) if len(jerk_mag) else 0
    heading_rate_p95 = (
        float(np.percentile(np.abs(heading_rate), 95)) if len(heading_rate) else 0.0
    )

    avg_speed_value = float(np.mean(speeds)) if len(speeds) else 0.0
    is_stationary = bool(
        path_length < STATIONARY_PATH_LENGTH
        and avg_speed_value < STATIONARY_AVG_SPEED
        and displacement < STATIONARY_DISPLACEMENT
    )

    return {
        "avg_speed": float(np.mean(speeds)) if len(speeds) else 0.0,
        "max_speed": float(np.max(speeds)) if len(speeds) else 0.0,
        "var_speed": float(np.var(speeds)) if len(speeds) else 0.0,
        "avg_accel": float(np.mean(accel_mag)) if len(accel_mag) else 0.0,
        "max_accel": float(np.max(accel_mag)) if len(accel_mag) else 0.0,
        "max_abs_accel": float(np.max(np.abs(accel_mag))) if len(accel_mag) else 0.0,
        "var_acceleration": float(np.var(accel_mag)) if len(accel_mag) else 0.0,
        "avg_jerk": mean_jerk,
        "max_abs_jerk": float(np.max(np.abs(jerk_mag))) if len(jerk_mag) else 0.0,
        "avg_long_accel": float(np.mean(long_accel)) if len(long_accel) else 0.0,
        "max_abs_long_accel": float(np.max(np.abs(long_accel))) if len(long_accel) else 0.0,
        "avg_lat_accel": float(np.mean(lat_accel)) if len(lat_accel) else 0.0,
        "max_abs_lat_accel": float(np.max(np.abs(lat_accel))) if len(lat_accel) else 0.0,
        "avg_long_jerk": float(np.mean(long_jerk)) if len(long_jerk) else 0.0,
        "max_abs_long_jerk": float(np.max(np.abs(long_jerk))) if len(long_jerk) else 0.0,
        "avg_lat_jerk": float(np.mean(lat_jerk)) if len(lat_jerk) else 0.0,
        "max_abs_lat_jerk": float(np.max(np.abs(lat_jerk))) if len(lat_jerk) else 0.0,
        "heading_change_signed": heading_change_signed,
        "heading_change_abs": heading_change_abs,
        "max_abs_heading_delta": float(np.max(np.abs(heading_delta))) if len(heading_delta) else 0.0,
        "max_abs_heading_rate": float(np.max(np.abs(heading_rate))) if len(heading_rate) else 0.0,
        "displacement": displacement,
        "path_length": path_length,
        "curvature_proxy": heading_change_abs / max(path_length, 1e-3),
        "curvature_signed": heading_change_signed / max(path_length, 1e-3),
        "gamma": gamma,
        "hard_brake_count": hard_brake_count,
        "hard_accel_count": hard_accel_count,
        "lateral_g_spike_count": lateral_g_spike_count,
        "high_jerk_count": high_jerk_count,
        "heading_rate_p95": heading_rate_p95,
        "n_valid_future_steps": n_valid_future_steps,
        "track_duration_s": track_duration_s,
        "is_stationary": is_stationary,
    }


def _close_encounter_count(
    obj_trajs_pos: np.ndarray,
    obj_trajs_mask: np.ndarray,
    center_index: int,
    *,
    radius: float = CLOSE_ENCOUNTER_RADIUS,
) -> int:
    pos = _to_numpy(obj_trajs_pos)[..., :2]
    mask = _to_numpy(obj_trajs_mask).astype(bool)
    if pos.ndim != 3 or mask.ndim != 2 or pos.shape[0] == 0:
        return 0

    num_objects, num_steps, _ = pos.shape
    if not (0 <= center_index < num_objects):
        return 0

    center_pos = pos[center_index]  # (num_steps, 2)
    center_mask = mask[center_index]

    neighbor_pos = np.delete(pos, center_index, axis=0)
    neighbor_mask = np.delete(mask, center_index, axis=0)
    if neighbor_pos.shape[0] == 0:
        return 0

    dist = np.linalg.norm(neighbor_pos - center_pos[None, :, :], axis=-1)
    valid = neighbor_mask & center_mask[None, :]
    close = valid & (dist < radius)
    step_has_close = np.any(close, axis=0)
    return int(np.count_nonzero(step_has_close))


def compute_social_features(
    obj_trajs_pos: np.ndarray,
    obj_trajs_mask: np.ndarray,
    obj_trajs: np.ndarray,
    center_index: int,
    *,
    neighbor_radius: float = 30.0,
) -> dict[str, float]:
    last_pos = _to_numpy(obj_trajs_pos)[:, -1, :2]
    current_valid = _to_numpy(obj_trajs_mask)[:, -1].astype(bool)
    obj_trajs = _to_numpy(obj_trajs)
    close_encounters = _close_encounter_count(obj_trajs_pos, obj_trajs_mask, center_index)

    center_pos = last_pos[center_index]
    center_vel = obj_trajs[center_index, -1, 25:27]

    neighbor_indices = np.flatnonzero(current_valid)
    neighbor_indices = neighbor_indices[neighbor_indices != center_index]
    if len(neighbor_indices) == 0:
        return {
            "neighbor_count_30m": 0.0,
            "min_neighbor_distance": np.inf,
            "inverse_min_neighbor_distance": 0.0,
            "max_closing_speed": 0.0,
            "min_ttc": np.inf,
            "inverse_min_ttc": 0.0,
            "close_encounter_count": close_encounters,
        }

    neighbor_pos = last_pos[neighbor_indices]
    rel_pos = neighbor_pos - center_pos
    distances = np.linalg.norm(rel_pos, axis=-1)
    in_radius = distances <= neighbor_radius
    if not np.any(in_radius):
        return {
            "neighbor_count_30m": 0.0,
            "min_neighbor_distance": float(np.min(distances)),
            "inverse_min_neighbor_distance": 1.0 / max(float(np.min(distances)), 1e-3),
            "max_closing_speed": 0.0,
            "min_ttc": np.inf,
            "inverse_min_ttc": 0.0,
            "close_encounter_count": close_encounters,
        }

    neighbor_indices = neighbor_indices[in_radius]
    rel_pos = rel_pos[in_radius]
    distances = distances[in_radius]
    neighbor_vel = obj_trajs[neighbor_indices, -1, 25:27]
    rel_vel = neighbor_vel - center_vel[None, :]
    closing_speed = -np.sum(rel_pos * rel_vel, axis=-1) / np.maximum(distances, 1e-3)
    closing_speed = np.maximum(closing_speed, 0.0)
    positive_ttc = distances / np.maximum(closing_speed, 1e-3)
    positive_ttc = positive_ttc[closing_speed > 1e-3]
    min_ttc = float(np.min(positive_ttc)) if len(positive_ttc) else np.inf

    min_distance = float(np.min(distances))
    return {
        "neighbor_count_30m": float(len(neighbor_indices)),
        "min_neighbor_distance": min_distance,
        "inverse_min_neighbor_distance": 1.0 / max(min_distance, 1e-3),
        "max_closing_speed": float(np.max(closing_speed)) if len(closing_speed) else 0.0,
        "min_ttc": min_ttc,
        "inverse_min_ttc": 0.0 if not np.isfinite(min_ttc) else 1.0 / max(min_ttc, 1e-3),
        "close_encounter_count": close_encounters,
    }


def compute_constant_velocity_error(
    center_past_positions: np.ndarray,
    center_past_mask: np.ndarray,
    center_gt_trajs: np.ndarray,
    center_gt_mask: np.ndarray,
) -> dict[str, float]:
    past = _first_two_valid_positions(center_past_positions[:, :2], center_past_mask)
    valid_idx = np.flatnonzero(center_gt_mask)
    if past is None or len(valid_idx) == 0:
        return {"cv_ade": np.nan, "cv_fde": np.nan, "cv_miss_rate": np.nan}

    prev_pos, last_pos = past
    velocity = (last_pos - prev_pos) / DT
    steps = np.arange(1, len(center_gt_mask) + 1, dtype=np.float32)[:, None]
    pred = last_pos[None, :] + velocity[None, :] * DT * steps

    gt = center_gt_trajs[:, :2]
    errors = np.linalg.norm(pred - gt, axis=-1)
    valid_errors = errors[valid_idx]
    final_idx = int(valid_idx[-1])
    fde = float(errors[final_idx])
    return {
        "cv_ade": float(np.mean(valid_errors)),
        "cv_fde": fde,
        "cv_miss_rate": float(fde > MISS_THRESHOLD_METERS),
    }


def extract_batch_rows(batch: dict[str, Any], *, dataset: str, split: str) -> list[dict[str, Any]]:
    inputs = batch["input_dict"]
    scenario_ids = [_normalize_string(value) for value in inputs["scenario_id"]]
    center_ids = [_normalize_string(value) for value in inputs["center_objects_id"]]
    center_types = _to_numpy(inputs["center_objects_type"])
    trajectory_types = _to_numpy(inputs["trajectory_type"])
    kalman = _to_numpy(inputs["kalman_difficulty"])
    center_gt_trajs = _to_numpy(inputs["center_gt_trajs"])
    center_gt_mask = _to_numpy(inputs["center_gt_trajs_mask"]).astype(bool)
    obj_trajs = _to_numpy(inputs["obj_trajs"])
    obj_trajs_mask = _to_numpy(inputs["obj_trajs_mask"]).astype(bool)
    obj_trajs_pos = _to_numpy(inputs["obj_trajs_pos"])
    obj_trajs_future_state = _to_numpy(inputs["obj_trajs_future_state"])
    obj_trajs_future_mask = _to_numpy(inputs["obj_trajs_future_mask"]).astype(bool)
    center_indices = _to_numpy(inputs["track_index_to_predict"]).astype(int)

    rows: list[dict[str, Any]] = []
    for idx, scenario_id in enumerate(scenario_ids):
        center_index = int(center_indices[idx])
        center_past_positions = obj_trajs_pos[idx, center_index]
        center_past_mask = obj_trajs_mask[idx, center_index]

        row = {
            "dataset": dataset,
            "split": split,
            "scenario_id": scenario_id,
            "center_objects_id": center_ids[idx],
            "center_objects_type": int(center_types[idx]),
            "trajectory_type": int(trajectory_types[idx]),
            "kalman_difficulty_2s": float(kalman[idx][0]),
            "kalman_difficulty_4s": float(kalman[idx][1]),
            "kalman_difficulty_6s": float(kalman[idx][2]),
        }
        row.update(compute_center_kinematics(center_gt_trajs[idx], center_gt_mask[idx]))
        row.update(compute_social_features(obj_trajs_pos[idx], obj_trajs_mask[idx], obj_trajs[idx], center_index))
        row.update(compute_constant_velocity_error(center_past_positions, center_past_mask, center_gt_trajs[idx], center_gt_mask[idx]))
        rows.append(row)

    return rows
