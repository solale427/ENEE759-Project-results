from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scenarionet.common_utils import read_dataset_summary, read_scenario


CATALOG = {
    "waymo": {
        "full_root": Path("/fs/nexus-projects/pc_driving/datasets/sn_womd"),
        "default_split": "validation",
    },
    "av2": {
        "full_root": Path("/fs/nexus-projects/pc_driving/datasets/argoverse2_sn"),
        "default_split": "val",
    },
}


def resolve_dataset_dir(dataset: str, split: str | None = None, dataset_root_override: str | None = None) -> Path:
    key = dataset.lower()
    if dataset_root_override:
        root = Path(dataset_root_override).expanduser().resolve()
        return root / (split or CATALOG[key]["default_split"])
    return CATALOG[key]["full_root"] / (split or CATALOG[key]["default_split"])


def _resolve_scenario_key(summary: dict, mapping: dict, scenario_id: str) -> tuple[str, dict]:
    if scenario_id in mapping and scenario_id in summary:
        return scenario_id, summary[scenario_id]
    if scenario_id in mapping:
        summary_entry = summary.get(scenario_id, {})
        return scenario_id, summary_entry

    for key, entry in summary.items():
        entry_scenario_id = str(entry.get("scenario_id", entry.get("id", "")))
        if entry_scenario_id == str(scenario_id):
            return key, entry

    raise KeyError(
        f"Scenario '{scenario_id}' was not found in dataset summary/mapping. "
        f"Use a matching split or dataset root override."
    )


def load_scenario_by_id(dataset: str, scenario_id: str, split: str | None = None, dataset_root_override: str | None = None) -> dict:
    dataset_dir = resolve_dataset_dir(dataset, split=split, dataset_root_override=dataset_root_override)
    summary, _, mapping = read_dataset_summary(str(dataset_dir))
    scenario_key, summary_entry = _resolve_scenario_key(summary, mapping, scenario_id)
    scenario = read_scenario(str(dataset_dir), mapping, scenario_key)
    return {
        "dataset_dir": dataset_dir,
        "scenario": scenario,
        "summary_entry": summary_entry,
        "scenario_key": scenario_key,
        "requested_scenario_id": scenario_id,
    }


def _normalize_track_id(track_id) -> str:
    return str(track_id)


def _target_track_ids(metadata: dict) -> set[str]:
    targets = set()
    for track_id, info in metadata.get("tracks_to_predict", {}).items():
        targets.add(_normalize_track_id(info.get("track_id", track_id)))
    return targets


def _interest_track_ids(metadata: dict) -> set[str]:
    return {_normalize_track_id(track_id) for track_id in metadata.get("objects_of_interest", [])}


def _valid_segments(positions: np.ndarray, valid_mask: np.ndarray) -> list[np.ndarray]:
    positions = np.asarray(positions)[..., :2]
    valid_mask = np.asarray(valid_mask).astype(bool).reshape(-1)
    segments = []
    start = None
    for idx, is_valid in enumerate(valid_mask):
        if is_valid and start is None:
            start = idx
        elif not is_valid and start is not None:
            if idx - start >= 2:
                segments.append(positions[start:idx])
            start = None
    if start is not None and len(valid_mask) - start >= 2:
        segments.append(positions[start:])
    return segments


def _scenario_center(scenario: dict, *, center_on_sdc: bool = True) -> np.ndarray:
    metadata = scenario["metadata"]
    current_idx = int(metadata.get("current_time_index", 0))
    tracks = scenario["tracks"]
    if center_on_sdc:
        sdc_id = metadata.get("sdc_id")
        if sdc_id in tracks:
            state = tracks[sdc_id]["state"]
            valid = np.asarray(state["valid"]).reshape(-1).astype(bool)
            if current_idx < len(valid) and valid[current_idx]:
                return np.asarray(state["position"])[current_idx, :2]

    positions = []
    for track in tracks.values():
        state = track["state"]
        valid = np.asarray(state["valid"]).reshape(-1).astype(bool)
        if current_idx < len(valid) and valid[current_idx]:
            positions.append(np.asarray(state["position"])[current_idx, :2])
    if positions:
        return np.mean(np.stack(positions, axis=0), axis=0)
    return np.zeros((2,), dtype=np.float32)


def _map_style(map_type: str) -> dict[str, object]:
    map_type = str(map_type).upper()
    if "CROSSWALK" in map_type:
        return {"color": "#d8b365", "linewidth": 1.2}
    if "STOP_SIGN" in map_type or "SPEED_BUMP" in map_type:
        return {"color": "#b15928", "linewidth": 1.2}
    if "ROAD_EDGE" in map_type or "BOUNDARY" in map_type or "MEDIAN" in map_type:
        return {"color": "#7f7f7f", "linewidth": 1.0}
    if "LANE" in map_type or "ROAD_LINE" in map_type:
        return {"color": "#c7c7c7", "linewidth": 0.9}
    return {"color": "#dddddd", "linewidth": 0.8}


def _track_color(track_type: str, role: str) -> str:
    if role == "sdc":
        return "#ff7f0e"
    if role == "target":
        return "#d62728"
    if role == "interest":
        return "#9467bd"
    track_type = str(track_type).upper()
    if "PEDESTRIAN" in track_type:
        return "#2ca02c"
    if "CYCLIST" in track_type or "MOTOR" in track_type:
        return "#17becf"
    if "VEHICLE" in track_type:
        return "#1f77b4"
    return "#7f7f7f"


def plot_scenario(
    scenario: dict,
    *,
    dataset: str,
    view_radius: float = 90.0,
    center_on_sdc: bool = True,
    show_track_ids: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    metadata = scenario["metadata"]
    current_idx = int(metadata.get("current_time_index", 0))
    targets = _target_track_ids(metadata)
    interests = _interest_track_ids(metadata)
    sdc_id = _normalize_track_id(metadata.get("sdc_id"))
    center_xy = _scenario_center(scenario, center_on_sdc=center_on_sdc)

    fig, ax = plt.subplots(figsize=(10, 10))
    for feature in scenario["map_features"].values():
        polyline = np.asarray(feature.get("polyline", []))
        if polyline.ndim != 2 or len(polyline) < 2:
            continue
        style = _map_style(feature.get("type", "UNKNOWN"))
        ax.plot(polyline[:, 0], polyline[:, 1], color=style["color"], linewidth=style["linewidth"], zorder=1)

    for track_id, track in scenario["tracks"].items():
        state = track["state"]
        positions = np.asarray(state["position"])
        valid = np.asarray(state["valid"]).reshape(-1).astype(bool)
        norm_id = _normalize_track_id(track_id)
        if norm_id == sdc_id:
            role = "sdc"
        elif norm_id in targets:
            role = "target"
        elif norm_id in interests:
            role = "interest"
        else:
            role = "other"
        color = _track_color(track.get("type", "UNKNOWN"), role)

        history_mask = valid.copy()
        history_mask[current_idx + 1 :] = False
        future_mask = valid.copy()
        future_mask[:current_idx] = False

        line_width = 2.6 if role in {"sdc", "target"} else 1.8
        for segment in _valid_segments(positions, history_mask):
            ax.plot(segment[:, 0], segment[:, 1], color=color, linewidth=line_width, zorder=3)
        for segment in _valid_segments(positions, future_mask):
            ax.plot(segment[:, 0], segment[:, 1], color=color, linewidth=max(1.0, line_width - 0.4), linestyle="--", zorder=4)

        if current_idx < len(valid) and valid[current_idx]:
            xy = positions[current_idx, :2]
            ax.scatter(xy[0], xy[1], color=color, s=36 if role in {"sdc", "target"} else 20, zorder=5)
            if show_track_ids:
                ax.text(xy[0], xy[1], norm_id, fontsize=7, color=color, zorder=6)

    ax.set_aspect("equal")
    ax.set_xlim(center_xy[0] - view_radius, center_xy[0] + view_radius)
    ax.set_ylim(center_xy[1] - view_radius, center_xy[1] + view_radius)
    ax.axis("off")
    ax.legend(
        handles=[
            Line2D([0], [0], color="#ff7f0e", lw=2.6, label="SDC"),
            Line2D([0], [0], color="#d62728", lw=2.6, label="Tracks to predict"),
            Line2D([0], [0], color="#9467bd", lw=2.0, label="Objects of interest"),
            Line2D([0], [0], color="#1f77b4", lw=1.8, label="Other vehicles"),
            Line2D([0], [0], color="#2ca02c", lw=1.8, label="Pedestrians"),
            Line2D([0], [0], color="#17becf", lw=1.8, label="Cyclists / motorcyclists"),
            Line2D([0], [0], color="black", lw=1.2, linestyle="--", label="Future segment"),
        ],
        loc="upper right",
        frameon=False,
    )
    ax.set_title(f"{dataset.upper()} | {metadata.get('scenario_id', scenario.get('id', 'unknown'))}")
    return fig, ax
