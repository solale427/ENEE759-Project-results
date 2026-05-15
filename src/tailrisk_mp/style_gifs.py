"""Render per-agent GIFs for driving-style clusters.

Thin wrapper around :mod:`tailrisk_mp.scenario_viz` that renders a time-lapse
GIF focused on a single center agent (``scenario_id`` + ``center_objects_id``),
coloured by its cluster assignment.

Used by ``notebooks/phase1_style_gifs.ipynb`` for a qualitative check that
the ``aggressive`` K-means cluster actually looks aggressive.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from tailrisk_mp.scenario_viz import (
    _map_style,
    _normalize_track_id,
    _scenario_center,
    _target_track_ids,
    _track_color,
    _valid_segments,
    load_scenario_by_id,
)


def _frame_center_xy(
    scenario: dict, center_track_id: str, frame: int, fallback: np.ndarray
) -> np.ndarray:
    """Return the agent's XY at ``frame`` if valid; otherwise the fallback."""
    tracks = scenario["tracks"]
    track = tracks.get(center_track_id)
    if track is None:
        for tid, t in tracks.items():
            if _normalize_track_id(tid) == str(center_track_id):
                track = t
                break
    if track is None:
        return fallback
    pos = np.asarray(track["state"]["position"])
    valid = np.asarray(track["state"]["valid"]).reshape(-1).astype(bool)
    if frame < len(valid) and valid[frame]:
        return pos[frame, :2]
    return fallback


def _draw_frame(
    ax: plt.Axes,
    scenario: dict,
    frame: int,
    center_track_id: str,
    view_radius: float,
) -> None:
    ax.clear()
    metadata = scenario["metadata"]
    targets = _target_track_ids(metadata)
    interests = {_normalize_track_id(x) for x in metadata.get("objects_of_interest", [])}
    sdc_id = _normalize_track_id(metadata.get("sdc_id"))
    center_track_norm = _normalize_track_id(center_track_id)

    for feature in scenario["map_features"].values():
        poly = np.asarray(feature.get("polyline", []))
        if poly.ndim != 2 or len(poly) < 2:
            continue
        style = _map_style(feature.get("type", "UNKNOWN"))
        ax.plot(poly[:, 0], poly[:, 1], color=style["color"], linewidth=style["linewidth"], zorder=1)

    fallback = _scenario_center(scenario)
    center_xy = _frame_center_xy(scenario, center_track_id, frame, fallback)

    for track_id, track in scenario["tracks"].items():
        state = track["state"]
        positions = np.asarray(state["position"])
        valid = np.asarray(state["valid"]).reshape(-1).astype(bool)
        norm_id = _normalize_track_id(track_id)
        is_center = norm_id == center_track_norm
        if is_center:
            role = "target"
        elif norm_id == sdc_id:
            role = "sdc"
        elif norm_id in targets:
            role = "target"
        elif norm_id in interests:
            role = "interest"
        else:
            role = "other"
        color = "#d62728" if is_center else _track_color(track.get("type", "UNKNOWN"), role)
        line_width = 3.0 if is_center else (2.2 if role in {"sdc", "target"} else 1.4)

        history = valid.copy(); history[frame + 1 :] = False
        future = valid.copy(); future[: frame + 1] = False

        for segment in _valid_segments(positions, history):
            ax.plot(segment[:, 0], segment[:, 1], color=color, linewidth=line_width, zorder=3)
        for segment in _valid_segments(positions, future):
            ax.plot(
                segment[:, 0], segment[:, 1],
                color=color, linewidth=max(1.0, line_width - 0.6),
                linestyle="--", alpha=0.4, zorder=2,
            )
        if frame < len(valid) and valid[frame]:
            xy = positions[frame, :2]
            ax.scatter(xy[0], xy[1], color=color,
                       s=90 if is_center else (36 if role in {"sdc", "target"} else 18),
                       zorder=5, edgecolors="black" if is_center else "none", linewidths=0.8)

    ax.set_aspect("equal")
    ax.set_xlim(center_xy[0] - view_radius, center_xy[0] + view_radius)
    ax.set_ylim(center_xy[1] - view_radius, center_xy[1] + view_radius)
    ax.axis("off")


def render_agent_gif(
    *,
    dataset: str,
    scenario_id: str,
    center_objects_id: str,
    split: str,
    output_path: Path,
    title: str = "",
    view_radius: float = 70.0,
    frame_stride: int = 2,
    fps: int = 10,
    dataset_root_override: Optional[str] = None,
    max_frames: Optional[int] = None,
) -> Path:
    """Render a GIF of one scenario centred on one predicted agent.

    Parameters
    ----------
    dataset:       'av2' or 'waymo'.
    scenario_id:   value of `scenario_id` column in the feature table.
    center_objects_id: value of `center_objects_id` column in the feature table.
    split:         'train' / 'val' / ... matching what was used at extraction.
    output_path:   destination .gif path (parents are created).
    title:         top-of-frame annotation (e.g. "aggressive | minFDE=8.3m").
    frame_stride:  render every Nth frame of the scenario.
    fps:           GIF frames per second.
    max_frames:    optional cap on rendered frames (for fast previews).

    Returns
    -------
    Path to the written GIF.
    """
    try:
        import imageio.v2 as imageio  # type: ignore
    except ImportError as err:  # pragma: no cover - runtime guard
        raise ImportError(
            "imageio is required for GIF rendering. "
            "Install with `pip install imageio` inside the tailrisk-mp-cu126 env."
        ) from err

    bundle = load_scenario_by_id(
        dataset=dataset,
        scenario_id=scenario_id,
        split=split,
        dataset_root_override=dataset_root_override,
    )
    scenario = bundle["scenario"]
    sample_track = next(iter(scenario["tracks"].values()))
    n_steps = len(np.asarray(sample_track["state"]["valid"]).reshape(-1))

    frames = list(range(0, n_steps, max(1, frame_stride)))
    if max_frames is not None:
        frames = frames[:max_frames]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="-", color="#d62728", lw=3.0,
               markeredgecolor="black", label="Target agent"),
        Line2D([0], [0], color="#ff7f0e", lw=2.0, label="SDC"),
        Line2D([0], [0], color="#1f77b4", lw=1.4, label="Other vehicles"),
        Line2D([0], [0], color="#2ca02c", lw=1.4, label="Pedestrians"),
        Line2D([0], [0], color="#7f7f7f", lw=1.0, label="Map"),
        Line2D([0], [0], color="black", lw=1.0, linestyle="--", label="Future"),
    ]

    fig, ax = plt.subplots(figsize=(6, 6), dpi=110)
    imgs = []
    for f in frames:
        _draw_frame(ax, scenario, f, center_objects_id, view_radius=view_radius)
        ax.set_title(f"{title}  t={f}", fontsize=9)
        ax.legend(handles=legend_handles, loc="upper right", fontsize=7, frameon=False)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
        buf.seek(0)
        imgs.append(imageio.imread(buf))
    plt.close(fig)

    imageio.mimsave(output_path, imgs, fps=fps, loop=0)
    return output_path


def sample_cluster_agents(
    df, *, cluster_column: str, cluster_value, n: int = 6,
    extra_filter=None, random_state: int = 0,
) -> "pd.DataFrame":
    """Pick ``n`` (scenario_id, center_objects_id) rows from a cluster.

    Uses a stratified sample so the selection is deterministic given
    ``random_state``. ``extra_filter`` can be a boolean Series aligned with
    ``df`` (e.g. top-minFDE rows within the cluster).
    """
    import pandas as pd

    subset = df[df[cluster_column] == cluster_value]
    if extra_filter is not None:
        subset = subset[extra_filter.reindex(subset.index, fill_value=False)]
    subset = subset.dropna(subset=["scenario_id", "center_objects_id"])
    n_sample = min(n, len(subset))
    if n_sample == 0:
        return subset.head(0)
    return subset.sample(n=n_sample, random_state=random_state)


def render_cluster_montage(
    df,
    *,
    cluster_column: str,
    clusters: Sequence,
    per_cluster: int = 4,
    dataset: str = "av2",
    split: str = "val",
    output_dir: Path,
    view_radius: float = 70.0,
    frame_stride: int = 2,
    fps: int = 10,
    max_frames: Optional[int] = 40,
    title_prefix: str = "",
    random_state: int = 0,
    dataset_root_override: Optional[str] = None,
) -> list[Path]:
    """Render ``per_cluster`` GIFs for each ``cluster`` value and return paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for cl in clusters:
        sampled = sample_cluster_agents(df, cluster_column=cluster_column,
                                        cluster_value=cl, n=per_cluster,
                                        random_state=random_state)
        for i, (_, row) in enumerate(sampled.iterrows()):
            tag = f"{title_prefix}cluster={cl}_{i:02d}"
            gif_path = output_dir / f"{tag}.gif"
            minfde = row.get("mtr_minfde6") if "mtr_minfde6" in row else None
            title_extra = f" minFDE6={minfde:.2f}" if isinstance(minfde, (float, np.floating)) and np.isfinite(minfde) else ""
            try:
                render_agent_gif(
                    dataset=dataset,
                    scenario_id=str(row["scenario_id"]),
                    center_objects_id=str(row["center_objects_id"]),
                    split=split,
                    output_path=gif_path,
                    title=f"cluster={cl}{title_extra}",
                    view_radius=view_radius,
                    frame_stride=frame_stride,
                    fps=fps,
                    max_frames=max_frames,
                    dataset_root_override=dataset_root_override,
                )
                paths.append(gif_path)
            except Exception as err:
                print(f"[style_gifs] skipped scenario={row['scenario_id']} "
                      f"agent={row['center_objects_id']}: {err}")
    return paths


__all__ = [
    "render_agent_gif",
    "render_cluster_montage",
    "sample_cluster_agents",
]
