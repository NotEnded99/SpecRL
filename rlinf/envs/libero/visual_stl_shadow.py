"""Shadow visual STL monitoring without changing reward."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rlinf.envs.libero.visual_stl_monitor import (
    estimate_visual_approach,
)
from rlinf.envs.libero.visual_in_margin import (
    VisualInConfig,
    visual_container_on_from_bounds,
    visual_point_contact_quantiles,
    visual_in_basket_margin,
    visual_in_caddy_back_margin,
    visual_in_drawer_margin,
    visual_in_microwave_margin,
    visual_in_microwave_occlusion_margin,
    visual_release_evidence_is_current,
)
from rlinf.envs.libero.visual_open_margin import (
    VisualOpenConfig,
    visual_open_margin,
)
from rlinf.envs.libero.visual_close_margin import (
    VisualCloseConfig,
    apply_visual_in_gate,
    visual_close_margin,
)
from rlinf.envs.libero.visual_drawer_panel_diagnostic import (
    drawer_panel_roi,
    make_depth_diagnostic,
    make_drawer_panel_overlay,
)
from rlinf.envs.libero.visual_drawer_close_margin import (
    VisualDrawerCloseConfig,
    visual_drawer_close_margin,
)
from rlinf.envs.libero.visual_target_selector import (
    select_visual_target_candidate,
)
from rlinf.envs.libero.visual_turnon_observer import (
    VisualTurnOnConfig,
    make_turnon_overlay,
    reset_relative_features,
    stove_roi_from_mask,
    visual_turnon_red_margin,
)
from rlinf.envs.libero.visual_microwave_diagnostic import (
    make_microwave_depth,
    make_microwave_overlay,
    microwave_roi,
)
from rlinf.envs.libero.visual_microwave_close_margin import (
    VisualMicrowaveCloseConfig,
    microwave_door_search_roi,
    visual_articulated_class,
    visual_microwave_dark_door_close_margin,
    visual_microwave_door_edge_close_margin,
)
from rlinf.envs.libero.visual_spatial_relation_margin import (
    VisualSpatialRelationConfig,
    visual_directional_region_margin,
)
from rlinf.envs.libero.yoloe_client import YoloeClient


def _mask_image_geometry(detection: dict | None) -> dict:
    """Return detector-image mask diagnostics without affecting margins."""
    nan = float("nan")
    empty = {
        "mask_pixels": 0,
        "mask_centroid_u_px": nan,
        "mask_centroid_v_px": nan,
        "mask_bbox_left_px": -1,
        "mask_bbox_top_px": -1,
        "mask_bbox_right_px": -1,
        "mask_bbox_bottom_px": -1,
    }
    if not detection or "mask" not in detection:
        return empty
    mask = np.asarray(detection["mask"], dtype=bool)
    if mask.ndim != 2 or not np.any(mask):
        return empty
    rows, cols = np.nonzero(mask)
    return {
        "mask_pixels": int(len(rows)),
        "mask_centroid_u_px": float(np.mean(cols)),
        "mask_centroid_v_px": float(np.mean(rows)),
        "mask_bbox_left_px": int(np.min(cols)),
        "mask_bbox_top_px": int(np.min(rows)),
        "mask_bbox_right_px": int(np.max(cols)),
        "mask_bbox_bottom_px": int(np.max(rows)),
    }


def _make_mask_probe_overlay(
    image_rgb: np.ndarray,
    detection: dict,
    color: tuple[int, int, int],
) -> np.ndarray:
    """Overlay one selected YOLOE mask for offline identity inspection."""
    image = np.asarray(image_rgb, dtype=np.uint8)
    mask = np.asarray(detection.get("mask"), dtype=bool)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("probe_image_must_be_rgb")
    if mask.shape != image.shape[:2]:
        raise ValueError("probe_mask_image_shape_mismatch")
    overlay = image.copy()
    color_array = np.asarray(color, dtype=np.float64)
    overlay[mask] = np.clip(
        0.45 * overlay[mask].astype(np.float64) + 0.55 * color_array,
        0.0,
        255.0,
    ).astype(np.uint8)
    geometry = _mask_image_geometry(detection)
    if geometry["mask_pixels"]:
        left = geometry["mask_bbox_left_px"]
        top = geometry["mask_bbox_top_px"]
        right = geometry["mask_bbox_right_px"]
        bottom = geometry["mask_bbox_bottom_px"]
        overlay[top : bottom + 1, left] = color_array
        overlay[top : bottom + 1, right] = color_array
        overlay[top, left : right + 1] = color_array
        overlay[bottom, left : right + 1] = color_array
    return overlay


def _is_caddy_visual_opening_candidate(
    *,
    container_geometry: str,
    target_source: str,
    geometry_margin: float,
    entry_tolerance_m: float,
    opening_margin: float,
    core_lateral_margin: float,
    min_core_lateral_margin_m: float,
) -> bool:
    """Start a fresh caddy handoff only near the intended compartment."""
    return bool(
        container_geometry == "caddy_back"
        and target_source == "fresh_rgbd"
        and np.isfinite(geometry_margin)
        and geometry_margin + float(entry_tolerance_m) >= 0.0
        and np.isfinite(opening_margin)
        and opening_margin >= 0.0
        and np.isfinite(core_lateral_margin)
        and (
            core_lateral_margin - float(min_core_lateral_margin_m)
            >= 0.0
        )
    )


def _caddy_opening_gate_margin(
    opening_margin: float,
    opening_tolerance_m: float,
) -> float:
    """Apply a caddy-only tolerance to the observable gripper delta."""
    if not np.isfinite(opening_margin):
        return float("nan")
    return float(opening_margin + float(opening_tolerance_m))


def _is_caddy_proxy_entry_opening_candidate(
    *,
    container_geometry: str,
    target_source: str,
    grasp_confirmed: bool,
    geometry_margin: float,
    entry_tolerance_m: float,
    opening_margin: float,
) -> bool:
    """Accept a confirmed attached proxy only near the caddy entrance."""
    return bool(
        container_geometry == "caddy_back"
        and target_source == "eef_attached_proxy"
        and grasp_confirmed
        and np.isfinite(geometry_margin)
        and geometry_margin + float(entry_tolerance_m) >= 0.0
        and np.isfinite(opening_margin)
        and opening_margin >= 0.0
    )


def _caddy_visual_opening_evidence(
    *,
    fresh_candidate: bool,
    proxy_candidate: bool,
    geometry_margin: float,
    fresh_entry_tolerance_m: float,
    proxy_entry_tolerance_m: float,
    opening_margin: float,
    core_lateral_margin: float,
    fresh_min_core_lateral_margin_m: float,
) -> float:
    """Return the signed margin of the selected caddy opening contract."""
    if fresh_candidate:
        return float(min(
            geometry_margin + float(fresh_entry_tolerance_m),
            opening_margin,
            core_lateral_margin
            - float(fresh_min_core_lateral_margin_m),
        ))
    if proxy_candidate:
        return float(min(
            geometry_margin + float(proxy_entry_tolerance_m),
            opening_margin,
        ))
    return float("nan")


def _next_microwave_settled_close_count(
    previous_count: int,
    close_progress: float,
    progress_threshold: float,
) -> int:
    """Count consecutive fresh near-close frames without weakening strict close."""
    if (
        np.isfinite(close_progress)
        and close_progress >= float(progress_threshold)
    ):
        return int(previous_count) + 1
    return 0


def _compose_microwave_settled_close_margin(
    strict_margin: float,
    close_progress: float,
    progress_threshold: float,
    settled_count: int,
    confirmation_steps: int,
) -> float:
    """Add a sustained near-close route while retaining the strict margin."""
    if (
        int(settled_count) >= int(confirmation_steps)
        and np.isfinite(close_progress)
    ):
        return float(max(
            strict_margin,
            close_progress - float(progress_threshold),
        ))
    return float(strict_margin)


def _caddy_visual_opening_max_age_steps(
    *,
    opening_source: str,
    standard_max_age_steps: int,
    fresh_entry_max_age_steps: int,
    proxy_max_age_steps: int,
) -> int:
    """Select the bounded age for the opening evidence source."""
    if opening_source == "fresh_rgbd_near_entry":
        return int(fresh_entry_max_age_steps)
    if opening_source == "confirmed_pick_proxy_near_entry":
        return int(proxy_max_age_steps)
    return int(standard_max_age_steps)


def _is_recent_visual_evidence(
    *,
    current_step: int,
    evidence_step: int | None,
    max_age_steps: int,
) -> bool:
    """Return whether a visual event is current and bounded in time."""
    return bool(
        evidence_step is not None
        and 0 <= int(current_step) - int(evidence_step) <= max_age_steps
    )


def _eef_target_surface_quantiles(
    points_world: np.ndarray,
    eef_world: np.ndarray,
) -> tuple[float, float, float]:
    """Summarize observable EEF distance to the target RGB-D surface."""
    nan = float("nan")
    try:
        points = np.asarray(points_world, dtype=np.float64)
        eef = np.asarray(eef_world, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return nan, nan, nan
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or eef.shape != (3,)
        or not np.all(np.isfinite(eef))
    ):
        return nan, nan, nan
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) == 0:
        return nan, nan, nan
    distances = np.linalg.norm(points - eef[None, :], axis=1)
    quantiles = np.quantile(distances, [0.01, 0.05, 0.10])
    return tuple(float(value) for value in quantiles)


def _is_caddy_complete_entry_progress(
    *,
    geometry_margin: float,
    opening_geometry_margin: float,
    core_lateral_margin: float,
) -> bool:
    """Recognize motion that carries the target fully across the entry."""
    return bool(
        np.isfinite(geometry_margin)
        and np.isfinite(opening_geometry_margin)
        and np.isfinite(core_lateral_margin)
        and geometry_margin >= 0.0
        and geometry_margin >= opening_geometry_margin
        and geometry_margin + opening_geometry_margin >= 0.0
        and core_lateral_margin >= 0.0
    )


def _caddy_dual_stage_handoff_margin(
    *,
    container_geometry: str,
    opening_recent: bool,
    target_source: str,
    opening_evidence: float,
    inside_xy_margin: float,
    above_floor_margin: float,
    core_lateral_margin: float,
    target_eef_distance_m: float,
    separation_threshold_m: float,
) -> float:
    """Fuse a recent entry observation with direct supported separation.

    A vertical book may correctly protrude above the caddy rim, so this
    route deliberately relies on horizontal containment and floor support
    instead of the full geometry margin.
    """
    if (
        container_geometry != "caddy_back"
        or not opening_recent
        or target_source != "fresh_rgbd"
    ):
        return float("nan")
    values = np.asarray(
        [
            opening_evidence,
            inside_xy_margin,
            above_floor_margin,
            core_lateral_margin,
            target_eef_distance_m - float(separation_threshold_m),
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        return float("nan")
    return float(np.min(values))


def _is_caddy_visual_handoff_candidate(
    *,
    container_geometry: str,
    opening_recent: bool,
    target_source: str,
    geometry_margin: float,
    opening_geometry_margin: float,
    core_lateral_margin: float,
    entry_tolerance_m: float,
    target_eef_distance_m: float,
    separation_threshold_m: float,
    opening_target_displacement_m: float,
    max_target_displacement_m: float,
    max_allowed_target_displacement_m: float,
    opening_age_steps: int,
    settled_confirmation_age_steps: int,
) -> bool:
    """Confirm actual separation or a settled visible placement.

    EEF distance from the cached opening point is intentionally insufficient:
    a still-grasped target can move with the EEF and create that displacement.
    """
    current_placement = bool(
        target_source == "fresh_rgbd"
        and np.isfinite(geometry_margin)
        and geometry_margin + float(entry_tolerance_m) >= 0.0
    )
    distance_stable = bool(
        np.isfinite(opening_target_displacement_m)
        and np.isfinite(max_target_displacement_m)
        and opening_target_displacement_m
        <= float(max_allowed_target_displacement_m)
        and max_target_displacement_m
        <= float(max_allowed_target_displacement_m)
    )
    complete_entry_progress = _is_caddy_complete_entry_progress(
        geometry_margin=geometry_margin,
        opening_geometry_margin=opening_geometry_margin,
        core_lateral_margin=core_lateral_margin,
    )
    target_stable = bool(distance_stable or complete_entry_progress)
    actual_separation = bool(
        opening_recent
        and current_placement
        and target_stable
        and np.isfinite(target_eef_distance_m)
        and target_eef_distance_m >= float(separation_threshold_m)
    )
    settled_placement = bool(
        opening_recent
        and current_placement
        and target_stable
        and int(opening_age_steps) >= int(settled_confirmation_age_steps)
    )
    return bool(
        container_geometry == "caddy_back"
        and (actual_separation or settled_placement)
    )


def _is_caddy_visual_handoff_contradiction(
    *,
    container_geometry: str,
    target_source: str,
    release_source: str,
    inside_xy_margin: float,
    above_floor_margin: float,
    opening_target_displacement_m: float,
    contradiction_margin_m: float,
    max_target_displacement_m: float,
) -> bool:
    """Reject a cached handoff only from corroborated fresh departure."""
    direct_departure = bool(
        (
            np.isfinite(inside_xy_margin)
            and inside_xy_margin <= -float(contradiction_margin_m)
        )
        or (
            np.isfinite(above_floor_margin)
            and above_floor_margin <= -float(contradiction_margin_m)
        )
    )
    displaced_from_opening = bool(
        np.isfinite(opening_target_displacement_m)
        and opening_target_displacement_m
        > float(max_target_displacement_m)
    )
    return bool(
        container_geometry == "caddy_back"
        and target_source == "fresh_rgbd"
        and release_source == "caddy_visual_handoff"
        and direct_departure
        and displaced_from_opening
    )


def _compose_caddy_visual_in_margin(
    *,
    geometry_margin: float,
    handoff_margin: float,
    release_latched: bool,
    release_evidence: float,
    unreleased_ceiling_m: float = 0.001,
    cold_start_margin: float = -1.0,
) -> float:
    """Expose dense geometry while keeping unreleased In strictly false."""
    if release_latched:
        evidence = (
            handoff_margin
            if np.isfinite(handoff_margin)
            else release_evidence
        )
        return float(
            evidence if np.isfinite(evidence) else cold_start_margin
        )
    if np.isfinite(geometry_margin):
        values = [
            float(geometry_margin),
            -abs(float(unreleased_ceiling_m)),
        ]
        if np.isfinite(handoff_margin):
            values.append(float(handoff_margin))
        return float(min(values))
    return float(cold_start_margin)


def _caddy_topology_temporal_diagnostic(
    *,
    container_geometry: str,
    target_source: str,
    record_valid: bool,
    topology_valid: bool,
    topology_back_margin_m: float,
    above_floor_margin_m: float,
    previous_count: int,
    confirmation_steps: int = 3,
) -> tuple[float, int, bool]:
    """Track persistent fresh topology evidence without changing reward."""
    steps = int(confirmation_steps)
    if steps < 1:
        raise ValueError("confirmation_steps must be >= 1")

    topology_margin = float(topology_back_margin_m)
    floor_margin = float(above_floor_margin_m)
    candidate_margin = (
        float(min(topology_margin, floor_margin))
        if np.isfinite(topology_margin) and np.isfinite(floor_margin)
        else float("nan")
    )
    frame_positive = bool(
        container_geometry == "caddy_back"
        and target_source == "fresh_rgbd"
        and record_valid
        and topology_valid
        and np.isfinite(candidate_margin)
        and candidate_margin >= 0.0
    )
    count = max(0, int(previous_count)) + 1 if frame_positive else 0
    return candidate_margin, count, bool(count >= steps)


def _caddy_dual_expert_temporal_diagnostic(
    *,
    container_geometry: str,
    target_source: str,
    record_valid: bool,
    topology_temporal_candidate: bool,
    central_back_margin_m: float,
    above_floor_margin_m: float,
    previous_central_count: int,
    central_confirmation_steps: int = 20,
) -> tuple[float, int, bool, bool, str]:
    """Fuse strict topology with a stable central-geometry fallback."""
    steps = int(central_confirmation_steps)
    if steps < 1:
        raise ValueError("central_confirmation_steps must be >= 1")

    central_margin = float(central_back_margin_m)
    floor_margin = float(above_floor_margin_m)
    candidate_margin = (
        float(min(central_margin, floor_margin))
        if np.isfinite(central_margin) and np.isfinite(floor_margin)
        else float("nan")
    )
    fresh_caddy_record = bool(
        container_geometry == "caddy_back"
        and target_source == "fresh_rgbd"
        and record_valid
    )
    central_positive = bool(
        fresh_caddy_record
        and np.isfinite(candidate_margin)
        and candidate_margin >= 0.0
    )
    central_count = (
        max(0, int(previous_central_count)) + 1
        if central_positive else 0
    )
    central_candidate = bool(central_count >= steps)
    topology_candidate = bool(
        fresh_caddy_record and topology_temporal_candidate
    )
    dual_candidate = bool(topology_candidate or central_candidate)
    route = (
        "topology"
        if topology_candidate
        else "central_stable" if central_candidate else "none"
    )
    return (
        candidate_margin,
        central_count,
        central_candidate,
        dual_candidate,
        route,
    )


def _caddy_cached_topology_temporal_diagnostic(
    *,
    container_geometry: str,
    target_source: str,
    record_valid: bool,
    topology_valid: bool,
    topology_lateral_low_m: float,
    topology_lateral_high_m: float,
    center_longitudinal_margin_m: float,
    center_lateral_offset_m: float,
    above_floor_margin_m: float,
    previous_half_widths_m: tuple[float, ...] | list[float],
    previous_count: int,
    width_quantile: float = 0.25,
    min_samples: int = 3,
    confirmation_steps: int = 3,
    max_history: int = 64,
) -> tuple[tuple[float, ...], float, float, int, bool]:
    """Reuse a conservative caddy opening width across fresh frames."""
    quantile = float(width_quantile)
    samples = int(min_samples)
    steps = int(confirmation_steps)
    history_limit = int(max_history)
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("width_quantile must be in [0, 1]")
    if samples < 1:
        raise ValueError("min_samples must be >= 1")
    if steps < 1:
        raise ValueError("confirmation_steps must be >= 1")
    if history_limit < samples:
        raise ValueError("max_history must be >= min_samples")

    history = [
        float(value)
        for value in previous_half_widths_m
        if np.isfinite(value) and float(value) > 0.0
    ][-history_limit:]
    fresh_caddy_record = bool(
        container_geometry == "caddy_back"
        and target_source == "fresh_rgbd"
        and record_valid
    )
    low = float(topology_lateral_low_m)
    high = float(topology_lateral_high_m)
    if (
        fresh_caddy_record
        and topology_valid
        and np.isfinite(low)
        and np.isfinite(high)
        and high > low
    ):
        # The caddy origin is the only stable center across camera-axis
        # jitter. Use the nearer divider instead of half the detected span,
        # which would overstate an off-center opening.
        history.append(min(abs(low), abs(high)))
        history = history[-history_limit:]

    cached_half_width = (
        float(np.quantile(np.asarray(history, dtype=np.float64), quantile))
        if len(history) >= samples else float("nan")
    )
    longitudinal = float(center_longitudinal_margin_m)
    lateral_offset = float(center_lateral_offset_m)
    floor = float(above_floor_margin_m)
    candidate_margin = (
        float(min(
            cached_half_width - abs(lateral_offset),
            longitudinal,
            floor,
        ))
        if (
            np.isfinite(cached_half_width)
            and np.isfinite(longitudinal)
            and np.isfinite(lateral_offset)
            and np.isfinite(floor)
        ) else float("nan")
    )
    frame_positive = bool(
        fresh_caddy_record
        and np.isfinite(candidate_margin)
        and candidate_margin >= 0.0
    )
    count = max(0, int(previous_count)) + 1 if frame_positive else 0
    return (
        tuple(history),
        cached_half_width,
        candidate_margin,
        count,
        bool(count >= steps),
    )


def _caddy_static_baseline_temporal_diagnostic(
    *,
    container_geometry: str,
    target_source: str,
    baseline_valid: bool,
    candidate_margin_m: float,
    previous_count: int,
    confirmation_steps: int = 3,
) -> tuple[float, int, bool]:
    """Track fixed-caddy evidence without changing canonical In."""
    steps = int(confirmation_steps)
    if steps < 1:
        raise ValueError("confirmation_steps must be >= 1")

    margin = float(candidate_margin_m)
    frame_positive = bool(
        container_geometry == "caddy_back"
        and target_source == "fresh_rgbd"
        and baseline_valid
        and np.isfinite(margin)
        and margin >= 0.0
    )
    count = max(0, int(previous_count)) + 1 if frame_positive else 0
    return margin, count, bool(count >= steps)


def _caddy_cached_topology_reobservation_diagnostic(
    *,
    container_geometry: str,
    target_source: str,
    record_valid: bool,
    cached_topology_candidate: bool,
    cached_topology_margin_m: float,
    previous_candidate_step: int,
    previous_bouts: int,
    previous_margin_m: float,
    current_step: int,
    min_gap_steps: int = 3,
    required_bouts: int = 2,
) -> tuple[int, int, int, float, bool]:
    """Confirm caddy containment across two independent observations."""
    gap_required = int(min_gap_steps)
    bouts_required = int(required_bouts)
    if gap_required < 1:
        raise ValueError("min_gap_steps must be >= 1")
    if bouts_required < 2:
        raise ValueError("required_bouts must be >= 2")

    last_step = int(previous_candidate_step)
    bouts = max(0, int(previous_bouts))
    latched_margin = float(previous_margin_m)
    step = int(current_step)
    gap_steps = max(0, step - last_step - 1) if last_step >= 0 else -1
    margin = float(cached_topology_margin_m)
    current_candidate = bool(
        container_geometry == "caddy_back"
        and target_source == "fresh_rgbd"
        and record_valid
        and cached_topology_candidate
        and np.isfinite(margin)
        and margin >= 0.0
    )
    if current_candidate:
        if bouts == 0:
            bouts = 1
        elif last_step >= 0 and gap_steps >= gap_required:
            bouts += 1
        last_step = step
        gap_steps = 0
        if bouts >= bouts_required:
            latched_margin = (
                max(latched_margin, margin)
                if np.isfinite(latched_margin)
                else margin
            )
    confirmed = bool(
        bouts >= bouts_required
        and np.isfinite(latched_margin)
        and latched_margin >= 0.0
    )
    return last_step, bouts, gap_steps, latched_margin, confirmed


def _promote_caddy_cached_topology_reobservation_margin(
    *,
    visual_in_margin_m: float,
    reobservation_margin_m: float,
    reobservation_candidate: bool,
) -> float:
    """Promote In from repeated fresh views of cached caddy topology."""
    current_margin = float(visual_in_margin_m)
    observed_margin = float(reobservation_margin_m)
    if not (
        reobservation_candidate
        and np.isfinite(observed_margin)
        and observed_margin >= 0.0
    ):
        return current_margin
    if not np.isfinite(current_margin):
        return observed_margin
    return float(max(current_margin, observed_margin))


def _caddy_confirmed_event_bridge_diagnostic(
    *,
    container_geometry: str,
    target_source: str,
    record_valid: bool,
    visual_in_confirmed: bool,
    visual_in_margin_m: float,
    previous_margin_m: float,
    previous_step: int,
    current_step: int,
) -> tuple[float, int, int, bool]:
    """Retain a visually confirmed caddy-In event for offline analysis.

    Goal evaluation uses success-once semantics. Retaining the measured
    positive margin lets diagnostics quantify events that are shorter than
    downstream smoothing, without changing the canonical In margin.
    """
    previous_margin = float(previous_margin_m)
    first_step = int(previous_step)
    latched = bool(
        container_geometry == "caddy_back"
        and np.isfinite(previous_margin)
        and previous_margin >= 0.0
        and first_step >= 0
    )
    current_margin = float(visual_in_margin_m)
    current_candidate = bool(
        container_geometry == "caddy_back"
        and target_source == "fresh_rgbd"
        and record_valid
        and visual_in_confirmed
        and np.isfinite(current_margin)
        and current_margin >= 0.0
    )
    if current_candidate:
        if latched:
            previous_margin = max(previous_margin, current_margin)
        else:
            previous_margin = current_margin
            first_step = int(current_step)
        latched = True
    if not latched:
        return float("nan"), -1, -1, False
    age_steps = max(0, int(current_step) - first_step)
    return previous_margin, first_step, age_steps, True


def _promote_caddy_confirmed_event_margin(
    *,
    visual_in_margin_m: float,
    confirmed_event_margin_m: float,
    confirmed_event_candidate: bool,
) -> float:
    """Preserve a confirmed caddy-In event in the canonical margin."""
    current_margin = float(visual_in_margin_m)
    event_margin = float(confirmed_event_margin_m)
    if not (
        confirmed_event_candidate
        and np.isfinite(event_margin)
        and event_margin >= 0.0
    ):
        return current_margin
    if not np.isfinite(current_margin):
        return event_margin
    return float(max(current_margin, event_margin))


def _terminal_grasp_candidate_margin(
    *,
    gripper_width_m: float,
    near_margin: float,
    close_margin: float,
    comove_margin: float,
    motion_margin: float,
    displacement_margin: float,
    max_width_m: float,
    comove_slack_m: float,
    motion_slack_m: float,
) -> float:
    """Return stable geometry margin after a terminal grasp gate passes."""
    values = (
        gripper_width_m,
        near_margin,
        close_margin,
        comove_margin,
        motion_margin,
        displacement_margin,
    )
    if not all(np.isfinite(value) for value in values):
        return float("nan")
    if gripper_width_m > max_width_m:
        return float("nan")
    if (
        near_margin < 0.0
        or close_margin < 0.0
        or displacement_margin < 0.0
        or comove_margin < -comove_slack_m
        or motion_margin < -motion_slack_m
    ):
        return float("nan")
    # Co-motion and EEF motion are noisy temporal derivatives. They gate the
    # candidate above, while the confirmed robustness magnitude comes only
    # from stable geometry. This prevents sub-millimetre derivative noise
    # from pinning a real grasp just below zero through the causal EMA.
    return float(min(near_margin, close_margin, displacement_margin))


@dataclass
class VisualSTLShadowRecord:
    step: int
    task_id: int
    target_object: str
    target_class: str
    selector: str
    valid: bool
    reason: str
    visual_fresh: bool
    visual_held: bool
    visual_hold_age_steps: int
    confidence: float
    mask_pixels: int
    depth_pixels: int
    two_finger_grasp: bool
    oracle_distance_m: float
    visual_distance_m: float
    oracle_robustness: float
    visual_robustness: float
    visual_surface_p01_distance_m: float
    visual_surface_p05_distance_m: float
    visual_surface_p10_distance_m: float
    visual_surface_p01_robustness: float
    visual_surface_p05_robustness: float
    visual_surface_p10_robustness: float
    oracle_pick_margin: float
    visual_pick_margin: float
    eef_x: float
    eef_y: float
    eef_z: float
    oracle_x: float
    oracle_y: float
    oracle_z: float
    visual_x: float
    visual_y: float
    visual_z: float
    gripper_width_m: float
    visual_near_margin: float
    visual_close_margin: float
    visual_comove_error_m: float
    visual_comove_margin: float
    eef_motion_m: float
    visual_motion_margin: float
    visual_displacement_m: float
    visual_displacement_margin: float
    visual_grasp_margin: float
    visual_grasp_history_ready: bool
    visual_grasp_confirmed: bool
    visual_grasp_terminal_candidate: bool
    visual_grasp_terminal_margin: float
    visual_grasp_terminal_count: int
    visual_grasp_terminal_confirmed: bool
    visual_grasp_confirmation_source: str
    on_target_object: str
    on_target_class: str
    oracle_on_margin: float
    visual_on_valid: bool
    visual_on_reason: str
    visual_on_target_confidence: float
    visual_on_xy_distance_m: float
    visual_on_xy_margin: float
    visual_on_completion_xy_margin: float
    visual_on_height_delta_m: float
    visual_on_above_margin: float
    visual_on_surface_gap_m: float
    visual_on_support_margin: float
    visual_on_release_margin: float
    visual_on_margin: float
    visual_on_confirmed: bool
    visual_on_object_source: str
    visual_on_attached: bool
    visual_on_released: bool
    visual_on_geometry_mode: str = "ordinary_surface"
    visual_on_inside_xy_margin: float = float("nan")
    visual_on_below_rim_margin: float = float("nan")
    visual_on_above_floor_margin: float = float("nan")
    # A stable positive visual geometry may be handed to a later observed
    # release when the object becomes occluded before the gripper opens.
    visual_on_handoff_used: bool = False
    visual_on_handoff_age_steps: int = -1
    visual_on_handoff_margin: float = float("nan")
    visual_on_contact_valid: bool = False
    visual_on_contact_reason: str = ""
    visual_on_contact_p01_m: float = float("nan")
    visual_on_contact_p05_m: float = float("nan")
    visual_on_contact_p10_m: float = float("nan")
    visual_on_contact_margin: float = float("nan")
    visual_on_object_center_x_m: float = float("nan")
    visual_on_object_center_y_m: float = float("nan")
    visual_on_object_center_z_m: float = float("nan")
    visual_on_support_center_x_m: float = float("nan")
    visual_on_support_center_y_m: float = float("nan")
    visual_on_support_center_z_m: float = float("nan")
    visual_on_support_locked: bool = False
    visual_on_object_bottom_z_m: float = float("nan")
    visual_on_support_z_p95_m: float = float("nan")
    visual_on_support_z_p99_m: float = float("nan")
    visual_on_support_z_p995_m: float = float("nan")
    visual_on_support_baseline_center_x_m: float = float("nan")
    visual_on_support_baseline_center_y_m: float = float("nan")
    visual_on_support_baseline_center_z_m: float = float("nan")
    visual_on_support_baseline_z_p95_m: float = float("nan")
    visual_on_support_center_drift_m: float = float("nan")
    visual_on_support_z_p95_drift_m: float = float("nan")
    visual_on_site_center_delta_q99_m: float = float("nan")
    visual_on_site_bottom_gap_q99_m: float = float("nan")
    visual_on_spatial_relation: str = ""
    visual_on_relative_x_m: float = float("nan")
    visual_on_relative_y_m: float = float("nan")
    visual_on_relative_z_m: float = float("nan")
    visual_on_anchor_x_q05_m: float = float("nan")
    visual_on_anchor_x_q95_m: float = float("nan")
    visual_on_anchor_y_q05_m: float = float("nan")
    visual_on_anchor_y_q95_m: float = float("nan")
    visual_on_anchor_u: float = float("nan")
    visual_on_anchor_v: float = float("nan")
    visual_on_spatial_x_margin: float = float("nan")
    visual_on_spatial_y_margin: float = float("nan")
    visual_on_spatial_z_margin: float = float("nan")
    visual_on_spatial_directional_offset_m: float = float("nan")
    visual_on_spatial_lateral_offset_m: float = float("nan")
    visual_on_spatial_anchor_direction_half_extent_m: float = float("nan")
    visual_on_spatial_anchor_lateral_half_extent_m: float = float("nan")
    visual_on_spatial_displacement_m: float = float("nan")
    visual_on_spatial_frame_motion_m: float = float("nan")
    visual_on_spatial_motion_observed: bool = False
    visual_on_spatial_stable: bool = False
    visual_on_spatial_pick_proxy_margin: float = float("nan")
    # Shadow-only q05/q95 footprint candidates. These fields never enter the
    # visual On margin, confirmation latch, AGM runtime, or reward.
    visual_on_spatial_footprint_box_valid: bool = False
    visual_on_spatial_footprint_box_reason: str = ""
    visual_on_spatial_footprint_x_margin: float = float("nan")
    visual_on_spatial_footprint_y_margin: float = float("nan")
    visual_on_spatial_footprint_bottom_z_margin: float = float("nan")
    visual_on_spatial_footprint_top_z_margin: float = float("nan")
    visual_on_spatial_footprint_full_z_margin: float = float("nan")
    visual_on_spatial_footprint_box_candidate_margin: float = float("nan")
    visual_on_spatial_footprint_directional_valid: bool = False
    visual_on_spatial_footprint_directional_reason: str = ""
    visual_on_spatial_footprint_directional_margin: float = float("nan")
    visual_on_spatial_footprint_lateral_margin: float = float("nan")
    visual_on_spatial_footprint_surface_margin: float = float("nan")
    visual_on_spatial_footprint_directional_candidate_margin: float = float("nan")
    visual_on_object_x_q05_m: float = float("nan")
    visual_on_object_x_q95_m: float = float("nan")
    visual_on_object_y_q05_m: float = float("nan")
    visual_on_object_y_q95_m: float = float("nan")
    visual_on_support_baseline_x_q05_m: float = float("nan")
    visual_on_support_baseline_x_q95_m: float = float("nan")
    visual_on_support_baseline_y_q05_m: float = float("nan")
    visual_on_support_baseline_y_q95_m: float = float("nan")
    visual_on_robust_xy_distance_m: float = float("nan")
    visual_on_robust_xy_margin: float = float("nan")
    visual_on_robust_candidate_margin: float = float("nan")


@dataclass
class VisualInShadowRecord:
    step: int
    task_id: int
    target_object: str
    target_class: str
    container_object: str
    container_class: str
    container_geometry: str
    valid: bool
    reason: str
    visual_fresh: bool
    target_source: str
    visual_hold_age_steps: int
    oracle_in_margin: float
    target_confidence: float
    container_confidence: float
    target_selector: str
    target_camera_name: str
    container_camera_name: str
    same_camera_geometry: bool
    target_candidate_count: int
    target_same_camera_candidate_count: int
    container_candidate_count: int
    container_same_camera_candidate_count: int
    target_mask_pixels: int
    target_mask_centroid_u_px: float
    target_mask_centroid_v_px: float
    target_mask_bbox_left_px: int
    target_mask_bbox_top_px: int
    target_mask_bbox_right_px: int
    target_mask_bbox_bottom_px: int
    container_mask_pixels: int
    container_mask_centroid_u_px: float
    container_mask_centroid_v_px: float
    container_mask_bbox_left_px: int
    container_mask_bbox_top_px: int
    container_mask_bbox_right_px: int
    container_mask_bbox_bottom_px: int
    target_center_world_x_m: float
    target_center_world_y_m: float
    target_center_world_z_m: float
    container_center_world_x_m: float
    container_center_world_y_m: float
    container_center_world_z_m: float
    target_container_center_distance_m: float
    inside_xy_margin: float
    caddy_center_longitudinal_margin: float
    caddy_center_lateral_margin: float
    caddy_center_back_offset_m: float
    caddy_center_lateral_offset_m: float
    caddy_back_half_extent_m: float
    caddy_lateral_half_extent_m: float
    caddy_central_lateral_margin_m: float
    caddy_central_back_margin_m: float
    caddy_topology_valid: bool
    caddy_topology_reason: str
    caddy_topology_confidence: float
    caddy_topology_lateral_low_m: float
    caddy_topology_lateral_high_m: float
    caddy_topology_lateral_margin_m: float
    caddy_topology_back_margin_m: float
    caddy_topology_footprint_valid: bool
    caddy_topology_footprint_longitudinal_margin_m: float
    caddy_topology_footprint_lateral_margin_m: float
    caddy_topology_footprint_below_rim_margin_m: float
    caddy_topology_footprint_above_floor_margin_m: float
    caddy_topology_footprint_margin_m: float
    caddy_topology_temporal_margin_m: float
    caddy_topology_temporal_count: int
    caddy_topology_temporal_candidate: bool
    caddy_central_temporal_margin_m: float
    caddy_central_temporal_count: int
    caddy_central_temporal_candidate: bool
    caddy_dual_expert_candidate: bool
    caddy_dual_expert_route: str
    caddy_cached_topology_half_width_m: float
    caddy_cached_topology_sample_count: int
    caddy_cached_topology_margin_m: float
    caddy_cached_topology_count: int
    caddy_cached_topology_candidate: bool
    caddy_static_baseline_camera_name: str
    caddy_static_baseline_same_camera: bool
    caddy_static_baseline_point_count: int
    caddy_static_baseline_center_drift_m: float
    caddy_static_baseline_valid: bool
    caddy_static_baseline_inside_xy_margin_m: float
    caddy_static_baseline_geometry_margin_m: float
    caddy_static_baseline_topology_valid: bool
    caddy_static_baseline_topology_back_margin_m: float
    caddy_static_baseline_below_rim_margin_m: float
    caddy_static_baseline_above_floor_margin_m: float
    caddy_static_baseline_candidate_margin_m: float
    caddy_static_baseline_temporal_count: int
    caddy_static_baseline_temporal_candidate: bool
    caddy_cached_topology_reobservation_bouts: int
    caddy_cached_topology_reobservation_gap_steps: int
    caddy_cached_topology_reobservation_margin_m: float
    caddy_cached_topology_reobservation_candidate: bool
    caddy_confirmed_event_bridge_margin_m: float
    caddy_confirmed_event_bridge_step: int
    caddy_confirmed_event_bridge_age_steps: int
    caddy_confirmed_event_bridge_candidate: bool
    caddy_core_longitudinal_margin: float
    caddy_core_lateral_margin: float
    caddy_full_longitudinal_margin: float
    caddy_full_lateral_margin: float
    below_rim_margin: float
    above_floor_margin: float
    entry_depth_margin: float
    target_eef_distance_m: float
    target_eef_surface_p01_m: float
    target_eef_surface_p05_m: float
    target_eef_surface_p10_m: float
    eef_world_x_m: float
    eef_world_y_m: float
    eef_world_z_m: float
    release_proximity_margin: float
    opening_margin: float
    release_margin: float
    release_event: bool
    release_latched: bool
    release_source: str
    caddy_release_candidate: bool
    caddy_release_count: int
    caddy_opening_count: int
    caddy_opening_candidate: bool
    caddy_opening_candidate_source: str
    caddy_opening_source: str
    caddy_opening_evidence: float
    caddy_opening_gate_margin: float
    caddy_core_lateral_gate_margin: float
    caddy_opening_max_age_steps: int
    caddy_opening_age_steps: int
    caddy_separation_age_steps: int
    caddy_opening_center_world_x_m: float
    caddy_opening_center_world_y_m: float
    caddy_opening_center_world_z_m: float
    caddy_fixed_target_distance_m: float
    caddy_opening_target_displacement_m: float
    caddy_opening_max_target_displacement_m: float
    caddy_actual_separation_margin: float
    caddy_settled_age_margin_steps: int
    caddy_handoff_route: str
    caddy_handoff_margin: float
    caddy_contradiction_count: int
    caddy_handoff_revoked: bool
    visual_in_margin: float
    visual_in_confirmed: bool


@dataclass
class VisualOpenShadowRecord:
    step: int
    task_id: int
    articulated_object: str
    cabinet_class: str
    drawer_level: str
    valid: bool
    reason: str
    visual_fresh: bool
    visual_hold_age_steps: int
    confidence: float
    mask_pixels: int
    depth_pixels: int
    baseline_ready: bool
    displacement_m: float
    displacement_threshold_m: float
    visual_open_margin: float
    visual_open_confirmed: bool
    extension_axis: str
    signed_extension_m: float
    extension_x_minus_m: float
    extension_x_plus_m: float
    extension_y_minus_m: float
    extension_y_plus_m: float
    reference_shift_m: float
    baseline_band_points: int
    current_band_points: int
    oracle_open_margin: float


@dataclass
class VisualCloseShadowRecord:
    step: int
    task_id: int
    articulated_object: str
    cabinet_class: str
    drawer_level: str
    valid: bool
    reason: str
    visual_fresh: bool
    visual_hold_age_steps: int
    confidence: float
    mask_pixels: int
    depth_pixels: int
    baseline_ready: bool
    retraction_m: float
    retraction_threshold_m: float
    raw_visual_close_margin: float
    visual_close_margin: float
    visual_in_gate_required: bool
    visual_in_gate_open: bool
    visual_close_confirmed: bool
    extension_axis: str
    signed_extension_m: float
    reference_shift_m: float
    baseline_band_points: int
    current_band_points: int
    retraction_q05_m: float
    retraction_q10_m: float
    retraction_q25_m: float
    retraction_q50_m: float
    retraction_q75_m: float
    retraction_q90_m: float
    retraction_q95_m: float
    oracle_close_margin: float
    close_mode: str
    panel_retraction_fraction: float
    panel_close_margin: float
    panel_baseline_edge_px: float
    panel_current_edge_px: float
    panel_reference_shift_fraction: float
    panel_baseline_pixels: int
    panel_current_pixels: int
    microwave_settled_close_count: int = 0
    microwave_settled_close_confirmed: bool = False
    microwave_close_route: str = "none"


@dataclass
class VisualTurnOnShadowRecord:
    step: int
    task_id: int
    articulated_object: str
    valid: bool
    reason: str
    confidence: float
    roi_ready: bool
    rgb_mae: float
    gray_mae: float
    edge_mae: float
    depth_mae: float
    max_cell_change: float
    max_cell_row: int
    max_cell_col: int
    red_activation_fraction: float
    visual_turnon_margin: float
    visual_turnon_confirmed: bool
    oracle_turnon_margin: float


def _object_class(object_name: str) -> str | None:
    lowered = str(object_name).lower()

    rules = (
        ("black_bowl", "black bowl"),
        ("akita", "black bowl"),
        ("plate", "plate"),
        ("ramekin", "ramekin"),
        ("cookies", "cookies"),
        ("moka", "moka pot"),
        ("storage_box", "storage box"),
        ("milk", "milk carton"),
        ("wine_bottle", "wine bottle"),
        ("coffee_mug", "red mug"),
        ("yellow_book", "yellow book"),
        ("porcelain_mug", "white mug"),
        ("cream_cheese", "cream cheese"),
        ("salad_dressing", "salad dressing"),
        ("bbq_sauce", "bbq sauce"),
        ("ketchup", "ketchup"),
        ("tomato_sauce", "tomato sauce"),
        ("butter", "butter"),
        ("orange_juice", "orange juice"),
        ("chocolate_pudding", "chocolate pudding"),
        ("alphabet_soup", "alphabet soup"),
        ("black_book", "black book"),
        ("black book", "black book"),
        ("basket", "basket"),
        ("desk_caddy", "desk caddy"),
        ("desk caddy", "desk caddy"),
        ("flat_stove", "stove"),
        ("stove", "stove"),
        ("wine_rack", "wine rack"),
        ("wine rack", "wine rack"),
        ("wooden_cabinet", "wooden cabinet"),
        ("wooden cabinet", "wooden cabinet"),
        ("white_cabinet", "white cabinet"),
        ("white cabinet", "white cabinet"),
        ("microwave", "microwave"),
        ("white_yellow_mug", "yellow and white mug"),
        ("yellow_white_mug", "yellow and white mug"),
        ("yellow and white mug", "yellow and white mug"),
    )

    for marker, class_name in rules:
        if marker in lowered:
            return class_name

    return None


class VisualSTLShadowMonitor:
    """YOLOE-based approach robustness trace recorder."""

    def __init__(
        self,
        *,
        yoloe_url: str = "http://127.0.0.1:8010",
        yoloe_timeout_s: float = 10.0,
        yoloe_batch_enabled: bool = True,
        yoloe_batch_max_size: int = 32,
        yoloe_raw_transport: bool = False,
        camera_name: str = "agentview",
        fallback_camera_name: str | None = None,
        fallback_min_confidence: float | None = None,
        mask_to_depth_orientation: str = "rot180",
        grasp_radius_m: float = 0.10,
        visual_center_radius_m: float | None = None,
        surface_radius_m: float = 0.05,
        min_confidence: float = 0.25,
        track_gate_m: float = 0.10,
        track_ema: float = 0.70,
        close_threshold_m: float = 0.08,
        comove_tolerance_m: float = 0.015,
        motion_min_m: float = 0.005,
        displacement_min_m: float = 0.010,
        hold_steps: int = 5,
        grasp_near_mode: str = "center",
        grasp_history_lag_steps: int = 3,
        grasp_confirmation_steps: int = 3,
        grasp_terminal_width_m: float = 0.050,
        grasp_terminal_comove_slack_m: float = 0.001,
        grasp_terminal_motion_slack_m: float = 0.0005,
        grasp_terminal_confirmation_steps: int = 2,
        on_xy_tolerance_m: float = 0.05,
        on_above_epsilon_m: float = 0.0,
        on_surface_tolerance_m: float = 0.03,
        on_ordinary_xy_tolerance_m: float = 0.030,
        on_ordinary_xy_uncertainty_m: float = 0.014,
        on_ordinary_completion_xy_uncertainty_m: float = 0.009,
        on_ordinary_min_above_m: float = -0.005,
        on_ordinary_surface_tolerance_m: float = 0.03,
        on_ordinary_require_release: bool = False,
        on_ordinary_require_contact: bool = False,
        on_ordinary_contact_tolerance_m: float = 0.01,
        on_ordinary_confirmation_steps: int = 5,
        on_ordinary_near_center_fallback_enabled: bool = True,
        on_ordinary_near_center_fallback_xy_m: float = 0.008,
        on_ordinary_near_center_fallback_contact_p01_m: float = 0.005,
        on_ordinary_near_center_fallback_confirmation_steps: int = 30,
        on_confirmation_steps: int = 3,
        on_site_lower_delta_m: float = -0.20,
        on_site_upper_delta_m: float = 0.10,
        on_front_x_lower_m: float = 0.12,
        on_front_x_upper_m: float = 0.24,
        on_front_y_lower_m: float = -0.04,
        on_front_y_upper_m: float = 0.05,
        on_front_z_lower_m: float = -0.03,
        on_front_z_upper_m: float = 0.03,
        on_front_confirmation_steps: int = 7,
        on_spatial_min_displacement_m: float = 0.03,
        on_spatial_stability_tolerance_m: float = 0.003,
        on_right_min_displacement_m: float = 0.135,
        on_right_released_min_displacement_m: float = 0.10,
        on_right_interior_margin_m: float = 0.015,
        on_right_recovery_slack_m: float = 0.015,
        on_right_recovery_min_displacement_m: float = 0.16,
        on_right_released_slack_m: float = 0.02,
        on_right_release_max_age_steps: int = 30,
        on_right_lateral_padding_m: float = 0.01,
        on_right_max_offset_m: float = 0.06,
        on_right_require_release: bool = True,
        on_right_pick_proxy: bool = True,
        on_right_confirmation_steps: int = 5,
        on_attach_width_m: float = 0.05,
        on_attach_confirmation_steps: int = 2,
        on_release_width_m: float = 0.06,
        on_release_hold_steps: int = 30,
        on_support_cache_steps: int = 0,
        on_require_fresh_after_release: bool = False,
        on_contact_tolerance_m: float = 0.0012,
        in_enabled: bool = False,
        in_confirmation_steps: int = 3,
        in_reacquire_gate_m: float = 0.30,
        in_release_history_steps: int = 5,
        in_release_delta_m: float = 0.003,
        in_positive_hold_steps: int = 1,
        in_caddy_release_separation_m: float = 0.10,
        in_caddy_release_confirmation_steps: int = 3,
        in_caddy_release_max_age_steps: int = 60,
        in_caddy_fresh_entry_tolerance_m: float = 0.03,
        in_caddy_opening_tolerance_m: float = 0.001,
        in_caddy_fresh_min_core_lateral_margin_m: float = 0.04,
        in_caddy_fresh_release_max_age_steps: int = 90,
        in_caddy_max_target_displacement_m: float = 0.04,
        in_caddy_settled_confirmation_age_steps: int = 60,
        in_caddy_latched_occlusion_hold_steps: int = 12,
        in_caddy_proxy_entry_tolerance_m: float = 0.03,
        in_caddy_proxy_opening_confirmation_steps: int = 2,
        in_caddy_proxy_release_max_age_steps: int = 90,
        in_caddy_unreleased_ceiling_m: float = 0.001,
        in_caddy_contradiction_margin_m: float = 0.03,
        in_caddy_contradiction_confirmation_steps: int = 3,
        in_config: VisualInConfig | None = None,
        open_enabled: bool = False,
        open_confirmation_steps: int = 3,
        open_hold_steps: int = 3,
        open_config: VisualOpenConfig | None = None,
        close_enabled: bool = False,
        close_require_in_confirmed: bool = False,
        close_confirmation_steps: int = 3,
        close_hold_steps: int = 3,
        close_config: VisualCloseConfig | None = None,
        close_panel_diagnostic: bool = False,
        close_panel_diagnostic_interval: int = 10,
        close_mode: str = "legacy_pointcloud",
        close_panel_config: VisualDrawerCloseConfig | None = None,
        microwave_close_config: VisualMicrowaveCloseConfig | None = None,
        close_microwave_settled_progress_threshold: float = 0.75,
        close_microwave_settled_confirmation_steps: int = 30,
        turnon_enabled: bool = False,
        turnon_config: VisualTurnOnConfig | None = None,
    ) -> None:
        self.client = YoloeClient(
            yoloe_url,
            timeout=float(yoloe_timeout_s),
            raw_transport=bool(yoloe_raw_transport),
        )
        self.yoloe_batch_enabled = bool(yoloe_batch_enabled)
        self.yoloe_batch_max_size = int(yoloe_batch_max_size)
        if self.yoloe_batch_max_size <= 0:
            raise ValueError("yoloe_batch_max_size must be positive")
        # Cleared and repopulated once per EnvGroup step. Values are raw
        # detector outputs, before per-atom confidence or geometry filtering.
        self._detection_cache = {}
        self._camera_estimate_cache_key = None
        self._camera_estimate_cache_value = None
        self.camera_name = camera_name
        self.fallback_camera_name = (
            None
            if fallback_camera_name in {None, "", camera_name}
            else str(fallback_camera_name)
        )
        self.mask_to_depth_orientation = str(
            mask_to_depth_orientation
        ).lower()
        if self.mask_to_depth_orientation not in {
            "rot180", "identity", "vflip", "hflip"
        }:
            raise ValueError(
                "mask_to_depth_orientation must be one of "
                "rot180, identity, vflip, hflip"
            )
        self.grasp_radius_m = float(grasp_radius_m)
        self.visual_center_radius_m = (
            self.grasp_radius_m
            if visual_center_radius_m is None
            else float(visual_center_radius_m)
        )
        self.surface_radius_m = float(surface_radius_m)

        if self.visual_center_radius_m <= 0.0:
            raise ValueError(
                "visual_center_radius_m must be positive"
            )
        self.min_confidence = float(min_confidence)
        self.fallback_min_confidence = float(
            min_confidence
            if fallback_min_confidence is None
            else fallback_min_confidence
        )
        self.track_gate_m = float(track_gate_m)
        self.track_ema = float(track_ema)
        self.close_threshold_m = float(close_threshold_m)
        self.comove_tolerance_m = float(
            comove_tolerance_m
        )
        self.motion_min_m = float(motion_min_m)
        self.displacement_min_m = float(displacement_min_m)
        self.hold_steps = int(hold_steps)
        self.grasp_near_mode = str(
            grasp_near_mode
        ).strip().lower()
        self.grasp_history_lag_steps = int(
            grasp_history_lag_steps
        )
        self.grasp_confirmation_steps = int(
            grasp_confirmation_steps
        )
        self.grasp_terminal_width_m = float(
            grasp_terminal_width_m
        )
        self.grasp_terminal_comove_slack_m = float(
            grasp_terminal_comove_slack_m
        )
        self.grasp_terminal_motion_slack_m = float(
            grasp_terminal_motion_slack_m
        )
        self.grasp_terminal_confirmation_steps = int(
            grasp_terminal_confirmation_steps
        )
        self.on_xy_tolerance_m = float(on_xy_tolerance_m)
        self.on_above_epsilon_m = float(on_above_epsilon_m)
        self.on_surface_tolerance_m = float(
            on_surface_tolerance_m
        )
        self.on_ordinary_xy_tolerance_m = float(
            on_ordinary_xy_tolerance_m
        )
        self.on_ordinary_xy_uncertainty_m = float(
            on_ordinary_xy_uncertainty_m
        )
        self.on_ordinary_completion_xy_uncertainty_m = float(
            on_ordinary_completion_xy_uncertainty_m
        )
        self.on_ordinary_min_above_m = float(
            on_ordinary_min_above_m
        )
        self.on_ordinary_surface_tolerance_m = float(
            on_ordinary_surface_tolerance_m
        )
        self.on_ordinary_require_release = bool(
            on_ordinary_require_release
        )
        self.on_ordinary_require_contact = bool(
            on_ordinary_require_contact
        )
        self.on_ordinary_contact_tolerance_m = float(
            on_ordinary_contact_tolerance_m
        )
        self.on_ordinary_confirmation_steps = int(
            on_ordinary_confirmation_steps
        )
        self.on_ordinary_near_center_fallback_enabled = bool(
            on_ordinary_near_center_fallback_enabled
        )
        self.on_ordinary_near_center_fallback_xy_m = float(
            on_ordinary_near_center_fallback_xy_m
        )
        self.on_ordinary_near_center_fallback_contact_p01_m = float(
            on_ordinary_near_center_fallback_contact_p01_m
        )
        self.on_ordinary_near_center_fallback_confirmation_steps = int(
            on_ordinary_near_center_fallback_confirmation_steps
        )
        self.on_confirmation_steps = int(on_confirmation_steps)
        self.on_site_lower_delta_m = float(on_site_lower_delta_m)
        self.on_site_upper_delta_m = float(on_site_upper_delta_m)
        self.on_front_bounds_m = (
            float(on_front_x_lower_m),
            float(on_front_x_upper_m),
            float(on_front_y_lower_m),
            float(on_front_y_upper_m),
            float(on_front_z_lower_m),
            float(on_front_z_upper_m),
        )
        self.on_front_confirmation_steps = int(
            on_front_confirmation_steps
        )
        self.on_spatial_min_displacement_m = float(
            on_spatial_min_displacement_m
        )
        self.on_spatial_stability_tolerance_m = float(
            on_spatial_stability_tolerance_m
        )
        self.on_right_min_displacement_m = float(
            on_right_min_displacement_m
        )
        self.on_right_released_min_displacement_m = float(
            on_right_released_min_displacement_m
        )
        self.on_right_interior_margin_m = float(
            on_right_interior_margin_m
        )
        self.on_right_recovery_slack_m = float(
            on_right_recovery_slack_m
        )
        self.on_right_recovery_min_displacement_m = float(
            on_right_recovery_min_displacement_m
        )
        self.on_right_released_slack_m = float(
            on_right_released_slack_m
        )
        self.on_right_release_max_age_steps = int(
            on_right_release_max_age_steps
        )
        self.on_right_lateral_padding_m = float(
            on_right_lateral_padding_m
        )
        self.on_right_max_offset_m = float(on_right_max_offset_m)
        self.on_right_require_release = bool(
            on_right_require_release
        )
        self.on_right_pick_proxy = bool(on_right_pick_proxy)
        self.on_right_confirmation_steps = int(
            on_right_confirmation_steps
        )
        self.on_attach_width_m = float(on_attach_width_m)
        self.on_attach_confirmation_steps = int(
            on_attach_confirmation_steps
        )
        self.on_release_width_m = float(on_release_width_m)
        self.on_release_hold_steps = int(on_release_hold_steps)
        self.on_support_cache_steps = int(on_support_cache_steps)
        self.on_require_fresh_after_release = bool(
            on_require_fresh_after_release
        )
        self.on_contact_tolerance_m = float(
            on_contact_tolerance_m
        )
        if self.on_site_lower_delta_m >= self.on_site_upper_delta_m:
            raise ValueError(
                "on_site_lower_delta_m must be below "
                "on_site_upper_delta_m"
            )
        for relation, bounds in (
            ("front", self.on_front_bounds_m),
        ):
            for lower_index, upper_index, label in (
                (0, 1, "x"),
                (2, 3, "y"),
                (4, 5, "z"),
            ):
                if bounds[lower_index] >= bounds[upper_index]:
                    raise ValueError(
                        f"on_{relation}_{label}_lower_m must be below "
                        f"on_{relation}_{label}_upper_m"
                    )
        self.in_enabled = bool(in_enabled)
        self.in_confirmation_steps = int(in_confirmation_steps)
        self.in_reacquire_gate_m = float(in_reacquire_gate_m)
        self.in_release_history_steps = int(in_release_history_steps)
        self.in_release_delta_m = float(in_release_delta_m)
        self.in_positive_hold_steps = int(in_positive_hold_steps)
        self.in_caddy_release_separation_m = float(
            in_caddy_release_separation_m
        )
        self.in_caddy_release_confirmation_steps = int(
            in_caddy_release_confirmation_steps
        )
        self.in_caddy_release_max_age_steps = int(
            in_caddy_release_max_age_steps
        )
        self.in_caddy_fresh_entry_tolerance_m = float(
            in_caddy_fresh_entry_tolerance_m
        )
        self.in_caddy_opening_tolerance_m = float(
            in_caddy_opening_tolerance_m
        )
        self.in_caddy_fresh_min_core_lateral_margin_m = float(
            in_caddy_fresh_min_core_lateral_margin_m
        )
        self.in_caddy_fresh_release_max_age_steps = int(
            in_caddy_fresh_release_max_age_steps
        )
        self.in_caddy_max_target_displacement_m = float(
            in_caddy_max_target_displacement_m
        )
        self.in_caddy_settled_confirmation_age_steps = int(
            in_caddy_settled_confirmation_age_steps
        )
        self.in_caddy_latched_occlusion_hold_steps = int(
            in_caddy_latched_occlusion_hold_steps
        )
        self.in_caddy_proxy_entry_tolerance_m = float(
            in_caddy_proxy_entry_tolerance_m
        )
        self.in_caddy_proxy_opening_confirmation_steps = int(
            in_caddy_proxy_opening_confirmation_steps
        )
        self.in_caddy_proxy_release_max_age_steps = int(
            in_caddy_proxy_release_max_age_steps
        )
        self.in_caddy_unreleased_ceiling_m = float(
            in_caddy_unreleased_ceiling_m
        )
        self.in_caddy_contradiction_margin_m = float(
            in_caddy_contradiction_margin_m
        )
        self.in_caddy_contradiction_confirmation_steps = int(
            in_caddy_contradiction_confirmation_steps
        )
        self.in_config = in_config or VisualInConfig()
        self.open_enabled = bool(open_enabled)
        self.open_confirmation_steps = int(open_confirmation_steps)
        self.open_hold_steps = int(open_hold_steps)
        self.open_config = open_config or VisualOpenConfig()
        self.close_enabled = bool(close_enabled)
        self.close_require_in_confirmed = bool(
            close_require_in_confirmed
        )
        self.close_confirmation_steps = int(close_confirmation_steps)
        self.close_hold_steps = int(close_hold_steps)
        self.close_config = close_config or VisualCloseConfig()
        self.close_panel_diagnostic = bool(close_panel_diagnostic)
        self.close_panel_diagnostic_interval = int(
            close_panel_diagnostic_interval
        )
        self.close_mode = str(close_mode).strip().lower()
        self.close_panel_config = (
            close_panel_config or VisualDrawerCloseConfig()
        )
        self.microwave_close_config = (
            microwave_close_config or VisualMicrowaveCloseConfig()
        )
        self.close_microwave_settled_progress_threshold = float(
            close_microwave_settled_progress_threshold
        )
        self.close_microwave_settled_confirmation_steps = int(
            close_microwave_settled_confirmation_steps
        )
        self.turnon_enabled = bool(turnon_enabled)
        self.turnon_config = turnon_config or VisualTurnOnConfig()

        if self.track_gate_m <= 0.0:
            raise ValueError("track_gate_m must be positive")
        if not 0.0 <= self.track_ema < 1.0:
            raise ValueError(
                "track_ema must be in [0, 1)"
            )
        if self.close_threshold_m <= 0.0:
            raise ValueError(
                "close_threshold_m must be positive"
            )
        if self.comove_tolerance_m <= 0.0:
            raise ValueError(
                "comove_tolerance_m must be positive"
            )
        if self.motion_min_m < 0.0:
            raise ValueError(
                "motion_min_m must be non-negative"
            )
        if self.displacement_min_m < 0.0:
            raise ValueError(
                "displacement_min_m must be non-negative"
            )
        if self.hold_steps < 0:
            raise ValueError("hold_steps must be non-negative")
        if self.grasp_near_mode not in {
            "center",
            "surface_p01",
            "surface_p05",
            "surface_p10",
        }:
            raise ValueError(
                "grasp_near_mode must be center, surface_p01, "
                "surface_p05, or surface_p10"
            )
        if self.grasp_history_lag_steps < 1:
            raise ValueError(
                "grasp_history_lag_steps must be >= 1"
            )
        if self.grasp_confirmation_steps < 1:
            raise ValueError(
                "grasp_confirmation_steps must be >= 1"
            )
        if self.grasp_terminal_width_m <= 0.0:
            raise ValueError(
                "grasp_terminal_width_m must be positive"
            )
        if self.grasp_terminal_comove_slack_m < 0.0:
            raise ValueError(
                "grasp_terminal_comove_slack_m must be non-negative"
            )
        if self.grasp_terminal_motion_slack_m < 0.0:
            raise ValueError(
                "grasp_terminal_motion_slack_m must be non-negative"
            )
        if self.grasp_terminal_confirmation_steps < 1:
            raise ValueError(
                "grasp_terminal_confirmation_steps must be >= 1"
            )
        if self.on_xy_tolerance_m <= 0.0:
            raise ValueError("on_xy_tolerance_m must be positive")
        if self.on_surface_tolerance_m <= 0.0:
            raise ValueError(
                "on_surface_tolerance_m must be positive"
            )
        if self.on_ordinary_xy_tolerance_m <= 0.0:
            raise ValueError(
                "on_ordinary_xy_tolerance_m must be positive"
            )
        if self.on_ordinary_xy_uncertainty_m < 0.0:
            raise ValueError(
                "on_ordinary_xy_uncertainty_m must be non-negative"
            )
        if self.on_ordinary_completion_xy_uncertainty_m < 0.0:
            raise ValueError(
                "on_ordinary_completion_xy_uncertainty_m must be "
                "non-negative"
            )
        if (
            self.on_ordinary_completion_xy_uncertainty_m
            > self.on_ordinary_xy_uncertainty_m
        ):
            raise ValueError(
                "on_ordinary_completion_xy_uncertainty_m must not exceed "
                "on_ordinary_xy_uncertainty_m"
            )
        if not np.isfinite(self.on_ordinary_min_above_m):
            raise ValueError(
                "on_ordinary_min_above_m must be finite"
            )
        if self.on_ordinary_surface_tolerance_m <= 0.0:
            raise ValueError(
                "on_ordinary_surface_tolerance_m must be positive"
            )
        if self.on_ordinary_contact_tolerance_m <= 0.0:
            raise ValueError(
                "on_ordinary_contact_tolerance_m must be positive"
            )
        if self.on_ordinary_confirmation_steps < 1:
            raise ValueError(
                "on_ordinary_confirmation_steps must be >= 1"
            )
        if self.on_ordinary_near_center_fallback_xy_m <= 0.0:
            raise ValueError(
                "on_ordinary_near_center_fallback_xy_m must be positive"
            )
        if self.on_ordinary_near_center_fallback_contact_p01_m <= 0.0:
            raise ValueError(
                "on_ordinary_near_center_fallback_contact_p01_m must be positive"
            )
        if self.on_ordinary_near_center_fallback_confirmation_steps < 1:
            raise ValueError(
                "on_ordinary_near_center_fallback_confirmation_steps must be >= 1"
            )
        if self.on_confirmation_steps < 1:
            raise ValueError("on_confirmation_steps must be >= 1")
        if self.on_front_confirmation_steps < 1:
            raise ValueError(
                "on_front_confirmation_steps must be >= 1"
            )
        if self.on_right_confirmation_steps < 1:
            raise ValueError(
                "on_right_confirmation_steps must be >= 1"
            )
        if self.on_spatial_min_displacement_m < 0.0:
            raise ValueError(
                "on_spatial_min_displacement_m must be non-negative"
            )
        if self.on_spatial_stability_tolerance_m <= 0.0:
            raise ValueError(
                "on_spatial_stability_tolerance_m must be positive"
            )
        if self.on_right_min_displacement_m < 0.0:
            raise ValueError(
                "on_right_min_displacement_m must be non-negative"
            )
        if not (
            0.0 <= self.on_right_released_min_displacement_m
            <= self.on_right_min_displacement_m
        ):
            raise ValueError(
                "on_right_released_min_displacement_m must be between "
                "zero and on_right_min_displacement_m"
            )
        if self.on_right_interior_margin_m < 0.0:
            raise ValueError(
                "on_right_interior_margin_m must be non-negative"
            )
        if self.on_right_recovery_slack_m < 0.0:
            raise ValueError(
                "on_right_recovery_slack_m must be non-negative"
            )
        if (
            self.on_right_recovery_min_displacement_m
            < self.on_right_min_displacement_m
        ):
            raise ValueError(
                "on_right_recovery_min_displacement_m must be at least "
                "on_right_min_displacement_m"
            )
        if self.on_right_released_slack_m < 0.0:
            raise ValueError(
                "on_right_released_slack_m must be non-negative"
            )
        if self.on_right_release_max_age_steps < 0:
            raise ValueError(
                "on_right_release_max_age_steps must be non-negative"
            )
        if self.on_right_lateral_padding_m < 0.0:
            raise ValueError(
                "on_right_lateral_padding_m must be non-negative"
            )
        if self.on_right_max_offset_m <= 0.0:
            raise ValueError(
                "on_right_max_offset_m must be positive"
            )
        if self.on_attach_width_m <= 0.0:
            raise ValueError("on_attach_width_m must be positive")
        if self.on_attach_confirmation_steps < 1:
            raise ValueError(
                "on_attach_confirmation_steps must be >= 1"
            )
        if self.on_release_width_m <= self.on_attach_width_m:
            raise ValueError(
                "on_release_width_m must exceed on_attach_width_m"
            )
        if self.on_release_hold_steps < 1:
            raise ValueError("on_release_hold_steps must be >= 1")
        if self.on_support_cache_steps < 0:
            raise ValueError(
                "on_support_cache_steps must be non-negative"
            )
        if self.on_contact_tolerance_m <= 0.0:
            raise ValueError(
                "on_contact_tolerance_m must be positive"
            )
        if self.in_confirmation_steps < 1:
            raise ValueError("in_confirmation_steps must be >= 1")
        if self.in_reacquire_gate_m <= self.track_gate_m:
            raise ValueError(
                "in_reacquire_gate_m must exceed track_gate_m"
            )
        if self.in_release_history_steps < 1:
            raise ValueError("in_release_history_steps must be >= 1")
        if self.in_release_delta_m <= 0.0:
            raise ValueError("in_release_delta_m must be positive")
        if self.in_positive_hold_steps < 0:
            raise ValueError("in_positive_hold_steps must be non-negative")
        if self.in_caddy_release_separation_m <= 0.0:
            raise ValueError(
                "in_caddy_release_separation_m must be positive"
            )
        if self.in_caddy_release_confirmation_steps < 1:
            raise ValueError(
                "in_caddy_release_confirmation_steps must be >= 1"
            )
        if self.in_caddy_release_max_age_steps < 1:
            raise ValueError(
                "in_caddy_release_max_age_steps must be >= 1"
            )
        if self.in_caddy_fresh_entry_tolerance_m < 0.0:
            raise ValueError(
                "in_caddy_fresh_entry_tolerance_m must be non-negative"
            )
        if self.in_caddy_opening_tolerance_m < 0.0:
            raise ValueError(
                "in_caddy_opening_tolerance_m must be non-negative"
            )
        if self.in_caddy_fresh_min_core_lateral_margin_m < 0.0:
            raise ValueError(
                "in_caddy_fresh_min_core_lateral_margin_m must be "
                "non-negative"
            )
        if (
            self.in_caddy_fresh_release_max_age_steps
            < self.in_caddy_release_max_age_steps
        ):
            raise ValueError(
                "in_caddy_fresh_release_max_age_steps must be at least "
                "in_caddy_release_max_age_steps"
            )
        if self.in_caddy_max_target_displacement_m <= 0.0:
            raise ValueError(
                "in_caddy_max_target_displacement_m must be positive"
            )
        if self.in_caddy_settled_confirmation_age_steps < 1:
            raise ValueError(
                "in_caddy_settled_confirmation_age_steps must be >= 1"
            )
        if (
            self.in_caddy_settled_confirmation_age_steps
            > self.in_caddy_fresh_release_max_age_steps
        ):
            raise ValueError(
                "in_caddy_settled_confirmation_age_steps must not exceed "
                "in_caddy_fresh_release_max_age_steps"
            )
        if self.in_caddy_latched_occlusion_hold_steps < 0:
            raise ValueError(
                "in_caddy_latched_occlusion_hold_steps must be non-negative"
            )
        if self.in_caddy_proxy_entry_tolerance_m < 0.0:
            raise ValueError(
                "in_caddy_proxy_entry_tolerance_m must be non-negative"
            )
        if self.in_caddy_proxy_opening_confirmation_steps < 1:
            raise ValueError(
                "in_caddy_proxy_opening_confirmation_steps must be >= 1"
            )
        if (
            self.in_caddy_proxy_release_max_age_steps
            < self.in_caddy_release_max_age_steps
        ):
            raise ValueError(
                "in_caddy_proxy_release_max_age_steps must be at least "
                "in_caddy_release_max_age_steps"
            )
        if self.in_caddy_unreleased_ceiling_m <= 0.0:
            raise ValueError(
                "in_caddy_unreleased_ceiling_m must be positive"
            )
        if self.in_caddy_contradiction_margin_m <= 0.0:
            raise ValueError(
                "in_caddy_contradiction_margin_m must be positive"
            )
        if self.in_caddy_contradiction_confirmation_steps < 1:
            raise ValueError(
                "in_caddy_contradiction_confirmation_steps must be >= 1"
            )
        if self.in_config.microwave_entry_tolerance_m < 0.0:
            raise ValueError(
                "microwave_entry_tolerance_m must be non-negative"
            )
        if not 0.5 < self.in_config.microwave_front_quantile < 1.0:
            raise ValueError(
                "microwave_front_quantile must be between 0.5 and 1.0"
            )
        if self.in_config.microwave_min_entry_depth_m < 0.0:
            raise ValueError(
                "microwave_min_entry_depth_m must be non-negative"
            )
        if (
            self.in_config.microwave_release_max_target_eef_distance_m
            <= 0.0
        ):
            raise ValueError(
                "microwave_release_max_target_eef_distance_m must be positive"
            )
        if self.in_config.microwave_occlusion_steps < 1:
            raise ValueError("microwave_occlusion_steps must be >= 1")
        if self.in_config.microwave_release_max_age_steps < 1:
            raise ValueError(
                "microwave_release_max_age_steps must be >= 1"
            )
        if self.open_confirmation_steps < 1:
            raise ValueError("open_confirmation_steps must be >= 1")
        if self.open_hold_steps < 0:
            raise ValueError("open_hold_steps must be non-negative")
        if self.close_confirmation_steps < 1:
            raise ValueError("close_confirmation_steps must be >= 1")
        if self.close_hold_steps < 0:
            raise ValueError("close_hold_steps must be non-negative")
        if not (
            0.0 < self.close_microwave_settled_progress_threshold
            < self.microwave_close_config.door_close_progress_threshold
        ):
            raise ValueError(
                "close_microwave_settled_progress_threshold must be positive "
                "and below the strict microwave close threshold"
            )
        if self.close_microwave_settled_confirmation_steps < 1:
            raise ValueError(
                "close_microwave_settled_confirmation_steps must be >= 1"
            )
        if self.close_panel_diagnostic_interval < 1:
            raise ValueError(
                "close_panel_diagnostic_interval must be >= 1"
            )
        if self.close_mode not in {
            "legacy_pointcloud",
            "fixed_panel_edge",
            "microwave_silhouette",
            "articulated_geometry",
        }:
            raise ValueError(
                "close_mode must be legacy_pointcloud, fixed_panel_edge, "
                "microwave_silhouette, or articulated_geometry"
            )

        self.calibrations: dict[int, dict] = {}
        self.fallback_calibrations: dict[int, dict] = {}
        self.fallback_camera_in_eef: dict[int, np.ndarray] = {}
        self.target_tracks: dict[int, np.ndarray] = {}
        self.grasp_reference_centers: dict[int, np.ndarray] = {}
        self.last_fresh_records: dict[
            int,
            VisualSTLShadowRecord,
        ] = {}
        self.last_fresh_steps: dict[int, int] = {}
        self.grasp_histories: dict[
            int,
            list[tuple[int, np.ndarray, np.ndarray]],
        ] = {}
        self.grasp_positive_counts: dict[int, int] = {}
        self.grasp_terminal_positive_counts: dict[int, int] = {}
        self.grasp_terminal_confirmed: dict[int, bool] = {}
        self.grasp_confirmation_sources: dict[int, str] = {}
        self.grasp_confirmed: dict[int, bool] = {}
        self.on_positive_counts: dict[int, int] = {}
        self.on_ordinary_near_center_fallback_counts: dict[int, int] = {}
        self.on_confirmation_sources: dict[int, str] = {}
        self.on_confirmed: dict[int, bool] = {}
        self.on_attach_counts: dict[int, int] = {}
        self.on_object_states: dict[int, dict] = {}
        self.on_release_observed: dict[int, bool] = {}
        self.on_release_steps: dict[int, int] = {}
        self.on_post_release_fresh: dict[int, bool] = {}
        self.on_container_geometry_positive_counts: dict[int, int] = {}
        self.on_container_geometry_last_steps: dict[int, int] = {}
        self.on_container_last_positive: dict[int, dict] = {}
        self.on_support_states: dict[
            tuple[int, str], dict
        ] = {}
        self.on_spatial_initial_centers: dict[int, np.ndarray] = {}
        self.on_spatial_previous_centers: dict[int, np.ndarray] = {}
        self.on_spatial_motion_observed: dict[int, bool] = {}
        self.records: dict[
            int,
            list[VisualSTLShadowRecord],
        ] = {}
        self.in_records: dict[int, list[VisualInShadowRecord]] = {}
        self.in_positive_counts: dict[int, int] = {}
        self.in_confirmed: dict[int, bool] = {}
        self.in_width_histories: dict[int, list[float]] = {}
        self.in_release_latched: dict[int, bool] = {}
        self.in_release_evidence: dict[int, float] = {}
        self.in_release_steps: dict[int, int] = {}
        self.in_release_sources: dict[int, str] = {}
        self.in_caddy_release_counts: dict[int, int] = {}
        self.in_caddy_dual_stage_release_counts: dict[int, int] = {}
        self.in_caddy_opening_counts: dict[int, int] = {}
        self.in_caddy_topology_temporal_counts: dict[int, int] = {}
        self.in_caddy_central_temporal_counts: dict[int, int] = {}
        self.in_caddy_cached_topology_half_widths: dict[
            int, list[float]
        ] = {}
        self.in_caddy_cached_topology_counts: dict[int, int] = {}
        self.in_caddy_static_baseline_points: dict[int, np.ndarray] = {}
        self.in_caddy_static_baseline_centers: dict[int, np.ndarray] = {}
        self.in_caddy_static_baseline_cameras: dict[int, str] = {}
        self.in_caddy_static_baseline_temporal_counts: dict[int, int] = {}
        self.in_caddy_reobservation_last_steps: dict[int, int] = {}
        self.in_caddy_reobservation_bouts: dict[int, int] = {}
        self.in_caddy_reobservation_margins: dict[int, float] = {}
        self.in_caddy_confirmed_event_bridge_margins: dict[int, float] = {}
        self.in_caddy_confirmed_event_bridge_steps: dict[int, int] = {}
        self.in_caddy_opening_steps: dict[int, int] = {}
        self.in_caddy_opening_centers: dict[int, np.ndarray] = {}
        self.in_caddy_opening_max_target_displacements: dict[int, float] = {}
        self.in_caddy_opening_evidence: dict[int, float] = {}
        self.in_caddy_opening_geometry_margins: dict[int, float] = {}
        self.in_caddy_opening_sources: dict[int, str] = {}
        self.in_caddy_opening_candidate_sources: dict[int, str] = {}
        self.in_caddy_observed_opening_steps: dict[int, int] = {}
        self.in_caddy_observed_opening_centers: dict[int, np.ndarray] = {}
        self.in_caddy_observed_opening_evidence: dict[int, float] = {}
        self.in_caddy_observed_opening_geometry_margins: dict[int, float] = {}
        self.in_caddy_observed_opening_sources: dict[int, str] = {}
        self.in_caddy_separation_steps: dict[int, int] = {}
        self.in_caddy_separation_evidence: dict[int, float] = {}
        self.in_caddy_contradiction_counts: dict[int, int] = {}
        self.in_caddy_probe_frames: dict[int, list[dict]] = {}
        self.in_microwave_baseline_points: dict[int, np.ndarray] = {}
        self.in_attached_points_eef: dict[int, np.ndarray] = {}
        self.in_attached_confidence: dict[int, float] = {}
        self.in_last_positive: dict[int, VisualInShadowRecord] = {}
        self.in_last_fresh: dict[int, VisualInShadowRecord] = {}
        self.in_hold_ages: dict[int, int] = {}
        self.open_records: dict[int, list[VisualOpenShadowRecord]] = {}
        self.open_baselines: dict[int, np.ndarray] = {}
        self.open_baseline_centers: dict[int, np.ndarray] = {}
        self.open_positive_counts: dict[int, int] = {}
        self.open_confirmed: dict[int, bool] = {}
        self.open_last_fresh: dict[int, VisualOpenShadowRecord] = {}
        self.open_last_fresh_steps: dict[int, int] = {}
        self.close_records: dict[int, list[VisualCloseShadowRecord]] = {}
        self.close_baselines: dict[int, np.ndarray] = {}
        self.close_baseline_centers: dict[int, np.ndarray] = {}
        self.close_track_centers: dict[int, np.ndarray] = {}
        self.close_positive_counts: dict[int, int] = {}
        self.close_confirmed: dict[int, bool] = {}
        self.close_last_fresh: dict[int, VisualCloseShadowRecord] = {}
        self.close_last_fresh_steps: dict[int, int] = {}
        self.close_panel_frames: dict[int, list[dict]] = {}
        self.close_panel_last_positive: dict[int, bool] = {}
        self.close_panel_fixed_rois: dict[int, object] = {}
        self.close_panel_baseline_masks: dict[int, np.ndarray] = {}
        self.close_microwave_settled_counts: dict[int, int] = {}
        self.microwave_fixed_rois: dict[int, object] = {}
        self.microwave_baseline_masks: dict[int, np.ndarray] = {}
        self.microwave_baseline_images: dict[int, np.ndarray] = {}
        self.microwave_baseline_depths: dict[int, np.ndarray] = {}
        self.turnon_records: dict[int, list[VisualTurnOnShadowRecord]] = {}
        self.turnon_rois: dict[int, object] = {}
        self.turnon_baseline_rgb: dict[int, np.ndarray] = {}
        self.turnon_baseline_depth: dict[int, np.ndarray] = {}
        self.turnon_frames: dict[int, list[dict]] = {}
        self.turnon_positive_counts: dict[int, int] = {}
        self.turnon_confirmed: dict[int, bool] = {}

    def reset_env(
        self,
        env_id: int,
        calibration: dict,
        fallback_calibration: dict | None = None,
        raw_obs: dict | None = None,
    ) -> None:
        env_id = int(env_id)
        self.calibrations[env_id] = calibration
        self.fallback_calibrations.pop(env_id, None)
        self.fallback_camera_in_eef.pop(env_id, None)
        if (
            self.fallback_camera_name is not None
            and fallback_calibration is not None
            and fallback_calibration.get("valid", False)
            and raw_obs is not None
        ):
            self.fallback_calibrations[env_id] = dict(
                fallback_calibration
            )
            world_from_eef = self._pose_matrix(
                np.asarray(raw_obs["robot0_eef_pos"]),
                self._eef_rotation(raw_obs),
            )
            world_from_camera = np.asarray(
                fallback_calibration["camera_to_world"],
                dtype=np.float64,
            ).reshape(4, 4)
            self.fallback_camera_in_eef[env_id] = (
                np.linalg.inv(world_from_eef)
                @ world_from_camera
            )
        self.records[env_id] = []
        self.in_records[env_id] = []
        self.open_records[env_id] = []
        self.close_records[env_id] = []
        self.close_panel_frames[env_id] = []
        self.close_panel_last_positive.pop(env_id, None)
        self.close_panel_fixed_rois.pop(env_id, None)
        self.close_panel_baseline_masks.pop(env_id, None)
        self.microwave_fixed_rois.pop(env_id, None)
        self.microwave_baseline_masks.pop(env_id, None)
        self.microwave_baseline_images.pop(env_id, None)
        self.microwave_baseline_depths.pop(env_id, None)
        self.close_microwave_settled_counts[env_id] = 0
        self.turnon_records[env_id] = []
        self.turnon_rois.pop(env_id, None)
        self.turnon_baseline_rgb.pop(env_id, None)
        self.turnon_baseline_depth.pop(env_id, None)
        self.turnon_frames[env_id] = []
        self.turnon_positive_counts[env_id] = 0
        self.turnon_confirmed[env_id] = False
        self.in_positive_counts[env_id] = 0
        self.in_confirmed[env_id] = False
        self.in_width_histories[env_id] = []
        self.in_release_latched[env_id] = False
        self.in_release_evidence[env_id] = float("nan")
        self.in_release_steps.pop(env_id, None)
        self.in_release_sources.pop(env_id, None)
        self.in_caddy_release_counts[env_id] = 0
        self.in_caddy_dual_stage_release_counts[env_id] = 0
        self.in_caddy_opening_counts[env_id] = 0
        self.in_caddy_topology_temporal_counts[env_id] = 0
        self.in_caddy_central_temporal_counts[env_id] = 0
        self.in_caddy_cached_topology_half_widths[env_id] = []
        self.in_caddy_cached_topology_counts[env_id] = 0
        self.in_caddy_static_baseline_points.pop(env_id, None)
        self.in_caddy_static_baseline_centers.pop(env_id, None)
        self.in_caddy_static_baseline_cameras.pop(env_id, None)
        self.in_caddy_static_baseline_temporal_counts[env_id] = 0
        self.in_caddy_reobservation_last_steps.pop(env_id, None)
        self.in_caddy_reobservation_bouts[env_id] = 0
        self.in_caddy_reobservation_margins.pop(env_id, None)
        self.in_caddy_confirmed_event_bridge_margins.pop(env_id, None)
        self.in_caddy_confirmed_event_bridge_steps.pop(env_id, None)
        self.in_caddy_opening_steps.pop(env_id, None)
        self.in_caddy_opening_centers.pop(env_id, None)
        self.in_caddy_opening_max_target_displacements.pop(env_id, None)
        self.in_caddy_opening_evidence.pop(env_id, None)
        self.in_caddy_opening_geometry_margins.pop(env_id, None)
        self.in_caddy_opening_sources.pop(env_id, None)
        self.in_caddy_opening_candidate_sources.pop(env_id, None)
        self.in_caddy_observed_opening_steps.pop(env_id, None)
        self.in_caddy_observed_opening_centers.pop(env_id, None)
        self.in_caddy_observed_opening_evidence.pop(env_id, None)
        self.in_caddy_observed_opening_geometry_margins.pop(env_id, None)
        self.in_caddy_observed_opening_sources.pop(env_id, None)
        self.in_caddy_separation_steps.pop(env_id, None)
        self.in_caddy_separation_evidence.pop(env_id, None)
        self.in_caddy_contradiction_counts[env_id] = 0
        self.in_caddy_probe_frames[env_id] = []
        self.in_microwave_baseline_points.pop(env_id, None)
        self.in_attached_points_eef.pop(env_id, None)
        self.in_attached_confidence.pop(env_id, None)
        self.in_last_positive.pop(env_id, None)
        self.in_last_fresh.pop(env_id, None)
        self.in_hold_ages[env_id] = 0
        self.open_baselines.pop(env_id, None)
        self.open_baseline_centers.pop(env_id, None)
        self.open_positive_counts[env_id] = 0
        self.open_confirmed[env_id] = False
        self.open_last_fresh.pop(env_id, None)
        self.open_last_fresh_steps.pop(env_id, None)
        self.close_baselines.pop(env_id, None)
        self.close_baseline_centers.pop(env_id, None)
        self.close_track_centers.pop(env_id, None)
        self.close_positive_counts[env_id] = 0
        self.close_confirmed[env_id] = False
        self.close_last_fresh.pop(env_id, None)
        self.close_last_fresh_steps.pop(env_id, None)
        self.target_tracks.pop(env_id, None)
        self.grasp_reference_centers.pop(env_id, None)
        self.last_fresh_records.pop(env_id, None)
        self.last_fresh_steps.pop(env_id, None)
        self.grasp_histories[env_id] = []
        self.grasp_positive_counts[env_id] = 0
        self.grasp_terminal_positive_counts[env_id] = 0
        self.grasp_terminal_confirmed[env_id] = False
        self.grasp_confirmation_sources[env_id] = "none"
        self.grasp_confirmed[env_id] = False
        self.on_positive_counts[env_id] = 0
        self.on_ordinary_near_center_fallback_counts[env_id] = 0
        self.on_confirmation_sources[env_id] = "none"
        self.on_confirmed[env_id] = False
        self.on_attach_counts[env_id] = 0
        self.on_object_states.pop(env_id, None)
        self.on_release_observed[env_id] = False
        self.on_release_steps.pop(env_id, None)
        self.on_post_release_fresh[env_id] = False
        self.on_container_geometry_positive_counts[env_id] = 0
        self.on_container_geometry_last_steps.pop(env_id, None)
        self.on_container_last_positive.pop(env_id, None)
        self.on_spatial_initial_centers.pop(env_id, None)
        self.on_spatial_previous_centers.pop(env_id, None)
        self.on_spatial_motion_observed[env_id] = False
        for key in list(self.on_support_states):
            if key[0] == env_id:
                self.on_support_states.pop(key, None)

    def begin_atom_stage(self, env_id: int, predicate: str) -> None:
        """Reset predicate-local gates while preserving object handoff state."""
        env_id = int(env_id)
        pred = str(predicate).strip().lower()
        if pred == "pick":
            self.grasp_histories[env_id] = []
            self.grasp_positive_counts[env_id] = 0
            self.grasp_terminal_positive_counts[env_id] = 0
            self.grasp_terminal_confirmed[env_id] = False
            self.grasp_confirmation_sources[env_id] = "none"
            self.grasp_confirmed[env_id] = False
            self.grasp_reference_centers.pop(env_id, None)
            return

        if pred == "on":
            self.on_positive_counts[env_id] = 0
            self.on_ordinary_near_center_fallback_counts[env_id] = 0
            self.on_confirmation_sources[env_id] = "none"
            self.on_confirmed[env_id] = False
            self.on_release_observed[env_id] = False
            self.on_release_steps.pop(env_id, None)
            self.on_post_release_fresh[env_id] = False
            self.on_container_geometry_positive_counts[env_id] = 0
            self.on_container_geometry_last_steps.pop(env_id, None)
            self.on_container_last_positive.pop(env_id, None)
            self.on_spatial_initial_centers.pop(env_id, None)
            self.on_spatial_previous_centers.pop(env_id, None)
            self.on_spatial_motion_observed[env_id] = False
            for key in list(self.on_support_states):
                if key[0] == env_id:
                    self.on_support_states.pop(key, None)
            return

        if pred == "in":
            self.in_positive_counts[env_id] = 0
            self.in_confirmed[env_id] = False
            self.in_width_histories[env_id] = []
            self.in_release_latched[env_id] = False
            self.in_release_evidence[env_id] = float("nan")
            self.in_release_steps.pop(env_id, None)
            self.in_release_sources.pop(env_id, None)
            self.in_caddy_release_counts[env_id] = 0
            self.in_caddy_dual_stage_release_counts[env_id] = 0
            self.in_caddy_opening_counts[env_id] = 0
            self.in_caddy_topology_temporal_counts[env_id] = 0
            self.in_caddy_central_temporal_counts[env_id] = 0
            self.in_caddy_cached_topology_half_widths[env_id] = []
            self.in_caddy_cached_topology_counts[env_id] = 0
            # Keep the pre-interaction caddy geometry across Pick -> In.
            self.in_caddy_static_baseline_temporal_counts[env_id] = 0
            self.in_caddy_reobservation_last_steps.pop(env_id, None)
            self.in_caddy_reobservation_bouts[env_id] = 0
            self.in_caddy_reobservation_margins.pop(env_id, None)
            self.in_caddy_confirmed_event_bridge_margins.pop(env_id, None)
            self.in_caddy_confirmed_event_bridge_steps.pop(env_id, None)
            self.in_caddy_opening_steps.pop(env_id, None)
            self.in_caddy_opening_centers.pop(env_id, None)
            self.in_caddy_opening_max_target_displacements.pop(env_id, None)
            self.in_caddy_opening_evidence.pop(env_id, None)
            self.in_caddy_opening_geometry_margins.pop(env_id, None)
            self.in_caddy_opening_sources.pop(env_id, None)
            self.in_caddy_opening_candidate_sources.pop(env_id, None)
            self.in_caddy_observed_opening_steps.pop(env_id, None)
            self.in_caddy_observed_opening_centers.pop(env_id, None)
            self.in_caddy_observed_opening_evidence.pop(env_id, None)
            self.in_caddy_observed_opening_geometry_margins.pop(
                env_id, None
            )
            self.in_caddy_observed_opening_sources.pop(env_id, None)
            self.in_caddy_separation_steps.pop(env_id, None)
            self.in_caddy_separation_evidence.pop(env_id, None)
            self.in_caddy_contradiction_counts[env_id] = 0
            self.in_microwave_baseline_points.pop(env_id, None)
            self.in_last_positive.pop(env_id, None)
            self.in_last_fresh.pop(env_id, None)
            self.in_hold_ages[env_id] = 0

    @staticmethod
    def _pose_matrix(
        position: np.ndarray,
        rotation: np.ndarray,
    ) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = np.asarray(
            rotation, dtype=np.float64
        ).reshape(3, 3)
        matrix[:3, 3] = np.asarray(
            position, dtype=np.float64
        ).reshape(3)
        return matrix

    @staticmethod
    def _points_world_to_eef(
        points_world: np.ndarray,
        eef: np.ndarray,
        eef_rotation: np.ndarray,
    ) -> np.ndarray:
        """Express visual RGB-D points in the observable EEF frame."""
        points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        position = np.asarray(eef, dtype=np.float64).reshape(3)
        rotation = np.asarray(
            eef_rotation, dtype=np.float64
        ).reshape(3, 3)
        points = points[np.all(np.isfinite(points), axis=1)]
        return (points - position) @ rotation

    @staticmethod
    def _points_eef_to_world(
        points_eef: np.ndarray,
        eef: np.ndarray,
        eef_rotation: np.ndarray,
    ) -> np.ndarray:
        """Transform a visual-to-EEF point cache back to world coordinates."""
        points = np.asarray(points_eef, dtype=np.float64).reshape(-1, 3)
        position = np.asarray(eef, dtype=np.float64).reshape(3)
        rotation = np.asarray(
            eef_rotation, dtype=np.float64
        ).reshape(3, 3)
        return points @ rotation.T + position

    @staticmethod
    def _summarize_visual_points(points_world: np.ndarray):
        """Summarize a finite visual point cloud in world coordinates."""
        points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        points = points[np.all(np.isfinite(points), axis=1)]
        if len(points) == 0:
            return None
        xy_q05, xy_q95 = np.quantile(points[:, :2], (0.05, 0.95), axis=0)
        z_p05, z_p95 = np.quantile(points[:, 2], (0.05, 0.95))
        return {
            "points_world": points,
            "z_p05_m": float(z_p05),
            "z_p95_m": float(z_p95),
            "xy_q05_m": np.asarray(xy_q05, dtype=np.float64),
            "xy_q95_m": np.asarray(xy_q95, dtype=np.float64),
        }

    def _cache_in_attached_target(
        self,
        *,
        env_id: int,
        selected,
        eef: np.ndarray,
        eef_rotation: np.ndarray,
        attached: bool,
    ) -> None:
        """Cache segmented target geometry only after visual attachment."""
        if not self.in_enabled or not attached or selected is None:
            return
        points_eef = self._points_world_to_eef(
            selected["estimate"].points_world,
            eef,
            eef_rotation,
        )
        if len(points_eef) < self.in_config.min_object_points:
            return
        env_id = int(env_id)
        self.in_attached_points_eef[env_id] = points_eef.copy()
        self.in_attached_confidence[env_id] = float(
            selected["detection"]["confidence"]
        )

    def _in_attached_proxy_selected(
        self,
        *,
        env_id: int,
        target_class: str,
        eef: np.ndarray,
        eef_rotation: np.ndarray,
    ):
        """Return a visual-to-EEF target proxy during grasp occlusion.

        The proxy is enabled only while the independently visual attachment
        state remains active.  It never reads object pose, contacts, sites,
        joints, the privileged predicate, or the policy action.
        """
        env_id = int(env_id)
        state = self.on_object_states.get(env_id, {})
        points_eef = self.in_attached_points_eef.get(env_id)
        if not bool(state.get("attached", False)) or points_eef is None:
            return None
        points_world = self._points_eef_to_world(
            points_eef,
            eef,
            eef_rotation,
        )
        if len(points_world) < self.in_config.min_object_points:
            return None
        center = np.median(points_world, axis=0)
        return {
            "detection": {
                "class_name": str(target_class),
                "confidence": float(
                    self.in_attached_confidence.get(env_id, 0.0)
                ),
            },
            "estimate": SimpleNamespace(
                points_world=points_world,
                center_world=center,
            ),
        }

    def _current_fallback_calibration(
        self,
        env_id: int,
        raw_obs: dict,
    ) -> dict | None:
        base = self.fallback_calibrations.get(int(env_id))
        camera_in_eef = self.fallback_camera_in_eef.get(int(env_id))
        if base is None or camera_in_eef is None:
            return None

        current = dict(base)
        world_from_eef = self._pose_matrix(
            np.asarray(raw_obs["robot0_eef_pos"]),
            self._eef_rotation(raw_obs),
        )
        current["camera_to_world"] = (
            world_from_eef @ camera_in_eef
        )
        return current

    @staticmethod
    def _detection_cache_key(image_rgb: np.ndarray):
        image_rgb = np.ascontiguousarray(image_rgb, dtype=np.uint8)
        return image_rgb.shape, image_rgb.tobytes()

    def prepare_inference_batch(self, images_rgb) -> None:
        """Prefetch one step of detector outputs without changing semantics."""
        self._detection_cache = {}
        if not self.yoloe_batch_enabled:
            return

        unique_images = []
        unique_keys = []
        seen_keys = set()
        for image_rgb in images_rgb:
            image_rgb = np.ascontiguousarray(image_rgb, dtype=np.uint8)
            key = self._detection_cache_key(image_rgb)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            unique_keys.append(key)
            unique_images.append(image_rgb)

        if not unique_images:
            return

        results = self.client.infer_batch(
            unique_images,
            max_batch_size=self.yoloe_batch_max_size,
            fallback_to_single=True,
        )
        if len(results) != len(unique_images):
            raise RuntimeError(
                "YOLOE batch result count mismatch: "
                f"{len(results)} != {len(unique_images)}"
            )
        self._detection_cache.update(zip(unique_keys, results))

    @staticmethod
    def target_class_for_atoms(atoms) -> str | None:
        """Return the manipulated visible class without mutating state."""
        for atom in atoms:
            if atom[0] in {"pick", "on", "in"} and atom[1] is not None:
                return _object_class(str(atom[1]))
        return None

    def prepare_missing_target_fallback_batch(self, requests) -> int:
        """Prefetch only fallback frames required by missing primary targets.

        Requests contain ``(primary_rgb, fallback_rgb, target_class)``.  The
        primary detector result is already cached.  This method only moves
        detector execution earlier and groups it; target selection, track
        gates, RGB-D geometry, margins, and rewards remain unchanged.
        """
        if not self.yoloe_batch_enabled:
            return 0

        pending_images = []
        pending_keys = []
        seen_keys = set()
        for primary_rgb, fallback_rgb, target_class in requests:
            if fallback_rgb is None or not target_class:
                continue
            primary_detections = self._infer_image(primary_rgb)
            primary_has_target = any(
                detection["class_name"] == target_class
                and detection["confidence"] >= self.min_confidence
                for detection in primary_detections
            )
            if primary_has_target:
                continue
            fallback_rgb = np.ascontiguousarray(fallback_rgb, dtype=np.uint8)
            key = self._detection_cache_key(fallback_rgb)
            if key in self._detection_cache or key in seen_keys:
                continue
            seen_keys.add(key)
            pending_keys.append(key)
            pending_images.append(fallback_rgb)

        # Preserve the established single-image endpoint when there is no
        # batch to form.  The later on-demand path will handle that one image.
        if len(pending_images) < 2:
            return 0

        results = self.client.infer_batch(
            pending_images,
            max_batch_size=self.yoloe_batch_max_size,
            fallback_to_single=True,
        )
        if len(results) != len(pending_images):
            raise RuntimeError(
                "YOLOE fallback batch result count mismatch: "
                f"{len(results)} != {len(pending_images)}"
            )
        self._detection_cache.update(zip(pending_keys, results))
        return len(pending_images)

    def _infer_image(self, image_rgb: np.ndarray) -> list[dict]:
        image_rgb = np.ascontiguousarray(image_rgb, dtype=np.uint8)
        key = self._detection_cache_key(image_rgb)
        if key not in self._detection_cache:
            self._detection_cache[key] = self.client.infer(image_rgb)
        return list(self._detection_cache[key])

    def _estimate_camera_objects(
        self,
        *,
        raw_obs: dict,
        camera_name: str,
        calibration: dict,
        eef: np.ndarray,
        min_confidence: float,
    ) -> list[dict]:
        image_rgb = np.ascontiguousarray(
            raw_obs[f"{camera_name}_image"][::-1, ::-1]
        )
        raw_depth = np.asarray(
            raw_obs[f"{camera_name}_depth"], dtype=np.float64
        ).squeeze()
        camera_to_world = np.asarray(
            calibration["camera_to_world"], dtype=np.float64
        )
        cache_key = (
            str(camera_name),
            image_rgb.shape,
            hash(image_rgb.tobytes()),
            hash(np.ascontiguousarray(raw_depth).tobytes()),
            hash(camera_to_world.tobytes()),
            float(min_confidence),
        )
        if cache_key == self._camera_estimate_cache_key:
            return list(self._camera_estimate_cache_value or ())
        detections = [
            detection
            for detection in self._infer_image(image_rgb)
            if detection["confidence"] >= min_confidence
        ]

        # The normalized-to-metric depth conversion depends on the camera
        # frame, not on an instance mask.  Reuse the exact established result
        # for every detection in this frame instead of recomputing a full
        # 256x256 array for each object.
        metric_depth = None
        try:
            near_value = float(calibration["near_m"])
            far_value = float(calibration["far_m"])
            if near_value > 0.0 and far_value > near_value:
                metric_depth = near_value / (
                    1.0
                    - raw_depth
                    * (1.0 - near_value / far_value)
                )
        except (KeyError, TypeError, ValueError):
            metric_depth = None

        estimated = []
        for detection in detections:
            mask = np.asarray(detection["mask"])
            if self.mask_to_depth_orientation == "rot180":
                mask_raw = np.ascontiguousarray(mask[::-1, ::-1])
            elif self.mask_to_depth_orientation == "vflip":
                mask_raw = np.ascontiguousarray(mask[::-1, :])
            elif self.mask_to_depth_orientation == "hflip":
                mask_raw = np.ascontiguousarray(mask[:, ::-1])
            else:
                mask_raw = np.ascontiguousarray(mask)

            estimate = estimate_visual_approach(
                sim=None,
                camera_name=camera_name,
                mask=mask_raw,
                raw_depth=raw_depth,
                eef_world=eef,
                intrinsic=calibration["intrinsic"],
                camera_to_world=calibration["camera_to_world"],
                near_m=calibration["near_m"],
                far_m=calibration["far_m"],
                grasp_radius_m=self.visual_center_radius_m,
                surface_radius_m=self.surface_radius_m,
                metric_depth=metric_depth,
            )
            if estimate.valid:
                estimated.append({
                    "detection": detection,
                    "estimate": estimate,
                    "camera_name": camera_name,
                    "image_rgb": image_rgb,
                })
        self._camera_estimate_cache_key = cache_key
        self._camera_estimate_cache_value = list(estimated)
        return estimated

    def _select_camera_target(
        self,
        *,
        env_id: int,
        estimated: list[dict],
        target_class: str,
        task_description: str,
        allow_unique_reacquire: bool = False,
    ) -> tuple[dict, str]:
        candidates = [
            item
            for item in estimated
            if item["detection"]["class_name"] == target_class
        ]
        if not candidates:
            raise RuntimeError(
                f"target_not_detected:{target_class}"
            )

        previous_center = self.target_tracks.get(int(env_id))
        if previous_center is None:
            selected, selector = select_visual_target_candidate(
                candidates=candidates,
                estimated=estimated,
                target_class=target_class,
                task_description=task_description,
            )
            return selected, f"lock_init:{selector}"

        selected = min(
            candidates,
            key=lambda item: np.linalg.norm(
                np.asarray(
                    item["estimate"].center_world,
                    dtype=np.float64,
                )
                - previous_center
            ),
        )
        track_distance = float(np.linalg.norm(
            np.asarray(
                selected["estimate"].center_world,
                dtype=np.float64,
            )
            - previous_center
        ))
        if track_distance > self.track_gate_m:
            if (
                allow_unique_reacquire
                and len(candidates) == 1
                and track_distance <= self.in_reacquire_gate_m
            ):
                return selected, (
                    "in_reacquire_unique:"
                    f"distance={track_distance:.6f}"
                )
            raise RuntimeError(
                "target_track_gate:"
                f"distance={track_distance:.6f}>"
                f"{self.track_gate_m:.6f}"
            )
        return selected, (
            "track_3d_nearest:"
            f"distance={track_distance:.6f}"
        )

    @staticmethod
    def _object_on_target_class(target_name: str | None) -> str | None:
        """Return the visible parent object for supported On regions.

        ``On`` atoms exported by LIBERO sometimes name a simulator region
        rather than the visible support object.  Only regions whose parent
        support has the same top-surface semantics as the generic visual On
        predicate are mapped here.  Other relations (for example, the table
        region in front of the stove or the region right of a plate) remain
        unsupported and require their own spatial predicate.
        """
        if target_name is None:
            return None
        lowered = str(target_name).lower()

        semantic_supports = (
            ("flat_stove_1_cook_region", "stove"),
            ("wine_rack_1_top_region", "wine rack"),
            ("main_table_stove_front_region", "stove"),
            ("living_room_table_plate_right_region", "plate"),
        )
        for marker, class_name in semantic_supports:
            if marker in lowered:
                return class_name

        if "region" in lowered or "site" in lowered:
            return None
        return _object_class(lowered)

    @staticmethod
    def _on_uses_support_footprint(
        support_class: str | None,
    ) -> bool:
        """Whether On XY should use the visible support footprint.

        A plate keeps the existing centre-distance predicate.  A stove,
        cabinet top, or wine-rack top is a functional support region: an
        object can be correctly placed away from the support object's RGB-D
        centre.  These classes therefore use distance to the robust projected
        support footprint instead.
        """
        return str(support_class or "").lower() in {
            "wooden cabinet",
            "white cabinet",
            "wine rack",
            "stove",
        }

    @staticmethod
    def _is_visual_site_on_target(
        target_name: str | None,
    ) -> bool:
        """Whether a canonical On target is a functional region site."""
        lowered = str(target_name or "").lower()
        return any(marker in lowered for marker in (
            "flat_stove_1_cook_region",
            "wooden_cabinet_1_top_side",
            "wine_rack_1_top_region",
        ))

    @staticmethod
    def _visual_spatial_relation_kind(
        target_name: str | None,
    ) -> str | None:
        """Resolve only the two audited LIBERO-40 spatial site relations."""
        lowered = str(target_name or "").lower()
        if "main_table_stove_front_region" in lowered:
            return "front_of_stove"
        if "living_room_table_plate_right_region" in lowered:
            return "right_of_plate"
        return None

    @classmethod
    def _locks_spatial_support(cls, target_name: str | None) -> bool:
        """Keep a static visual anchor for named spatial-region sites.

        The physical plate/stove is static, while later task actions can
        occlude its mask and bias a freshly reconstructed center.  Locking
        the first clean RGB-D support estimate mirrors the fixed site frame
        without reading a simulator site, object pose, joint, or contact.
        """
        return cls._visual_spatial_relation_kind(target_name) is not None

    @staticmethod
    def _spatial_relation_diagnostics(
        object_center,
        anchor_center,
        anchor_lower_xy,
        anchor_upper_xy,
    ) -> dict:
        """Raw visual geometry for calibrating a semantic spatial site.

        Every argument comes from RGB-D reconstruction.  This helper accepts
        no simulator site pose, size, object pose, contact, or Oracle value.
        """
        obj = np.asarray(object_center, dtype=np.float64).reshape(3)
        anchor = np.asarray(anchor_center, dtype=np.float64).reshape(3)
        lower = np.asarray(anchor_lower_xy, dtype=np.float64).reshape(2)
        upper = np.asarray(anchor_upper_xy, dtype=np.float64).reshape(2)
        span = upper - lower
        normalized = np.divide(
            obj[:2] - lower,
            span,
            out=np.full(2, np.nan, dtype=np.float64),
            where=np.abs(span) > 1.0e-9,
        )
        return {
            "relative_x_m": float(obj[0] - anchor[0]),
            "relative_y_m": float(obj[1] - anchor[1]),
            "relative_z_m": float(obj[2] - anchor[2]),
            "anchor_x_q05_m": float(lower[0]),
            "anchor_x_q95_m": float(upper[0]),
            "anchor_y_q05_m": float(lower[1]),
            "anchor_y_q95_m": float(upper[1]),
            "anchor_u": float(normalized[0]),
            "anchor_v": float(normalized[1]),
        }

    @staticmethod
    def _axis_aligned_spatial_margin(
        relative_xyz,
        bounds,
    ) -> dict:
        """Signed AABB containment using only visual relative geometry."""
        x, y, z = np.asarray(
            relative_xyz,
            dtype=np.float64,
        ).reshape(3)
        xlo, xhi, ylo, yhi, zlo, zhi = (
            float(value) for value in bounds
        )
        x_margin = float(min(x - xlo, xhi - x))
        y_margin = float(min(y - ylo, yhi - y))
        z_margin = float(min(z - zlo, zhi - z))
        return {
            "x_margin": x_margin,
            "y_margin": y_margin,
            "z_margin": z_margin,
            "margin": float(min(x_margin, y_margin, z_margin)),
        }

    def _spatial_footprint_diagnostics(
        self,
        *,
        relation: str | None,
        object_lower_xy,
        object_upper_xy,
        object_bottom_z_m: float,
        object_top_z_m: float,
        anchor_center,
        anchor_lower_xy,
        anchor_upper_xy,
        anchor_top_z_m: float,
        camera_to_world,
    ) -> dict:
        """Return visual-only q05/q95 footprint candidates for diagnostics."""
        nan = float("nan")
        result = {
            "box_valid": False,
            "box_reason": "not_front_of_stove",
            "x_margin": nan,
            "y_margin": nan,
            "bottom_z_margin": nan,
            "top_z_margin": nan,
            "full_z_margin": nan,
            "box_candidate_margin": nan,
            "directional_valid": False,
            "directional_reason": "spatial_relation_unavailable",
            "directional_margin": nan,
            "lateral_margin": nan,
            "surface_margin": nan,
            "directional_candidate_margin": nan,
        }
        relation = str(relation or "")
        lower = np.asarray(object_lower_xy, dtype=np.float64).reshape(2)
        upper = np.asarray(object_upper_xy, dtype=np.float64).reshape(2)
        anchor = np.asarray(anchor_center, dtype=np.float64).reshape(3)
        object_z = np.asarray(
            (object_bottom_z_m, object_top_z_m), dtype=np.float64
        )
        footprint_valid = bool(
            np.all(np.isfinite(lower))
            and np.all(np.isfinite(upper))
            and np.all(upper >= lower)
            and np.all(np.isfinite(anchor))
            and np.all(np.isfinite(object_z))
        )
        if relation == "front_of_stove":
            if footprint_valid:
                xlo, xhi, ylo, yhi, zlo, zhi = (
                    float(value) for value in self.on_front_bounds_m
                )
                relative_lower = lower - anchor[:2]
                relative_upper = upper - anchor[:2]
                relative_bottom = float(object_z[0] - anchor[2])
                relative_top = float(object_z[1] - anchor[2])
                result.update({
                    "box_valid": True,
                    "box_reason": "",
                    "x_margin": float(min(
                        relative_lower[0] - xlo,
                        xhi - relative_upper[0],
                    )),
                    "y_margin": float(min(
                        relative_lower[1] - ylo,
                        yhi - relative_upper[1],
                    )),
                    "bottom_z_margin": float(min(
                        relative_bottom - zlo,
                        zhi - relative_bottom,
                    )),
                    "top_z_margin": float(min(
                        relative_top - zlo,
                        zhi - relative_top,
                    )),
                    "full_z_margin": float(min(
                        relative_bottom - zlo,
                        zhi - relative_top,
                    )),
                })
                result["box_candidate_margin"] = float(min(
                    result["x_margin"],
                    result["y_margin"],
                    result["full_z_margin"],
                ))
            else:
                result["box_reason"] = "object_footprint_unavailable"

        camera = np.asarray(camera_to_world, dtype=np.float64)
        anchor_lower = np.asarray(anchor_lower_xy, dtype=np.float64).reshape(2)
        anchor_upper = np.asarray(anchor_upper_xy, dtype=np.float64).reshape(2)
        if relation not in {"front_of_stove", "right_of_plate"}:
            return result
        if not footprint_valid:
            result["directional_reason"] = "object_footprint_unavailable"
            return result
        if camera.shape != (4, 4) or not np.all(np.isfinite(camera)):
            result["directional_reason"] = "camera_calibration_unavailable"
            return result
        corners = (
            (lower[0], lower[1]),
            (lower[0], upper[1]),
            (upper[0], lower[1]),
            (upper[0], upper[1]),
        )
        config = (
            VisualSpatialRelationConfig(
                max_offset_m=self.on_right_max_offset_m,
                lateral_padding_m=self.on_right_lateral_padding_m,
            )
            if relation == "right_of_plate"
            else VisualSpatialRelationConfig()
        )
        estimates = [
            visual_directional_region_margin(
                relation=relation,
                object_center_world=(xy[0], xy[1], object_z.mean()),
                object_bottom_z_m=float(object_z[0]),
                anchor_center_world=anchor,
                anchor_lower_xy_world=anchor_lower,
                anchor_upper_xy_world=anchor_upper,
                anchor_top_z_m=float(anchor_top_z_m),
                camera_position_world=camera[:3, 3],
                camera_right_world=camera[:3, 0],
                config=config,
            )
            for xy in corners
        ]
        invalid = next((item for item in estimates if not item.valid), None)
        if invalid is not None:
            result["directional_reason"] = str(invalid.reason)
            return result
        result.update({
            "directional_valid": True,
            "directional_reason": "",
            "directional_margin": float(min(
                item.directional_margin for item in estimates
            )),
            "lateral_margin": float(min(
                item.lateral_margin for item in estimates
            )),
            "surface_margin": float(min(
                item.surface_margin for item in estimates
            )),
        })
        result["directional_candidate_margin"] = float(min(
            result["directional_margin"],
            result["lateral_margin"],
            result["surface_margin"],
        ))
        return result

    @staticmethod
    def _site_vertical_margin(
        delta_z_m: float,
        lower_delta_m: float,
        upper_delta_m: float,
    ) -> float:
        """Signed containment margin for a visual site-height band."""
        delta = float(delta_z_m)
        lower = float(lower_delta_m)
        upper = float(upper_delta_m)
        return float(min(delta - lower, upper - delta))

    @staticmethod
    def _site_release_margin(
        current_release_margin: float,
        release_observed: bool,
    ) -> float:
        """Keep visual release evidence latched for canonical site-On.

        Closing the gripper again after placing an object must not invalidate
        the already-observed release event.  The latch is episode-local and
        is set only by the visual/proprioceptive attachment-to-release
        transition; it does not use contact, simulator state, or Oracle data.
        """
        margin = float(current_release_margin)
        if bool(release_observed):
            return float(max(margin, 1.0e-6))
        return margin

    @staticmethod
    def _support_footprint_xy_distance(
        object_xy,
        lower_xy,
        upper_xy,
    ) -> float:
        """Euclidean distance outside an axis-aligned visual XY footprint.

        The value is zero inside the footprint and positive outside.  All
        arguments are derived from RGB-D observations; no simulator site or
        object geometry is accepted.
        """
        point = np.asarray(object_xy, dtype=np.float64).reshape(2)
        lower = np.asarray(lower_xy, dtype=np.float64).reshape(2)
        upper = np.asarray(upper_xy, dtype=np.float64).reshape(2)
        if not (
            np.all(np.isfinite(point))
            and np.all(np.isfinite(lower))
            and np.all(np.isfinite(upper))
            and np.all(upper >= lower)
        ):
            return float("nan")
        outside = np.maximum(
            np.maximum(lower - point, point - upper),
            0.0,
        )
        return float(np.linalg.norm(outside))

    @staticmethod
    def _ordinary_effective_xy_tolerance(
        semantic_tolerance_m: float,
        observation_uncertainty_m: float,
    ) -> float:
        """Combine task semantics with an explicit visual error budget."""
        return float(semantic_tolerance_m + observation_uncertainty_m)

    @staticmethod
    def _visual_xy_bounds(estimate) -> tuple[np.ndarray, np.ndarray]:
        """Return robust visual XY bounds without simulator geometry."""
        direct_lower = np.asarray(
            getattr(estimate, "xy_q05_m", (np.nan, np.nan)),
            dtype=np.float64,
        ).reshape(2)
        direct_upper = np.asarray(
            getattr(estimate, "xy_q95_m", (np.nan, np.nan)),
            dtype=np.float64,
        ).reshape(2)
        if (
            np.all(np.isfinite(direct_lower))
            and np.all(np.isfinite(direct_upper))
            and np.all(direct_upper >= direct_lower)
        ):
            return direct_lower, direct_upper

        points = np.asarray(
            getattr(estimate, "points_world", None),
            dtype=np.float64,
        )
        if points.ndim == 2 and points.shape[1] >= 2:
            finite_xy = points[
                np.all(np.isfinite(points[:, :2]), axis=1),
                :2,
            ]
            if len(finite_xy) > 0:
                lower, upper = np.quantile(
                    finite_xy,
                    (0.05, 0.95),
                    axis=0,
                )
                return lower, upper
        unavailable = np.full(2, np.nan, dtype=np.float64)
        return unavailable.copy(), unavailable.copy()

    @classmethod
    def _visual_xy_quantile_midpoint(cls, estimate) -> tuple[
        np.ndarray, np.ndarray, np.ndarray
    ]:
        """Return q05/q95 and their midpoint from visual points only."""
        lower, upper = cls._visual_xy_bounds(estimate)
        midpoint = (
            0.5 * (lower + upper)
            if np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
            else np.full(2, np.nan, dtype=np.float64)
        )
        return lower, upper, midpoint

    def _on_object_estimate(
        self,
        *,
        env_id: int,
        step: int,
        eef: np.ndarray,
        eef_rotation: np.ndarray,
        gripper_width_m: float,
        center=None,
        z_p05_m=float("nan"),
        z_p95_m=float("nan"),
        xy_q05_m=(float("nan"), float("nan")),
        xy_q95_m=(float("nan"), float("nan")),
        points_world=None,
        near_margin=float("nan"),
        fresh: bool,
    ):
        """Handoff an RGB-D object track to EEF proprioception."""
        env_id = int(env_id)
        step = int(step)
        eef = np.asarray(eef, dtype=np.float64).reshape(3)
        eef_rotation = np.asarray(
            eef_rotation, dtype=np.float64
        ).reshape(3, 3)
        state = self.on_object_states.get(env_id)

        if center is not None:
            center = np.asarray(center, dtype=np.float64).reshape(3)
            if not np.all(np.isfinite(center)):
                center = None
        xy_q05_m = np.asarray(xy_q05_m, dtype=np.float64).reshape(2)
        xy_q95_m = np.asarray(xy_q95_m, dtype=np.float64).reshape(2)
        xy_bounds_valid = bool(
            np.all(np.isfinite(xy_q05_m))
            and np.all(np.isfinite(xy_q95_m))
            and np.all(xy_q95_m >= xy_q05_m)
        )
        fresh_points = np.asarray(
            points_world,
            dtype=np.float64,
        )
        if fresh_points.ndim == 2 and fresh_points.shape[1] == 3:
            fresh_points = fresh_points[
                np.all(np.isfinite(fresh_points), axis=1)
            ]
        else:
            fresh_points = np.empty((0, 3), dtype=np.float64)

        if state is not None and state.get("released", False):
            if fresh and center is not None:
                fresh_summary = self._summarize_visual_points(fresh_points)
                if fresh_summary is not None:
                    z_p05_m = fresh_summary["z_p05_m"]
                    z_p95_m = fresh_summary["z_p95_m"]
                    xy_q05_m = fresh_summary["xy_q05_m"]
                    xy_q95_m = fresh_summary["xy_q95_m"]
                state.update({
                    "last_step": step,
                    "last_center": center.copy(),
                    "last_z_p05_m": float(z_p05_m),
                    "last_z_p95_m": float(z_p95_m),
                    "last_xy_q05_m": xy_q05_m.copy(),
                    "last_xy_q95_m": xy_q95_m.copy(),
                    "frozen_center": center.copy(),
                    "frozen_z_p05_m": float(z_p05_m),
                    "frozen_z_p95_m": float(z_p95_m),
                    "frozen_xy_q05_m": xy_q05_m.copy(),
                    "frozen_xy_q95_m": xy_q95_m.copy(),
                    "frozen_points_world": fresh_points.copy(),
                    # The hold age now means age since the latest visual
                    # observation, not age since the gripper opened.
                    "release_step": step,
                    "fresh_after_release": True,
                })
                self.on_post_release_fresh[env_id] = True
                return SimpleNamespace(
                    center_world=tuple(center),
                    z_p05_m=float(z_p05_m),
                    z_p95_m=float(z_p95_m),
                    xy_q05_m=tuple(float(x) for x in xy_q05_m),
                    xy_q95_m=tuple(float(x) for x in xy_q95_m),
                    points_world=fresh_points.copy(),
                ), "fresh_rgbd", state

            age = step - int(state["release_step"])
            if age <= self.on_release_hold_steps:
                frozen = np.asarray(state["frozen_center"])
                frozen_points = np.asarray(
                    state.get("frozen_points_world", np.empty((0, 3))),
                    dtype=np.float64,
                ).reshape(-1, 3)
                return SimpleNamespace(
                    center_world=tuple(frozen),
                    z_p05_m=float(state["frozen_z_p05_m"]),
                    z_p95_m=float(state["frozen_z_p95_m"]),
                    xy_q05_m=tuple(
                        float(x) for x in state.get(
                            "frozen_xy_q05_m", (np.nan, np.nan)
                        )
                    ),
                    xy_q95_m=tuple(
                        float(x) for x in state.get(
                            "frozen_xy_q95_m", (np.nan, np.nan)
                        )
                    ),
                    points_world=frozen_points.copy(),
                ), "eef_release_hold", state

            self.on_object_states.pop(env_id, None)
            state = None

        if state is not None and state.get("attached", False):
            if fresh and center is not None:
                state["relative_center_eef"] = (
                    eef_rotation.T @ (center - eef)
                )
                state["z_p05_offset"] = float(z_p05_m - center[2])
                state["z_p95_offset"] = float(z_p95_m - center[2])
                if xy_bounds_valid:
                    state["xy_q05_offset"] = (
                        xy_q05_m - center[:2]
                    )
                    state["xy_q95_offset"] = (
                        xy_q95_m - center[:2]
                    )
                if len(fresh_points) > 0:
                    state["points_eef"] = self._points_world_to_eef(
                        fresh_points,
                        eef,
                        eef_rotation,
                    )

            predicted = eef + eef_rotation @ np.asarray(
                state["relative_center_eef"]
            )
            predicted_z_p05 = float(
                predicted[2] + state["z_p05_offset"]
            )
            predicted_z_p95 = float(
                predicted[2] + state["z_p95_offset"]
            )
            predicted_xy_q05 = (
                predicted[:2]
                + np.asarray(
                    state.get(
                        "xy_q05_offset", (np.nan, np.nan)
                    ),
                    dtype=np.float64,
                )
            )
            predicted_xy_q95 = (
                predicted[:2]
                + np.asarray(
                    state.get(
                        "xy_q95_offset", (np.nan, np.nan)
                    ),
                    dtype=np.float64,
                )
            )
            predicted_points = np.empty((0, 3), dtype=np.float64)
            points_eef = np.asarray(
                state.get("points_eef", np.empty((0, 3))),
                dtype=np.float64,
            ).reshape(-1, 3)
            if len(points_eef) > 0:
                predicted_points = self._points_eef_to_world(
                    points_eef,
                    eef,
                    eef_rotation,
                )
                predicted_summary = self._summarize_visual_points(
                    predicted_points
                )
                if predicted_summary is not None:
                    predicted_z_p05 = predicted_summary["z_p05_m"]
                    predicted_z_p95 = predicted_summary["z_p95_m"]
                    predicted_xy_q05 = predicted_summary["xy_q05_m"]
                    predicted_xy_q95 = predicted_summary["xy_q95_m"]

            if gripper_width_m >= self.on_release_width_m:
                state.update({
                    "attached": False,
                    "released": True,
                    "release_step": step,
                    "frozen_center": predicted.copy(),
                    "frozen_z_p05_m": predicted_z_p05,
                    "frozen_z_p95_m": predicted_z_p95,
                    "frozen_xy_q05_m": predicted_xy_q05.copy(),
                    "frozen_xy_q95_m": predicted_xy_q95.copy(),
                    "frozen_points_world": predicted_points.copy(),
                    "fresh_after_release": False,
                })
                self.on_release_observed[env_id] = True
                self.on_release_steps[env_id] = step
                self.on_post_release_fresh[env_id] = False
                source = "eef_release"
            else:
                source = "eef_attached"

            return SimpleNamespace(
                center_world=tuple(predicted),
                z_p05_m=predicted_z_p05,
                z_p95_m=predicted_z_p95,
                xy_q05_m=tuple(float(x) for x in predicted_xy_q05),
                xy_q95_m=tuple(float(x) for x in predicted_xy_q95),
                points_world=(
                    predicted_points.copy()
                    if len(predicted_points) > 0
                    else fresh_points.copy()
                ),
            ), source, state

        attach_evidence = bool(
            center is not None
            and np.isfinite(near_margin)
            and float(near_margin) >= 0.0
            and gripper_width_m <= self.on_attach_width_m
        )
        if attach_evidence:
            count = self.on_attach_counts.get(env_id, 0) + 1
        else:
            count = 0
        self.on_attach_counts[env_id] = count

        if center is not None:
            # The released state may have expired while the object was
            # occluded.  Preserve the episode-level release latch so a later
            # visual reacquisition can still satisfy the gate.
            if (
                fresh
                and self.on_release_observed.get(env_id, False)
            ):
                self.on_post_release_fresh[env_id] = True
            state = {
                "attached": False,
                "released": False,
                "last_step": step,
                "last_center": center.copy(),
                "last_z_p05_m": float(z_p05_m),
                "last_z_p95_m": float(z_p95_m),
                "last_xy_q05_m": xy_q05_m.copy(),
                "last_xy_q95_m": xy_q95_m.copy(),
            }
            self.on_object_states[env_id] = state

        if (
            attach_evidence
            and count >= self.on_attach_confirmation_steps
            and center is not None
        ):
            state.update({
                "attached": True,
                "released": False,
                "relative_center_eef": (
                    eef_rotation.T @ (center - eef)
                ),
                "z_p05_offset": float(z_p05_m - center[2]),
                "z_p95_offset": float(z_p95_m - center[2]),
                "xy_q05_offset": xy_q05_m - center[:2],
                "xy_q95_offset": xy_q95_m - center[:2],
                "points_eef": self._points_world_to_eef(
                    fresh_points,
                    eef,
                    eef_rotation,
                ),
            })
            return SimpleNamespace(
                center_world=tuple(center),
                z_p05_m=float(z_p05_m),
                z_p95_m=float(z_p95_m),
                xy_q05_m=tuple(float(x) for x in xy_q05_m),
                xy_q95_m=tuple(float(x) for x in xy_q95_m),
                points_world=fresh_points.copy(),
            ), "eef_attach", state

        if fresh and center is not None:
            return SimpleNamespace(
                center_world=tuple(center),
                z_p05_m=float(z_p05_m),
                z_p95_m=float(z_p95_m),
                xy_q05_m=tuple(float(x) for x in xy_q05_m),
                xy_q95_m=tuple(float(x) for x in xy_q95_m),
                points_world=fresh_points.copy(),
            ), "fresh_rgbd", state

        return None, "object_unavailable", state

    @staticmethod
    def _eef_rotation(raw_obs: dict) -> np.ndarray:
        """Convert the proprioceptive xyzw EEF quaternion to rotation."""
        quaternion = np.asarray(
            raw_obs.get(
                "robot0_eef_quat",
                [0.0, 0.0, 0.0, 1.0],
            ),
            dtype=np.float64,
        ).reshape(4)
        norm = float(np.linalg.norm(quaternion))
        if norm <= 1.0e-12:
            return np.eye(3, dtype=np.float64)
        x, y, z, w = quaternion / norm
        return np.asarray([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ], dtype=np.float64)

    @staticmethod
    def _uses_container_on_geometry(
        support_class: str | None,
    ) -> bool:
        """Limit containment-as-contact to explicitly audited supports."""
        return str(support_class or "").strip().lower() in {
            "black bowl",
        }

    @staticmethod
    def _final_container_on_observation(
        *,
        geometry_valid: bool,
        contact_valid: bool,
        handoff_used: bool,
        margin: float,
        reason: str,
    ) -> tuple[bool, float, str]:
        """Export audited Container-On only with visual contact evidence.

        Missing current contact and a missing contact-validated handoff mean
        that the atom is unobserved, not violated.  Returning NaN together
        with ``valid=False`` lets the visual AGM runtime freeze the active
        stage instead of treating perception dropout as negative reward.
        """
        reason = str(reason or "")
        if not geometry_valid:
            detail = reason or "invalid_geometry"
            return (
                False,
                float("nan"),
                f"container_geometry_unavailable:{detail}",
            )
        if not contact_valid and not handoff_used:
            detail = reason or "contact_not_observed"
            return (
                False,
                float("nan"),
                f"container_contact_unavailable:{detail}",
            )
        if not np.isfinite(float(margin)):
            return (
                False,
                float("nan"),
                "container_contact_unavailable:nonfinite_margin",
            )
        return True, float(margin), reason

    @staticmethod
    def _on_confirmation_gate(
        *,
        support_class: str | None,
        object_source: str,
        require_fresh_after_release: bool,
        release_observed: bool,
        post_release_fresh: bool,
        geometric_margin: float,
    ) -> tuple[bool, str]:
        """Gate container ``On`` without simulator or Oracle state.

        A release proxy alone cannot confirm ``On``.  A post-release fresh
        RGB-D observation passes here; the caller may separately admit its
        bounded stable pre-release visual handoff for audited containers.
        """
        if not require_fresh_after_release:
            return True, ""
        if release_observed and post_release_fresh:
            return True, ""

        return False, "fresh_after_release_required"

    def _spatial_motion_gate(
        self,
        *,
        env_id: int,
        object_center: np.ndarray,
    ) -> dict:
        """Require observable transport followed by a stable target pose."""
        env_id = int(env_id)
        center = np.asarray(object_center, dtype=np.float64).reshape(3)
        initial = self.on_spatial_initial_centers.setdefault(
            env_id, center.copy()
        )
        previous = self.on_spatial_previous_centers.get(env_id)
        displacement = float(np.linalg.norm(center - initial))
        frame_motion = (
            float(np.linalg.norm(center - previous))
            if previous is not None
            else float("nan")
        )
        self.on_spatial_previous_centers[env_id] = center.copy()
        if displacement >= self.on_spatial_min_displacement_m:
            self.on_spatial_motion_observed[env_id] = True
        motion_observed = bool(
            self.on_spatial_motion_observed.get(env_id, False)
        )
        stable = bool(
            motion_observed
            and np.isfinite(frame_motion)
            and frame_motion <= self.on_spatial_stability_tolerance_m
        )
        return {
            "displacement_m": displacement,
            "frame_motion_m": frame_motion,
            "motion_observed": motion_observed,
            "stable": stable,
        }

    def _spatial_pick_proxy_margin(
        self,
        *,
        object_source: str,
        gripper_width_m: float,
        spatial_motion: dict,
    ) -> float:
        """Contact-free pick fallback after a visual-to-EEF handoff."""
        if not self.on_right_pick_proxy:
            return float("nan")
        if object_source not in {
            "eef_attach", "eef_attached", "eef_release",
            "eef_release_hold",
        }:
            return float("nan")
        if not bool(spatial_motion.get("motion_observed", False)):
            return float("nan")
        displacement = float(
            spatial_motion.get("displacement_m", float("nan"))
        )
        if not np.isfinite(displacement):
            return float("nan")
        return float(min(
            self.close_threshold_m - float(gripper_width_m),
            displacement - self.on_spatial_min_displacement_m,
        ))

    def _right_spatial_confirmation_gate(
        self,
        *,
        geometric_margin: float,
        spatial_motion: dict,
        release_observed: bool,
    ) -> tuple[bool, str, float]:
        """Return task36 confirmation and signed visual evidence."""
        if not bool(spatial_motion.get("motion_observed", False)):
            return False, "spatial_motion_required", float("nan")
        displacement = float(
            spatial_motion.get("displacement_m", float("nan"))
        )
        frame_motion = float(
            spatial_motion.get("frame_motion_m", float("nan"))
        )
        if not np.isfinite(geometric_margin):
            return False, "spatial_geometry_required", float("nan")
        if not np.isfinite(displacement):
            return False, "spatial_transport_required", float("nan")
        if not np.isfinite(frame_motion):
            return False, "spatial_stability_required", float("nan")

        stability_margin = float(
            self.on_spatial_stability_tolerance_m - frame_motion
        )
        confident_margin = float(min(
            geometric_margin - self.on_right_interior_margin_m,
            displacement - self.on_right_min_displacement_m,
            stability_margin,
        ))
        recovery_margin = float(min(
            geometric_margin + self.on_right_recovery_slack_m,
            displacement - self.on_right_recovery_min_displacement_m,
            stability_margin,
        ))
        released_margin = (
            float(min(
                geometric_margin + self.on_right_released_slack_m,
                displacement - self.on_right_released_min_displacement_m,
                stability_margin,
            ))
            if release_observed
            else float("-inf")
        )
        combined_margin = float(max(
            confident_margin,
            recovery_margin,
            released_margin,
        ))
        if self.on_right_require_release and not release_observed:
            return False, "spatial_release_required", float(min(
                combined_margin, -1.0e-6
            ))
        if combined_margin < 0.0:
            if stability_margin < 0.0:
                reason = "spatial_stability_required"
            elif max(
                displacement - self.on_right_min_displacement_m,
                displacement
                - self.on_right_recovery_min_displacement_m,
                (
                    displacement
                    - self.on_right_released_min_displacement_m
                    if release_observed
                    else float("-inf")
                ),
            ) < 0.0:
                reason = "spatial_transport_required"
            else:
                reason = "spatial_geometry_confidence_required"
            return False, reason, combined_margin
        return True, "", combined_margin

    def _right_release_is_recent(
        self,
        *,
        step: int,
        release_step: int | None,
    ) -> bool:
        """Bound the proprioceptive release relaxation in time."""
        if release_step is None:
            return False
        age = int(step) - int(release_step)
        return 0 <= age <= self.on_right_release_max_age_steps

    def _visual_on_components(
        self,
        *,
        env_id: int,
        step: int,
        object_estimate,
        object_source: str,
        support_estimate,
        support_class: str | None,
        on_target_object: str | None,
        gripper_width_m: float,
    ) -> dict:
        """Simulator-contact-free On evidence from two RGB-D masks."""
        object_center = np.asarray(
            object_estimate.center_world,
            dtype=np.float64,
        )
        support_center = np.asarray(
            support_estimate.center_world,
            dtype=np.float64,
        )
        baseline_center = np.asarray(
            getattr(
                support_estimate,
                "baseline_center_world",
                support_estimate.center_world,
            ),
            dtype=np.float64,
        )
        baseline_z_p95 = float(
            getattr(
                support_estimate,
                "baseline_z_p95_m",
                support_estimate.z_p95_m,
            )
        )
        (
            object_xy_lower,
            object_xy_upper,
            object_xy_midpoint,
        ) = self._visual_xy_quantile_midpoint(object_estimate)
        baseline_points_world = getattr(
            support_estimate,
            "baseline_points_world",
            getattr(support_estimate, "points_world", None),
        )
        (
            support_baseline_xy_lower,
            support_baseline_xy_upper,
            support_baseline_xy_midpoint,
        ) = self._visual_xy_quantile_midpoint(SimpleNamespace(
            points_world=baseline_points_world,
        ))
        robust_xy_distance = float("nan")
        if (
            np.all(np.isfinite(object_xy_midpoint))
            and np.all(np.isfinite(support_baseline_xy_midpoint))
        ):
            robust_xy_distance = float(np.linalg.norm(
                object_xy_midpoint - support_baseline_xy_midpoint
            ))
        ordinary_xy_tolerance = self._ordinary_effective_xy_tolerance(
            self.on_ordinary_xy_tolerance_m,
            getattr(self, "on_ordinary_xy_uncertainty_m", 0.0),
        )
        ordinary_completion_xy_tolerance = (
            self._ordinary_effective_xy_tolerance(
                self.on_ordinary_xy_tolerance_m,
                getattr(
                    self,
                    "on_ordinary_completion_xy_uncertainty_m",
                    0.0,
                ),
            )
        )
        ordinary_completion_xy_margin = float("nan")
        robust_xy_margin = float(
            ordinary_xy_tolerance - robust_xy_distance
        )
        robust_candidate_margin = float("nan")
        xy_distance = float("nan")
        if self._on_uses_support_footprint(support_class):
            xy_distance = self._support_footprint_xy_distance(
                object_center[:2],
                getattr(support_estimate, "xy_q05_m", (np.nan, np.nan)),
                getattr(support_estimate, "xy_q95_m", (np.nan, np.nan)),
            )
        if not np.isfinite(xy_distance):
            xy_distance = float(np.linalg.norm(
                object_center[:2] - support_center[:2]
            ))
        xy_margin = float(self.on_xy_tolerance_m - xy_distance)
        height_delta = float(object_center[2] - support_center[2])
        above_margin = float(
            height_delta - self.on_above_epsilon_m
        )
        surface_gap = float(abs(
            object_estimate.z_p05_m
            - support_estimate.z_p95_m
        ))
        support_z_p99 = float(
            getattr(support_estimate, "z_p99_m", np.nan)
        )
        support_z_p995 = float(
            getattr(support_estimate, "z_p995_m", np.nan)
        )
        site_center_delta_q99 = float(
            object_center[2] - support_z_p99
        )
        site_bottom_gap_q99 = float(
            object_estimate.z_p05_m - support_z_p99
        )
        site_mode = self._is_visual_site_on_target(on_target_object)
        spatial_relation = self._visual_spatial_relation_kind(
            on_target_object
        )
        spatial = self._spatial_relation_diagnostics(
            object_center,
            support_center,
            getattr(support_estimate, "xy_q05_m", (np.nan, np.nan)),
            getattr(support_estimate, "xy_q95_m", (np.nan, np.nan)),
        )
        spatial_margins = {
            "valid": False,
            "reason": "",
            "x_margin": float("nan"),
            "y_margin": float("nan"),
            "z_margin": float("nan"),
            "margin": float("nan"),
            "directional_offset_m": float("nan"),
            "lateral_offset_m": float("nan"),
            "anchor_direction_half_extent_m": float("nan"),
            "anchor_lateral_half_extent_m": float("nan"),
        }
        spatial_motion = {
            "displacement_m": float("nan"),
            "frame_motion_m": float("nan"),
            "motion_observed": False,
            "stable": False,
        }
        spatial_pick_proxy_margin = float("nan")
        spatial_footprint = self._spatial_footprint_diagnostics(
            relation=None,
            object_lower_xy=object_xy_lower,
            object_upper_xy=object_xy_upper,
            object_bottom_z_m=float(object_estimate.z_p05_m),
            object_top_z_m=float(object_estimate.z_p95_m),
            anchor_center=support_center,
            anchor_lower_xy=getattr(
                support_estimate, "xy_q05_m", (np.nan, np.nan)
            ),
            anchor_upper_xy=getattr(
                support_estimate, "xy_q95_m", (np.nan, np.nan)
            ),
            anchor_top_z_m=support_z_p99,
            camera_to_world=np.full((4, 4), np.nan),
        )
        container_mode = self._uses_container_on_geometry(
            support_class
        )
        container_geometry = None
        contact_margin = float("nan")
        contact_diagnostics = SimpleNamespace(
            valid=False,
            reason=(
                "fresh_rgbd_required"
                if container_mode
                else "container_geometry_required"
            ),
            p01_m=float("nan"),
            p05_m=float("nan"),
            p10_m=float("nan"),
        )
        if container_mode:
            container_geometry = visual_container_on_from_bounds(
                object_center_world=object_center,
                object_xy_q05_m=getattr(
                    object_estimate,
                    "xy_q05_m",
                    (np.nan, np.nan),
                ),
                object_xy_q95_m=getattr(
                    object_estimate,
                    "xy_q95_m",
                    (np.nan, np.nan),
                ),
                object_bottom_z_m=float(object_estimate.z_p05_m),
                object_top_z_m=float(object_estimate.z_p95_m),
                bowl_center_world=support_center,
                bowl_xy_q05_m=getattr(
                    support_estimate,
                    "xy_q05_m",
                    (np.nan, np.nan),
                ),
                bowl_xy_q95_m=getattr(
                    support_estimate,
                    "xy_q95_m",
                    (np.nan, np.nan),
                ),
                bowl_floor_z_m=float(support_estimate.z_p05_m),
                bowl_rim_z_m=float(support_estimate.z_p95_m),
                center_xy_tolerance_m=0.03,
                config=self.in_config,
            )
            if (
                getattr(object_estimate, "points_world", None)
                is not None
                and getattr(support_estimate, "points_world", None)
                is not None
            ):
                contact_diagnostics = visual_point_contact_quantiles(
                    getattr(object_estimate, "points_world", None),
                    getattr(support_estimate, "points_world", None),
                )
                if contact_diagnostics.valid:
                    contact_margin = float(
                        self.on_contact_tolerance_m
                        - contact_diagnostics.p05_m
                    )
        elif (
            getattr(object_estimate, "points_world", None) is not None
            and getattr(
                support_estimate,
                "baseline_points_world",
                getattr(support_estimate, "points_world", None),
            ) is not None
        ):
            # The frozen first clean support cloud prevents a placed object
            # from contaminating its own visual contact reference.
            contact_diagnostics = visual_point_contact_quantiles(
                getattr(object_estimate, "points_world", None),
                getattr(
                    support_estimate,
                    "baseline_points_world",
                    getattr(support_estimate, "points_world", None),
                ),
            )
            if contact_diagnostics.valid:
                contact_margin = float(
                    self.on_ordinary_contact_tolerance_m
                    - contact_diagnostics.p05_m
                )
        if spatial_relation is not None:
            relative_xyz = (
                spatial["relative_x_m"],
                spatial["relative_y_m"],
                spatial["relative_z_m"],
            )
            if (
                spatial_relation == "front_of_stove"
                and np.all(np.isfinite(relative_xyz))
            ):
                axis_margins = self._axis_aligned_spatial_margin(
                    relative_xyz,
                    self.on_front_bounds_m,
                )
                spatial_margins.update({
                    "valid": True,
                    "reason": "",
                    "x_margin": float(axis_margins["x_margin"]),
                    "y_margin": float(axis_margins["y_margin"]),
                    "z_margin": float(axis_margins["z_margin"]),
                    "margin": float(axis_margins["margin"]),
                })
            elif spatial_relation == "front_of_stove":
                spatial_margins["reason"] = (
                    "relative_geometry_unavailable"
                )

            # Directional sites are expressed in a frame derived from the
            # visible support and calibrated camera. This remains invariant
            # to randomized world placement, unlike a task-specific AABB.
            calibration = self.calibrations.get(int(env_id), {})
            camera_to_world = np.asarray(
                calibration.get(
                    "camera_to_world", np.full((4, 4), np.nan)
                ),
                dtype=np.float64,
            )
            if camera_to_world.shape == (4, 4):
                spatial_estimate = visual_directional_region_margin(
                    relation=spatial_relation,
                    object_center_world=object_center,
                    object_bottom_z_m=float(object_estimate.z_p05_m),
                    anchor_center_world=support_center,
                    anchor_lower_xy_world=getattr(
                        support_estimate,
                        "xy_q05_m",
                        (np.nan, np.nan),
                    ),
                    anchor_upper_xy_world=getattr(
                        support_estimate,
                        "xy_q95_m",
                        (np.nan, np.nan),
                    ),
                    anchor_top_z_m=support_z_p99,
                    camera_position_world=camera_to_world[:3, 3],
                    camera_right_world=camera_to_world[:3, 0],
                    config=(
                        VisualSpatialRelationConfig(
                            max_offset_m=self.on_right_max_offset_m,
                            lateral_padding_m=(
                                self.on_right_lateral_padding_m
                            )
                        )
                        if spatial_relation == "right_of_plate"
                        else VisualSpatialRelationConfig()
                    ),
                )
                if spatial_relation == "right_of_plate":
                    spatial_margins.update({
                        "valid": bool(spatial_estimate.valid),
                        "reason": str(spatial_estimate.reason),
                        "x_margin": float(
                            spatial_estimate.directional_margin
                        ),
                        "y_margin": float(
                            spatial_estimate.lateral_margin
                        ),
                        "z_margin": float(
                            spatial_estimate.surface_margin
                        ),
                        "margin": float(spatial_estimate.margin),
                    })
                spatial_margins.update({
                    "directional_offset_m": float(
                        spatial_estimate.directional_offset_m
                    ),
                    "lateral_offset_m": float(
                        spatial_estimate.lateral_offset_m
                    ),
                    "anchor_direction_half_extent_m": float(
                        spatial_estimate.anchor_direction_half_extent_m
                    ),
                    "anchor_lateral_half_extent_m": float(
                        spatial_estimate.anchor_lateral_half_extent_m
                    ),
                })
            if spatial_relation == "right_of_plate":
                spatial_motion = self._spatial_motion_gate(
                    env_id=env_id,
                    object_center=object_center,
                )
                spatial_pick_proxy_margin = (
                    self._spatial_pick_proxy_margin(
                        object_source=object_source,
                        gripper_width_m=gripper_width_m,
                        spatial_motion=spatial_motion,
                    )
                )
            spatial_footprint = self._spatial_footprint_diagnostics(
                relation=spatial_relation,
                object_lower_xy=object_xy_lower,
                object_upper_xy=object_xy_upper,
                object_bottom_z_m=float(object_estimate.z_p05_m),
                object_top_z_m=float(object_estimate.z_p95_m),
                anchor_center=support_center,
                anchor_lower_xy=getattr(
                    support_estimate, "xy_q05_m", (np.nan, np.nan)
                ),
                anchor_upper_xy=getattr(
                    support_estimate, "xy_q95_m", (np.nan, np.nan)
                ),
                anchor_top_z_m=support_z_p99,
                camera_to_world=camera_to_world,
            )
        if site_mode:
            support_margin = self._site_vertical_margin(
                site_center_delta_q99,
                self.on_site_lower_delta_m,
                self.on_site_upper_delta_m,
            )
        else:
            support_margin = float(
                self.on_surface_tolerance_m - surface_gap
            )
        release_margin = float(
            gripper_width_m - self.on_release_width_m
        )
        if site_mode:
            release_margin = self._site_release_margin(
                release_margin,
                self.on_release_observed.get(int(env_id), False),
            )
        if spatial_relation is not None:
            xy_margin = float(min(
                spatial_margins["x_margin"],
                spatial_margins["y_margin"],
            ))
            support_margin = float(spatial_margins["z_margin"])
            # The senior's site-under predicate is geometric and contains no
            # gripper-release term. Keep the visual surrogate aligned with
            # that semantics; temporal confirmation handles transient passes.
            on_margin = float(spatial_margins["margin"])
        elif site_mode:
            on_margin = float(min(
                xy_margin,
                support_margin,
                release_margin,
            ))
        elif container_mode:
            if container_geometry.valid:
                xy_distance = float(
                    container_geometry.center_xy_distance
                )
                xy_margin = float(
                    container_geometry.center_xy_margin
                )
                height_delta = float(
                    container_geometry.center_z_margin
                )
                above_margin = height_delta
                support_margin = float(min(
                    container_geometry.inside_xy_margin,
                    container_geometry.below_rim_margin,
                    container_geometry.above_floor_margin,
                ))
                on_margin = (
                    float(min(
                        container_geometry.margin,
                        contact_margin,
                    ))
                    if contact_diagnostics.valid
                    else -1.0e-6
                )
            else:
                on_margin = float("nan")
        else:
            # Ordinary object-on-support geometry uses the first clean RGB-D
            # support estimate.  Refreshing a plate after placement includes
            # the placed object in the support mask and raises its apparent
            # center/top, which caused the dominant false negatives.
            xy_distance = float(np.linalg.norm(
                object_center[:2] - baseline_center[:2]
            ))
            xy_margin = float(
                ordinary_xy_tolerance - xy_distance
            )
            ordinary_completion_xy_margin = float(
                ordinary_completion_xy_tolerance - xy_distance
            )
            height_delta = float(
                object_center[2] - baseline_center[2]
            )
            above_margin = float(
                height_delta - self.on_ordinary_min_above_m
            )
            surface_gap = float(abs(
                object_estimate.z_p05_m - baseline_z_p95
            ))
            support_margin = float(
                self.on_ordinary_surface_tolerance_m - surface_gap
            )
            on_margin = float(min(
                xy_margin,
                above_margin,
                support_margin,
            ))
            if self.on_ordinary_require_contact:
                on_margin = (
                    float(min(on_margin, contact_margin))
                    if contact_diagnostics.valid
                    else float("nan")
                )
            robust_candidate_margin = float(min(
                robust_xy_margin,
                above_margin,
                support_margin,
            ))
            if self.on_ordinary_require_contact:
                robust_candidate_margin = (
                    float(min(robust_candidate_margin, contact_margin))
                    if contact_diagnostics.valid
                    else float("nan")
                )
        env_id = int(env_id)
        step = int(step)
        handoff_used = False
        handoff_age_steps = -1
        handoff_margin = float("nan")

        if (
            container_mode
            and container_geometry.valid
            and contact_diagnostics.valid
            and object_source in {"eef_attach", "eef_attached"}
        ):
            if np.isfinite(on_margin) and on_margin >= 0.0:
                previous_step = self.on_container_geometry_last_steps.get(
                    env_id
                )
                geometry_count = (
                    self.on_container_geometry_positive_counts.get(
                        env_id, 0
                    ) + 1
                    if previous_step == step - 1
                    else 1
                )
                self.on_container_geometry_last_steps[env_id] = step
            else:
                geometry_count = 0
                self.on_container_geometry_last_steps.pop(env_id, None)
            self.on_container_geometry_positive_counts[env_id] = (
                geometry_count
            )
            if geometry_count >= 3:
                self.on_container_last_positive[env_id] = {
                    "step": step,
                    "margin": float(on_margin),
                }
        gate_reason = ""
        if spatial_relation is not None:
            confirmation_allowed = bool(spatial_margins["valid"])
            if spatial_relation == "right_of_plate":
                release_step = self.on_release_steps.get(env_id)
                release_recent = self._right_release_is_recent(
                    step=step,
                    release_step=release_step,
                )
                right_allowed, gate_reason, right_margin = (
                    self._right_spatial_confirmation_gate(
                        geometric_margin=on_margin,
                        spatial_motion=spatial_motion,
                        release_observed=release_recent,
                    )
                )
                on_margin = right_margin
                confirmation_allowed = bool(
                    confirmation_allowed and right_allowed
                )
        elif site_mode:
            # Cabinet/rack placement hides the target after release.  Use the
            # visual-to-EEF handoff and observed gripper release, but never
            # require a simulator contact or object pose.
            confirmation_allowed = bool(
                self.on_release_observed.get(env_id, False)
            )
        elif container_mode:
            confirmation_allowed, gate_reason = (
                self._on_confirmation_gate(
                    support_class=support_class,
                    object_source=object_source,
                    require_fresh_after_release=(
                        self.on_require_fresh_after_release
                    ),
                    release_observed=self.on_release_observed.get(
                        env_id, False
                    ),
                    post_release_fresh=self.on_post_release_fresh.get(
                        env_id, False
                    ),
                    geometric_margin=on_margin,
                )
            )
        else:
            confirmation_allowed = bool(
                not self.on_ordinary_require_release
                or self.on_release_observed.get(env_id, False)
            )
            if not confirmation_allowed:
                gate_reason = "ordinary_release_required"
            if (
                confirmation_allowed
                and self.on_ordinary_require_contact
                and not contact_diagnostics.valid
            ):
                confirmation_allowed = False
                gate_reason = "ordinary_contact_required"

        if (
            container_mode
            and not confirmation_allowed
            and self.on_release_observed.get(env_id, False)
            and not self.on_post_release_fresh.get(env_id, False)
            and object_source in {"eef_release", "eef_release_hold"}
        ):
            cached = self.on_container_last_positive.get(env_id)
            if cached is not None:
                age = step - int(cached["step"])
                cached_margin = float(cached["margin"])
                if (
                    0 <= age <= self.on_release_hold_steps
                    and np.isfinite(cached_margin)
                    and cached_margin >= 0.0
                ):
                    on_margin = cached_margin
                    confirmation_allowed = True
                    handoff_used = True
                    handoff_age_steps = age
                    handoff_margin = cached_margin
                    gate_reason = (
                        "container_pre_release_handoff:"
                        f"age={age}"
                    )
        # Keep individual geometric components observable, but make the
        # final atomic margin strictly negative until a post-release RGB-D
        # observation has occurred.  This protects both the shadow
        # confirmation state and any future reward consumer.
        if not confirmation_allowed:
            on_margin = float(min(on_margin, -1.0e-6))

        container_reason = (
            ""
            if not container_mode or container_geometry.valid
            else f"visual_container_on_invalid:{container_geometry.reason}"
        )
        final_container_reason = ""
        if spatial_relation is not None:
            valid = bool(spatial_margins["valid"])
            geometry_mode = f"spatial_relation:{spatial_relation}"
        elif site_mode:
            valid = True
            geometry_mode = "site_surface"
        elif container_mode:
            geometry_mode = "container_contact_proxy"
            contact_reason = str(
                getattr(contact_diagnostics, "reason", "") or ""
            )
            observation_reason = (
                gate_reason
                if handoff_used
                else (container_reason or contact_reason or gate_reason)
            )
            valid, on_margin, final_container_reason = (
                self._final_container_on_observation(
                    geometry_valid=bool(container_geometry.valid),
                    contact_valid=bool(contact_diagnostics.valid),
                    handoff_used=bool(handoff_used),
                    margin=on_margin,
                    reason=observation_reason,
                )
            )
        else:
            valid = bool(
                not self.on_ordinary_require_contact
                or contact_diagnostics.valid
            )
            geometry_mode = "ordinary_surface"

        # Unknown visual samples freeze the atom-level confirmation state.
        # A valid negative observation still resets the consecutive count.
        confirmation_margin = on_margin
        if not site_mode and not container_mode:
            confirmation_margin = float(min(
                on_margin,
                ordinary_completion_xy_margin,
            ))
        count = self.on_positive_counts.get(env_id, 0)
        fallback_count = self.on_ordinary_near_center_fallback_counts.get(
            env_id, 0
        )
        if valid:
            if confirmation_margin >= 0.0 and confirmation_allowed:
                count += 1
            else:
                count = 0
            self.on_positive_counts[env_id] = count
            ordinary_fallback_candidate = bool(
                self.on_ordinary_near_center_fallback_enabled
                and spatial_relation is None
                and not site_mode
                and not container_mode
                and confirmation_allowed
                and object_source == "fresh_rgbd"
                and contact_diagnostics.valid
                and np.isfinite(xy_distance)
                and xy_distance
                <= self.on_ordinary_near_center_fallback_xy_m
                and np.isfinite(contact_diagnostics.p01_m)
                and contact_diagnostics.p01_m
                <= self.on_ordinary_near_center_fallback_contact_p01_m
                and np.isfinite(above_margin)
                and above_margin >= 0.0
                and np.isfinite(support_margin)
                and support_margin >= 0.0
            )
            fallback_count = (
                fallback_count + 1
                if ordinary_fallback_candidate
                else 0
            )
            self.on_ordinary_near_center_fallback_counts[env_id] = (
                fallback_count
            )
        required_confirmation_steps = (
            self.on_front_confirmation_steps
            if spatial_relation == "front_of_stove"
            else (
                self.on_right_confirmation_steps
                if spatial_relation == "right_of_plate"
                else (
                    self.on_ordinary_confirmation_steps
                    if not site_mode and not container_mode
                    else self.on_confirmation_steps
                )
            )
        )
        if not self.on_confirmed.get(env_id, False):
            if count >= required_confirmation_steps:
                self.on_confirmed[env_id] = True
                self.on_confirmation_sources[env_id] = "standard"
            elif (
                spatial_relation is None
                and not site_mode
                and not container_mode
                and fallback_count
                >= self.on_ordinary_near_center_fallback_confirmation_steps
            ):
                self.on_confirmed[env_id] = True
                self.on_confirmation_sources[env_id] = (
                    "near_center_p01_fallback"
                )
        confirmed = bool(self.on_confirmed.get(env_id, False))
        return {
            "valid": valid,
            "xy_distance_m": xy_distance,
            "xy_margin": xy_margin,
            "completion_xy_margin": ordinary_completion_xy_margin,
            "height_delta_m": height_delta,
            "above_margin": above_margin,
            "surface_gap_m": surface_gap,
            "support_margin": support_margin,
            "release_margin": release_margin,
            "margin": on_margin,
            "confirmed": confirmed,
            "reason": (
                (
                    (
                        f"visual_spatial_relation_gate:{gate_reason}"
                        if gate_reason
                        else f"visual_spatial_relation:{spatial_relation}"
                    )
                    if spatial_margins["valid"]
                    else (
                        "visual_spatial_relation_invalid:"
                        f"{spatial_relation}:{spatial_margins['reason']}"
                    )
                )
                if spatial_relation is not None else (
                    (
                        final_container_reason
                        if container_mode
                        else (container_reason or gate_reason)
                    )
                )
            ),
            "geometry_mode": geometry_mode,
            "inside_xy_margin": float(
                container_geometry.inside_xy_margin
                if container_mode and container_geometry.valid
                else np.nan
            ),
            "below_rim_margin": float(
                container_geometry.below_rim_margin
                if container_mode and container_geometry.valid
                else np.nan
            ),
            "above_floor_margin": float(
                container_geometry.above_floor_margin
                if container_mode and container_geometry.valid
                else np.nan
            ),
            "handoff_used": handoff_used,
            "handoff_age_steps": handoff_age_steps,
            "handoff_margin": handoff_margin,
            "contact_valid": bool(contact_diagnostics.valid),
            "contact_reason": str(contact_diagnostics.reason),
            "contact_p01_m": float(contact_diagnostics.p01_m),
            "contact_p05_m": float(contact_diagnostics.p05_m),
            "contact_p10_m": float(contact_diagnostics.p10_m),
            "contact_margin": contact_margin,
            "fresh_after_release": bool(
                self.on_post_release_fresh.get(env_id, False)
            ),
            "object_center_x_m": float(object_center[0]),
            "object_center_y_m": float(object_center[1]),
            "object_center_z_m": float(object_center[2]),
            "support_center_x_m": float(support_center[0]),
            "support_center_y_m": float(support_center[1]),
            "support_center_z_m": float(support_center[2]),
            "support_locked": bool(spatial_relation is not None),
            "object_bottom_z_m": float(object_estimate.z_p05_m),
            "support_z_p95_m": float(support_estimate.z_p95_m),
            "support_z_p99_m": support_z_p99,
            "support_z_p995_m": support_z_p995,
            "support_baseline_center_x_m": float(baseline_center[0]),
            "support_baseline_center_y_m": float(baseline_center[1]),
            "support_baseline_center_z_m": float(baseline_center[2]),
            "support_baseline_z_p95_m": baseline_z_p95,
            "support_center_drift_m": float(np.linalg.norm(
                support_center - baseline_center
            )),
            "support_z_p95_drift_m": float(
                support_estimate.z_p95_m - baseline_z_p95
            ),
            "site_center_delta_q99_m": site_center_delta_q99,
            "site_bottom_gap_q99_m": site_bottom_gap_q99,
            "spatial_relation": str(spatial_relation or ""),
            "anchor_u": float(spatial["anchor_u"]),
            "anchor_v": float(spatial["anchor_v"]),
            "spatial_x_margin": float(spatial_margins["x_margin"]),
            "spatial_y_margin": float(spatial_margins["y_margin"]),
            "spatial_z_margin": float(spatial_margins["z_margin"]),
            "spatial_directional_offset_m": float(
                spatial_margins["directional_offset_m"]
            ),
            "spatial_lateral_offset_m": float(
                spatial_margins["lateral_offset_m"]
            ),
            "spatial_anchor_direction_half_extent_m": float(
                spatial_margins["anchor_direction_half_extent_m"]
            ),
            "spatial_anchor_lateral_half_extent_m": float(
                spatial_margins["anchor_lateral_half_extent_m"]
            ),
            "spatial_displacement_m": float(
                spatial_motion["displacement_m"]
            ),
            "spatial_frame_motion_m": float(
                spatial_motion["frame_motion_m"]
            ),
            "spatial_motion_observed": bool(
                spatial_motion["motion_observed"]
            ),
            "spatial_stable": bool(spatial_motion["stable"]),
            "spatial_pick_proxy_margin": float(
                spatial_pick_proxy_margin
            ),
            "spatial_footprint_box_valid": bool(
                spatial_footprint["box_valid"]
            ),
            "spatial_footprint_box_reason": str(
                spatial_footprint["box_reason"]
            ),
            "spatial_footprint_x_margin": float(
                spatial_footprint["x_margin"]
            ),
            "spatial_footprint_y_margin": float(
                spatial_footprint["y_margin"]
            ),
            "spatial_footprint_bottom_z_margin": float(
                spatial_footprint["bottom_z_margin"]
            ),
            "spatial_footprint_top_z_margin": float(
                spatial_footprint["top_z_margin"]
            ),
            "spatial_footprint_full_z_margin": float(
                spatial_footprint["full_z_margin"]
            ),
            "spatial_footprint_box_candidate_margin": float(
                spatial_footprint["box_candidate_margin"]
            ),
            "spatial_footprint_directional_valid": bool(
                spatial_footprint["directional_valid"]
            ),
            "spatial_footprint_directional_reason": str(
                spatial_footprint["directional_reason"]
            ),
            "spatial_footprint_directional_margin": float(
                spatial_footprint["directional_margin"]
            ),
            "spatial_footprint_lateral_margin": float(
                spatial_footprint["lateral_margin"]
            ),
            "spatial_footprint_surface_margin": float(
                spatial_footprint["surface_margin"]
            ),
            "spatial_footprint_directional_candidate_margin": float(
                spatial_footprint["directional_candidate_margin"]
            ),
            "object_x_q05_m": float(object_xy_lower[0]),
            "object_x_q95_m": float(object_xy_upper[0]),
            "object_y_q05_m": float(object_xy_lower[1]),
            "object_y_q95_m": float(object_xy_upper[1]),
            "support_baseline_x_q05_m": float(
                support_baseline_xy_lower[0]
            ),
            "support_baseline_x_q95_m": float(
                support_baseline_xy_upper[0]
            ),
            "support_baseline_y_q05_m": float(
                support_baseline_xy_lower[1]
            ),
            "support_baseline_y_q95_m": float(
                support_baseline_xy_upper[1]
            ),
            "robust_xy_distance_m": robust_xy_distance,
            "robust_xy_margin": robust_xy_margin,
            "robust_candidate_margin": robust_candidate_margin,
            **spatial,
        }

    def _resolve_on_support(
        self,
        *,
        env_id: int,
        step: int,
        support_identity: str,
        support_class: str,
        support_candidates: list[dict],
        object_center: np.ndarray,
    ) -> tuple[object | None, float, str]:
        """Use fresh RGB-D support geometry, or its visual-only cache.

        Support objects such as a plate are commonly occluded after placement.
        A fresh YOLOE+depth estimate refreshes the cache.  When the support is
        hidden, the cached world-frame geometry is reused for a bounded number
        of steps.  No simulator object/site/joint/contact state is consulted.
        """
        env_id = int(env_id)
        step = int(step)
        key = (env_id, str(support_identity))
        cached = self.on_support_states.get(key)
        support_track_reason = ""

        # A named support must keep its visual identity when another instance
        # of the same class remains visible.  This is common with two plates:
        # once the target plate is covered by the placed object, selecting the
        # only remaining plate would silently move the cached support across
        # the table.
        if (
            support_candidates
            and cached is not None
            and cached["support_class"] == str(support_class)
            and not self._locks_spatial_support(support_identity)
        ):
            cached_center = np.asarray(
                cached["center_world"], dtype=np.float64
            )
            tracked = min(
                support_candidates,
                key=lambda item: np.linalg.norm(
                    np.asarray(
                        item["estimate"].center_world,
                        dtype=np.float64,
                    )[:2]
                    - cached_center[:2]
                ),
            )
            track_distance = float(np.linalg.norm(
                np.asarray(
                    tracked["estimate"].center_world,
                    dtype=np.float64,
                )[:2]
                - cached_center[:2]
            ))
            if track_distance <= self.track_gate_m:
                support_candidates = [tracked]
            else:
                support_candidates = []
                support_track_reason = (
                    "support_track_gate:"
                    f"distance={track_distance:.6f}>"
                    f"{self.track_gate_m:.6f}"
                )

        if support_candidates:
            support = min(
                support_candidates,
                key=lambda item: np.linalg.norm(
                    np.asarray(
                        item["estimate"].center_world,
                        dtype=np.float64,
                    )[:2]
                    - np.asarray(object_center, dtype=np.float64)[:2]
                ),
            )
            estimate = support["estimate"]
            points_world = np.asarray(
                getattr(estimate, "points_world", None),
                dtype=np.float64,
            )
            if (
                points_world.ndim == 2
                and points_world.shape[1] >= 2
                and len(points_world) > 0
            ):
                finite_xy = points_world[
                    np.all(np.isfinite(points_world[:, :2]), axis=1),
                    :2,
                ]
            else:
                finite_xy = np.empty((0, 2), dtype=np.float64)
            if len(finite_xy) > 0:
                xy_q05_m, xy_q95_m = np.quantile(
                    finite_xy,
                    (0.05, 0.95),
                    axis=0,
                )
            else:
                xy_q05_m = np.full(2, np.nan, dtype=np.float64)
                xy_q95_m = np.full(2, np.nan, dtype=np.float64)
            points_z = points_world[
                np.isfinite(points_world[:, 2]), 2
            ] if points_world.ndim == 2 and points_world.shape[1] >= 3 else np.empty(0)
            if len(points_z) > 0:
                z_p99_m, z_p995_m = np.quantile(
                    points_z,
                    (0.99, 0.995),
                )
            else:
                z_p99_m = float("nan")
                z_p995_m = float("nan")
            # The fresh estimate is also consumed immediately, so attach the
            # visual footprint without changing VisualApproachEstimate.
            estimate = SimpleNamespace(
                center_world=tuple(estimate.center_world),
                z_p05_m=float(estimate.z_p05_m),
                z_p95_m=float(estimate.z_p95_m),
                xy_q05_m=tuple(float(x) for x in xy_q05_m),
                xy_q95_m=tuple(float(x) for x in xy_q95_m),
                z_p99_m=float(z_p99_m),
                z_p995_m=float(z_p995_m),
                points_world=points_world.copy(),
            )
            confidence = float(
                support["detection"]["confidence"]
            )
            cached = self.on_support_states.get(key)
            if (
                cached is not None
                and cached["support_class"] == str(support_class)
            ):
                baseline_center_world = tuple(cached.get(
                    "baseline_center_world", cached["center_world"]
                ))
                baseline_z_p95_m = float(cached.get(
                    "baseline_z_p95_m", cached["z_p95_m"]
                ))
                baseline_points_world = np.asarray(
                    cached.get("baseline_points_world", points_world),
                    dtype=np.float64,
                ).copy()
            else:
                baseline_center_world = tuple(estimate.center_world)
                baseline_z_p95_m = float(estimate.z_p95_m)
                baseline_points_world = points_world.copy()
            estimate.baseline_center_world = baseline_center_world
            estimate.baseline_z_p95_m = baseline_z_p95_m
            estimate.baseline_points_world = baseline_points_world
            if (
                self._locks_spatial_support(support_identity)
                and cached is not None
                and cached["support_class"] == str(support_class)
            ):
                locked_estimate = SimpleNamespace(
                    center_world=tuple(cached["center_world"]),
                    z_p05_m=float(cached["z_p05_m"]),
                    z_p95_m=float(cached["z_p95_m"]),
                    xy_q05_m=tuple(cached["xy_q05_m"]),
                    xy_q95_m=tuple(cached["xy_q95_m"]),
                    z_p99_m=float(cached["z_p99_m"]),
                    z_p995_m=float(cached["z_p995_m"]),
                    baseline_center_world=baseline_center_world,
                    baseline_z_p95_m=baseline_z_p95_m,
                    baseline_points_world=baseline_points_world,
                )
                age = step - int(cached["step"])
                return (
                    locked_estimate,
                    confidence,
                    f"spatial_support_lock:age={age};fresh_seen=1",
                )
            self.on_support_states[key] = {
                "support_class": str(support_class),
                "step": step,
                "confidence": confidence,
                "center_world": tuple(estimate.center_world),
                "z_p05_m": float(estimate.z_p05_m),
                "z_p95_m": float(estimate.z_p95_m),
                "xy_q05_m": tuple(estimate.xy_q05_m),
                "xy_q95_m": tuple(estimate.xy_q95_m),
                "z_p99_m": float(estimate.z_p99_m),
                "z_p995_m": float(estimate.z_p995_m),
                "baseline_center_world": baseline_center_world,
                "baseline_z_p95_m": baseline_z_p95_m,
                "baseline_points_world": baseline_points_world,
            }
            return estimate, confidence, ""

        if cached is None or self.on_support_cache_steps <= 0:
            return None, float("nan"), (
                f"on_target_not_detected:{support_class}"
            )

        age = step - int(cached["step"])
        if (
            age < 0
            or age > self.on_support_cache_steps
            or cached["support_class"] != str(support_class)
        ):
            return None, float("nan"), (
                f"on_target_not_detected:{support_class}"
            )

        estimate = SimpleNamespace(
            center_world=tuple(cached["center_world"]),
            z_p05_m=float(cached["z_p05_m"]),
            z_p95_m=float(cached["z_p95_m"]),
            xy_q05_m=tuple(
                cached.get("xy_q05_m", (np.nan, np.nan))
            ),
            xy_q95_m=tuple(
                cached.get("xy_q95_m", (np.nan, np.nan))
            ),
            z_p99_m=float(cached.get("z_p99_m", np.nan)),
            z_p995_m=float(cached.get("z_p995_m", np.nan)),
            baseline_center_world=tuple(cached.get(
                "baseline_center_world", cached["center_world"]
            )),
            baseline_z_p95_m=float(cached.get(
                "baseline_z_p95_m", cached["z_p95_m"]
            )),
            baseline_points_world=np.asarray(
                cached.get("baseline_points_world", np.empty((0, 3))),
                dtype=np.float64,
            ).copy(),
        )
        return (
            estimate,
            float(cached["confidence"]),
            (
                f"support_cache:age={age}"
                + (
                    f";{support_track_reason}"
                    if support_track_reason else ""
                )
            ),
        )

    def _empty_on_fields(
        self,
        env_id: int,
        on_target_object: str | None,
        oracle_on_margin: float,
        reason: str,
    ) -> dict:
        nan = float("nan")
        return {
            "on_target_object": str(on_target_object or ""),
            "on_target_class": str(
                self._object_on_target_class(on_target_object) or ""
            ),
            "oracle_on_margin": float(oracle_on_margin),
            "visual_on_valid": False,
            "visual_on_reason": str(reason),
            "visual_on_target_confidence": nan,
            "visual_on_xy_distance_m": nan,
            "visual_on_xy_margin": nan,
            "visual_on_completion_xy_margin": nan,
            "visual_on_height_delta_m": nan,
            "visual_on_above_margin": nan,
            "visual_on_surface_gap_m": nan,
            "visual_on_support_margin": nan,
            "visual_on_release_margin": nan,
            "visual_on_margin": nan,
            "visual_on_confirmed": bool(
                self.on_confirmed.get(int(env_id), False)
            ),
            "visual_on_object_source": "unavailable",
            "visual_on_attached": bool(
                self.on_object_states.get(int(env_id), {}).get(
                    "attached", False
                )
            ),
            "visual_on_released": bool(
                self.on_object_states.get(int(env_id), {}).get(
                    "released", False
                )
            ),
            "visual_on_geometry_mode": "unavailable",
            "visual_on_inside_xy_margin": nan,
            "visual_on_below_rim_margin": nan,
            "visual_on_above_floor_margin": nan,
            "visual_on_handoff_used": False,
            "visual_on_handoff_age_steps": -1,
            "visual_on_handoff_margin": nan,
            "visual_on_contact_valid": False,
            "visual_on_contact_reason": "unavailable",
            "visual_on_contact_p01_m": nan,
            "visual_on_contact_p05_m": nan,
            "visual_on_contact_p10_m": nan,
            "visual_on_contact_margin": nan,
            "visual_on_spatial_displacement_m": nan,
            "visual_on_spatial_frame_motion_m": nan,
            "visual_on_spatial_motion_observed": bool(
                self.on_spatial_motion_observed.get(int(env_id), False)
            ),
            "visual_on_spatial_stable": False,
            "visual_on_spatial_pick_proxy_margin": nan,
        }

    @staticmethod
    def _gripper_width(raw_obs: dict) -> float:
        """Read proprioceptive gripper width, never contact state."""
        qpos = np.asarray(
            raw_obs["robot0_gripper_qpos"],
            dtype=np.float64,
        ).reshape(-1)
        if len(qpos) < 2:
            raise ValueError(
                "robot0_gripper_qpos must contain two joints"
            )
        return float(qpos[0] - qpos[1])

    def _grasp_near_margin(self, estimate) -> float:
        """Select the configured visual proximity predicate."""
        fields = {
            "center": "robustness",
            "surface_p01": "surface_p01_robustness",
            "surface_p05": "surface_p05_robustness",
            "surface_p10": "surface_p10_robustness",
        }
        return float(getattr(
            estimate,
            fields[self.grasp_near_mode],
        ))

    def _break_grasp_confirmation(self, env_id: int) -> None:
        """A missing visual frame breaks consecutive evidence."""
        self.grasp_positive_counts[int(env_id)] = 0
        self.grasp_terminal_positive_counts[int(env_id)] = 0

    def _visual_grasp_components(
        self,
        *,
        env_id: int,
        step: int,
        visual_center: np.ndarray,
        eef: np.ndarray,
        gripper_width_m: float,
        near_margin: float,
    ) -> dict:
        """Compute contact-free visual-proprioceptive grasp margins."""
        env_id = int(env_id)
        step = int(step)
        center = np.asarray(
            visual_center,
            dtype=np.float64,
        ).reshape(3)
        eef = np.asarray(eef, dtype=np.float64).reshape(3)

        close_margin = float(
            self.close_threshold_m - gripper_width_m
        )
        nan = float("nan")
        result = {
            "near_margin": float(near_margin),
            "close_margin": close_margin,
            "comove_error_m": nan,
            "comove_margin": nan,
            "eef_motion_m": nan,
            "motion_margin": nan,
            "displacement_m": nan,
            "displacement_margin": nan,
            "grasp_margin": nan,
            "history_ready": False,
            "terminal_candidate": False,
            "terminal_margin": nan,
            "terminal_count": int(
                self.grasp_terminal_positive_counts.get(env_id, 0)
            ),
            "terminal_confirmed": bool(
                self.grasp_terminal_confirmed.get(env_id, False)
            ),
            "confirmation_source": str(
                self.grasp_confirmation_sources.get(env_id, "none")
            ),
            "confirmed": bool(
                self.grasp_confirmed.get(env_id, False)
            ),
        }

        history = self.grasp_histories.setdefault(env_id, [])
        reference_center = self.grasp_reference_centers.setdefault(
            env_id,
            center.copy(),
        )
        displacement = float(
            np.linalg.norm(center - reference_center)
        )
        displacement_margin = float(
            displacement - self.displacement_min_m
        )
        result.update({
            "displacement_m": displacement,
            "displacement_margin": displacement_margin,
        })
        lag = self.grasp_history_lag_steps

        # Large detection gaps must not be treated as consecutive motion.
        if history and step - history[-1][0] > 2 * lag:
            history.clear()
            self._break_grasp_confirmation(env_id)

        history.append((step, center.copy(), eef.copy()))
        max_history = lag + 1
        if len(history) > max_history:
            del history[:-max_history]

        if len(history) < max_history:
            self._break_grasp_confirmation(env_id)
            result["terminal_count"] = 0
            return result

        previous_step, previous_center, previous_eef = history[0]
        if step - previous_step > 2 * lag:
            self._break_grasp_confirmation(env_id)
            result["terminal_count"] = 0
            return result

        relative_now = center - eef
        relative_previous = previous_center - previous_eef
        comove_error = float(
            np.linalg.norm(relative_now - relative_previous)
        )
        eef_motion = float(np.linalg.norm(eef - previous_eef))
        comove_margin = float(
            self.comove_tolerance_m - comove_error
        )
        motion_margin = float(
            eef_motion - self.motion_min_m
        )
        standard_grasp_margin = float(min(
            near_margin,
            close_margin,
            comove_margin,
            motion_margin,
            displacement_margin,
        ))

        if standard_grasp_margin >= 0.0:
            count = self.grasp_positive_counts.get(env_id, 0) + 1
        else:
            count = 0
        self.grasp_positive_counts[env_id] = count

        terminal_margin = _terminal_grasp_candidate_margin(
            gripper_width_m=gripper_width_m,
            near_margin=near_margin,
            close_margin=close_margin,
            comove_margin=comove_margin,
            motion_margin=motion_margin,
            displacement_margin=displacement_margin,
            max_width_m=self.grasp_terminal_width_m,
            comove_slack_m=self.grasp_terminal_comove_slack_m,
            motion_slack_m=self.grasp_terminal_motion_slack_m,
        )
        terminal_candidate = bool(np.isfinite(terminal_margin))
        if terminal_candidate:
            terminal_count = (
                self.grasp_terminal_positive_counts.get(env_id, 0) + 1
            )
        else:
            terminal_count = 0
        self.grasp_terminal_positive_counts[env_id] = terminal_count
        terminal_confirmed = bool(
            terminal_count >= self.grasp_terminal_confirmation_steps
        )
        if terminal_confirmed:
            self.grasp_terminal_confirmed[env_id] = True

        grasp_margin = standard_grasp_margin
        if terminal_confirmed:
            grasp_margin = float(max(grasp_margin, terminal_margin))

        if count >= self.grasp_confirmation_steps:
            self.grasp_confirmed[env_id] = True
            if self.grasp_confirmation_sources.get(env_id, "none") == "none":
                self.grasp_confirmation_sources[env_id] = "standard"
        elif terminal_confirmed:
            self.grasp_confirmed[env_id] = True
            if self.grasp_confirmation_sources.get(env_id, "none") == "none":
                self.grasp_confirmation_sources[env_id] = "terminal_closure"

        result.update({
            "comove_error_m": comove_error,
            "comove_margin": comove_margin,
            "eef_motion_m": eef_motion,
            "motion_margin": motion_margin,
            "displacement_m": displacement,
            "displacement_margin": displacement_margin,
            "grasp_margin": grasp_margin,
            "history_ready": True,
            "terminal_candidate": terminal_candidate,
            "terminal_margin": terminal_margin,
            "terminal_count": terminal_count,
            "terminal_confirmed": bool(
                self.grasp_terminal_confirmed.get(env_id, False)
            ),
            "confirmation_source": str(
                self.grasp_confirmation_sources.get(env_id, "none")
            ),
            "confirmed": bool(
                self.grasp_confirmed.get(env_id, False)
            ),
        })
        return result

    def _invalid_record(
        self,
        *,
        env_id,
        step,
        task_id,
        target_object,
        target_class,
        reason,
        two_finger_grasp,
        oracle_distance_m,
        oracle_pick_margin,
        on_target_object=None,
        oracle_on_margin=float("nan"),
        eef,
        oracle_center,
        gripper_width_m=float("nan"),
    ) -> VisualSTLShadowRecord:
        oracle_rho = (
            self.grasp_radius_m
            - float(oracle_distance_m)
        )

        nan = float("nan")

        return VisualSTLShadowRecord(
            step=int(step),
            task_id=int(task_id),
            target_object=str(target_object),
            target_class=str(target_class or ""),
            selector="unavailable",
            valid=False,
            reason=str(reason),
            visual_fresh=False,
            visual_held=False,
            visual_hold_age_steps=0,
            confidence=nan,
            mask_pixels=0,
            depth_pixels=0,
            two_finger_grasp=bool(two_finger_grasp),
            oracle_distance_m=float(
                oracle_distance_m
            ),
            visual_distance_m=nan,
            oracle_robustness=oracle_rho,
            visual_robustness=nan,
            visual_surface_p01_distance_m=nan,
            visual_surface_p05_distance_m=nan,
            visual_surface_p10_distance_m=nan,
            visual_surface_p01_robustness=nan,
            visual_surface_p05_robustness=nan,
            visual_surface_p10_robustness=nan,
            oracle_pick_margin=float(
                oracle_pick_margin
            ),
            visual_pick_margin=nan,
            eef_x=float(eef[0]),
            eef_y=float(eef[1]),
            eef_z=float(eef[2]),
            oracle_x=float(oracle_center[0]),
            oracle_y=float(oracle_center[1]),
            oracle_z=float(oracle_center[2]),
            visual_x=nan,
            visual_y=nan,
            visual_z=nan,
            gripper_width_m=float(gripper_width_m),
            visual_near_margin=nan,
            visual_close_margin=(
                float(self.close_threshold_m - gripper_width_m)
                if np.isfinite(gripper_width_m)
                else nan
            ),
            visual_comove_error_m=nan,
            visual_comove_margin=nan,
            eef_motion_m=nan,
            visual_motion_margin=nan,
            visual_displacement_m=nan,
            visual_displacement_margin=nan,
            visual_grasp_margin=nan,
            visual_grasp_history_ready=False,
            visual_grasp_confirmed=bool(
                self.grasp_confirmed.get(int(env_id), False)
            ),
            visual_grasp_terminal_candidate=False,
            visual_grasp_terminal_margin=nan,
            visual_grasp_terminal_count=int(
                self.grasp_terminal_positive_counts.get(int(env_id), 0)
            ),
            visual_grasp_terminal_confirmed=bool(
                self.grasp_terminal_confirmed.get(int(env_id), False)
            ),
            visual_grasp_confirmation_source=str(
                self.grasp_confirmation_sources.get(int(env_id), "none")
            ),
            **self._empty_on_fields(
                env_id,
                on_target_object,
                oracle_on_margin,
                "visual_target_unavailable",
            ),
        )

    def _held_record(
        self,
        *,
        env_id,
        step,
        task_id,
        target_object,
        target_class,
        reason,
        two_finger_grasp,
        oracle_distance_m,
        oracle_pick_margin,
        on_target_object,
        oracle_on_margin,
        eef,
        oracle_center,
        gripper_width_m,
    ) -> VisualSTLShadowRecord | None:
        """Bridge a short target dropout without inventing fresh evidence."""
        env_id = int(env_id)
        previous = self.last_fresh_records.get(env_id)
        previous_step = self.last_fresh_steps.get(env_id)

        if (
            self.hold_steps == 0
            or previous is None
            or previous_step is None
            or previous.target_object != str(target_object)
        ):
            return None

        hold_age = int(step) - int(previous_step)
        if hold_age < 1 or hold_age > self.hold_steps:
            return None

        center = np.asarray(
            [previous.visual_x, previous.visual_y, previous.visual_z],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(center)):
            return None

        eef = np.asarray(eef, dtype=np.float64).reshape(3)
        center_distance = float(np.linalg.norm(center - eef))

        def held_surface(previous_distance):
            offset = max(
                0.0,
                float(previous.visual_distance_m)
                - float(previous_distance),
            )
            return max(0.0, center_distance - offset)

        p01_distance = held_surface(
            previous.visual_surface_p01_distance_m
        )
        p05_distance = held_surface(
            previous.visual_surface_p05_distance_m
        )
        p10_distance = held_surface(
            previous.visual_surface_p10_distance_m
        )
        center_rho = self.visual_center_radius_m - center_distance
        p01_rho = self.surface_radius_m - p01_distance
        p05_rho = self.surface_radius_m - p05_distance
        p10_rho = self.surface_radius_m - p10_distance
        near_fields = {
            "center": center_rho,
            "surface_p01": p01_rho,
            "surface_p05": p05_rho,
            "surface_p10": p10_rho,
        }
        near_margin = float(near_fields[self.grasp_near_mode])
        close_margin = float(
            self.close_threshold_m - gripper_width_m
        )
        reference = self.grasp_reference_centers.get(env_id)
        displacement = (
            float(np.linalg.norm(center - reference))
            if reference is not None
            else float("nan")
        )
        displacement_margin = (
            displacement - self.displacement_min_m
            if np.isfinite(displacement)
            else float("nan")
        )

        dynamic_margins = (
            previous.visual_comove_margin,
            previous.visual_motion_margin,
            displacement_margin,
        )
        history_ready = bool(
            previous.visual_grasp_history_ready
            and all(np.isfinite(value) for value in dynamic_margins)
        )
        grasp_margin = (
            float(min(
                near_margin,
                close_margin,
                *dynamic_margins,
            ))
            if history_ready
            else float("nan")
        )
        terminal_margin = (
            _terminal_grasp_candidate_margin(
                gripper_width_m=gripper_width_m,
                near_margin=near_margin,
                close_margin=close_margin,
                comove_margin=float(previous.visual_comove_margin),
                motion_margin=float(previous.visual_motion_margin),
                displacement_margin=displacement_margin,
                max_width_m=self.grasp_terminal_width_m,
                comove_slack_m=self.grasp_terminal_comove_slack_m,
                motion_slack_m=self.grasp_terminal_motion_slack_m,
            )
            if history_ready
            else float("nan")
        )
        if (
            previous.visual_grasp_terminal_confirmed
            and np.isfinite(terminal_margin)
        ):
            grasp_margin = float(max(grasp_margin, terminal_margin))

        # Held frames make the trace continuous, but never increment the
        # consecutive-confirmation counter. Only a fresh YOLOE observation
        # can contribute new evidence.
        confidence_scale = max(
            0.0,
            1.0 - hold_age / float(self.hold_steps + 1),
        )

        return VisualSTLShadowRecord(
            step=int(step),
            task_id=int(task_id),
            target_object=str(target_object),
            target_class=str(target_class or ""),
            selector=f"hold_last_fresh:age={hold_age}",
            valid=True,
            reason=str(reason),
            visual_fresh=False,
            visual_held=True,
            visual_hold_age_steps=hold_age,
            confidence=float(previous.confidence * confidence_scale),
            mask_pixels=0,
            depth_pixels=0,
            two_finger_grasp=bool(two_finger_grasp),
            oracle_distance_m=float(oracle_distance_m),
            visual_distance_m=center_distance,
            oracle_robustness=(
                self.grasp_radius_m - float(oracle_distance_m)
            ),
            visual_robustness=center_rho,
            visual_surface_p01_distance_m=p01_distance,
            visual_surface_p05_distance_m=p05_distance,
            visual_surface_p10_distance_m=p10_distance,
            visual_surface_p01_robustness=p01_rho,
            visual_surface_p05_robustness=p05_rho,
            visual_surface_p10_robustness=p10_rho,
            oracle_pick_margin=float(oracle_pick_margin),
            visual_pick_margin=grasp_margin,
            eef_x=float(eef[0]),
            eef_y=float(eef[1]),
            eef_z=float(eef[2]),
            oracle_x=float(oracle_center[0]),
            oracle_y=float(oracle_center[1]),
            oracle_z=float(oracle_center[2]),
            visual_x=float(center[0]),
            visual_y=float(center[1]),
            visual_z=float(center[2]),
            gripper_width_m=float(gripper_width_m),
            visual_near_margin=near_margin,
            visual_close_margin=close_margin,
            visual_comove_error_m=float(
                previous.visual_comove_error_m
            ),
            visual_comove_margin=float(
                previous.visual_comove_margin
            ),
            eef_motion_m=float(previous.eef_motion_m),
            visual_motion_margin=float(
                previous.visual_motion_margin
            ),
            visual_displacement_m=displacement,
            visual_displacement_margin=float(
                displacement_margin
            ),
            visual_grasp_margin=grasp_margin,
            visual_grasp_history_ready=history_ready,
            visual_grasp_confirmed=bool(
                self.grasp_confirmed.get(env_id, False)
            ),
            # A held frame propagates an already confirmed continuous
            # terminal margin, but is never a new terminal candidate.
            visual_grasp_terminal_candidate=False,
            visual_grasp_terminal_margin=float(terminal_margin),
            visual_grasp_terminal_count=int(
                self.grasp_terminal_positive_counts.get(env_id, 0)
            ),
            visual_grasp_terminal_confirmed=bool(
                self.grasp_terminal_confirmed.get(env_id, False)
            ),
            visual_grasp_confirmation_source=str(
                self.grasp_confirmation_sources.get(env_id, "none")
            ),
            **self._empty_on_fields(
                env_id,
                on_target_object,
                oracle_on_margin,
                "held_pick_frame_not_on_evidence",
            ),
        )

    def _record_visual_in(
        self,
        *,
        env_id: int,
        step: int,
        task_id: int,
        target_object: str,
        container_object: str | None,
        oracle_in_margin: float,
        gripper_width_m: float,
        eef: np.ndarray,
        selected=None,
        estimated=(),
        reason: str = "",
        target_source: str = "unavailable",
        target_selector: str = "unavailable",
    ) -> None:
        """Record In from RGB-D geometry or its visual-to-EEF handoff."""
        if not self.in_enabled or container_object is None:
            return

        env_id = int(env_id)
        target_class = _object_class(target_object) or ""
        container_class = _object_class(container_object) or ""
        container_text = str(container_object).lower()
        if "drawer" in container_text or "cabinet" in container_text:
            container_geometry = "drawer"
        elif "microwave" in container_text:
            container_geometry = "microwave"
        elif "caddy" in container_text:
            container_geometry = "caddy_back"
        else:
            container_geometry = "basket"
        eef_world = np.asarray(eef, dtype=np.float64).reshape(3)
        previous_topology_temporal_count = int(
            self.in_caddy_topology_temporal_counts.get(env_id, 0)
        )
        previous_central_temporal_count = int(
            self.in_caddy_central_temporal_counts.get(env_id, 0)
        )
        previous_cached_topology_count = int(
            self.in_caddy_cached_topology_counts.get(env_id, 0)
        )
        previous_static_baseline_count = int(
            self.in_caddy_static_baseline_temporal_counts.get(env_id, 0)
        )
        previous_cached_topology_half_widths = tuple(
            self.in_caddy_cached_topology_half_widths.get(env_id, ())
        )
        # Any unavailable or non-fresh frame must break all dwell streaks.
        self.in_caddy_topology_temporal_counts[env_id] = 0
        self.in_caddy_central_temporal_counts[env_id] = 0
        self.in_caddy_cached_topology_counts[env_id] = 0
        self.in_caddy_static_baseline_temporal_counts[env_id] = 0
        if container_geometry != "caddy_back":
            self.in_caddy_cached_topology_half_widths[env_id] = []
            self.in_caddy_reobservation_last_steps.pop(env_id, None)
            self.in_caddy_reobservation_bouts[env_id] = 0
            self.in_caddy_reobservation_margins.pop(env_id, None)
            self.in_caddy_confirmed_event_bridge_margins.pop(env_id, None)
            self.in_caddy_confirmed_event_bridge_steps.pop(env_id, None)
            self.in_caddy_release_counts[env_id] = 0
            self.in_caddy_dual_stage_release_counts[env_id] = 0
            self.in_caddy_opening_counts[env_id] = 0
            self.in_caddy_opening_steps.pop(env_id, None)
            self.in_caddy_opening_centers.pop(env_id, None)
            self.in_caddy_opening_max_target_displacements.pop(env_id, None)
            self.in_caddy_opening_evidence.pop(env_id, None)
            self.in_caddy_opening_geometry_margins.pop(env_id, None)
            self.in_caddy_opening_sources.pop(env_id, None)
            self.in_caddy_opening_candidate_sources.pop(env_id, None)
            self.in_caddy_observed_opening_steps.pop(env_id, None)
            self.in_caddy_observed_opening_centers.pop(env_id, None)
            self.in_caddy_observed_opening_evidence.pop(env_id, None)
            self.in_caddy_observed_opening_geometry_margins.pop(
                env_id, None
            )
            self.in_caddy_observed_opening_sources.pop(env_id, None)
            self.in_caddy_separation_steps.pop(env_id, None)
            self.in_caddy_separation_evidence.pop(env_id, None)
            self.in_caddy_contradiction_counts[env_id] = 0
        nan = float("nan")
        cached_opening_center = self.in_caddy_opening_centers.get(env_id)
        fields = dict(
            container_geometry=container_geometry,
            valid=False,
            reason=str(reason or "visual_in_unavailable"),
            visual_fresh=False,
            target_source=str(target_source),
            visual_hold_age_steps=0,
            target_confidence=nan,
            container_confidence=nan,
            target_selector=str(target_selector),
            target_camera_name="unavailable",
            container_camera_name="unavailable",
            same_camera_geometry=False,
            target_candidate_count=0,
            target_same_camera_candidate_count=0,
            container_candidate_count=0,
            container_same_camera_candidate_count=0,
            target_mask_pixels=0,
            target_mask_centroid_u_px=nan,
            target_mask_centroid_v_px=nan,
            target_mask_bbox_left_px=-1,
            target_mask_bbox_top_px=-1,
            target_mask_bbox_right_px=-1,
            target_mask_bbox_bottom_px=-1,
            container_mask_pixels=0,
            container_mask_centroid_u_px=nan,
            container_mask_centroid_v_px=nan,
            container_mask_bbox_left_px=-1,
            container_mask_bbox_top_px=-1,
            container_mask_bbox_right_px=-1,
            container_mask_bbox_bottom_px=-1,
            target_center_world_x_m=nan,
            target_center_world_y_m=nan,
            target_center_world_z_m=nan,
            container_center_world_x_m=nan,
            container_center_world_y_m=nan,
            container_center_world_z_m=nan,
            target_container_center_distance_m=nan,
            inside_xy_margin=nan,
            caddy_center_longitudinal_margin=nan,
            caddy_center_lateral_margin=nan,
            caddy_center_back_offset_m=nan,
            caddy_center_lateral_offset_m=nan,
            caddy_back_half_extent_m=nan,
            caddy_lateral_half_extent_m=nan,
            caddy_central_lateral_margin_m=nan,
            caddy_central_back_margin_m=nan,
            caddy_topology_valid=False,
            caddy_topology_reason="not_available",
            caddy_topology_confidence=nan,
            caddy_topology_lateral_low_m=nan,
            caddy_topology_lateral_high_m=nan,
            caddy_topology_lateral_margin_m=nan,
            caddy_topology_back_margin_m=nan,
            caddy_topology_footprint_valid=False,
            caddy_topology_footprint_longitudinal_margin_m=nan,
            caddy_topology_footprint_lateral_margin_m=nan,
            caddy_topology_footprint_below_rim_margin_m=nan,
            caddy_topology_footprint_above_floor_margin_m=nan,
            caddy_topology_footprint_margin_m=nan,
            caddy_topology_temporal_margin_m=nan,
            caddy_topology_temporal_count=0,
            caddy_topology_temporal_candidate=False,
            caddy_central_temporal_margin_m=nan,
            caddy_central_temporal_count=0,
            caddy_central_temporal_candidate=False,
            caddy_dual_expert_candidate=False,
            caddy_dual_expert_route="none",
            caddy_cached_topology_half_width_m=nan,
            caddy_cached_topology_sample_count=0,
            caddy_cached_topology_margin_m=nan,
            caddy_cached_topology_count=0,
            caddy_cached_topology_candidate=False,
            caddy_static_baseline_camera_name=str(
                self.in_caddy_static_baseline_cameras.get(
                    env_id, "unavailable"
                )
            ),
            caddy_static_baseline_same_camera=False,
            caddy_static_baseline_point_count=int(len(
                self.in_caddy_static_baseline_points.get(
                    env_id, np.empty((0, 3), dtype=np.float64)
                )
            )),
            caddy_static_baseline_center_drift_m=nan,
            caddy_static_baseline_valid=False,
            caddy_static_baseline_inside_xy_margin_m=nan,
            caddy_static_baseline_geometry_margin_m=nan,
            caddy_static_baseline_topology_valid=False,
            caddy_static_baseline_topology_back_margin_m=nan,
            caddy_static_baseline_below_rim_margin_m=nan,
            caddy_static_baseline_above_floor_margin_m=nan,
            caddy_static_baseline_candidate_margin_m=nan,
            caddy_static_baseline_temporal_count=0,
            caddy_static_baseline_temporal_candidate=False,
            caddy_cached_topology_reobservation_bouts=int(
                self.in_caddy_reobservation_bouts.get(env_id, 0)
            ),
            caddy_cached_topology_reobservation_gap_steps=(
                max(
                    0,
                    int(step)
                    - int(self.in_caddy_reobservation_last_steps[env_id])
                    - 1,
                )
                if env_id in self.in_caddy_reobservation_last_steps
                else -1
            ),
            caddy_cached_topology_reobservation_margin_m=float(
                self.in_caddy_reobservation_margins.get(env_id, nan)
            ),
            caddy_cached_topology_reobservation_candidate=bool(
                env_id in self.in_caddy_reobservation_margins
            ),
            caddy_confirmed_event_bridge_margin_m=float(
                self.in_caddy_confirmed_event_bridge_margins.get(env_id, nan)
            ),
            caddy_confirmed_event_bridge_step=int(
                self.in_caddy_confirmed_event_bridge_steps.get(env_id, -1)
            ),
            caddy_confirmed_event_bridge_age_steps=(
                int(step) - int(self.in_caddy_confirmed_event_bridge_steps[env_id])
                if env_id in self.in_caddy_confirmed_event_bridge_steps
                else -1
            ),
            caddy_confirmed_event_bridge_candidate=bool(
                env_id in self.in_caddy_confirmed_event_bridge_margins
            ),
            caddy_core_longitudinal_margin=nan,
            caddy_core_lateral_margin=nan,
            caddy_full_longitudinal_margin=nan,
            caddy_full_lateral_margin=nan,
            below_rim_margin=nan,
            above_floor_margin=nan,
            entry_depth_margin=nan,
            target_eef_distance_m=nan,
            target_eef_surface_p01_m=nan,
            target_eef_surface_p05_m=nan,
            target_eef_surface_p10_m=nan,
            eef_world_x_m=float(eef_world[0]),
            eef_world_y_m=float(eef_world[1]),
            eef_world_z_m=float(eef_world[2]),
            release_proximity_margin=nan,
            opening_margin=nan,
            release_margin=nan,
            release_event=False,
            release_latched=bool(
                self.in_release_latched.get(env_id, False)
            ),
            release_source=str(
                self.in_release_sources.get(env_id, "none")
            ),
            caddy_release_candidate=False,
            caddy_release_count=int(
                max(
                    self.in_caddy_release_counts.get(env_id, 0),
                    self.in_caddy_dual_stage_release_counts.get(env_id, 0),
                )
            ),
            caddy_opening_count=int(
                self.in_caddy_opening_counts.get(env_id, 0)
            ),
            caddy_opening_candidate=False,
            caddy_opening_candidate_source="none",
            caddy_opening_source=str(
                self.in_caddy_opening_sources.get(env_id, "none")
            ),
            caddy_opening_evidence=float(
                self.in_caddy_opening_evidence.get(env_id, nan)
            ),
            caddy_opening_gate_margin=nan,
            caddy_core_lateral_gate_margin=nan,
            caddy_opening_max_age_steps=(
                _caddy_visual_opening_max_age_steps(
                    opening_source=str(
                        self.in_caddy_opening_sources.get(env_id, "none")
                    ),
                    standard_max_age_steps=(
                        self.in_caddy_release_max_age_steps
                    ),
                    fresh_entry_max_age_steps=(
                        self.in_caddy_fresh_release_max_age_steps
                    ),
                    proxy_max_age_steps=(
                        self.in_caddy_proxy_release_max_age_steps
                    ),
                )
            ),
            caddy_opening_age_steps=-1,
            caddy_separation_age_steps=-1,
            caddy_opening_center_world_x_m=(
                float(cached_opening_center[0])
                if cached_opening_center is not None else nan
            ),
            caddy_opening_center_world_y_m=(
                float(cached_opening_center[1])
                if cached_opening_center is not None else nan
            ),
            caddy_opening_center_world_z_m=(
                float(cached_opening_center[2])
                if cached_opening_center is not None else nan
            ),
            caddy_fixed_target_distance_m=nan,
            caddy_opening_target_displacement_m=nan,
            caddy_opening_max_target_displacement_m=float(
                self.in_caddy_opening_max_target_displacements.get(
                    env_id, nan
                )
            ),
            caddy_actual_separation_margin=nan,
            caddy_settled_age_margin_steps=-1,
            caddy_handoff_route="none",
            caddy_handoff_margin=nan,
            caddy_contradiction_count=int(
                self.in_caddy_contradiction_counts.get(env_id, 0)
            ),
            caddy_handoff_revoked=False,
            visual_in_margin=nan,
            visual_in_confirmed=bool(
                self.in_confirmed.get(env_id, False)
            ),
        )

        # Width is proprioception, not simulator privilege.  Keep it on every
        # step so a target that becomes visually occluded at the microwave
        # entrance can still be paired with the observable release event.
        widths = self.in_width_histories.setdefault(env_id, [])
        recent_min = (
            min(widths[-self.in_release_history_steps:])
            if widths else float(gripper_width_m)
        )
        opening_margin = float(
            gripper_width_m - recent_min - self.in_release_delta_m
        )
        fields["opening_margin"] = opening_margin

        if selected is not None and not reason:
            candidates = [
                item for item in estimated
                if item["detection"]["class_name"] == container_class
            ]
            if not candidates:
                fields["reason"] = (
                    f"in_container_not_detected:{container_class}"
                )
            else:
                container = max(
                    candidates,
                    key=lambda item: float(
                        item["detection"]["confidence"]
                    ),
                )
                target_camera_name = str(
                    selected.get("camera_name", target_source)
                )
                container_camera_name = str(
                    container.get("camera_name", "unavailable")
                )
                target_candidates = [
                    item for item in estimated
                    if item["detection"]["class_name"] == target_class
                ]
                target_mask = _mask_image_geometry(
                    selected.get("detection")
                )
                container_mask = _mask_image_geometry(
                    container.get("detection")
                )
                target_center = np.asarray(
                    selected["estimate"].center_world,
                    dtype=np.float64,
                ).reshape(3)
                container_center = np.asarray(
                    container["estimate"].center_world,
                    dtype=np.float64,
                ).reshape(3)
                fields.update(
                    target_selector=str(target_selector),
                    target_camera_name=target_camera_name,
                    container_camera_name=container_camera_name,
                    same_camera_geometry=(
                        target_camera_name == container_camera_name
                    ),
                    target_candidate_count=len(target_candidates),
                    target_same_camera_candidate_count=sum(
                        str(item.get("camera_name", ""))
                        == target_camera_name
                        for item in target_candidates
                    ),
                    container_candidate_count=len(candidates),
                    container_same_camera_candidate_count=sum(
                        str(item.get("camera_name", ""))
                        == target_camera_name
                        for item in candidates
                    ),
                    target_mask_pixels=target_mask["mask_pixels"],
                    target_mask_centroid_u_px=(
                        target_mask["mask_centroid_u_px"]
                    ),
                    target_mask_centroid_v_px=(
                        target_mask["mask_centroid_v_px"]
                    ),
                    target_mask_bbox_left_px=(
                        target_mask["mask_bbox_left_px"]
                    ),
                    target_mask_bbox_top_px=(
                        target_mask["mask_bbox_top_px"]
                    ),
                    target_mask_bbox_right_px=(
                        target_mask["mask_bbox_right_px"]
                    ),
                    target_mask_bbox_bottom_px=(
                        target_mask["mask_bbox_bottom_px"]
                    ),
                    container_mask_pixels=container_mask["mask_pixels"],
                    container_mask_centroid_u_px=(
                        container_mask["mask_centroid_u_px"]
                    ),
                    container_mask_centroid_v_px=(
                        container_mask["mask_centroid_v_px"]
                    ),
                    container_mask_bbox_left_px=(
                        container_mask["mask_bbox_left_px"]
                    ),
                    container_mask_bbox_top_px=(
                        container_mask["mask_bbox_top_px"]
                    ),
                    container_mask_bbox_right_px=(
                        container_mask["mask_bbox_right_px"]
                    ),
                    container_mask_bbox_bottom_px=(
                        container_mask["mask_bbox_bottom_px"]
                    ),
                    target_center_world_x_m=float(target_center[0]),
                    target_center_world_y_m=float(target_center[1]),
                    target_center_world_z_m=float(target_center[2]),
                    container_center_world_x_m=float(container_center[0]),
                    container_center_world_y_m=float(container_center[1]),
                    container_center_world_z_m=float(container_center[2]),
                    target_container_center_distance_m=float(
                        np.linalg.norm(target_center - container_center)
                    ),
                )
                if container_geometry == "drawer":
                    estimate = visual_in_drawer_margin(
                        selected["estimate"].points_world,
                        container["estimate"].points_world,
                        gripper_width_m,
                        drawer_level=self._drawer_level(
                            container_object, ""
                        ),
                        config=self.in_config,
                    )
                elif container_geometry == "microwave":
                    # The appliance body is static until the final Close
                    # action.  Lock its first stable RGB-D point cloud so
                    # later YOLOE mask-shape changes cannot move the visual
                    # container envelope by 10--20 cm and create In events.
                    if env_id not in self.in_microwave_baseline_points:
                        self.in_microwave_baseline_points[env_id] = (
                            np.asarray(
                                container["estimate"].points_world,
                                dtype=np.float64,
                            ).copy()
                        )
                    calibration = self.calibrations.get(env_id, {})
                    camera_to_world = np.asarray(
                        calibration.get(
                            "camera_to_world", np.full((4, 4), np.nan)
                        ),
                        dtype=np.float64,
                    )
                    camera_position = (
                        camera_to_world[:3, 3]
                        if camera_to_world.shape == (4, 4)
                        else np.full(3, np.nan)
                    )
                    estimate = visual_in_microwave_margin(
                        selected["estimate"].points_world,
                        self.in_microwave_baseline_points[env_id],
                        gripper_width_m,
                        self.in_config,
                        camera_position_world=camera_position,
                    )
                elif container_geometry == "caddy_back":
                    calibration = self.calibrations.get(env_id, {})
                    camera_to_world = np.asarray(
                        calibration.get(
                            "camera_to_world", np.full((4, 4), np.nan)
                        ),
                        dtype=np.float64,
                    )
                    camera_position = (
                        camera_to_world[:3, 3]
                        if camera_to_world.shape == (4, 4)
                        else np.full(3, np.nan)
                    )
                    if env_id not in self.in_caddy_static_baseline_points:
                        baseline_points = np.asarray(
                            container["estimate"].points_world,
                            dtype=np.float64,
                        )
                        if (
                            baseline_points.ndim == 2
                            and baseline_points.shape[1] == 3
                            and len(baseline_points) > 0
                            and np.all(np.isfinite(baseline_points))
                        ):
                            self.in_caddy_static_baseline_points[env_id] = (
                                baseline_points.copy()
                            )
                            self.in_caddy_static_baseline_centers[env_id] = (
                                container_center.copy()
                            )
                            self.in_caddy_static_baseline_cameras[env_id] = (
                                container_camera_name
                            )
                    estimate = visual_in_caddy_back_margin(
                        selected["estimate"].points_world,
                        container["estimate"].points_world,
                        gripper_width_m,
                        camera_position_world=camera_position,
                        config=self.in_config,
                    )
                    static_baseline = (
                        self.in_caddy_static_baseline_points.get(env_id)
                    )
                    static_estimate = (
                        visual_in_caddy_back_margin(
                            selected["estimate"].points_world,
                            static_baseline,
                            gripper_width_m,
                            camera_position_world=camera_position,
                            config=self.in_config,
                        )
                        if static_baseline is not None else None
                    )
                    static_valid = bool(
                        static_estimate is not None
                        and static_estimate.valid
                    )
                    static_geometry_margin = (
                        float(min(
                            static_estimate.inside_xy_margin,
                            static_estimate.below_rim_margin,
                            static_estimate.above_floor_margin,
                        ))
                        if static_valid else nan
                    )
                    static_topology_valid = bool(
                        static_valid
                        and static_estimate.caddy_topology_valid
                    )
                    static_candidate_margin = (
                        float(min(
                            static_geometry_margin,
                            static_estimate.caddy_topology_back_margin_m,
                        ))
                        if static_topology_valid else nan
                    )
                    (
                        static_candidate_margin,
                        static_temporal_count,
                        static_temporal_candidate,
                    ) = _caddy_static_baseline_temporal_diagnostic(
                        container_geometry=container_geometry,
                        target_source=str(target_source),
                        baseline_valid=static_topology_valid,
                        candidate_margin_m=static_candidate_margin,
                        previous_count=previous_static_baseline_count,
                    )
                    self.in_caddy_static_baseline_temporal_counts[env_id] = (
                        static_temporal_count
                    )
                    baseline_center = (
                        self.in_caddy_static_baseline_centers.get(env_id)
                    )
                    baseline_camera_name = str(
                        self.in_caddy_static_baseline_cameras.get(
                            env_id, "unavailable"
                        )
                    )
                    fields.update(
                        caddy_static_baseline_camera_name=(
                            baseline_camera_name
                        ),
                        caddy_static_baseline_same_camera=bool(
                            baseline_camera_name == target_camera_name
                        ),
                        caddy_static_baseline_point_count=int(
                            len(static_baseline)
                            if static_baseline is not None else 0
                        ),
                        caddy_static_baseline_center_drift_m=(
                            float(np.linalg.norm(
                                container_center - baseline_center
                            ))
                            if baseline_center is not None else nan
                        ),
                        caddy_static_baseline_valid=static_valid,
                        caddy_static_baseline_inside_xy_margin_m=(
                            float(static_estimate.inside_xy_margin)
                            if static_valid else nan
                        ),
                        caddy_static_baseline_geometry_margin_m=(
                            static_geometry_margin
                        ),
                        caddy_static_baseline_topology_valid=(
                            static_topology_valid
                        ),
                        caddy_static_baseline_topology_back_margin_m=(
                            float(
                                static_estimate.caddy_topology_back_margin_m
                            )
                            if static_topology_valid else nan
                        ),
                        caddy_static_baseline_below_rim_margin_m=(
                            float(static_estimate.below_rim_margin)
                            if static_valid else nan
                        ),
                        caddy_static_baseline_above_floor_margin_m=(
                            float(static_estimate.above_floor_margin)
                            if static_valid else nan
                        ),
                        caddy_static_baseline_candidate_margin_m=(
                            static_candidate_margin
                        ),
                        caddy_static_baseline_temporal_count=int(
                            static_temporal_count
                        ),
                        caddy_static_baseline_temporal_candidate=bool(
                            static_temporal_candidate
                        ),
                    )
                else:
                    estimate = visual_in_basket_margin(
                        selected["estimate"].points_world,
                        container["estimate"].points_world,
                        gripper_width_m,
                        self.in_config,
                    )
                if estimate.valid:
                    geometry_components = [
                        estimate.inside_xy_margin,
                        estimate.below_rim_margin,
                        estimate.above_floor_margin,
                    ]
                    if container_geometry == "microwave":
                        geometry_components.append(
                            estimate.entry_depth_margin
                        )
                    geometry_margin = float(min(geometry_components))
                    contradiction_opening_displacement_m = (
                        float(np.linalg.norm(
                            np.asarray(
                                selected["estimate"].center_world,
                                dtype=np.float64,
                            ).reshape(3)
                            - cached_opening_center
                        ))
                        if cached_opening_center is not None else nan
                    )
                    caddy_handoff_revoked = False
                    caddy_contradiction = (
                        _is_caddy_visual_handoff_contradiction(
                            container_geometry=container_geometry,
                            target_source=str(target_source),
                            release_source=str(
                                self.in_release_sources.get(env_id, "none")
                            ),
                            inside_xy_margin=estimate.inside_xy_margin,
                            above_floor_margin=estimate.above_floor_margin,
                            opening_target_displacement_m=(
                                contradiction_opening_displacement_m
                            ),
                            contradiction_margin_m=(
                                self.in_caddy_contradiction_margin_m
                            ),
                            max_target_displacement_m=(
                                self.in_caddy_max_target_displacement_m
                            ),
                        )
                    )
                    if caddy_contradiction:
                        self.in_caddy_contradiction_counts[env_id] = (
                            self.in_caddy_contradiction_counts.get(env_id, 0)
                            + 1
                        )
                    elif (
                        container_geometry == "caddy_back"
                        and str(target_source) == "fresh_rgbd"
                    ):
                        self.in_caddy_contradiction_counts[env_id] = 0
                    if (
                        self.in_caddy_contradiction_counts.get(env_id, 0)
                        >= self.in_caddy_contradiction_confirmation_steps
                    ):
                        # A handoff is an occlusion bridge, not permission to
                        # ignore later direct visual evidence.  Clear both the
                        # release latch and its stale geometric antecedents.
                        self.in_release_latched[env_id] = False
                        self.in_release_evidence[env_id] = nan
                        self.in_release_steps.pop(env_id, None)
                        self.in_release_sources.pop(env_id, None)
                        self.in_positive_counts[env_id] = 0
                        self.in_confirmed[env_id] = False
                        self.in_last_positive.pop(env_id, None)
                        self.in_caddy_release_counts[env_id] = 0
                        self.in_caddy_dual_stage_release_counts[env_id] = 0
                        self.in_caddy_opening_counts[env_id] = 0
                        self.in_caddy_opening_steps.pop(env_id, None)
                        self.in_caddy_opening_centers.pop(env_id, None)
                        self.in_caddy_opening_max_target_displacements.pop(
                            env_id, None
                        )
                        self.in_caddy_opening_evidence.pop(env_id, None)
                        self.in_caddy_opening_geometry_margins.pop(
                            env_id, None
                        )
                        self.in_caddy_opening_sources.pop(env_id, None)
                        self.in_caddy_opening_candidate_sources.pop(
                            env_id, None
                        )
                        self.in_caddy_observed_opening_steps.pop(
                            env_id, None
                        )
                        self.in_caddy_observed_opening_centers.pop(
                            env_id, None
                        )
                        self.in_caddy_observed_opening_evidence.pop(
                            env_id, None
                        )
                        self.in_caddy_observed_opening_geometry_margins.pop(
                            env_id, None
                        )
                        self.in_caddy_observed_opening_sources.pop(
                            env_id, None
                        )
                        self.in_caddy_separation_steps.pop(env_id, None)
                        self.in_caddy_separation_evidence.pop(env_id, None)
                        self.in_caddy_contradiction_counts[env_id] = 0
                        caddy_handoff_revoked = True
                    target_eef_distance_m = float(np.linalg.norm(
                        np.asarray(
                            selected["estimate"].center_world,
                            dtype=np.float64,
                        ).reshape(3)
                        - np.asarray(eef, dtype=np.float64).reshape(3)
                    ))
                    (
                        target_eef_surface_p01_m,
                        target_eef_surface_p05_m,
                        target_eef_surface_p10_m,
                    ) = (
                        _eef_target_surface_quantiles(
                            selected["estimate"].points_world,
                            eef_world,
                        )
                        if str(target_source) == "fresh_rgbd"
                        else (nan, nan, nan)
                    )
                    release_proximity_margin = (
                        float(
                            self.in_config
                            .microwave_release_max_target_eef_distance_m
                            - target_eef_distance_m
                        )
                        if container_geometry == "microwave"
                        else nan
                    )
                    # Other containers can use the absolute width transition
                    # immediately. A thin book in a caddy may be released at
                    # a much smaller width, so caddy release is confirmed by
                    # the bounded visual handoff below instead.
                    width_release_event = bool(
                        container_geometry != "caddy_back"
                        and
                        geometry_margin >= 0.0
                        and estimate.release_margin >= 0.0
                        and opening_margin >= 0.0
                        and (
                            container_geometry != "microwave"
                            or release_proximity_margin >= 0.0
                        )
                    )
                    caddy_core_lateral_gate_margin = (
                        float(
                            estimate.caddy_core_lateral_margin
                            - self.in_caddy_fresh_min_core_lateral_margin_m
                        )
                        if container_geometry == "caddy_back"
                        else nan
                    )
                    caddy_opening_gate_margin = (
                        _caddy_opening_gate_margin(
                            opening_margin,
                            self.in_caddy_opening_tolerance_m,
                        )
                        if container_geometry == "caddy_back"
                        else opening_margin
                    )
                    fresh_opening_candidate = (
                        _is_caddy_visual_opening_candidate(
                            container_geometry=container_geometry,
                            target_source=str(target_source),
                            geometry_margin=geometry_margin,
                            entry_tolerance_m=(
                                self.in_caddy_fresh_entry_tolerance_m
                            ),
                            opening_margin=caddy_opening_gate_margin,
                            core_lateral_margin=(
                                estimate.caddy_core_lateral_margin
                            ),
                            min_core_lateral_margin_m=(
                                self.in_caddy_fresh_min_core_lateral_margin_m
                            ),
                        )
                    )
                    proxy_opening_candidate = (
                        _is_caddy_proxy_entry_opening_candidate(
                            container_geometry=container_geometry,
                            target_source=str(target_source),
                            grasp_confirmed=bool(
                                self.grasp_confirmed.get(env_id, False)
                            ),
                            geometry_margin=geometry_margin,
                            entry_tolerance_m=(
                                self.in_caddy_proxy_entry_tolerance_m
                            ),
                            opening_margin=opening_margin,
                        )
                    )
                    caddy_opening_candidate = bool(
                        fresh_opening_candidate
                        or proxy_opening_candidate
                    )
                    current_opening_evidence = (
                        _caddy_visual_opening_evidence(
                            fresh_candidate=fresh_opening_candidate,
                            proxy_candidate=proxy_opening_candidate,
                            geometry_margin=geometry_margin,
                            fresh_entry_tolerance_m=(
                                self.in_caddy_fresh_entry_tolerance_m
                            ),
                            proxy_entry_tolerance_m=(
                                self.in_caddy_proxy_entry_tolerance_m
                            ),
                            opening_margin=(
                                caddy_opening_gate_margin
                                if fresh_opening_candidate
                                else opening_margin
                            ),
                            core_lateral_margin=(
                                estimate.caddy_core_lateral_margin
                            ),
                            fresh_min_core_lateral_margin_m=(
                                self.in_caddy_fresh_min_core_lateral_margin_m
                            ),
                        )
                    )
                    opening_candidate_source = (
                        "fresh_rgbd_near_entry"
                        if fresh_opening_candidate
                        else (
                            "confirmed_pick_proxy_near_entry"
                            if proxy_opening_candidate else "none"
                        )
                    )
                    previous_candidate_source = (
                        self.in_caddy_opening_candidate_sources.get(
                            env_id, "none"
                        )
                    )
                    opening_count = (
                        self.in_caddy_opening_counts.get(env_id, 0) + 1
                        if (
                            caddy_opening_candidate
                            and opening_candidate_source
                            == previous_candidate_source
                        )
                        else (1 if caddy_opening_candidate else 0)
                    )
                    self.in_caddy_opening_counts[env_id] = opening_count
                    self.in_caddy_opening_candidate_sources[env_id] = (
                        opening_candidate_source
                    )
                    opening_confirmation_steps = (
                        self.in_caddy_proxy_opening_confirmation_steps
                        if proxy_opening_candidate
                        else self.in_caddy_release_confirmation_steps
                    )
                    if (
                        opening_count
                        == opening_confirmation_steps
                    ):
                        self.in_caddy_opening_steps[env_id] = int(step)
                        self.in_caddy_opening_centers[env_id] = np.asarray(
                            selected["estimate"].center_world,
                            dtype=np.float64,
                        ).reshape(3).copy()
                        self.in_caddy_opening_max_target_displacements[
                            env_id
                        ] = 0.0
                        self.in_caddy_opening_evidence[env_id] = (
                            current_opening_evidence
                        )
                        self.in_caddy_opening_geometry_margins[env_id] = (
                            geometry_margin
                        )
                        self.in_caddy_opening_sources[env_id] = (
                            opening_candidate_source
                        )

                    observed_opening_step = (
                        self.in_caddy_observed_opening_steps.get(env_id)
                    )
                    observed_opening_source = (
                        self.in_caddy_observed_opening_sources.get(
                            env_id, "none"
                        )
                    )
                    observed_opening_max_age_steps = (
                        _caddy_visual_opening_max_age_steps(
                            opening_source=str(observed_opening_source),
                            standard_max_age_steps=(
                                self.in_caddy_release_max_age_steps
                            ),
                            fresh_entry_max_age_steps=(
                                self.in_caddy_fresh_release_max_age_steps
                            ),
                            proxy_max_age_steps=(
                                self.in_caddy_proxy_release_max_age_steps
                            ),
                        )
                    )
                    observed_opening_recent = _is_recent_visual_evidence(
                        current_step=step,
                        evidence_step=observed_opening_step,
                        max_age_steps=observed_opening_max_age_steps,
                    )
                    observed_opening_center = (
                        self.in_caddy_observed_opening_centers.get(env_id)
                    )
                    observed_opening_displacement_m = (
                        float(np.linalg.norm(
                            np.asarray(
                                selected["estimate"].center_world,
                                dtype=np.float64,
                            ).reshape(3)
                            - observed_opening_center
                        ))
                        if observed_opening_center is not None else nan
                    )
                    observed_opening_departed = bool(
                        str(target_source) == "fresh_rgbd"
                        and (
                            estimate.inside_xy_margin
                            <= -self.in_caddy_contradiction_margin_m
                            or estimate.above_floor_margin
                            <= -self.in_caddy_contradiction_margin_m
                        )
                        and np.isfinite(observed_opening_displacement_m)
                        and observed_opening_displacement_m
                        > self.in_caddy_max_target_displacement_m
                    )
                    if (
                        observed_opening_step is not None
                        and (
                            not observed_opening_recent
                            or observed_opening_departed
                        )
                    ):
                        self.in_caddy_observed_opening_steps.pop(
                            env_id, None
                        )
                        self.in_caddy_observed_opening_centers.pop(
                            env_id, None
                        )
                        self.in_caddy_observed_opening_evidence.pop(
                            env_id, None
                        )
                        self.in_caddy_observed_opening_geometry_margins.pop(
                            env_id, None
                        )
                        self.in_caddy_observed_opening_sources.pop(
                            env_id, None
                        )
                        observed_opening_recent = False
                        observed_opening_center = None
                    if caddy_opening_candidate:
                        observed_opening_step = int(step)
                        observed_opening_center = np.asarray(
                            selected["estimate"].center_world,
                            dtype=np.float64,
                        ).reshape(3).copy()
                        self.in_caddy_observed_opening_steps[env_id] = (
                            observed_opening_step
                        )
                        self.in_caddy_observed_opening_centers[env_id] = (
                            observed_opening_center
                        )
                        self.in_caddy_observed_opening_evidence[env_id] = (
                            current_opening_evidence
                        )
                        self.in_caddy_observed_opening_geometry_margins[
                            env_id
                        ] = geometry_margin
                        self.in_caddy_observed_opening_sources[env_id] = (
                            opening_candidate_source
                        )
                        observed_opening_recent = True

                    dual_stage_handoff_margin = (
                        _caddy_dual_stage_handoff_margin(
                            container_geometry=container_geometry,
                            opening_recent=observed_opening_recent,
                            target_source=str(target_source),
                            opening_evidence=(
                                self.in_caddy_observed_opening_evidence.get(
                                    env_id, nan
                                )
                            ),
                            inside_xy_margin=estimate.inside_xy_margin,
                            above_floor_margin=estimate.above_floor_margin,
                            core_lateral_margin=(
                                estimate.caddy_core_lateral_margin
                            ),
                            target_eef_distance_m=target_eef_distance_m,
                            separation_threshold_m=(
                                self.in_caddy_release_separation_m
                            ),
                        )
                    )
                    fresh_separation = bool(
                        np.isfinite(dual_stage_handoff_margin)
                        and dual_stage_handoff_margin >= 0.0
                    )
                    if fresh_separation:
                        self.in_caddy_separation_steps[env_id] = int(step)
                        self.in_caddy_separation_evidence[env_id] = float(
                            target_eef_distance_m
                            - self.in_caddy_release_separation_m
                        )

                    opening_step = self.in_caddy_opening_steps.get(env_id)
                    opening_source = self.in_caddy_opening_sources.get(
                        env_id, "none"
                    )
                    opening_max_age_steps = (
                        _caddy_visual_opening_max_age_steps(
                            opening_source=str(opening_source),
                            standard_max_age_steps=(
                                self.in_caddy_release_max_age_steps
                            ),
                            fresh_entry_max_age_steps=(
                                self.in_caddy_fresh_release_max_age_steps
                            ),
                            proxy_max_age_steps=(
                                self.in_caddy_proxy_release_max_age_steps
                            ),
                        )
                    )
                    separation_step = (
                        self.in_caddy_separation_steps.get(env_id)
                    )
                    opening_recent = _is_recent_visual_evidence(
                        current_step=step,
                        evidence_step=opening_step,
                        max_age_steps=opening_max_age_steps,
                    )
                    separation_recent = _is_recent_visual_evidence(
                        current_step=step,
                        evidence_step=separation_step,
                        max_age_steps=self.in_caddy_release_max_age_steps,
                    )
                    dual_stage_separation_handoff = bool(
                        fresh_separation and separation_recent
                    )
                    opening_center = self.in_caddy_opening_centers.get(env_id)
                    fixed_target_distance_m = (
                        float(np.linalg.norm(
                            np.asarray(eef, dtype=np.float64).reshape(3)
                            - opening_center
                        ))
                        if opening_center is not None
                        else nan
                    )
                    opening_target_displacement_m = (
                        float(np.linalg.norm(
                            np.asarray(
                                selected["estimate"].center_world,
                                dtype=np.float64,
                            ).reshape(3)
                            - opening_center
                        ))
                        if opening_center is not None
                        else nan
                    )
                    if (
                        str(target_source) == "fresh_rgbd"
                        and np.isfinite(opening_target_displacement_m)
                    ):
                        self.in_caddy_opening_max_target_displacements[
                            env_id
                        ] = max(
                            self.in_caddy_opening_max_target_displacements.get(
                                env_id, 0.0
                            ),
                            opening_target_displacement_m,
                        )
                    max_target_displacement_m = float(
                        self.in_caddy_opening_max_target_displacements.get(
                            env_id, nan
                        )
                    )
                    opening_age_steps = (
                        int(step) - int(opening_step)
                        if opening_step is not None else -1
                    )
                    opening_geometry_margin = float(
                        self.in_caddy_opening_geometry_margins.get(
                            env_id, nan
                        )
                    )
                    distance_stable = bool(
                        opening_recent
                        and np.isfinite(opening_target_displacement_m)
                        and np.isfinite(max_target_displacement_m)
                        and opening_target_displacement_m
                        <= self.in_caddy_max_target_displacement_m
                        and max_target_displacement_m
                        <= self.in_caddy_max_target_displacement_m
                    )
                    complete_entry_progress = bool(
                        opening_recent
                        and str(target_source) == "fresh_rgbd"
                        and _is_caddy_complete_entry_progress(
                            geometry_margin=geometry_margin,
                            opening_geometry_margin=(
                                opening_geometry_margin
                            ),
                            core_lateral_margin=(
                                estimate.caddy_core_lateral_margin
                            ),
                        )
                    )
                    target_stable = bool(
                        distance_stable or complete_entry_progress
                    )
                    current_placement = bool(
                        str(target_source) == "fresh_rgbd"
                        and geometry_margin
                        + self.in_caddy_fresh_entry_tolerance_m >= 0.0
                    )
                    actual_separation_handoff = bool(
                        opening_recent
                        and current_placement
                        and target_stable
                        and target_eef_distance_m
                        >= self.in_caddy_release_separation_m
                    )
                    settled_placement_handoff = bool(
                        opening_recent
                        and current_placement
                        and target_stable
                        and opening_age_steps
                        >= self.in_caddy_settled_confirmation_age_steps
                    )
                    standard_caddy_release_candidate = (
                        _is_caddy_visual_handoff_candidate(
                            container_geometry=container_geometry,
                            opening_recent=opening_recent,
                            target_source=str(target_source),
                            geometry_margin=geometry_margin,
                            opening_geometry_margin=(
                                opening_geometry_margin
                            ),
                            core_lateral_margin=(
                                estimate.caddy_core_lateral_margin
                            ),
                            entry_tolerance_m=(
                                self.in_caddy_fresh_entry_tolerance_m
                            ),
                            target_eef_distance_m=target_eef_distance_m,
                            separation_threshold_m=(
                                self.in_caddy_release_separation_m
                            ),
                            opening_target_displacement_m=(
                                opening_target_displacement_m
                            ),
                            max_target_displacement_m=(
                                max_target_displacement_m
                            ),
                            max_allowed_target_displacement_m=(
                                self.in_caddy_max_target_displacement_m
                            ),
                            opening_age_steps=opening_age_steps,
                            settled_confirmation_age_steps=(
                                self.in_caddy_settled_confirmation_age_steps
                            ),
                        )
                    )
                    self.in_caddy_release_counts[env_id] = (
                        self.in_caddy_release_counts.get(env_id, 0) + 1
                        if standard_caddy_release_candidate else 0
                    )
                    self.in_caddy_dual_stage_release_counts[env_id] = (
                        self.in_caddy_dual_stage_release_counts.get(
                            env_id, 0
                        ) + 1
                        if dual_stage_separation_handoff else 0
                    )
                    caddy_release_candidate = bool(
                        standard_caddy_release_candidate
                        or dual_stage_separation_handoff
                    )
                    standard_caddy_release_event = bool(
                        standard_caddy_release_candidate
                        and self.in_caddy_release_counts[env_id]
                        >= self.in_caddy_release_confirmation_steps
                    )
                    dual_stage_release_event = bool(
                        dual_stage_separation_handoff
                        and self.in_caddy_dual_stage_release_counts[env_id]
                        >= self.in_caddy_release_confirmation_steps
                    )
                    caddy_release_event = bool(
                        (
                            standard_caddy_release_event
                            or dual_stage_release_event
                        )
                        and not self.in_release_latched.get(env_id, False)
                    )
                    if dual_stage_release_event:
                        observed_step = (
                            self.in_caddy_observed_opening_steps.get(env_id)
                        )
                        observed_center = (
                            self.in_caddy_observed_opening_centers.get(env_id)
                        )
                        if observed_step is not None:
                            self.in_caddy_opening_steps[env_id] = int(
                                observed_step
                            )
                        if observed_center is not None:
                            self.in_caddy_opening_centers[env_id] = (
                                observed_center.copy()
                            )
                        self.in_caddy_opening_evidence[env_id] = float(
                            self.in_caddy_observed_opening_evidence.get(
                                env_id, nan
                            )
                        )
                        self.in_caddy_opening_geometry_margins[env_id] = (
                            float(
                                self.in_caddy_observed_opening_geometry_margins.get(
                                    env_id, nan
                                )
                            )
                        )
                        self.in_caddy_opening_sources[env_id] = str(
                            self.in_caddy_observed_opening_sources.get(
                                env_id, "none"
                            )
                        )
                        self.in_caddy_opening_max_target_displacements[
                            env_id
                        ] = 0.0
                    caddy_handoff_margin = nan
                    caddy_handoff_route = "none"
                    distance_placement_evidence = [
                        self.in_caddy_opening_evidence.get(env_id, nan),
                        geometry_margin
                        + self.in_caddy_fresh_entry_tolerance_m,
                        self.in_caddy_max_target_displacement_m
                        - opening_target_displacement_m,
                        self.in_caddy_max_target_displacement_m
                        - max_target_displacement_m,
                    ]
                    progress_placement_evidence = [
                        self.in_caddy_opening_evidence.get(env_id, nan),
                        geometry_margin,
                        geometry_margin - opening_geometry_margin,
                        geometry_margin + opening_geometry_margin,
                        estimate.caddy_core_lateral_margin,
                    ]
                    placement_evidence = (
                        distance_placement_evidence
                        if distance_stable
                        else progress_placement_evidence
                    )
                    if actual_separation_handoff:
                        caddy_handoff_route = "actual_separation"
                        caddy_handoff_margin = float(min(
                            *placement_evidence,
                            target_eef_distance_m
                            - self.in_caddy_release_separation_m,
                        ))
                    elif settled_placement_handoff:
                        caddy_handoff_route = "settled_placement"
                        caddy_handoff_margin = float(min(
                            *placement_evidence,
                        ))
                    elif dual_stage_separation_handoff:
                        caddy_handoff_route = "dual_stage_separation"
                        caddy_handoff_margin = float(
                            dual_stage_handoff_margin
                        )
                    release_event = bool(
                        width_release_event or caddy_release_event
                    )
                    microwave_release_near_entry = bool(
                        container_geometry == "microwave"
                        and estimate.inside_xy_margin >= -0.04
                        and estimate.below_rim_margin >= 0.0
                        and estimate.above_floor_margin >= 0.0
                        and estimate.entry_depth_margin >= 0.0
                        and estimate.release_margin >= 0.0
                        and opening_margin >= 0.0
                        and release_proximity_margin >= 0.0
                    )
                    if release_event or microwave_release_near_entry:
                        self.in_release_latched[env_id] = True
                        self.in_release_steps[env_id] = int(step)
                        if caddy_release_event and not width_release_event:
                            self.in_release_sources[env_id] = (
                                "caddy_visual_handoff"
                            )
                            self.in_release_evidence[env_id] = float(
                                caddy_handoff_margin
                            )
                        else:
                            self.in_release_sources[env_id] = (
                                "gripper_width_opening"
                            )
                            release_evidence = [
                                estimate.release_margin,
                                opening_margin,
                            ]
                            if container_geometry == "microwave":
                                release_evidence.append(
                                    release_proximity_margin
                                )
                            self.in_release_evidence[env_id] = float(
                                min(release_evidence)
                            )
                    release_step = self.in_release_steps.get(env_id)
                    release_is_current = bool(
                        self.in_release_latched.get(env_id, False)
                        and (
                            container_geometry != "microwave"
                            or self.in_confirmed.get(env_id, False)
                            or visual_release_evidence_is_current(
                                current_step=step,
                                release_step=release_step,
                                max_age_steps=(
                                    self.in_config
                                    .microwave_release_max_age_steps
                                ),
                            )
                        )
                    )
                    release_latched = bool(
                        self.in_release_latched.get(env_id, False)
                    )
                    if container_geometry == "caddy_back":
                        margin = _compose_caddy_visual_in_margin(
                            geometry_margin=geometry_margin,
                            handoff_margin=caddy_handoff_margin,
                            release_latched=release_latched,
                            release_evidence=self.in_release_evidence.get(
                                env_id, nan
                            ),
                            unreleased_ceiling_m=(
                                self.in_caddy_unreleased_ceiling_m
                            ),
                        )
                        release_margin = float(margin)
                    else:
                        release_margin = (
                            float(self.in_release_evidence[env_id])
                            if release_is_current
                            else float(min(
                                estimate.release_margin,
                                opening_margin,
                                *(
                                    [release_proximity_margin]
                                    if container_geometry == "microwave"
                                    else []
                                ),
                            ))
                        )
                        margin = float(
                            geometry_margin
                            if (
                                not self.in_config.require_release
                                or (
                                    container_geometry == "microwave"
                                    and not self.in_config
                                    .microwave_require_release
                                )
                            )
                            else min(geometry_margin, release_margin)
                        )
                    (
                        topology_temporal_margin,
                        topology_temporal_count,
                        topology_temporal_candidate,
                    ) = _caddy_topology_temporal_diagnostic(
                        container_geometry=container_geometry,
                        target_source=str(target_source),
                        record_valid=True,
                        topology_valid=bool(
                            estimate.caddy_topology_valid
                        ),
                        topology_back_margin_m=float(
                            estimate.caddy_topology_back_margin_m
                        ),
                        above_floor_margin_m=float(
                            estimate.above_floor_margin
                        ),
                        previous_count=previous_topology_temporal_count,
                    )
                    self.in_caddy_topology_temporal_counts[env_id] = (
                        topology_temporal_count
                    )
                    (
                        central_temporal_margin,
                        central_temporal_count,
                        central_temporal_candidate,
                        dual_expert_candidate,
                        dual_expert_route,
                    ) = _caddy_dual_expert_temporal_diagnostic(
                        container_geometry=container_geometry,
                        target_source=str(target_source),
                        record_valid=True,
                        topology_temporal_candidate=bool(
                            topology_temporal_candidate
                        ),
                        central_back_margin_m=float(
                            estimate.caddy_central_back_margin_m
                        ),
                        above_floor_margin_m=float(
                            estimate.above_floor_margin
                        ),
                        previous_central_count=(
                            previous_central_temporal_count
                        ),
                    )
                    self.in_caddy_central_temporal_counts[env_id] = (
                        central_temporal_count
                    )
                    (
                        cached_topology_half_widths,
                        cached_topology_half_width,
                        cached_topology_margin,
                        cached_topology_count,
                        cached_topology_candidate,
                    ) = _caddy_cached_topology_temporal_diagnostic(
                        container_geometry=container_geometry,
                        target_source=str(target_source),
                        record_valid=True,
                        topology_valid=bool(
                            estimate.caddy_topology_valid
                        ),
                        topology_lateral_low_m=float(
                            estimate.caddy_topology_lateral_low_m
                        ),
                        topology_lateral_high_m=float(
                            estimate.caddy_topology_lateral_high_m
                        ),
                        center_longitudinal_margin_m=float(
                            estimate.caddy_center_longitudinal_margin
                        ),
                        center_lateral_offset_m=float(
                            estimate.caddy_center_lateral_offset_m
                        ),
                        above_floor_margin_m=float(
                            estimate.above_floor_margin
                        ),
                        previous_half_widths_m=(
                            previous_cached_topology_half_widths
                        ),
                        previous_count=previous_cached_topology_count,
                    )
                    self.in_caddy_cached_topology_half_widths[env_id] = (
                        list(cached_topology_half_widths)
                    )
                    self.in_caddy_cached_topology_counts[env_id] = (
                        cached_topology_count
                    )
                    (
                        reobservation_last_step,
                        reobservation_bouts,
                        reobservation_gap_steps,
                        reobservation_margin,
                        reobservation_candidate,
                    ) = _caddy_cached_topology_reobservation_diagnostic(
                        container_geometry=container_geometry,
                        target_source=str(target_source),
                        record_valid=True,
                        cached_topology_candidate=bool(
                            cached_topology_candidate
                        ),
                        cached_topology_margin_m=float(
                            cached_topology_margin
                        ),
                        previous_candidate_step=int(
                            self.in_caddy_reobservation_last_steps.get(
                                env_id, -1
                            )
                        ),
                        previous_bouts=int(
                            self.in_caddy_reobservation_bouts.get(env_id, 0)
                        ),
                        previous_margin_m=float(
                            self.in_caddy_reobservation_margins.get(
                                env_id, nan
                            )
                        ),
                        current_step=int(step),
                    )
                    if reobservation_last_step >= 0:
                        self.in_caddy_reobservation_last_steps[env_id] = (
                            reobservation_last_step
                        )
                    self.in_caddy_reobservation_bouts[env_id] = (
                        reobservation_bouts
                    )
                    if reobservation_candidate:
                        self.in_caddy_reobservation_margins[env_id] = (
                            reobservation_margin
                        )
                    margin = (
                        _promote_caddy_cached_topology_reobservation_margin(
                            visual_in_margin_m=float(margin),
                            reobservation_margin_m=float(
                                reobservation_margin
                            ),
                            reobservation_candidate=bool(
                                reobservation_candidate
                            ),
                        )
                    )
                    positive = margin >= 0.0
                    self.in_positive_counts[env_id] = (
                        self.in_positive_counts.get(env_id, 0) + 1
                        if positive else 0
                    )
                    if (
                        self.in_positive_counts[env_id]
                        >= self.in_confirmation_steps
                    ):
                        self.in_confirmed[env_id] = True
                    (
                        confirmed_event_bridge_margin,
                        confirmed_event_bridge_step,
                        confirmed_event_bridge_age_steps,
                        confirmed_event_bridge_candidate,
                    ) = _caddy_confirmed_event_bridge_diagnostic(
                        container_geometry=container_geometry,
                        target_source=str(target_source),
                        record_valid=True,
                        visual_in_confirmed=bool(
                            self.in_confirmed.get(env_id, False)
                        ),
                        visual_in_margin_m=float(margin),
                        previous_margin_m=float(
                            self.in_caddy_confirmed_event_bridge_margins.get(
                                env_id, nan
                            )
                        ),
                        previous_step=int(
                            self.in_caddy_confirmed_event_bridge_steps.get(
                                env_id, -1
                            )
                        ),
                        current_step=int(step),
                    )
                    if confirmed_event_bridge_candidate:
                        self.in_caddy_confirmed_event_bridge_margins[env_id] = (
                            float(confirmed_event_bridge_margin)
                        )
                        self.in_caddy_confirmed_event_bridge_steps[env_id] = (
                            int(confirmed_event_bridge_step)
                        )
                    margin = _promote_caddy_confirmed_event_margin(
                        visual_in_margin_m=float(margin),
                        confirmed_event_margin_m=float(
                            confirmed_event_bridge_margin
                        ),
                        confirmed_event_candidate=bool(
                            confirmed_event_bridge_candidate
                        ),
                    )
                    fields.update(
                        valid=True,
                        reason="",
                        visual_fresh=(
                            str(target_source) == "fresh_rgbd"
                        ),
                        target_source=str(target_source),
                        visual_hold_age_steps=0,
                        target_confidence=float(
                            selected["detection"]["confidence"]
                        ),
                        container_confidence=float(
                            container["detection"]["confidence"]
                        ),
                        inside_xy_margin=float(
                            estimate.inside_xy_margin
                        ),
                        caddy_center_longitudinal_margin=float(
                            estimate.caddy_center_longitudinal_margin
                        ),
                        caddy_center_lateral_margin=float(
                            estimate.caddy_center_lateral_margin
                        ),
                        caddy_center_back_offset_m=float(
                            estimate.caddy_center_back_offset_m
                        ),
                        caddy_center_lateral_offset_m=float(
                            estimate.caddy_center_lateral_offset_m
                        ),
                        caddy_back_half_extent_m=float(
                            estimate.caddy_back_half_extent_m
                        ),
                        caddy_lateral_half_extent_m=float(
                            estimate.caddy_lateral_half_extent_m
                        ),
                        caddy_central_lateral_margin_m=float(
                            estimate.caddy_central_lateral_margin_m
                        ),
                        caddy_central_back_margin_m=float(
                            estimate.caddy_central_back_margin_m
                        ),
                        caddy_topology_valid=bool(
                            estimate.caddy_topology_valid
                        ),
                        caddy_topology_reason=str(
                            estimate.caddy_topology_reason
                        ),
                        caddy_topology_confidence=float(
                            estimate.caddy_topology_confidence
                        ),
                        caddy_topology_lateral_low_m=float(
                            estimate.caddy_topology_lateral_low_m
                        ),
                        caddy_topology_lateral_high_m=float(
                            estimate.caddy_topology_lateral_high_m
                        ),
                        caddy_topology_lateral_margin_m=float(
                            estimate.caddy_topology_lateral_margin_m
                        ),
                        caddy_topology_back_margin_m=float(
                            estimate.caddy_topology_back_margin_m
                        ),
                        caddy_topology_footprint_valid=bool(
                            estimate.caddy_topology_footprint_valid
                        ),
                        caddy_topology_footprint_longitudinal_margin_m=float(
                            estimate.caddy_topology_footprint_longitudinal_margin_m
                        ),
                        caddy_topology_footprint_lateral_margin_m=float(
                            estimate.caddy_topology_footprint_lateral_margin_m
                        ),
                        caddy_topology_footprint_below_rim_margin_m=float(
                            estimate.caddy_topology_footprint_below_rim_margin_m
                        ),
                        caddy_topology_footprint_above_floor_margin_m=float(
                            estimate.caddy_topology_footprint_above_floor_margin_m
                        ),
                        caddy_topology_footprint_margin_m=float(
                            estimate.caddy_topology_footprint_margin_m
                        ),
                        caddy_topology_temporal_margin_m=float(
                            topology_temporal_margin
                        ),
                        caddy_topology_temporal_count=int(
                            topology_temporal_count
                        ),
                        caddy_topology_temporal_candidate=bool(
                            topology_temporal_candidate
                        ),
                        caddy_central_temporal_margin_m=float(
                            central_temporal_margin
                        ),
                        caddy_central_temporal_count=int(
                            central_temporal_count
                        ),
                        caddy_central_temporal_candidate=bool(
                            central_temporal_candidate
                        ),
                        caddy_dual_expert_candidate=bool(
                            dual_expert_candidate
                        ),
                        caddy_dual_expert_route=str(
                            dual_expert_route
                        ),
                        caddy_cached_topology_half_width_m=float(
                            cached_topology_half_width
                        ),
                        caddy_cached_topology_sample_count=len(
                            cached_topology_half_widths
                        ),
                        caddy_cached_topology_margin_m=float(
                            cached_topology_margin
                        ),
                        caddy_cached_topology_count=int(
                            cached_topology_count
                        ),
                        caddy_cached_topology_candidate=bool(
                            cached_topology_candidate
                        ),
                        caddy_cached_topology_reobservation_bouts=int(
                            reobservation_bouts
                        ),
                        caddy_cached_topology_reobservation_gap_steps=int(
                            reobservation_gap_steps
                        ),
                        caddy_cached_topology_reobservation_margin_m=float(
                            reobservation_margin
                        ),
                        caddy_cached_topology_reobservation_candidate=bool(
                            reobservation_candidate
                        ),
                        caddy_confirmed_event_bridge_margin_m=float(
                            confirmed_event_bridge_margin
                        ),
                        caddy_confirmed_event_bridge_step=int(
                            confirmed_event_bridge_step
                        ),
                        caddy_confirmed_event_bridge_age_steps=int(
                            confirmed_event_bridge_age_steps
                        ),
                        caddy_confirmed_event_bridge_candidate=bool(
                            confirmed_event_bridge_candidate
                        ),
                        caddy_core_longitudinal_margin=float(
                            estimate.caddy_core_longitudinal_margin
                        ),
                        caddy_core_lateral_margin=float(
                            estimate.caddy_core_lateral_margin
                        ),
                        caddy_full_longitudinal_margin=float(
                            estimate.caddy_full_longitudinal_margin
                        ),
                        caddy_full_lateral_margin=float(
                            estimate.caddy_full_lateral_margin
                        ),
                        below_rim_margin=float(
                            estimate.below_rim_margin
                        ),
                        above_floor_margin=float(
                            estimate.above_floor_margin
                        ),
                        entry_depth_margin=float(
                            estimate.entry_depth_margin
                        ),
                        target_eef_distance_m=target_eef_distance_m,
                        target_eef_surface_p01_m=(
                            target_eef_surface_p01_m
                        ),
                        target_eef_surface_p05_m=(
                            target_eef_surface_p05_m
                        ),
                        target_eef_surface_p10_m=(
                            target_eef_surface_p10_m
                        ),
                        release_proximity_margin=float(
                            release_proximity_margin
                        ),
                        opening_margin=float(opening_margin),
                        release_margin=release_margin,
                        release_event=bool(
                            release_event or microwave_release_near_entry
                        ),
                        release_latched=bool(
                            self.in_release_latched.get(env_id, False)
                        ),
                        release_source=str(
                            self.in_release_sources.get(env_id, "none")
                        ),
                        caddy_release_candidate=bool(
                            caddy_release_candidate
                        ),
                        caddy_release_count=int(
                            max(
                                self.in_caddy_release_counts.get(env_id, 0),
                                self.in_caddy_dual_stage_release_counts.get(
                                    env_id, 0
                                ),
                            )
                        ),
                        caddy_opening_count=int(
                            self.in_caddy_opening_counts.get(env_id, 0)
                        ),
                        caddy_opening_candidate=bool(
                            caddy_opening_candidate
                        ),
                        caddy_opening_candidate_source=str(
                            opening_candidate_source
                        ),
                        caddy_opening_source=str(
                            self.in_caddy_opening_sources.get(
                                env_id, "none"
                            )
                        ),
                        caddy_opening_evidence=float(
                            self.in_caddy_opening_evidence.get(env_id, nan)
                        ),
                        caddy_opening_gate_margin=float(
                            caddy_opening_gate_margin
                        ),
                        caddy_core_lateral_gate_margin=float(
                            caddy_core_lateral_gate_margin
                        ),
                        caddy_opening_max_age_steps=int(
                            opening_max_age_steps
                        ),
                        caddy_opening_age_steps=(
                            int(step) - int(opening_step)
                            if opening_step is not None else -1
                        ),
                        caddy_separation_age_steps=(
                            int(step) - int(separation_step)
                            if separation_step is not None else -1
                        ),
                        caddy_opening_center_world_x_m=(
                            float(opening_center[0])
                            if opening_center is not None else nan
                        ),
                        caddy_opening_center_world_y_m=(
                            float(opening_center[1])
                            if opening_center is not None else nan
                        ),
                        caddy_opening_center_world_z_m=(
                            float(opening_center[2])
                            if opening_center is not None else nan
                        ),
                        caddy_fixed_target_distance_m=float(
                            fixed_target_distance_m
                        ),
                        caddy_opening_target_displacement_m=float(
                            opening_target_displacement_m
                        ),
                        caddy_opening_max_target_displacement_m=float(
                            max_target_displacement_m
                        ),
                        caddy_actual_separation_margin=float(
                            target_eef_distance_m
                            - self.in_caddy_release_separation_m
                        ),
                        caddy_settled_age_margin_steps=(
                            opening_age_steps
                            - self.in_caddy_settled_confirmation_age_steps
                            if opening_step is not None else -1
                        ),
                        caddy_handoff_route=str(caddy_handoff_route),
                        caddy_handoff_margin=float(
                            caddy_handoff_margin
                        ),
                        caddy_contradiction_count=int(
                            self.in_caddy_contradiction_counts.get(
                                env_id, 0
                            )
                        ),
                        caddy_handoff_revoked=bool(
                            caddy_handoff_revoked
                        ),
                        visual_in_margin=margin,
                        visual_in_confirmed=bool(
                            self.in_confirmed.get(env_id, False)
                        ),
                    )
                    should_capture_caddy_probe = bool(
                        container_geometry == "caddy_back"
                        and str(target_source) == "fresh_rgbd"
                        and (
                            caddy_opening_candidate
                            or caddy_release_candidate
                            or release_event
                            or margin >= 0.0
                        )
                        and "image_rgb" in selected
                        and "image_rgb" in container
                    )
                    probe_frames = self.in_caddy_probe_frames.setdefault(
                        env_id, []
                    )
                    if should_capture_caddy_probe and len(probe_frames) < 48:
                        target_overlay = _make_mask_probe_overlay(
                            selected["image_rgb"],
                            selected["detection"],
                            (255, 40, 40),
                        )
                        container_overlay = _make_mask_probe_overlay(
                            container["image_rgb"],
                            container["detection"],
                            (40, 220, 70),
                        )
                        probe_frames.append({
                            "step": int(step),
                            "target_camera_name": target_camera_name,
                            "container_camera_name": container_camera_name,
                            "same_camera_geometry": bool(
                                target_camera_name == container_camera_name
                            ),
                            "target_selector": str(target_selector),
                            "target_candidate_count": len(
                                target_candidates
                            ),
                            "container_candidate_count": len(candidates),
                            "target_confidence": float(
                                selected["detection"]["confidence"]
                            ),
                            "container_confidence": float(
                                container["detection"]["confidence"]
                            ),
                            "inside_xy_margin": float(
                                estimate.inside_xy_margin
                            ),
                            "caddy_core_lateral_margin": float(
                                estimate.caddy_core_lateral_margin
                            ),
                            "caddy_core_lateral_gate_margin": float(
                                caddy_core_lateral_gate_margin
                            ),
                            "caddy_opening_gate_margin": float(
                                caddy_opening_gate_margin
                            ),
                            "caddy_opening_candidate": bool(
                                caddy_opening_candidate
                            ),
                            "caddy_opening_candidate_source": str(
                                opening_candidate_source
                            ),
                            "eef_world": [
                                float(eef_world[0]),
                                float(eef_world[1]),
                                float(eef_world[2]),
                            ],
                            "caddy_opening_center_world": (
                                [
                                    float(opening_center[0]),
                                    float(opening_center[1]),
                                    float(opening_center[2]),
                                ]
                                if opening_center is not None else None
                            ),
                            "caddy_fixed_target_distance_m": float(
                                fixed_target_distance_m
                            ),
                            "target_eef_distance_m": float(
                                target_eef_distance_m
                            ),
                            "caddy_opening_target_displacement_m": float(
                                opening_target_displacement_m
                            ),
                            "caddy_opening_max_target_displacement_m": float(
                                max_target_displacement_m
                            ),
                            "caddy_handoff_route": str(
                                caddy_handoff_route
                            ),
                            "visual_in_margin": float(margin),
                            "caddy_release_candidate": bool(
                                caddy_release_candidate
                            ),
                            "release_event": bool(release_event),
                            "target_overlay": target_overlay,
                            "container_overlay": container_overlay,
                        })
                else:
                    fields["reason"] = estimate.reason

        if not fields["valid"]:
            if container_geometry == "caddy_back":
                self.in_caddy_release_counts[env_id] = 0
                self.in_caddy_dual_stage_release_counts[env_id] = 0
                fields["caddy_release_count"] = 0
            # A mug normally disappears behind the microwave body exactly as
            # it crosses the opening.  Treat only a short, continuous miss as
            last_fresh = self.in_last_fresh.get(env_id)
            fresh_age = (
                int(step) - int(last_fresh.step)
                if last_fresh is not None else -1
            )
            target_occluded = "target_not_detected:" in fields["reason"]
            release_margin = float(
                self.in_release_evidence[env_id]
                if (
                    self.in_release_latched.get(env_id, False)
                    and (
                        container_geometry != "microwave"
                        or self.in_confirmed.get(env_id, False)
                        or visual_release_evidence_is_current(
                            current_step=step,
                            release_step=self.in_release_steps.get(env_id),
                            max_age_steps=(
                                self.in_config
                                .microwave_release_max_age_steps
                            ),
                        )
                    )
                )
                else min(
                    gripper_width_m
                    - self.in_config.release_width_threshold_m,
                    opening_margin,
                    *(
                        [last_fresh.release_proximity_margin]
                        if (
                            container_geometry == "microwave"
                            and last_fresh is not None
                        )
                        else []
                    ),
                )
            )
            occlusion_estimate = visual_in_microwave_occlusion_margin(
                last_inside_xy_margin=(
                    last_fresh.inside_xy_margin
                    if last_fresh is not None else nan
                ),
                last_below_rim_margin=(
                    last_fresh.below_rim_margin
                    if last_fresh is not None else nan
                ),
                last_above_floor_margin=(
                    last_fresh.above_floor_margin
                    if last_fresh is not None else nan
                ),
                last_entry_depth_margin=(
                    last_fresh.entry_depth_margin
                    if last_fresh is not None else nan
                ),
                release_evidence_margin=release_margin,
                occlusion_age_steps=fresh_age,
                config=self.in_config,
            )
            occlusion_entry = bool(
                container_geometry == "microwave"
                and target_occluded
                and occlusion_estimate.valid
            )
            latched_occlusion = bool(
                container_geometry == "microwave"
                and target_occluded
                and self.in_confirmed.get(env_id, False)
                and self.in_last_positive.get(env_id) is not None
            )
            if occlusion_entry or latched_occlusion:
                previous = (
                    self.in_last_positive.get(env_id)
                    if latched_occlusion else last_fresh
                )
                adjusted_inside = float(
                    previous.inside_xy_margin
                    if latched_occlusion else
                    occlusion_estimate.inside_xy_margin
                )
                effective_release = float(
                    previous.release_margin
                    if latched_occlusion else release_margin
                )
                margin = float(min(
                    adjusted_inside,
                    previous.below_rim_margin,
                    previous.above_floor_margin,
                    previous.entry_depth_margin,
                    *(
                        [effective_release]
                        if self.in_config.require_release
                        else []
                    ),
                ))
                fields.update(
                    valid=True,
                    reason=(
                        "microwave_in_latched_occlusion"
                        if latched_occlusion else
                        occlusion_estimate.reason
                    ),
                    visual_fresh=False,
                    target_source="microwave_occlusion",
                    visual_hold_age_steps=max(fresh_age, 0),
                    target_confidence=previous.target_confidence,
                    container_confidence=previous.container_confidence,
                    inside_xy_margin=adjusted_inside,
                    below_rim_margin=previous.below_rim_margin,
                    above_floor_margin=previous.above_floor_margin,
                    entry_depth_margin=previous.entry_depth_margin,
                    release_proximity_margin=(
                        previous.release_proximity_margin
                    ),
                    release_margin=effective_release,
                    release_event=bool(occlusion_entry),
                    release_latched=True,
                    visual_in_margin=margin,
                )
                self.in_release_latched[env_id] = True
                if occlusion_entry:
                    self.in_release_steps[env_id] = int(step)
                self.in_release_evidence[env_id] = effective_release
                self.in_positive_counts[env_id] = (
                    self.in_positive_counts.get(env_id, 0) + 1
                    if margin >= 0.0 else 0
                )
                if (
                    self.in_positive_counts[env_id]
                    >= self.in_confirmation_steps
                ):
                    self.in_confirmed[env_id] = True
                fields["visual_in_confirmed"] = bool(
                    self.in_confirmed.get(env_id, False)
                )

        if not fields["valid"]:
            previous = self.in_last_positive.get(env_id)
            age = self.in_hold_ages.get(env_id, 0) + 1
            release_step = self.in_release_steps.get(env_id)
            caddy_occlusion_age = (
                int(step) - int(release_step)
                if release_step is not None else -1
            )
            caddy_latched_occlusion = bool(
                previous is not None
                and previous.container_geometry == "caddy_back"
                and self.in_release_latched.get(env_id, False)
                and self.in_release_sources.get(env_id, "none")
                == "caddy_visual_handoff"
                and 0 <= caddy_occlusion_age
                <= self.in_caddy_latched_occlusion_hold_steps
            )
            can_hold = (
                previous is not None
                and self.in_release_latched.get(env_id, False)
                and (
                    caddy_latched_occlusion
                    or age <= self.in_positive_hold_steps
                )
            )
            if can_hold:
                self.in_hold_ages[env_id] = age
                fields.update(
                    valid=bool(caddy_latched_occlusion),
                    reason=(
                        (
                            "caddy_latched_occlusion:"
                            f"age={caddy_occlusion_age};cause="
                        )
                        if caddy_latched_occlusion
                        else f"in_positive_hold:age={age};cause="
                    ) + str(fields["reason"]),
                    target_source=(
                        "caddy_latched_occlusion"
                        if caddy_latched_occlusion
                        else "positive_hold"
                    ),
                    visual_fresh=False,
                    visual_hold_age_steps=(
                        caddy_occlusion_age
                        if caddy_latched_occlusion else age
                    ),
                    target_confidence=previous.target_confidence,
                    container_confidence=previous.container_confidence,
                    inside_xy_margin=previous.inside_xy_margin,
                    caddy_center_longitudinal_margin=(
                        previous.caddy_center_longitudinal_margin
                    ),
                    caddy_center_lateral_margin=(
                        previous.caddy_center_lateral_margin
                    ),
                    caddy_center_back_offset_m=(
                        previous.caddy_center_back_offset_m
                    ),
                    caddy_center_lateral_offset_m=(
                        previous.caddy_center_lateral_offset_m
                    ),
                    caddy_back_half_extent_m=(
                        previous.caddy_back_half_extent_m
                    ),
                    caddy_lateral_half_extent_m=(
                        previous.caddy_lateral_half_extent_m
                    ),
                    caddy_central_lateral_margin_m=(
                        previous.caddy_central_lateral_margin_m
                    ),
                    caddy_central_back_margin_m=(
                        previous.caddy_central_back_margin_m
                    ),
                    caddy_topology_valid=previous.caddy_topology_valid,
                    caddy_topology_reason=previous.caddy_topology_reason,
                    caddy_topology_confidence=(
                        previous.caddy_topology_confidence
                    ),
                    caddy_topology_lateral_low_m=(
                        previous.caddy_topology_lateral_low_m
                    ),
                    caddy_topology_lateral_high_m=(
                        previous.caddy_topology_lateral_high_m
                    ),
                    caddy_topology_lateral_margin_m=(
                        previous.caddy_topology_lateral_margin_m
                    ),
                    caddy_topology_back_margin_m=(
                        previous.caddy_topology_back_margin_m
                    ),
                    caddy_topology_footprint_valid=(
                        previous.caddy_topology_footprint_valid
                    ),
                    caddy_topology_footprint_longitudinal_margin_m=(
                        previous.caddy_topology_footprint_longitudinal_margin_m
                    ),
                    caddy_topology_footprint_lateral_margin_m=(
                        previous.caddy_topology_footprint_lateral_margin_m
                    ),
                    caddy_topology_footprint_below_rim_margin_m=(
                        previous.caddy_topology_footprint_below_rim_margin_m
                    ),
                    caddy_topology_footprint_above_floor_margin_m=(
                        previous.caddy_topology_footprint_above_floor_margin_m
                    ),
                    caddy_topology_footprint_margin_m=(
                        previous.caddy_topology_footprint_margin_m
                    ),
                    caddy_core_longitudinal_margin=(
                        previous.caddy_core_longitudinal_margin
                    ),
                    caddy_core_lateral_margin=(
                        previous.caddy_core_lateral_margin
                    ),
                    caddy_full_longitudinal_margin=(
                        previous.caddy_full_longitudinal_margin
                    ),
                    caddy_full_lateral_margin=(
                        previous.caddy_full_lateral_margin
                    ),
                    below_rim_margin=previous.below_rim_margin,
                    above_floor_margin=previous.above_floor_margin,
                    entry_depth_margin=previous.entry_depth_margin,
                    target_eef_distance_m=(
                        previous.target_eef_distance_m
                    ),
                    target_eef_surface_p01_m=(
                        previous.target_eef_surface_p01_m
                    ),
                    target_eef_surface_p05_m=(
                        previous.target_eef_surface_p05_m
                    ),
                    target_eef_surface_p10_m=(
                        previous.target_eef_surface_p10_m
                    ),
                    release_proximity_margin=(
                        previous.release_proximity_margin
                    ),
                    release_margin=previous.release_margin,
                    release_event=False,
                    release_latched=True,
                    release_source=previous.release_source,
                    caddy_release_candidate=False,
                    caddy_release_count=int(
                        max(
                            self.in_caddy_release_counts.get(env_id, 0),
                            self.in_caddy_dual_stage_release_counts.get(
                                env_id, 0
                            ),
                        )
                    ),
                    caddy_opening_target_displacement_m=(
                        previous.caddy_opening_target_displacement_m
                    ),
                    caddy_opening_max_target_displacement_m=(
                        previous.caddy_opening_max_target_displacement_m
                    ),
                    caddy_actual_separation_margin=(
                        previous.caddy_actual_separation_margin
                    ),
                    caddy_settled_age_margin_steps=(
                        previous.caddy_settled_age_margin_steps
                    ),
                    caddy_handoff_route=previous.caddy_handoff_route,
                    caddy_handoff_margin=previous.caddy_handoff_margin,
                    visual_in_margin=previous.visual_in_margin,
                )
                self.in_positive_counts[env_id] = (
                    self.in_positive_counts.get(env_id, 0) + 1
                )
                if (
                    self.in_positive_counts[env_id]
                    >= self.in_confirmation_steps
                ):
                    self.in_confirmed[env_id] = True
                fields["visual_in_confirmed"] = bool(
                    self.in_confirmed.get(env_id, False)
                )
            else:
                self.in_positive_counts[env_id] = 0
                self.in_hold_ages[env_id] = 0

        record = VisualInShadowRecord(
            step=int(step),
            task_id=int(task_id),
            target_object=str(target_object),
            target_class=target_class,
            container_object=str(container_object),
            container_class=container_class,
            oracle_in_margin=float(oracle_in_margin),
            **fields,
        )
        self.in_records.setdefault(env_id, []).append(record)
        if record.visual_fresh:
            self.in_last_fresh[env_id] = record
        if (
            (
                record.visual_fresh
                or record.target_source == "eef_attached_proxy"
                or record.reason.startswith(
                    "microwave_entry_occlusion:"
                )
            )
            and
            np.isfinite(record.visual_in_margin)
            and record.visual_in_margin >= 0.0
        ):
            self.in_last_positive[env_id] = record
            self.in_hold_ages[env_id] = 0
        widths.append(float(gripper_width_m))
        if len(widths) > self.in_release_history_steps:
            del widths[:-self.in_release_history_steps]

    @staticmethod
    def _drawer_level(
        articulated_object: str | None,
        task_description: str,
    ) -> str:
        text = (
            f"{articulated_object or ''} {task_description or ''}"
        ).lower()
        if "microwave" in text:
            return "microwave"
        for level in ("middle", "top", "bottom"):
            if level in text:
                return level
        return "middle"

    @staticmethod
    def _open_cabinet_class(
        articulated_object: str | None,
        task_description: str,
    ) -> str:
        return visual_articulated_class(
            articulated_object, task_description
        )

    def _append_open_invalid(
        self,
        *,
        env_id: int,
        step: int,
        task_id: int,
        articulated_object: str,
        cabinet_class: str,
        drawer_level: str,
        oracle_open_margin: float,
        reason: str,
    ) -> VisualOpenShadowRecord:
        """Append an invalid/short-held Open sample without new evidence."""
        env_id = int(env_id)
        previous = self.open_last_fresh.get(env_id)
        previous_step = self.open_last_fresh_steps.get(env_id)
        hold_age = (
            int(step) - int(previous_step)
            if previous is not None and previous_step is not None
            else 0
        )
        can_hold = bool(
            previous is not None
            and hold_age >= 1
            and hold_age <= self.open_hold_steps
        )
        nan = float("nan")
        if str(cabinet_class) == "microwave":
            self.close_microwave_settled_counts[env_id] = 0
        if can_hold:
            record = replace(
                previous,
                step=int(step),
                valid=True,
                reason=f"open_hold:age={hold_age};cause={reason}",
                visual_fresh=False,
                visual_hold_age_steps=hold_age,
                confidence=float(
                    previous.confidence
                    * max(
                        0.0,
                        1.0
                        - hold_age / float(self.open_hold_steps + 1),
                    )
                ),
                oracle_open_margin=float(oracle_open_margin),
            )
        else:
            self.open_positive_counts[env_id] = 0
            record = VisualOpenShadowRecord(
                step=int(step),
                task_id=int(task_id),
                articulated_object=str(articulated_object or ""),
                cabinet_class=str(cabinet_class),
                drawer_level=str(drawer_level),
                valid=False,
                reason=str(reason),
                visual_fresh=False,
                visual_hold_age_steps=0,
                confidence=nan,
                mask_pixels=0,
                depth_pixels=0,
                baseline_ready=env_id in self.open_baselines,
                displacement_m=nan,
                displacement_threshold_m=float(
                    self.open_config.displacement_threshold_m
                ),
                visual_open_margin=nan,
                visual_open_confirmed=bool(
                    self.open_confirmed.get(env_id, False)
                ),
                extension_axis="",
                signed_extension_m=nan,
                extension_x_minus_m=nan,
                extension_x_plus_m=nan,
                extension_y_minus_m=nan,
                extension_y_plus_m=nan,
                reference_shift_m=nan,
                baseline_band_points=0,
                current_band_points=0,
                oracle_open_margin=float(oracle_open_margin),
            )
        self.open_records.setdefault(env_id, []).append(record)
        return record

    def evaluate_open(
        self,
        *,
        env_id: int,
        step: int,
        task_id: int,
        task_description: str,
        articulated_object: str | None,
        raw_obs: dict,
        oracle_open_margin: float = float("nan"),
    ) -> VisualOpenShadowRecord | None:
        """Record one Open estimate from cabinet RGB-D only.

        ``oracle_open_margin`` is copied to the shadow CSV for offline
        comparison. It is never read by the visual estimator or confirmation
        state machine.
        """
        if not self.open_enabled:
            return None

        env_id = int(env_id)
        articulated = str(articulated_object or "")
        level = self._drawer_level(articulated, task_description)
        cabinet_class = self._open_cabinet_class(
            articulated, task_description
        )
        calibration = self.calibrations.get(env_id)
        if calibration is None or not calibration.get("valid", False):
            return self._append_open_invalid(
                env_id=env_id,
                step=step,
                task_id=task_id,
                articulated_object=articulated,
                cabinet_class=cabinet_class,
                drawer_level=level,
                oracle_open_margin=oracle_open_margin,
                reason="camera_calibration_unavailable",
            )

        try:
            eef = np.asarray(
                raw_obs["robot0_eef_pos"], dtype=np.float64
            ).reshape(3)
            estimated = self._estimate_camera_objects(
                raw_obs=raw_obs,
                camera_name=self.camera_name,
                calibration=calibration,
                eef=eef,
                min_confidence=self.min_confidence,
            )
            candidates = [
                item
                for item in estimated
                if item["detection"]["class_name"] == cabinet_class
            ]
            if not candidates:
                raise RuntimeError(
                    f"open_cabinet_not_detected:{cabinet_class}"
                )

            baseline_center = self.open_baseline_centers.get(env_id)
            if baseline_center is None:
                selected = max(
                    candidates,
                    key=lambda item: float(
                        item["detection"]["confidence"]
                    ),
                )
            else:
                selected = min(
                    candidates,
                    key=lambda item: np.linalg.norm(
                        np.asarray(
                            item["estimate"].center_world,
                            dtype=np.float64,
                        )
                        - baseline_center
                    ),
                )

            estimate = selected["estimate"]
            points = np.asarray(
                estimate.points_world, dtype=np.float64
            ).reshape(-1, 3)
            center = np.asarray(
                estimate.center_world, dtype=np.float64
            ).reshape(3)
            if env_id not in self.open_baselines:
                self.open_baselines[env_id] = points.copy()
                self.open_baseline_centers[env_id] = center.copy()

            result = visual_open_margin(
                self.open_baselines[env_id],
                points,
                drawer_level=level,
                config=self.open_config,
            )
            if not result.valid:
                raise RuntimeError(result.reason)

            if result.margin >= 0.0:
                count = self.open_positive_counts.get(env_id, 0) + 1
            else:
                count = 0
            self.open_positive_counts[env_id] = count
            if count >= self.open_confirmation_steps:
                self.open_confirmed[env_id] = True

            record = VisualOpenShadowRecord(
                step=int(step),
                task_id=int(task_id),
                articulated_object=articulated,
                cabinet_class=cabinet_class,
                drawer_level=level,
                valid=True,
                reason="",
                visual_fresh=True,
                visual_hold_age_steps=0,
                confidence=float(
                    selected["detection"]["confidence"]
                ),
                mask_pixels=int(estimate.mask_pixels),
                depth_pixels=int(estimate.depth_pixels),
                baseline_ready=True,
                displacement_m=float(result.displacement_m),
                displacement_threshold_m=float(
                    self.open_config.displacement_threshold_m
                ),
                visual_open_margin=float(result.margin),
                visual_open_confirmed=bool(
                    self.open_confirmed.get(env_id, False)
                ),
                extension_axis=str(result.extension_axis),
                signed_extension_m=float(result.signed_extension_m),
                extension_x_minus_m=float(result.extension_x_minus_m),
                extension_x_plus_m=float(result.extension_x_plus_m),
                extension_y_minus_m=float(result.extension_y_minus_m),
                extension_y_plus_m=float(result.extension_y_plus_m),
                reference_shift_m=float(result.reference_shift_m),
                baseline_band_points=int(result.baseline_band_points),
                current_band_points=int(result.current_band_points),
                oracle_open_margin=float(oracle_open_margin),
            )
            self.open_last_fresh[env_id] = record
            self.open_last_fresh_steps[env_id] = int(step)
            self.open_records.setdefault(env_id, []).append(record)
            return record
        except Exception as error:
            return self._append_open_invalid(
                env_id=env_id,
                step=step,
                task_id=task_id,
                articulated_object=articulated,
                cabinet_class=cabinet_class,
                drawer_level=level,
                oracle_open_margin=oracle_open_margin,
                reason=(
                    f"{type(error).__name__}:{str(error)[:160]}"
                ),
            )

    def _append_close_invalid(
        self,
        *,
        env_id: int,
        step: int,
        task_id: int,
        articulated_object: str,
        cabinet_class: str,
        drawer_level: str,
        oracle_close_margin: float,
        reason: str,
    ) -> VisualCloseShadowRecord:
        """Append invalid Close evidence or a short visual hold."""
        env_id = int(env_id)
        previous = self.close_last_fresh.get(env_id)
        previous_step = self.close_last_fresh_steps.get(env_id)
        hold_age = (
            int(step) - int(previous_step)
            if previous is not None and previous_step is not None
            else 0
        )
        can_hold = bool(
            previous is not None
            and 1 <= hold_age <= self.close_hold_steps
        )
        nan = float("nan")
        if can_hold:
            record = replace(
                previous,
                step=int(step),
                valid=True,
                reason=f"close_hold:age={hold_age};cause={reason}",
                visual_fresh=False,
                visual_hold_age_steps=hold_age,
                confidence=float(
                    previous.confidence
                    * max(
                        0.0,
                        1.0
                        - hold_age / float(self.close_hold_steps + 1),
                    )
                ),
                oracle_close_margin=float(oracle_close_margin),
            )
        else:
            self.close_positive_counts[env_id] = 0
            record = VisualCloseShadowRecord(
                step=int(step),
                task_id=int(task_id),
                articulated_object=str(articulated_object or ""),
                cabinet_class=str(cabinet_class),
                drawer_level=str(drawer_level),
                valid=False,
                reason=str(reason),
                visual_fresh=False,
                visual_hold_age_steps=0,
                confidence=nan,
                mask_pixels=0,
                depth_pixels=0,
                baseline_ready=env_id in self.close_baselines,
                retraction_m=nan,
                retraction_threshold_m=float(
                    self.close_config.retraction_threshold_m
                ),
                raw_visual_close_margin=nan,
                visual_close_margin=nan,
                visual_in_gate_required=bool(
                    self.close_require_in_confirmed
                ),
                visual_in_gate_open=bool(
                    (not self.close_require_in_confirmed)
                    or self.in_confirmed.get(env_id, False)
                ),
                visual_close_confirmed=bool(
                    self.close_confirmed.get(env_id, False)
                ),
                extension_axis="",
                signed_extension_m=nan,
                reference_shift_m=nan,
                baseline_band_points=0,
                current_band_points=0,
                retraction_q05_m=nan,
                retraction_q10_m=nan,
                retraction_q25_m=nan,
                retraction_q50_m=nan,
                retraction_q75_m=nan,
                retraction_q90_m=nan,
                retraction_q95_m=nan,
                oracle_close_margin=float(oracle_close_margin),
                close_mode=self.close_mode,
                panel_retraction_fraction=nan,
                panel_close_margin=nan,
                panel_baseline_edge_px=nan,
                panel_current_edge_px=nan,
                panel_reference_shift_fraction=nan,
                panel_baseline_pixels=0,
                panel_current_pixels=0,
            )
        self.close_records.setdefault(env_id, []).append(record)
        return record

    def _record_close_panel_diagnostic(
        self,
        *,
        env_id: int,
        step: int,
        drawer_level: str,
        raw_obs: dict,
        selected: dict | None,
        raw_close_margin: float,
        oracle_close_margin: float,
        reason: str = "",
    ) -> None:
        """Cache fixed-ROI RGB/depth even while YOLOE loses the cabinet."""
        if not self.close_panel_diagnostic:
            return
        positive = bool(float(raw_close_margin) >= 0.0)
        previous = self.close_panel_last_positive.get(int(env_id))
        self.close_panel_last_positive[int(env_id)] = positive
        should_save = bool(
            int(step) == 1
            or int(step) % self.close_panel_diagnostic_interval == 0
            or previous is None
            or previous != positive
        )
        if not should_save:
            return

        image_rgb = np.ascontiguousarray(
            raw_obs[f"{self.camera_name}_image"][::-1, ::-1]
        )
        current_roi = None
        mask = np.zeros(image_rgb.shape[:2], dtype=bool)
        confidence = float("nan")
        if selected is not None:
            mask = np.asarray(
                selected["detection"]["mask"], dtype=bool
            )
            confidence = float(selected["detection"]["confidence"])
            is_microwave = (
                selected["detection"]["class_name"] == "microwave"
            )
            current_roi = (
                microwave_roi(mask) if is_microwave
                else drawer_panel_roi(mask, drawer_level=drawer_level)
            )
            if (
                int(env_id) not in self.close_panel_fixed_rois
                and current_roi.valid
                and not is_microwave
            ):
                self.close_panel_fixed_rois[int(env_id)] = current_roi
            if int(env_id) not in self.microwave_fixed_rois and current_roi.valid and is_microwave:
                self.microwave_fixed_rois[int(env_id)] = current_roi
        is_microwave = bool(
            drawer_level == "microwave"
            or self.close_mode == "microwave_silhouette"
            or (selected is not None and selected["detection"]["class_name"] == "microwave")
        )
        fixed_roi = (
            self.microwave_fixed_rois.get(int(env_id)) if is_microwave
            else self.close_panel_fixed_rois.get(int(env_id))
        )
        roi = fixed_roi if fixed_roi is not None else current_roi
        if roi is None:
            roi = microwave_roi(mask) if is_microwave else drawer_panel_roi(mask, drawer_level=drawer_level)
        diagnostic_roi = (
            microwave_door_search_roi(
                roi, image_rgb.shape[:2], self.microwave_close_config
            )
            if is_microwave and roi.valid else roi
        )
        overlay = (
            make_microwave_overlay(
                image_rgb, mask, np.zeros_like(mask), diagnostic_roi
            )
            if is_microwave
            else make_drawer_panel_overlay(image_rgb, mask, diagnostic_roi)
        )
        raw_depth = np.asarray(
            raw_obs[f"{self.camera_name}_depth"]
        ).squeeze()[::-1, ::-1]
        depth_overlay = (
            make_microwave_depth(raw_depth, diagnostic_roi)
            if is_microwave
            else make_depth_diagnostic(raw_depth, diagnostic_roi)
        )
        self.close_panel_frames.setdefault(int(env_id), []).append({
            "step": int(step),
            "confidence": confidence,
            "raw_visual_close_margin": float(raw_close_margin),
            "oracle_close_margin": float(oracle_close_margin),
            "reason": str(reason),
            "roi": asdict(diagnostic_roi),
            "overlay": overlay,
            "depth_overlay": depth_overlay,
        })

    def _record_close_panel_invalid(
        self,
        *,
        env_id: int,
        step: int,
        drawer_level: str,
        raw_obs: dict,
        oracle_close_margin: float,
        reason: str,
    ) -> None:
        if not self.close_panel_diagnostic:
            return
        if (
            int(step) == 1
            or int(step) % self.close_panel_diagnostic_interval == 0
        ):
            self._record_close_panel_diagnostic(
                env_id=env_id,
                step=step,
                drawer_level=drawer_level,
                raw_obs=raw_obs,
                selected=None,
                raw_close_margin=float("nan"),
                oracle_close_margin=oracle_close_margin,
                reason=reason,
            )

    def evaluate_close(
        self,
        *,
        env_id: int,
        step: int,
        task_id: int,
        task_description: str,
        articulated_object: str | None,
        raw_obs: dict,
        oracle_close_margin: float = float("nan"),
    ) -> VisualCloseShadowRecord | None:
        """Record reset-open drawer retraction from cabinet RGB-D only.

        The oracle value is copied solely for offline comparison and is never
        read by the estimator or confirmation state machine.
        """
        if not self.close_enabled:
            return None

        env_id = int(env_id)
        articulated = str(articulated_object or "")
        level = self._drawer_level(articulated, task_description)
        cabinet_class = self._open_cabinet_class(
            articulated, task_description
        )
        calibration = self.calibrations.get(env_id)
        invalid_fields = dict(
            env_id=env_id,
            step=step,
            task_id=task_id,
            articulated_object=articulated,
            cabinet_class=cabinet_class,
            drawer_level=level,
            oracle_close_margin=oracle_close_margin,
        )
        if calibration is None or not calibration.get("valid", False):
            return self._append_close_invalid(
                **invalid_fields,
                reason="camera_calibration_unavailable",
            )

        try:
            eef = np.asarray(
                raw_obs["robot0_eef_pos"], dtype=np.float64
            ).reshape(3)
            image_rgb = None
            image_depth = None
            if cabinet_class == "microwave":
                image_rgb = np.ascontiguousarray(
                    raw_obs[f"{self.camera_name}_image"][::-1, ::-1]
                )
                raw_depth = np.asarray(
                    raw_obs[f"{self.camera_name}_depth"]
                ).squeeze()
                if self.mask_to_depth_orientation == "rot180":
                    image_depth = np.ascontiguousarray(
                        raw_depth[::-1, ::-1]
                    )
                elif self.mask_to_depth_orientation == "vflip":
                    image_depth = np.ascontiguousarray(raw_depth[::-1, :])
                elif self.mask_to_depth_orientation == "hflip":
                    image_depth = np.ascontiguousarray(raw_depth[:, ::-1])
                else:
                    image_depth = np.ascontiguousarray(raw_depth)
            estimated = self._estimate_camera_objects(
                raw_obs=raw_obs,
                camera_name=self.camera_name,
                calibration=calibration,
                eef=eef,
                min_confidence=self.min_confidence,
            )
            candidates = [
                item for item in estimated
                if item["detection"]["class_name"] == cabinet_class
            ]
            baseline_ready = env_id in self.microwave_baseline_images
            if not candidates and not (
                cabinet_class == "microwave" and baseline_ready
            ):
                raise RuntimeError(
                    f"close_cabinet_not_detected:{cabinet_class}"
                )

            track_center = self.close_track_centers.get(env_id)
            selected = None
            if candidates and track_center is None:
                selected = max(
                    candidates,
                    key=lambda item: float(
                        item["detection"]["confidence"]
                    ),
                )
            elif candidates:
                selected = min(
                    candidates,
                    key=lambda item: np.linalg.norm(
                        np.asarray(
                            item["estimate"].center_world,
                            dtype=np.float64,
                        )
                        - track_center
                    ),
                )

            estimate = selected["estimate"] if selected is not None else None
            points = None
            if estimate is not None:
                points = np.asarray(
                    estimate.points_world, dtype=np.float64
                ).reshape(-1, 3)
                center = np.asarray(
                    estimate.center_world, dtype=np.float64
                ).reshape(3)
                if track_center is None:
                    self.close_track_centers[env_id] = center.copy()
                else:
                    self.close_track_centers[env_id] = (
                        self.track_ema * track_center
                        + (1.0 - self.track_ema) * center
                    )
                if env_id not in self.close_baselines:
                    self.close_baselines[env_id] = points.copy()
                    self.close_baseline_centers[env_id] = center.copy()

            if selected is not None:
                current_mask = np.asarray(
                    selected["detection"]["mask"], dtype=bool
                )
                confidence = float(selected["detection"]["confidence"])
                mask_pixels = int(estimate.mask_pixels)
                depth_pixels = int(estimate.depth_pixels)
            else:
                current_mask = np.zeros(image_rgb.shape[:2], dtype=bool)
                confidence = float("nan")
                mask_pixels = 0
                depth_pixels = 0
            if cabinet_class == "microwave":
                if env_id not in self.microwave_baseline_masks:
                    if selected is None:
                        raise RuntimeError(
                            "close_cabinet_not_detected:microwave"
                        )
                    fixed_roi = microwave_roi(current_mask)
                    if not fixed_roi.valid:
                        raise RuntimeError(fixed_roi.reason)
                    self.microwave_baseline_masks[env_id] = current_mask.copy()
                    self.microwave_baseline_images[env_id] = image_rgb.copy()
                    self.microwave_baseline_depths[env_id] = image_depth.copy()
                    self.microwave_fixed_rois[env_id] = fixed_roi
                panel_result = visual_microwave_dark_door_close_margin(
                    self.microwave_baseline_images[env_id],
                    image_rgb,
                    self.microwave_fixed_rois[env_id],
                    config=self.microwave_close_config,
                    baseline_mask_pixels=int(np.count_nonzero(
                        self.microwave_baseline_masks[env_id]
                    )),
                    current_mask_pixels=int(np.count_nonzero(current_mask)),
                )
                if not panel_result.valid:
                    panel_result = visual_microwave_door_edge_close_margin(
                        self.microwave_baseline_images[env_id],
                        image_rgb,
                        self.microwave_fixed_rois[env_id],
                        config=self.microwave_close_config,
                        baseline_mask_pixels=int(np.count_nonzero(
                            self.microwave_baseline_masks[env_id]
                        )),
                        current_mask_pixels=int(np.count_nonzero(
                            current_mask
                        )),
                        reset_raw_depth=(
                            self.microwave_baseline_depths[env_id]
                        ),
                        current_raw_depth=image_depth,
                    )
                    if not panel_result.valid:
                        raise RuntimeError(
                            "microwave_door_geometry:"
                            f"{panel_result.reason}"
                        )
                nan = float("nan")
                result = SimpleNamespace(
                    retraction_m=panel_result.retraction_fraction,
                    margin=panel_result.margin,
                    extension_axis="image_width",
                    signed_extension_m=panel_result.retraction_fraction,
                    reference_shift_m=panel_result.reference_shift_fraction,
                    baseline_band_points=panel_result.baseline_panel_pixels,
                    current_band_points=panel_result.current_panel_pixels,
                    retraction_q05_m=nan, retraction_q10_m=nan,
                    retraction_q25_m=nan, retraction_q50_m=nan,
                    retraction_q75_m=nan, retraction_q90_m=nan,
                    retraction_q95_m=nan,
                )
            else:
                result = visual_close_margin(
                    self.close_baselines[env_id],
                    points,
                    drawer_level=level,
                    config=self.close_config,
                )
                if not result.valid:
                    raise RuntimeError(result.reason)
                if env_id not in self.close_panel_baseline_masks:
                    fixed_roi = drawer_panel_roi(
                        current_mask, drawer_level=level
                    )
                    if not fixed_roi.valid:
                        raise RuntimeError(fixed_roi.reason)
                    self.close_panel_baseline_masks[env_id] = current_mask.copy()
                    self.close_panel_fixed_rois[env_id] = fixed_roi
                panel_result = visual_drawer_close_margin(
                    self.close_panel_baseline_masks[env_id],
                    current_mask,
                    self.close_panel_fixed_rois[env_id],
                    config=self.close_panel_config,
                )
                if self.close_mode == "fixed_panel_edge" and not panel_result.valid:
                    raise RuntimeError(f"panel:{panel_result.reason}")
            if cabinet_class == "microwave":
                raw_margin = float(panel_result.margin)
                settled_count = _next_microwave_settled_close_count(
                    self.close_microwave_settled_counts.get(env_id, 0),
                    panel_result.retraction_fraction,
                    self.close_microwave_settled_progress_threshold,
                )
                self.close_microwave_settled_counts[env_id] = settled_count
                settled_confirmed = bool(
                    settled_count
                    >= self.close_microwave_settled_confirmation_steps
                )
                raw_margin = _compose_microwave_settled_close_margin(
                    raw_margin,
                    panel_result.retraction_fraction,
                    self.close_microwave_settled_progress_threshold,
                    settled_count,
                    self.close_microwave_settled_confirmation_steps,
                )
                microwave_close_route = (
                    "strict"
                    if panel_result.margin >= 0.0
                    else (
                        "settled_near_close"
                        if settled_confirmed else "none"
                    )
                )
            elif self.close_mode == "fixed_panel_edge":
                raw_margin = float(panel_result.margin)
                settled_count = 0
                settled_confirmed = False
                microwave_close_route = "not_microwave"
            else:
                raw_margin = float(result.margin)
                settled_count = 0
                settled_confirmed = False
                microwave_close_route = "not_microwave"

            visual_in_gate_open = bool(
                (not self.close_require_in_confirmed)
                or self.in_confirmed.get(env_id, False)
            )
            effective_margin = apply_visual_in_gate(
                raw_margin,
                require_in_confirmed=self.close_require_in_confirmed,
                visual_in_confirmed=visual_in_gate_open,
            )
            self._record_close_panel_diagnostic(
                env_id=env_id,
                step=step,
                drawer_level=level,
                raw_obs=raw_obs,
                selected=selected,
                raw_close_margin=raw_margin,
                oracle_close_margin=oracle_close_margin,
            )
            count = (
                self.close_positive_counts.get(env_id, 0) + 1
                if effective_margin >= 0.0 else 0
            )
            self.close_positive_counts[env_id] = count
            if count >= self.close_confirmation_steps:
                self.close_confirmed[env_id] = True

            record = VisualCloseShadowRecord(
                step=int(step),
                task_id=int(task_id),
                articulated_object=articulated,
                cabinet_class=cabinet_class,
                drawer_level=level,
                valid=True,
                reason=("" if visual_in_gate_open else "await_visual_in"),
                visual_fresh=True,
                visual_hold_age_steps=0,
                confidence=confidence,
                mask_pixels=mask_pixels,
                depth_pixels=depth_pixels,
                baseline_ready=True,
                retraction_m=float(result.retraction_m),
                retraction_threshold_m=float(
                    self.microwave_close_config.door_close_progress_threshold
                    if cabinet_class == "microwave"
                    else self.close_config.retraction_threshold_m
                ),
                raw_visual_close_margin=raw_margin,
                visual_close_margin=float(effective_margin),
                visual_in_gate_required=bool(
                    self.close_require_in_confirmed
                ),
                visual_in_gate_open=visual_in_gate_open,
                visual_close_confirmed=bool(
                    self.close_confirmed.get(env_id, False)
                ),
                extension_axis=str(result.extension_axis),
                signed_extension_m=float(result.signed_extension_m),
                reference_shift_m=float(result.reference_shift_m),
                baseline_band_points=int(result.baseline_band_points),
                current_band_points=int(result.current_band_points),
                retraction_q05_m=float(result.retraction_q05_m),
                retraction_q10_m=float(result.retraction_q10_m),
                retraction_q25_m=float(result.retraction_q25_m),
                retraction_q50_m=float(result.retraction_q50_m),
                retraction_q75_m=float(result.retraction_q75_m),
                retraction_q90_m=float(result.retraction_q90_m),
                retraction_q95_m=float(result.retraction_q95_m),
                oracle_close_margin=float(oracle_close_margin),
                close_mode=self.close_mode,
                panel_retraction_fraction=float(
                    panel_result.retraction_fraction
                ),
                panel_close_margin=float(panel_result.margin),
                panel_baseline_edge_px=float(
                    panel_result.baseline_edge_px
                ),
                panel_current_edge_px=float(
                    panel_result.current_edge_px
                ),
                panel_reference_shift_fraction=float(
                    panel_result.reference_shift_fraction
                ),
                panel_baseline_pixels=int(
                    panel_result.baseline_panel_pixels
                ),
                panel_current_pixels=int(
                    panel_result.current_panel_pixels
                ),
                microwave_settled_close_count=int(settled_count),
                microwave_settled_close_confirmed=bool(settled_confirmed),
                microwave_close_route=str(microwave_close_route),
            )
            self.close_last_fresh[env_id] = record
            self.close_last_fresh_steps[env_id] = int(step)
            self.close_records.setdefault(env_id, []).append(record)
            return record
        except Exception as error:
            self._record_close_panel_invalid(
                env_id=env_id,
                step=step,
                drawer_level=level,
                raw_obs=raw_obs,
                oracle_close_margin=oracle_close_margin,
                reason=f"{type(error).__name__}:{str(error)[:160]}",
            )
            return self._append_close_invalid(
                **invalid_fields,
                reason=f"{type(error).__name__}:{str(error)[:160]}",
            )

    def evaluate_turnon(
        self,
        *,
        env_id: int,
        step: int,
        task_id: int,
        articulated_object: str | None,
        raw_obs: dict,
        oracle_turnon_margin: float = float("nan"),
    ) -> VisualTurnOnShadowRecord | None:
        """Probe stove-state observability from RGB-D, never from joint qpos.

        The oracle argument is copied only into the returned record for later
        comparison.  It is never passed to the ROI or feature functions.
        """
        if not self.turnon_enabled:
            return None
        env_id = int(env_id)
        nan = float("nan")
        image_rgb = np.ascontiguousarray(
            raw_obs[f"{self.camera_name}_image"][::-1, ::-1]
        )
        raw_depth = np.asarray(
            raw_obs[f"{self.camera_name}_depth"]
        ).squeeze()[::-1, ::-1]
        confidence = nan
        reason = ""
        features = None
        try:
            detections = [
                detection for detection in self._infer_image(image_rgb)
                if detection["confidence"] >= self.min_confidence
                and detection["class_name"] == "stove"
            ]
            if detections:
                selected = max(
                    detections, key=lambda item: float(item["confidence"])
                )
                confidence = float(selected["confidence"])
                if env_id not in self.turnon_rois:
                    roi = stove_roi_from_mask(
                        selected["mask"],
                        padding_fraction=(
                            self.turnon_config.roi_padding_fraction
                        ),
                        min_mask_pixels=self.turnon_config.min_mask_pixels,
                    )
                    if not roi.valid:
                        raise RuntimeError(roi.reason)
                    self.turnon_rois[env_id] = roi
            roi = self.turnon_rois.get(env_id)
            if roi is None:
                raise RuntimeError("stove_not_detected_before_roi_lock")

            if env_id not in self.turnon_baseline_rgb:
                self.turnon_baseline_rgb[env_id] = image_rgb.copy()
                self.turnon_baseline_depth[env_id] = raw_depth.copy()
            features = reset_relative_features(
                self.turnon_baseline_rgb[env_id],
                image_rgb,
                roi,
                baseline_depth=self.turnon_baseline_depth[env_id],
                current_depth=raw_depth,
                grid_rows=self.turnon_config.grid_rows,
                grid_cols=self.turnon_config.grid_cols,
            )
            if not features.valid:
                raise RuntimeError(features.reason)
            turnon = visual_turnon_red_margin(
                image_rgb,
                roi,
                red_min_intensity=self.turnon_config.red_min_intensity,
                red_excess_threshold=(
                    self.turnon_config.red_excess_threshold
                ),
                red_fraction_threshold=(
                    self.turnon_config.red_fraction_threshold
                ),
            )
            if not turnon.valid:
                raise RuntimeError(turnon.reason)
            positive_count = (
                self.turnon_positive_counts.get(env_id, 0) + 1
                if turnon.margin >= 0.0 else 0
            )
            self.turnon_positive_counts[env_id] = positive_count
            if positive_count >= self.turnon_config.confirmation_steps:
                self.turnon_confirmed[env_id] = True
            record = VisualTurnOnShadowRecord(
                step=int(step), task_id=int(task_id),
                articulated_object=str(articulated_object or "stove"),
                valid=True, reason="", confidence=confidence,
                roi_ready=True,
                rgb_mae=features.rgb_mae,
                gray_mae=features.gray_mae,
                edge_mae=features.edge_mae,
                depth_mae=features.depth_mae,
                max_cell_change=features.max_cell_change,
                max_cell_row=features.max_cell_row,
                max_cell_col=features.max_cell_col,
                red_activation_fraction=turnon.red_fraction,
                visual_turnon_margin=turnon.margin,
                visual_turnon_confirmed=self.turnon_confirmed.get(
                    env_id, False
                ),
                oracle_turnon_margin=float(oracle_turnon_margin),
            )
        except Exception as error:
            reason = f"{type(error).__name__}:{str(error)[:160]}"
            record = VisualTurnOnShadowRecord(
                step=int(step), task_id=int(task_id),
                articulated_object=str(articulated_object or "stove"),
                valid=False, reason=reason, confidence=confidence,
                roi_ready=env_id in self.turnon_rois,
                rgb_mae=nan, gray_mae=nan, edge_mae=nan,
                depth_mae=nan, max_cell_change=nan,
                max_cell_row=-1, max_cell_col=-1,
                red_activation_fraction=nan,
                visual_turnon_margin=nan,
                visual_turnon_confirmed=self.turnon_confirmed.get(
                    env_id, False
                ),
                oracle_turnon_margin=float(oracle_turnon_margin),
            )
            self.turnon_positive_counts[env_id] = 0
        self.turnon_records.setdefault(env_id, []).append(record)

        if (
            int(step) == 1
            or int(step) % self.turnon_config.diagnostic_interval == 0
        ):
            roi = self.turnon_rois.get(env_id)
            overlay = (
                make_turnon_overlay(
                    image_rgb, roi,
                    grid_rows=self.turnon_config.grid_rows,
                    grid_cols=self.turnon_config.grid_cols,
                )
                if roi is not None else image_rgb
            )
            self.turnon_frames.setdefault(env_id, []).append({
                "step": int(step),
                "valid": bool(record.valid),
                "reason": str(record.reason),
                "rgb_mae": float(record.rgb_mae),
                "max_cell_change": float(record.max_cell_change),
                "red_activation_fraction": float(
                    record.red_activation_fraction
                ),
                "visual_turnon_margin": float(
                    record.visual_turnon_margin
                ),
                "visual_turnon_confirmed": bool(
                    record.visual_turnon_confirmed
                ),
                "overlay": overlay,
            })
        return record

    def evaluate(
        self,
        *,
        env_id: int,
        step: int,
        task_id: int,
        task_description: str,
        target_object: str,
        raw_obs: dict,
        oracle_distance_m: float = float("nan"),
        oracle_pick_margin: float = float("nan"),
        on_target_object: str | None = None,
        oracle_on_margin: float = float("nan"),
        in_target_object: str | None = None,
        oracle_in_margin: float = float("nan"),
        oracle_center=None,
        two_finger_grasp: bool = False,
    ) -> VisualSTLShadowRecord:
        env_id = int(env_id)
        target_class = _object_class(target_object)
        eef = np.asarray(
            raw_obs["robot0_eef_pos"],
            dtype=np.float64,
        ).reshape(3)
        eef_rotation = self._eef_rotation(raw_obs)
        gripper_width_m = self._gripper_width(raw_obs)
        if oracle_center is None:
            oracle_center = np.full(3, np.nan)
        else:
            oracle_center = np.asarray(
                oracle_center,
                dtype=np.float64,
            ).reshape(3)

        calibration = self.calibrations.get(env_id)

        if (
            calibration is None
            or not calibration.get("valid", False)
        ):
            record = self._invalid_record(
                env_id=env_id,
                step=step,
                task_id=task_id,
                target_object=target_object,
                target_class=target_class,
                reason="camera_calibration_unavailable",
                two_finger_grasp=two_finger_grasp,
                oracle_distance_m=oracle_distance_m,
                oracle_pick_margin=oracle_pick_margin,
                on_target_object=on_target_object,
                oracle_on_margin=oracle_on_margin,
                eef=eef,
                oracle_center=oracle_center,
                gripper_width_m=gripper_width_m,
            )
            self.records.setdefault(env_id, []).append(
                record
            )
            self._record_visual_in(
                env_id=env_id, step=step, task_id=task_id,
                target_object=target_object,
                container_object=in_target_object,
                oracle_in_margin=oracle_in_margin,
                gripper_width_m=gripper_width_m,
                eef=eef,
                reason="camera_calibration_unavailable",
            )
            return record

        if target_class is None:
            record = self._invalid_record(
                env_id=env_id,
                step=step,
                task_id=task_id,
                target_object=target_object,
                target_class="",
                reason="unsupported_target_class",
                two_finger_grasp=two_finger_grasp,
                oracle_distance_m=oracle_distance_m,
                oracle_pick_margin=oracle_pick_margin,
                on_target_object=on_target_object,
                oracle_on_margin=oracle_on_margin,
                eef=eef,
                oracle_center=oracle_center,
                gripper_width_m=gripper_width_m,
            )
            self.records.setdefault(env_id, []).append(
                record
            )
            self._record_visual_in(
                env_id=env_id, step=step, task_id=task_id,
                target_object=target_object,
                container_object=in_target_object,
                oracle_in_margin=oracle_in_margin,
                gripper_width_m=gripper_width_m,
                eef=eef,
                reason="unsupported_target_class",
            )
            return record

        try:
            estimated = self._estimate_camera_objects(
                raw_obs=raw_obs,
                camera_name=self.camera_name,
                calibration=calibration,
                eef=eef,
                min_confidence=self.min_confidence,
            )
            all_estimated = list(estimated)

            try:
                selected, selector = self._select_camera_target(
                    env_id=env_id,
                    estimated=estimated,
                    target_class=target_class,
                    task_description=task_description,
                    allow_unique_reacquire=(
                        self.in_enabled and in_target_object is not None
                    ),
                )
            except RuntimeError as primary_error:
                primary_reason = str(primary_error)
                fallback_calibration = (
                    self._current_fallback_calibration(
                        env_id,
                        raw_obs,
                    )
                )
                can_fallback = (
                    fallback_calibration is not None
                    and self.fallback_camera_name is not None
                    and (
                        primary_reason.startswith(
                            "target_not_detected:"
                        )
                        or primary_reason.startswith(
                            "target_track_gate:"
                        )
                    )
                )
                if not can_fallback:
                    raise

                fallback_estimated = (
                    self._estimate_camera_objects(
                        raw_obs=raw_obs,
                        camera_name=self.fallback_camera_name,
                        calibration=fallback_calibration,
                        eef=eef,
                        min_confidence=(
                            self.fallback_min_confidence
                        ),
                    )
                )
                all_estimated.extend(fallback_estimated)
                selected, fallback_selector = (
                    self._select_camera_target(
                        env_id=env_id,
                        estimated=fallback_estimated,
                        target_class=target_class,
                        task_description=task_description,
                        allow_unique_reacquire=(
                            self.in_enabled
                            and in_target_object is not None
                        ),
                    )
                )
                selector = (
                    f"fallback_{self.fallback_camera_name}:"
                    f"{fallback_selector}"
                )

            detection = selected["detection"]
            estimate = selected["estimate"]
            visual_center = np.asarray(
                estimate.center_world,
                dtype=np.float64,
            )

            previous_center = self.target_tracks.get(env_id)
            if previous_center is None:
                tracked_center = visual_center.copy()
            else:
                tracked_center = (
                    self.track_ema * previous_center
                    + (1.0 - self.track_ema)
                    * visual_center
                )

            self.target_tracks[env_id] = tracked_center

            grasp_components = self._visual_grasp_components(
                env_id=env_id,
                step=step,
                visual_center=visual_center,
                eef=eef,
                gripper_width_m=gripper_width_m,
                near_margin=self._grasp_near_margin(estimate),
            )
            runtime_grasp_margin = float(
                grasp_components["grasp_margin"]
            )

            object_xy_q05_m, object_xy_q95_m = (
                self._visual_xy_bounds(estimate)
            )
            (
                object_on_estimate,
                object_on_source,
                object_on_state,
            ) = self._on_object_estimate(
                env_id=env_id,
                step=step,
                eef=eef,
                eef_rotation=eef_rotation,
                gripper_width_m=gripper_width_m,
                center=visual_center,
                z_p05_m=estimate.z_p05_m,
                z_p95_m=estimate.z_p95_m,
                xy_q05_m=object_xy_q05_m,
                xy_q95_m=object_xy_q95_m,
                points_world=getattr(
                    estimate, "points_world", None
                ),
                near_margin=grasp_components["near_margin"],
                fresh=True,
            )
            # Cache attached RGB-D geometry before a later In stage starts.
            # The object may already be occluded when that stage activates.
            self._cache_in_attached_target(
                env_id=env_id,
                selected=selected,
                eef=eef,
                eef_rotation=eef_rotation,
                attached=bool(
                    (object_on_state or {}).get("attached", False)
                ),
            )

            on_target_class = self._object_on_target_class(
                on_target_object
            )
            if on_target_object is None:
                self.on_positive_counts[env_id] = 0
                on_fields = self._empty_on_fields(
                    env_id,
                    on_target_object,
                    oracle_on_margin,
                    "no_object_on_atom",
                )
            elif object_on_estimate is None:
                on_fields = self._empty_on_fields(
                    env_id,
                    on_target_object,
                    oracle_on_margin,
                    f"on_object_unavailable:{object_on_source}",
                )
            elif on_target_class is None:
                self.on_positive_counts[env_id] = 0
                on_fields = self._empty_on_fields(
                    env_id,
                    on_target_object,
                    oracle_on_margin,
                    "unsupported_semantic_region_target",
                )
            else:
                support_candidates = [
                    item
                    for item in estimated
                    if (
                        item is not selected
                        and item["detection"]["class_name"]
                        == on_target_class
                    )
                ]
                support_estimate, support_confidence, support_reason = (
                    self._resolve_on_support(
                        env_id=env_id,
                        step=step,
                        support_identity=str(on_target_object),
                        support_class=on_target_class,
                        support_candidates=support_candidates,
                        object_center=visual_center,
                    )
                )
                if support_estimate is None:
                    on_fields = self._empty_on_fields(
                        env_id,
                        on_target_object,
                        oracle_on_margin,
                        support_reason,
                    )
                else:
                    components = self._visual_on_components(
                        env_id=env_id,
                        step=step,
                        object_estimate=object_on_estimate,
                        object_source=object_on_source,
                        support_estimate=support_estimate,
                        support_class=on_target_class,
                        on_target_object=on_target_object,
                        gripper_width_m=gripper_width_m,
                    )
                    spatial_pick_proxy = float(
                        components["spatial_pick_proxy_margin"]
                    )
                    if np.isfinite(spatial_pick_proxy):
                        runtime_grasp_margin = (
                            max(runtime_grasp_margin, spatial_pick_proxy)
                            if np.isfinite(runtime_grasp_margin)
                            else spatial_pick_proxy
                        )
                    on_fields = {
                        "on_target_object": str(on_target_object),
                        "on_target_class": on_target_class,
                        "oracle_on_margin": float(oracle_on_margin),
                        "visual_on_valid": bool(
                            components.get("valid", True)
                        ),
                        "visual_on_reason": (
                            components.get("reason") or support_reason
                        ),
                        "visual_on_target_confidence": support_confidence,
                        "visual_on_xy_distance_m": float(
                            components["xy_distance_m"]
                        ),
                        "visual_on_xy_margin": float(
                            components["xy_margin"]
                        ),
                        "visual_on_completion_xy_margin": float(
                            components["completion_xy_margin"]
                        ),
                        "visual_on_height_delta_m": float(
                            components["height_delta_m"]
                        ),
                        "visual_on_above_margin": float(
                            components["above_margin"]
                        ),
                        "visual_on_surface_gap_m": float(
                            components["surface_gap_m"]
                        ),
                        "visual_on_support_margin": float(
                            components["support_margin"]
                        ),
                        "visual_on_release_margin": float(
                            components["release_margin"]
                        ),
                        "visual_on_margin": float(
                            components["margin"]
                        ),
                        "visual_on_confirmed": bool(
                            components["confirmed"]
                        ),
                        "visual_on_object_source": object_on_source,
                        "visual_on_attached": bool(
                            (object_on_state or {}).get("attached", False)
                        ),
                        "visual_on_released": bool(
                            (object_on_state or {}).get("released", False)
                        ),
                        "visual_on_geometry_mode": str(
                            components["geometry_mode"]
                        ),
                        "visual_on_inside_xy_margin": float(
                            components["inside_xy_margin"]
                        ),
                        "visual_on_below_rim_margin": float(
                            components["below_rim_margin"]
                        ),
                        "visual_on_above_floor_margin": float(
                            components["above_floor_margin"]
                        ),
                        "visual_on_handoff_used": bool(
                            components["handoff_used"]
                        ),
                        "visual_on_handoff_age_steps": int(
                            components["handoff_age_steps"]
                        ),
                        "visual_on_handoff_margin": float(
                            components["handoff_margin"]
                        ),
                        "visual_on_contact_valid": bool(
                            components["contact_valid"]
                        ),
                        "visual_on_contact_reason": str(
                            components["contact_reason"]
                        ),
                        "visual_on_contact_p01_m": float(
                            components["contact_p01_m"]
                        ),
                        "visual_on_contact_p05_m": float(
                            components["contact_p05_m"]
                        ),
                        "visual_on_contact_p10_m": float(
                            components["contact_p10_m"]
                        ),
                        "visual_on_contact_margin": float(
                            components["contact_margin"]
                        ),
                        "visual_on_object_center_x_m": float(
                            components["object_center_x_m"]
                        ),
                        "visual_on_object_center_y_m": float(
                            components["object_center_y_m"]
                        ),
                        "visual_on_object_center_z_m": float(
                            components["object_center_z_m"]
                        ),
                        "visual_on_support_center_x_m": float(
                            components["support_center_x_m"]
                        ),
                        "visual_on_support_center_y_m": float(
                            components["support_center_y_m"]
                        ),
                        "visual_on_support_center_z_m": float(
                            components["support_center_z_m"]
                        ),
                        "visual_on_support_locked": bool(
                            components["support_locked"]
                        ),
                        "visual_on_object_bottom_z_m": float(
                            components["object_bottom_z_m"]
                        ),
                        "visual_on_support_z_p95_m": float(
                            components["support_z_p95_m"]
                        ),
                        "visual_on_support_z_p99_m": float(
                            components["support_z_p99_m"]
                        ),
                        "visual_on_support_z_p995_m": float(
                            components["support_z_p995_m"]
                        ),
                        "visual_on_support_baseline_center_x_m": float(
                            components["support_baseline_center_x_m"]
                        ),
                        "visual_on_support_baseline_center_y_m": float(
                            components["support_baseline_center_y_m"]
                        ),
                        "visual_on_support_baseline_center_z_m": float(
                            components["support_baseline_center_z_m"]
                        ),
                        "visual_on_support_baseline_z_p95_m": float(
                            components["support_baseline_z_p95_m"]
                        ),
                        "visual_on_support_center_drift_m": float(
                            components["support_center_drift_m"]
                        ),
                        "visual_on_support_z_p95_drift_m": float(
                            components["support_z_p95_drift_m"]
                        ),
                        "visual_on_site_center_delta_q99_m": float(
                            components["site_center_delta_q99_m"]
                        ),
                        "visual_on_site_bottom_gap_q99_m": float(
                            components["site_bottom_gap_q99_m"]
                        ),
                        "visual_on_spatial_relation": str(
                            components["spatial_relation"]
                        ),
                        "visual_on_relative_x_m": float(
                            components["relative_x_m"]
                        ),
                        "visual_on_relative_y_m": float(
                            components["relative_y_m"]
                        ),
                        "visual_on_relative_z_m": float(
                            components["relative_z_m"]
                        ),
                        "visual_on_anchor_x_q05_m": float(
                            components["anchor_x_q05_m"]
                        ),
                        "visual_on_anchor_x_q95_m": float(
                            components["anchor_x_q95_m"]
                        ),
                        "visual_on_anchor_y_q05_m": float(
                            components["anchor_y_q05_m"]
                        ),
                        "visual_on_anchor_y_q95_m": float(
                            components["anchor_y_q95_m"]
                        ),
                        "visual_on_anchor_u": float(
                            components["anchor_u"]
                        ),
                        "visual_on_anchor_v": float(
                            components["anchor_v"]
                        ),
                        "visual_on_spatial_x_margin": float(
                            components["spatial_x_margin"]
                        ),
                        "visual_on_spatial_y_margin": float(
                            components["spatial_y_margin"]
                        ),
                        "visual_on_spatial_z_margin": float(
                            components["spatial_z_margin"]
                        ),
                        "visual_on_spatial_directional_offset_m": float(
                            components["spatial_directional_offset_m"]
                        ),
                        "visual_on_spatial_lateral_offset_m": float(
                            components["spatial_lateral_offset_m"]
                        ),
                        "visual_on_spatial_anchor_direction_half_extent_m": float(
                            components[
                                "spatial_anchor_direction_half_extent_m"
                            ]
                        ),
                        "visual_on_spatial_anchor_lateral_half_extent_m": float(
                            components[
                                "spatial_anchor_lateral_half_extent_m"
                            ]
                        ),
                        "visual_on_spatial_displacement_m": float(
                            components["spatial_displacement_m"]
                        ),
                        "visual_on_spatial_frame_motion_m": float(
                            components["spatial_frame_motion_m"]
                        ),
                        "visual_on_spatial_motion_observed": bool(
                            components["spatial_motion_observed"]
                        ),
                        "visual_on_spatial_stable": bool(
                            components["spatial_stable"]
                        ),
                        "visual_on_spatial_pick_proxy_margin": float(
                            components["spatial_pick_proxy_margin"]
                        ),
                        "visual_on_spatial_footprint_box_valid": bool(
                            components["spatial_footprint_box_valid"]
                        ),
                        "visual_on_spatial_footprint_box_reason": str(
                            components["spatial_footprint_box_reason"]
                        ),
                        "visual_on_spatial_footprint_x_margin": float(
                            components["spatial_footprint_x_margin"]
                        ),
                        "visual_on_spatial_footprint_y_margin": float(
                            components["spatial_footprint_y_margin"]
                        ),
                        "visual_on_spatial_footprint_bottom_z_margin": float(
                            components[
                                "spatial_footprint_bottom_z_margin"
                            ]
                        ),
                        "visual_on_spatial_footprint_top_z_margin": float(
                            components[
                                "spatial_footprint_top_z_margin"
                            ]
                        ),
                        "visual_on_spatial_footprint_full_z_margin": float(
                            components[
                                "spatial_footprint_full_z_margin"
                            ]
                        ),
                        "visual_on_spatial_footprint_box_candidate_margin": float(
                            components[
                                "spatial_footprint_box_candidate_margin"
                            ]
                        ),
                        "visual_on_spatial_footprint_directional_valid": bool(
                            components[
                                "spatial_footprint_directional_valid"
                            ]
                        ),
                        "visual_on_spatial_footprint_directional_reason": str(
                            components[
                                "spatial_footprint_directional_reason"
                            ]
                        ),
                        "visual_on_spatial_footprint_directional_margin": float(
                            components[
                                "spatial_footprint_directional_margin"
                            ]
                        ),
                        "visual_on_spatial_footprint_lateral_margin": float(
                            components[
                                "spatial_footprint_lateral_margin"
                            ]
                        ),
                        "visual_on_spatial_footprint_surface_margin": float(
                            components[
                                "spatial_footprint_surface_margin"
                            ]
                        ),
                        "visual_on_spatial_footprint_directional_candidate_margin": float(
                            components[
                                "spatial_footprint_directional_candidate_margin"
                            ]
                        ),
                        "visual_on_object_x_q05_m": float(
                            components["object_x_q05_m"]
                        ),
                        "visual_on_object_x_q95_m": float(
                            components["object_x_q95_m"]
                        ),
                        "visual_on_object_y_q05_m": float(
                            components["object_y_q05_m"]
                        ),
                        "visual_on_object_y_q95_m": float(
                            components["object_y_q95_m"]
                        ),
                        "visual_on_support_baseline_x_q05_m": float(
                            components["support_baseline_x_q05_m"]
                        ),
                        "visual_on_support_baseline_x_q95_m": float(
                            components["support_baseline_x_q95_m"]
                        ),
                        "visual_on_support_baseline_y_q05_m": float(
                            components["support_baseline_y_q05_m"]
                        ),
                        "visual_on_support_baseline_y_q95_m": float(
                            components["support_baseline_y_q95_m"]
                        ),
                        "visual_on_robust_xy_distance_m": float(
                            components["robust_xy_distance_m"]
                        ),
                        "visual_on_robust_xy_margin": float(
                            components["robust_xy_margin"]
                        ),
                        "visual_on_robust_candidate_margin": float(
                            components["robust_candidate_margin"]
                        ),
                    }

            record = VisualSTLShadowRecord(
                step=int(step),
                task_id=int(task_id),
                target_object=str(target_object),
                target_class=target_class,
                selector=selector,
                valid=True,
                reason="",
                visual_fresh=True,
                visual_held=False,
                visual_hold_age_steps=0,
                confidence=float(
                    detection["confidence"]
                ),
                mask_pixels=int(
                    estimate.mask_pixels
                ),
                depth_pixels=int(
                    estimate.depth_pixels
                ),
                two_finger_grasp=bool(
                    two_finger_grasp
                ),
                oracle_distance_m=float(
                    oracle_distance_m
                ),
                visual_distance_m=float(
                    estimate.distance_m
                ),
                oracle_robustness=(
                    self.grasp_radius_m
                    - float(oracle_distance_m)
                ),
                visual_robustness=float(
                    estimate.robustness
                ),
                visual_surface_p01_distance_m=float(
                    estimate.surface_p01_distance_m
                ),
                visual_surface_p05_distance_m=float(
                    estimate.surface_p05_distance_m
                ),
                visual_surface_p10_distance_m=float(
                    estimate.surface_p10_distance_m
                ),
                visual_surface_p01_robustness=float(
                    estimate.surface_p01_robustness
                ),
                visual_surface_p05_robustness=float(
                    estimate.surface_p05_robustness
                ),
                visual_surface_p10_robustness=float(
                    estimate.surface_p10_robustness
                ),
                oracle_pick_margin=float(
                    oracle_pick_margin
                ),
                visual_pick_margin=float(
                    runtime_grasp_margin
                ),
                eef_x=float(eef[0]),
                eef_y=float(eef[1]),
                eef_z=float(eef[2]),
                oracle_x=float(oracle_center[0]),
                oracle_y=float(oracle_center[1]),
                oracle_z=float(oracle_center[2]),
                visual_x=float(visual_center[0]),
                visual_y=float(visual_center[1]),
                visual_z=float(visual_center[2]),
                gripper_width_m=gripper_width_m,
                visual_near_margin=float(
                    grasp_components["near_margin"]
                ),
                visual_close_margin=float(
                    grasp_components["close_margin"]
                ),
                visual_comove_error_m=float(
                    grasp_components["comove_error_m"]
                ),
                visual_comove_margin=float(
                    grasp_components["comove_margin"]
                ),
                eef_motion_m=float(
                    grasp_components["eef_motion_m"]
                ),
                visual_motion_margin=float(
                    grasp_components["motion_margin"]
                ),
                visual_displacement_m=float(
                    grasp_components["displacement_m"]
                ),
                visual_displacement_margin=float(
                    grasp_components["displacement_margin"]
                ),
                visual_grasp_margin=float(
                    runtime_grasp_margin
                ),
                visual_grasp_history_ready=bool(
                    grasp_components["history_ready"]
                ),
                visual_grasp_confirmed=bool(
                    grasp_components["confirmed"]
                ),
                visual_grasp_terminal_candidate=bool(
                    grasp_components["terminal_candidate"]
                ),
                visual_grasp_terminal_margin=float(
                    grasp_components["terminal_margin"]
                ),
                visual_grasp_terminal_count=int(
                    grasp_components["terminal_count"]
                ),
                visual_grasp_terminal_confirmed=bool(
                    grasp_components["terminal_confirmed"]
                ),
                visual_grasp_confirmation_source=str(
                    grasp_components["confirmation_source"]
                ),
                **on_fields,
            )
            self._record_visual_in(
                env_id=env_id,
                step=step,
                task_id=task_id,
                target_object=target_object,
                container_object=in_target_object,
                oracle_in_margin=oracle_in_margin,
                gripper_width_m=gripper_width_m,
                eef=eef,
                selected=selected,
                estimated=all_estimated,
                target_source="fresh_rgbd",
                target_selector=selector,
            )
            self.last_fresh_records[env_id] = record
            self.last_fresh_steps[env_id] = int(step)

        except Exception as error:
            reason = (
                f"{type(error).__name__}:"
                f"{str(error)[:160]}"
            )
            in_selected = locals().get("selected")
            in_reason = reason
            in_target_source = "unavailable"
            if (
                in_target_object is not None
                and (
                    "target_not_detected:" in reason
                    or "target_track_gate:" in reason
                )
            ):
                proxy = self._in_attached_proxy_selected(
                    env_id=env_id,
                    target_class=target_class,
                    eef=eef,
                    eef_rotation=eef_rotation,
                )
                if proxy is not None:
                    in_selected = proxy
                    in_reason = ""
                    in_target_source = "eef_attached_proxy"
            self._record_visual_in(
                env_id=env_id,
                step=step,
                task_id=task_id,
                target_object=target_object,
                container_object=in_target_object,
                oracle_in_margin=oracle_in_margin,
                gripper_width_m=gripper_width_m,
                eef=eef,
                selected=in_selected,
                estimated=locals().get("all_estimated", []),
                reason=in_reason,
                target_source=in_target_source,
                target_selector=(
                    "eef_attached_proxy"
                    if in_target_source == "eef_attached_proxy"
                    else str(locals().get("selector", "unavailable"))
                ),
            )
            can_hold = (
                "target_not_detected:" in reason
                or "target_track_gate:" in reason
            )
            record = (
                self._held_record(
                    env_id=env_id,
                    step=step,
                    task_id=task_id,
                    target_object=target_object,
                    target_class=target_class,
                    reason=reason,
                    two_finger_grasp=two_finger_grasp,
                    oracle_distance_m=oracle_distance_m,
                    oracle_pick_margin=oracle_pick_margin,
                    on_target_object=on_target_object,
                    oracle_on_margin=oracle_on_margin,
                    eef=eef,
                    oracle_center=oracle_center,
                    gripper_width_m=gripper_width_m,
                )
                if can_hold
                else None
            )
            if record is None:
                self._break_grasp_confirmation(env_id)
                record = self._invalid_record(
                env_id=env_id,
                step=step,
                task_id=task_id,
                target_object=target_object,
                target_class=target_class,
                reason=reason,
                two_finger_grasp=two_finger_grasp,
                oracle_distance_m=oracle_distance_m,
                oracle_pick_margin=oracle_pick_margin,
                on_target_object=on_target_object,
                oracle_on_margin=oracle_on_margin,
                eef=eef,
                oracle_center=oracle_center,
                gripper_width_m=gripper_width_m,
                )

            # The pick trace may be unavailable because the object is
            # occluded by the gripper. On monitoring remains possible after
            # a visual-to-EEF handoff, provided that the support is visible.
            support_class = self._object_on_target_class(
                on_target_object
            )
            state = self.on_object_states.get(env_id, {})
            held_center = (
                np.asarray(
                    [record.visual_x, record.visual_y, record.visual_z],
                    dtype=np.float64,
                )
                if record.visual_held
                else None
            )
            object_on_estimate, object_source, state = (
                self._on_object_estimate(
                    env_id=env_id,
                    step=step,
                    eef=eef,
                    eef_rotation=eef_rotation,
                    gripper_width_m=gripper_width_m,
                    center=held_center,
                    z_p05_m=state.get(
                        "last_z_p05_m", float("nan")
                    ),
                    z_p95_m=state.get(
                        "last_z_p95_m", float("nan")
                    ),
                    near_margin=(
                        record.visual_near_margin
                        if record.visual_held
                        else float("nan")
                    ),
                    fresh=False,
                )
            )
            support_candidates = [
                item
                for item in locals().get("estimated", [])
                if (
                    support_class is not None
                    and item["detection"]["class_name"]
                    == support_class
                )
            ]
            support_estimate = None
            support_confidence = float("nan")
            support_reason = (
                f"on_target_not_detected:{support_class}"
                if support_class is not None
                else "unsupported_semantic_region_target"
            )
            if object_on_estimate is not None and support_class is not None:
                object_center = np.asarray(
                    object_on_estimate.center_world,
                    dtype=np.float64,
                )
                (
                    support_estimate,
                    support_confidence,
                    support_reason,
                ) = self._resolve_on_support(
                    env_id=env_id,
                    step=step,
                    support_identity=str(on_target_object),
                    support_class=support_class,
                    support_candidates=support_candidates,
                    object_center=object_center,
                )
            if object_on_estimate is not None and support_estimate is not None:
                components = self._visual_on_components(
                    env_id=env_id,
                    step=step,
                    object_estimate=object_on_estimate,
                    object_source=object_source,
                    support_estimate=support_estimate,
                    support_class=support_class,
                    on_target_object=on_target_object,
                    gripper_width_m=gripper_width_m,
                )
                spatial_pick_proxy = float(
                    components["spatial_pick_proxy_margin"]
                )
                runtime_grasp_margin = float(record.visual_grasp_margin)
                if np.isfinite(spatial_pick_proxy):
                    runtime_grasp_margin = (
                        max(runtime_grasp_margin, spatial_pick_proxy)
                        if np.isfinite(runtime_grasp_margin)
                        else spatial_pick_proxy
                    )
                record = replace(
                    record,
                    valid=bool(
                        record.valid or np.isfinite(spatial_pick_proxy)
                    ),
                    reason=(
                        record.reason
                        if not np.isfinite(spatial_pick_proxy)
                        else "eef_attached_spatial_pick_proxy"
                    ),
                    visual_pick_margin=runtime_grasp_margin,
                    visual_grasp_margin=runtime_grasp_margin,
                    on_target_object=str(on_target_object or ""),
                    on_target_class=str(support_class or ""),
                    oracle_on_margin=float(oracle_on_margin),
                    visual_on_valid=bool(
                        components.get("valid", True)
                    ),
                    visual_on_reason=(
                        components.get("reason") or support_reason
                    ),
                    visual_on_target_confidence=support_confidence,
                    visual_on_xy_distance_m=float(
                        components["xy_distance_m"]
                    ),
                    visual_on_xy_margin=float(components["xy_margin"]),
                    visual_on_completion_xy_margin=float(
                        components["completion_xy_margin"]
                    ),
                    visual_on_height_delta_m=float(
                        components["height_delta_m"]
                    ),
                    visual_on_above_margin=float(
                        components["above_margin"]
                    ),
                    visual_on_surface_gap_m=float(
                        components["surface_gap_m"]
                    ),
                    visual_on_support_margin=float(
                        components["support_margin"]
                    ),
                    visual_on_release_margin=float(
                        components["release_margin"]
                    ),
                    visual_on_margin=float(components["margin"]),
                    visual_on_confirmed=bool(components["confirmed"]),
                    visual_on_object_source=object_source,
                    visual_on_attached=bool(
                        (state or {}).get("attached", False)
                    ),
                    visual_on_released=bool(
                        (state or {}).get("released", False)
                    ),
                    visual_on_geometry_mode=str(
                        components["geometry_mode"]
                    ),
                    visual_on_inside_xy_margin=float(
                        components["inside_xy_margin"]
                    ),
                    visual_on_below_rim_margin=float(
                        components["below_rim_margin"]
                    ),
                    visual_on_above_floor_margin=float(
                        components["above_floor_margin"]
                    ),
                    visual_on_handoff_used=bool(
                        components["handoff_used"]
                    ),
                    visual_on_handoff_age_steps=int(
                        components["handoff_age_steps"]
                    ),
                    visual_on_handoff_margin=float(
                        components["handoff_margin"]
                    ),
                    visual_on_contact_valid=bool(
                        components["contact_valid"]
                    ),
                    visual_on_contact_reason=str(
                        components["contact_reason"]
                    ),
                    visual_on_contact_p01_m=float(
                        components["contact_p01_m"]
                    ),
                    visual_on_contact_p05_m=float(
                        components["contact_p05_m"]
                    ),
                    visual_on_contact_p10_m=float(
                        components["contact_p10_m"]
                    ),
                    visual_on_contact_margin=float(
                        components["contact_margin"]
                    ),
                    visual_on_object_center_x_m=float(
                        components["object_center_x_m"]
                    ),
                    visual_on_object_center_y_m=float(
                        components["object_center_y_m"]
                    ),
                    visual_on_object_center_z_m=float(
                        components["object_center_z_m"]
                    ),
                    visual_on_support_center_x_m=float(
                        components["support_center_x_m"]
                    ),
                    visual_on_support_center_y_m=float(
                        components["support_center_y_m"]
                    ),
                    visual_on_support_center_z_m=float(
                        components["support_center_z_m"]
                    ),
                    visual_on_support_locked=bool(
                        components["support_locked"]
                    ),
                    visual_on_object_bottom_z_m=float(
                        components["object_bottom_z_m"]
                    ),
                    visual_on_support_z_p95_m=float(
                        components["support_z_p95_m"]
                    ),
                    visual_on_support_z_p99_m=float(
                        components["support_z_p99_m"]
                    ),
                    visual_on_support_z_p995_m=float(
                        components["support_z_p995_m"]
                    ),
                    visual_on_support_baseline_center_x_m=float(
                        components["support_baseline_center_x_m"]
                    ),
                    visual_on_support_baseline_center_y_m=float(
                        components["support_baseline_center_y_m"]
                    ),
                    visual_on_support_baseline_center_z_m=float(
                        components["support_baseline_center_z_m"]
                    ),
                    visual_on_support_baseline_z_p95_m=float(
                        components["support_baseline_z_p95_m"]
                    ),
                    visual_on_support_center_drift_m=float(
                        components["support_center_drift_m"]
                    ),
                    visual_on_support_z_p95_drift_m=float(
                        components["support_z_p95_drift_m"]
                    ),
                    visual_on_site_center_delta_q99_m=float(
                        components["site_center_delta_q99_m"]
                    ),
                    visual_on_site_bottom_gap_q99_m=float(
                        components["site_bottom_gap_q99_m"]
                    ),
                    visual_on_spatial_relation=str(
                        components["spatial_relation"]
                    ),
                    visual_on_relative_x_m=float(
                        components["relative_x_m"]
                    ),
                    visual_on_relative_y_m=float(
                        components["relative_y_m"]
                    ),
                    visual_on_relative_z_m=float(
                        components["relative_z_m"]
                    ),
                    visual_on_anchor_x_q05_m=float(
                        components["anchor_x_q05_m"]
                    ),
                    visual_on_anchor_x_q95_m=float(
                        components["anchor_x_q95_m"]
                    ),
                    visual_on_anchor_y_q05_m=float(
                        components["anchor_y_q05_m"]
                    ),
                    visual_on_anchor_y_q95_m=float(
                        components["anchor_y_q95_m"]
                    ),
                    visual_on_anchor_u=float(
                        components["anchor_u"]
                    ),
                    visual_on_anchor_v=float(
                        components["anchor_v"]
                    ),
                    visual_on_spatial_x_margin=float(
                        components["spatial_x_margin"]
                    ),
                    visual_on_spatial_y_margin=float(
                        components["spatial_y_margin"]
                    ),
                    visual_on_spatial_z_margin=float(
                        components["spatial_z_margin"]
                    ),
                    visual_on_spatial_directional_offset_m=float(
                        components["spatial_directional_offset_m"]
                    ),
                    visual_on_spatial_lateral_offset_m=float(
                        components["spatial_lateral_offset_m"]
                    ),
                    visual_on_spatial_anchor_direction_half_extent_m=float(
                        components[
                            "spatial_anchor_direction_half_extent_m"
                        ]
                    ),
                    visual_on_spatial_anchor_lateral_half_extent_m=float(
                        components[
                            "spatial_anchor_lateral_half_extent_m"
                        ]
                    ),
                    visual_on_spatial_displacement_m=float(
                        components["spatial_displacement_m"]
                    ),
                    visual_on_spatial_frame_motion_m=float(
                        components["spatial_frame_motion_m"]
                    ),
                    visual_on_spatial_motion_observed=bool(
                        components["spatial_motion_observed"]
                    ),
                    visual_on_spatial_stable=bool(
                        components["spatial_stable"]
                    ),
                    visual_on_spatial_pick_proxy_margin=float(
                        components["spatial_pick_proxy_margin"]
                    ),
                    visual_on_spatial_footprint_box_valid=bool(
                        components["spatial_footprint_box_valid"]
                    ),
                    visual_on_spatial_footprint_box_reason=str(
                        components["spatial_footprint_box_reason"]
                    ),
                    visual_on_spatial_footprint_x_margin=float(
                        components["spatial_footprint_x_margin"]
                    ),
                    visual_on_spatial_footprint_y_margin=float(
                        components["spatial_footprint_y_margin"]
                    ),
                    visual_on_spatial_footprint_bottom_z_margin=float(
                        components[
                            "spatial_footprint_bottom_z_margin"
                        ]
                    ),
                    visual_on_spatial_footprint_top_z_margin=float(
                        components["spatial_footprint_top_z_margin"]
                    ),
                    visual_on_spatial_footprint_full_z_margin=float(
                        components["spatial_footprint_full_z_margin"]
                    ),
                    visual_on_spatial_footprint_box_candidate_margin=float(
                        components[
                            "spatial_footprint_box_candidate_margin"
                        ]
                    ),
                    visual_on_spatial_footprint_directional_valid=bool(
                        components[
                            "spatial_footprint_directional_valid"
                        ]
                    ),
                    visual_on_spatial_footprint_directional_reason=str(
                        components[
                            "spatial_footprint_directional_reason"
                        ]
                    ),
                    visual_on_spatial_footprint_directional_margin=float(
                        components[
                            "spatial_footprint_directional_margin"
                        ]
                    ),
                    visual_on_spatial_footprint_lateral_margin=float(
                        components[
                            "spatial_footprint_lateral_margin"
                        ]
                    ),
                    visual_on_spatial_footprint_surface_margin=float(
                        components[
                            "spatial_footprint_surface_margin"
                        ]
                    ),
                    visual_on_spatial_footprint_directional_candidate_margin=float(
                        components[
                            "spatial_footprint_directional_candidate_margin"
                        ]
                    ),
                    visual_on_object_x_q05_m=float(
                        components["object_x_q05_m"]
                    ),
                    visual_on_object_x_q95_m=float(
                        components["object_x_q95_m"]
                    ),
                    visual_on_object_y_q05_m=float(
                        components["object_y_q05_m"]
                    ),
                    visual_on_object_y_q95_m=float(
                        components["object_y_q95_m"]
                    ),
                    visual_on_support_baseline_x_q05_m=float(
                        components["support_baseline_x_q05_m"]
                    ),
                    visual_on_support_baseline_x_q95_m=float(
                        components["support_baseline_x_q95_m"]
                    ),
                    visual_on_support_baseline_y_q05_m=float(
                        components["support_baseline_y_q05_m"]
                    ),
                    visual_on_support_baseline_y_q95_m=float(
                        components["support_baseline_y_q95_m"]
                    ),
                    visual_on_robust_xy_distance_m=float(
                        components["robust_xy_distance_m"]
                    ),
                    visual_on_robust_xy_margin=float(
                        components["robust_xy_margin"]
                    ),
                    visual_on_robust_candidate_margin=float(
                        components["robust_candidate_margin"]
                    ),
                )
            else:
                # Unknown visual samples freeze confirmation progress. A
                # valid negative sample still resets it in components().
                pass

        self.records.setdefault(env_id, []).append(record)
        return record

    def _dump_visual_in_episode(
        self,
        *,
        env_id: int,
        output_prefix,
        task_description: str,
        success: bool,
    ) -> dict | None:
        records = self.in_records.get(int(env_id), [])
        if not records:
            return None

        source_prefix = Path(output_prefix)
        name = source_prefix.name
        if name.startswith("visual_stl_"):
            name = "visual_in_" + name[len("visual_stl_"):]
        else:
            name = "visual_in_" + name
        prefix = source_prefix.with_name(name)
        fieldnames = list(VisualInShadowRecord.__dataclass_fields__)
        with prefix.with_suffix(".csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow(asdict(record))

        valid = [record for record in records if record.valid]
        compared = [
            record for record in valid
            if np.isfinite(record.oracle_in_margin)
            and np.isfinite(record.visual_in_margin)
        ]
        stats = {
            "task_description": task_description,
            "success": bool(success),
            "steps": len(records),
            "valid_steps": len(valid),
            "valid_rate": len(valid) / len(records),
            "visual_in_positive_steps": sum(
                record.valid and record.visual_in_margin >= 0.0
                for record in records
            ),
            "visual_in_confirmed": any(
                record.visual_in_confirmed for record in records
            ),
            "first_visual_in_confirmed_step": next(
                (
                    record.step for record in records
                    if record.visual_in_confirmed
                ),
                None,
            ),
            "first_oracle_in_positive_step": next(
                (
                    record.step for record in records
                    if np.isfinite(record.oracle_in_margin)
                    and record.oracle_in_margin >= 0.0
                ),
                None,
            ),
            "sign_agreement": (
                float(np.mean([
                    (record.oracle_in_margin >= 0.0)
                    == (record.visual_in_margin >= 0.0)
                    for record in compared
                ]))
                if compared else None
            ),
            "invalid_reasons": {
                reason: sum(
                    (not record.valid) and record.reason == reason
                    for record in records
                )
                for reason in sorted({
                    record.reason for record in records
                    if not record.valid
                })
            },
        }
        prefix.with_suffix(".json").write_text(
            json.dumps(stats, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        probe_frames = self.in_caddy_probe_frames.get(int(env_id), [])
        if probe_frames:
            frame_dir = prefix.with_name(
                prefix.name + "_caddy_cross_camera_frames"
            )
            frame_dir.mkdir(parents=True, exist_ok=True)
            from PIL import Image
            metadata = []
            for item in probe_frames:
                step = int(item["step"])
                target_camera = str(item["target_camera_name"]).replace(
                    "/", "_"
                )
                container_camera = str(
                    item["container_camera_name"]
                ).replace("/", "_")
                Image.fromarray(item["target_overlay"]).save(
                    frame_dir
                    / f"step_{step:04d}_target_{target_camera}.png"
                )
                Image.fromarray(item["container_overlay"]).save(
                    frame_dir
                    / f"step_{step:04d}_container_{container_camera}.png"
                )
                metadata.append({
                    key: value for key, value in item.items()
                    if key not in {"target_overlay", "container_overlay"}
                })
            (frame_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        steps = [record.step for record in records]
        figure, axis = plt.subplots(figsize=(10, 5))
        for field, label in (
            ("inside_xy_margin", "inside XY"),
            (
                "caddy_core_longitudinal_margin",
                "caddy core longitudinal",
            ),
            ("caddy_core_lateral_margin", "caddy core lateral"),
            (
                "caddy_full_longitudinal_margin",
                "caddy full longitudinal",
            ),
            ("caddy_full_lateral_margin", "caddy full lateral"),
            (
                "caddy_topology_footprint_margin_m",
                "caddy topology footprint",
            ),
            (
                "caddy_static_baseline_candidate_margin_m",
                "caddy static baseline candidate",
            ),
            ("below_rim_margin", "below rim"),
            ("above_floor_margin", "above floor"),
            ("entry_depth_margin", "past entry plane"),
            ("release_proximity_margin", "release near object"),
            ("release_margin", "release"),
            ("visual_in_margin", "combined visual In"),
            ("oracle_in_margin", "oracle In (diagnostic)"),
        ):
            axis.plot(
                steps,
                [getattr(record, field) for record in records],
                label=label,
                linewidth=2 if field == "visual_in_margin" else 1,
            )
        axis.axhline(0.0, color="black", linestyle="--")
        axis.set_xlabel("environment step")
        axis.set_ylabel("In margin")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8, ncol=2)
        figure.suptitle(
            f"{task_description}\nsuccess={success}, "
            f"valid={len(valid)}/{len(records)}"
        )
        figure.tight_layout()
        figure.savefig(prefix.with_suffix(".png"), dpi=160)
        plt.close(figure)
        return stats

    def _dump_visual_open_episode(
        self,
        *,
        env_id: int,
        output_prefix,
        task_description: str,
        success: bool,
    ) -> dict | None:
        records = self.open_records.get(int(env_id), [])
        if not records:
            return None

        source_prefix = Path(output_prefix)
        name = source_prefix.name
        if name.startswith("visual_stl_"):
            name = "visual_open_" + name[len("visual_stl_"):]
        else:
            name = "visual_open_" + name
        prefix = source_prefix.with_name(name)
        fieldnames = list(VisualOpenShadowRecord.__dataclass_fields__)
        with prefix.with_suffix(".csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow(asdict(record))

        valid = [
            record for record in records
            if record.valid and np.isfinite(record.visual_open_margin)
        ]
        compared = [
            record for record in valid
            if np.isfinite(record.oracle_open_margin)
        ]
        stats = {
            "task_description": task_description,
            "success": bool(success),
            "steps": len(records),
            "valid_steps": len(valid),
            "valid_rate": len(valid) / len(records),
            "visual_open_positive_steps": int(sum(
                record.visual_open_margin >= 0.0 for record in valid
            )),
            "visual_open_confirmed": bool(any(
                record.visual_open_confirmed for record in records
            )),
            "first_visual_open_confirmed_step": next(
                (
                    record.step for record in records
                    if record.visual_open_confirmed
                ),
                None,
            ),
            "first_oracle_open_positive_step": next(
                (
                    record.step for record in records
                    if np.isfinite(record.oracle_open_margin)
                    and record.oracle_open_margin >= 0.0
                ),
                None,
            ),
            "sign_agreement": (
                float(np.mean([
                    (record.oracle_open_margin >= 0.0)
                    == (record.visual_open_margin >= 0.0)
                    for record in compared
                ]))
                if compared else None
            ),
            "config": {
                "displacement_threshold_m": (
                    self.open_config.displacement_threshold_m
                ),
                "bound_quantile": self.open_config.bound_quantile,
                "extension_axis": self.open_config.extension_axis,
                "confirmation_steps": self.open_confirmation_steps,
                "hold_steps": self.open_hold_steps,
            },
            "extension_axes": {
                axis: int(sum(
                    record.extension_axis == axis for record in valid
                ))
                for axis in sorted({
                    record.extension_axis for record in valid
                })
            },
            "invalid_reasons": {
                reason: int(sum(
                    (not record.valid) and record.reason == reason
                    for record in records
                ))
                for reason in sorted({
                    record.reason for record in records
                    if not record.valid
                })
            },
        }
        prefix.with_suffix(".json").write_text(
            json.dumps(stats, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        steps = [record.step for record in records]
        figure, axes = plt.subplots(
            2, 1, figsize=(10, 7), sharex=True
        )
        axes[0].plot(
            steps,
            [record.displacement_m for record in records],
            label="visual drawer displacement",
            linewidth=2,
        )
        axes[0].axhline(
            self.open_config.displacement_threshold_m,
            color="black",
            linestyle="--",
            label="visual threshold",
        )
        axes[0].set_ylabel("displacement (m)")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)
        axes[1].plot(
            steps,
            [record.visual_open_margin for record in records],
            label="visual Open",
            linewidth=2,
        )
        axes[1].plot(
            steps,
            [record.oracle_open_margin for record in records],
            label="oracle Open (diagnostic)",
            linewidth=1.2,
        )
        axes[1].axhline(0.0, color="black", linestyle="--")
        axes[1].set_xlabel("environment step")
        axes[1].set_ylabel("Open margin")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)
        figure.suptitle(
            f"{task_description}\nsuccess={success}, "
            f"valid={len(valid)}/{len(records)}"
        )
        figure.tight_layout()
        figure.savefig(prefix.with_suffix(".png"), dpi=160)
        plt.close(figure)
        return stats

    def _dump_visual_close_episode(
        self,
        *,
        env_id: int,
        output_prefix,
        task_description: str,
        success: bool,
    ) -> dict | None:
        records = self.close_records.get(int(env_id), [])
        if not records:
            return None

        source_prefix = Path(output_prefix)
        name = source_prefix.name
        if name.startswith("visual_stl_"):
            name = "visual_close_" + name[len("visual_stl_"):]
        else:
            name = "visual_close_" + name
        prefix = source_prefix.with_name(name)
        fieldnames = list(VisualCloseShadowRecord.__dataclass_fields__)
        with prefix.with_suffix(".csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow(asdict(record))

        valid = [
            record for record in records
            if record.valid and np.isfinite(record.visual_close_margin)
        ]
        compared = [
            record for record in valid
            if np.isfinite(record.oracle_close_margin)
        ]
        stats = {
            "task_description": task_description,
            "success": bool(success),
            "steps": len(records),
            "valid_steps": len(valid),
            "valid_rate": len(valid) / len(records),
            "visual_close_positive_steps": int(sum(
                record.visual_close_margin >= 0.0 for record in valid
            )),
            "raw_visual_close_positive_steps": int(sum(
                record.raw_visual_close_margin >= 0.0 for record in valid
            )),
            "in_gate_blocked_positive_steps": int(sum(
                record.raw_visual_close_margin >= 0.0
                and record.visual_close_margin < 0.0
                and not record.visual_in_gate_open
                for record in valid
            )),
            "first_visual_in_gate_open_step": next(
                (
                    record.step for record in records
                    if record.visual_in_gate_open
                ),
                None,
            ),
            "visual_close_confirmed": bool(any(
                record.visual_close_confirmed for record in records
            )),
            "first_visual_close_confirmed_step": next(
                (
                    record.step for record in records
                    if record.visual_close_confirmed
                ),
                None,
            ),
            "microwave_settled_close_confirmed": bool(any(
                record.microwave_settled_close_confirmed
                for record in records
            )),
            "first_microwave_settled_close_step": next(
                (
                    record.step for record in records
                    if record.microwave_settled_close_confirmed
                ),
                None,
            ),
            "first_oracle_close_positive_step": next(
                (
                    record.step for record in records
                    if np.isfinite(record.oracle_close_margin)
                    and record.oracle_close_margin >= 0.0
                ),
                None,
            ),
            "sign_agreement": (
                float(np.mean([
                    (record.oracle_close_margin >= 0.0)
                    == (record.visual_close_margin >= 0.0)
                    for record in compared
                ]))
                if compared else None
            ),
            "config": {
                "retraction_threshold_m": (
                    self.close_config.retraction_threshold_m
                ),
                "bound_quantile": self.close_config.bound_quantile,
                "extension_axis": self.close_config.extension_axis,
                "front_quantile": self.close_config.front_quantile,
                "confirmation_steps": self.close_confirmation_steps,
                "microwave_settled_progress_threshold": (
                    self.close_microwave_settled_progress_threshold
                ),
                "microwave_settled_confirmation_steps": (
                    self.close_microwave_settled_confirmation_steps
                ),
                "hold_steps": self.close_hold_steps,
                "require_visual_in_confirmed": (
                    self.close_require_in_confirmed
                ),
            },
            "invalid_reasons": {
                reason: int(sum(
                    (not record.valid) and record.reason == reason
                    for record in records
                ))
                for reason in sorted({
                    record.reason for record in records
                    if not record.valid
                })
            },
        }
        prefix.with_suffix(".json").write_text(
            json.dumps(stats, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        steps = [record.step for record in records]
        figure, axes = plt.subplots(
            2, 1, figsize=(10, 7), sharex=True
        )
        axes[0].plot(
            steps,
            [record.retraction_m for record in records],
            label="visual drawer retraction",
            linewidth=2,
        )
        axes[0].axhline(
            self.close_config.retraction_threshold_m,
            color="black", linestyle="--", label="visual threshold",
        )
        axes[0].set_ylabel("retraction (m)")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)
        axes[1].plot(
            steps,
            [record.visual_close_margin for record in records],
            label="visual Close (after In gate)", linewidth=2,
        )
        axes[1].plot(
            steps,
            [record.raw_visual_close_margin for record in records],
            label="raw visual Close", linewidth=1, alpha=0.7,
        )
        axes[1].plot(
            steps,
            [record.oracle_close_margin for record in records],
            label="oracle Close (diagnostic)", linewidth=1.2,
        )
        axes[1].axhline(0.0, color="black", linestyle="--")
        axes[1].set_xlabel("environment step")
        axes[1].set_ylabel("Close margin")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)
        figure.suptitle(
            f"{task_description}\nsuccess={success}, "
            f"valid={len(valid)}/{len(records)}"
        )
        figure.tight_layout()
        figure.savefig(prefix.with_suffix(".png"), dpi=160)
        plt.close(figure)
        panel_frames = self.close_panel_frames.get(int(env_id), [])
        if panel_frames:
            panel_dir = prefix.with_name(prefix.name + "_panel_frames")
            panel_dir.mkdir(parents=True, exist_ok=True)
            metadata = []
            from PIL import Image
            for item in panel_frames:
                step = int(item["step"])
                Image.fromarray(item["overlay"]).save(
                    panel_dir / f"step_{step:04d}.png"
                )
                Image.fromarray(item["depth_overlay"]).save(
                    panel_dir / f"step_{step:04d}_depth.png"
                )
                metadata.append({
                    key: value for key, value in item.items()
                    if key not in {"overlay", "depth_overlay"}
                })
            (panel_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        return stats

    def _dump_visual_turnon_episode(
        self,
        *,
        env_id: int,
        output_prefix,
        task_description: str,
        success: bool,
    ) -> dict | None:
        records = self.turnon_records.get(int(env_id), [])
        if not records:
            return None
        source = Path(output_prefix)
        suffix = source.name[len("visual_stl_"):] if source.name.startswith(
            "visual_stl_"
        ) else source.name
        prefix = source.with_name("visual_turnon_" + suffix)
        fieldnames = list(VisualTurnOnShadowRecord.__dataclass_fields__)
        with prefix.with_suffix(".csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow(asdict(record))

        valid = [record for record in records if record.valid]
        missing = float("nan")
        max_rgb = max((record.rgb_mae for record in valid), default=missing)
        max_cell = max(
            (record.max_cell_change for record in valid), default=missing
        )
        config = self.turnon_config
        change_candidate = bool(
            np.isfinite(max_rgb)
            and np.isfinite(max_cell)
            and (max_rgb >= config.blocked_global_change
                 or max_cell >= config.blocked_cell_change)
        )
        positive = [
            record for record in valid
            if np.isfinite(record.visual_turnon_margin)
            and record.visual_turnon_margin >= 0.0
        ]
        first_visual = next(
            (
                record.step for record in valid
                if record.visual_turnon_confirmed
            ),
            None,
        )
        first_oracle = next(
            (
                record.step for record in records
                if np.isfinite(record.oracle_turnon_margin)
                and record.oracle_turnon_margin >= 0.0
            ),
            None,
        )
        comparable = [
            record for record in valid
            if np.isfinite(record.visual_turnon_margin)
            and np.isfinite(record.oracle_turnon_margin)
        ]
        sign_agreement = (
            sum(
                (record.visual_turnon_margin >= 0.0)
                == (record.oracle_turnon_margin >= 0.0)
                for record in comparable
            ) / len(comparable)
            if comparable else None
        )
        stats = {
            "task_description": task_description,
            "success": bool(success),
            "steps": len(records),
            "valid_steps": len(valid),
            "valid_rate": len(valid) / len(records),
            "max_reset_relative_rgb_mae": max_rgb,
            "max_reset_relative_cell_change": max_cell,
            "visual_change_candidate": change_candidate,
            "visual_turnon_positive_steps": len(positive),
            "visual_turnon_confirmed": bool(first_visual is not None),
            "first_visual_turnon_confirmed_step": first_visual,
            "first_oracle_turnon_positive_step": first_oracle,
            "confirmation_delay": (
                first_visual - first_oracle
                if first_visual is not None and first_oracle is not None
                else None
            ),
            "sign_agreement": sign_agreement,
            "blocked_reason": (
                "" if change_candidate
                else "no_reset_relative_visual_change_above_probe_floor"
            ),
            "oracle_label_location": "CSV only",
            "reward_enabled": False,
            "estimator_status": "shadow_visual_margin",
            "config": {
                "red_min_intensity": config.red_min_intensity,
                "red_excess_threshold": config.red_excess_threshold,
                "red_fraction_threshold": config.red_fraction_threshold,
                "confirmation_steps": config.confirmation_steps,
            },
        }
        prefix.with_suffix(".json").write_text(
            json.dumps(stats, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        frames = self.turnon_frames.get(int(env_id), [])
        if frames:
            frame_dir = prefix.with_name(prefix.name + "_frames")
            frame_dir.mkdir(parents=True, exist_ok=True)
            from PIL import Image
            metadata = []
            for item in frames:
                Image.fromarray(item["overlay"]).save(
                    frame_dir / f"step_{item['step']:04d}.png"
                )
                metadata.append({
                    key: value for key, value in item.items()
                    if key != "overlay"
                })
            (frame_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        return stats

    def dump_episode(
        self,
        *,
        env_id: int,
        output_prefix,
        task_description: str,
        success: bool,
    ) -> dict:
        records = self.records.get(int(env_id), [])
        prefix = Path(output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)

        in_stats = self._dump_visual_in_episode(
            env_id=env_id,
            output_prefix=output_prefix,
            task_description=task_description,
            success=success,
        )

        open_stats = self._dump_visual_open_episode(
            env_id=env_id,
            output_prefix=output_prefix,
            task_description=task_description,
            success=success,
        )

        close_stats = self._dump_visual_close_episode(
            env_id=env_id,
            output_prefix=output_prefix,
            task_description=task_description,
            success=success,
        )

        turnon_stats = self._dump_visual_turnon_episode(
            env_id=env_id,
            output_prefix=output_prefix,
            task_description=task_description,
            success=success,
        )

        if not records:
            return {
                "valid_steps": 0,
                "visual_in": in_stats,
                "visual_open": open_stats,
                "visual_close": close_stats,
                "visual_turnon": turnon_stats,
            }

        fieldnames = list(
            VisualSTLShadowRecord.__dataclass_fields__
        )

        with prefix.with_suffix(".csv").open(
            "w",
            newline="",
            encoding="utf-8",
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=fieldnames,
            )
            writer.writeheader()
            for record in records:
                writer.writerow(asdict(record))

        valid = [record for record in records if record.valid]
        oracle_valid = [
            record
            for record in valid
            if (
                np.isfinite(record.oracle_distance_m)
                and np.isfinite(record.oracle_robustness)
            )
        ]
        error = np.asarray(
            [
                record.visual_distance_m
                - record.oracle_distance_m
                for record in oracle_valid
            ],
            dtype=np.float64,
        )

        stats = {
            "task_description": task_description,
            "success": bool(success),
            "steps": len(records),
            "valid_steps": len(valid),
            "valid_rate": (
                len(valid) / len(records)
                if records else 0.0
            ),
            "oracle_comparison_steps": len(oracle_valid),
            "distance_mae_m": (
                float(np.mean(np.abs(error)))
                if len(error) else None
            ),
            "distance_rmse_m": (
                float(np.sqrt(np.mean(error ** 2)))
                if len(error) else None
            ),
            "distance_bias_m": (
                float(np.mean(error))
                if len(error) else None
            ),
            "robustness_sign_agreement": (
                float(np.mean([
                    (
                        record.oracle_robustness >= 0
                    )
                    == (
                        record.visual_robustness >= 0
                    )
                    for record in oracle_valid
                ]))
                if oracle_valid else None
            ),
            "visual_in": in_stats,
            "visual_open": open_stats,
            "visual_close": close_stats,
        }

        if len(oracle_valid) >= 2:
            oracle = np.asarray([
                record.oracle_distance_m
                for record in oracle_valid
            ])
            visual = np.asarray([
                record.visual_distance_m
                for record in oracle_valid
            ])
            if (
                np.std(oracle) > 1.0e-12
                and np.std(visual) > 1.0e-12
            ):
                stats["distance_pearson"] = float(
                    np.corrcoef(oracle, visual)[0, 1]
                )
            else:
                stats["distance_pearson"] = None
        else:
            stats["distance_pearson"] = None

        first_grasp_step = next(
            (
                record.step
                for record in records
                if record.two_finger_grasp
            ),
            None,
        )
        pregrasp_valid = [
            record
            for record in valid
            if (
                first_grasp_step is None
                or record.step < first_grasp_step
            )
        ]

        def method_stats(
            method_records,
            distance_field,
            robustness_field,
        ):
            method_records = [
                record
                for record in method_records
                if (
                    np.isfinite(record.oracle_distance_m)
                    and np.isfinite(
                        getattr(record, distance_field)
                    )
                    and np.isfinite(
                        getattr(record, robustness_field)
                    )
                )
            ]

            if not method_records:
                return {
                    "steps": 0,
                    "mae_m": None,
                    "rmse_m": None,
                    "bias_m": None,
                    "pearson": None,
                    "sign_agreement": None,
                    "oracle_crossing_step": None,
                    "method_crossing_step": None,
                    "crossing_lag_steps": None,
                }

            oracle_values = np.asarray([
                record.oracle_distance_m
                for record in method_records
            ])
            method_values = np.asarray([
                getattr(record, distance_field)
                for record in method_records
            ])
            method_error = (
                method_values - oracle_values
            )

            pearson = None
            if (
                len(method_records) >= 2
                and np.std(oracle_values) > 1.0e-12
                and np.std(method_values) > 1.0e-12
            ):
                pearson = float(
                    np.corrcoef(
                        oracle_values,
                        method_values,
                    )[0, 1]
                )

            oracle_crossing = next(
                (
                    record.step
                    for record in method_records
                    if record.oracle_robustness >= 0.0
                ),
                None,
            )
            method_crossing = next(
                (
                    record.step
                    for record in method_records
                    if getattr(
                        record,
                        robustness_field,
                    ) >= 0.0
                ),
                None,
            )

            return {
                "steps": len(method_records),
                "mae_m": float(
                    np.mean(np.abs(method_error))
                ),
                "rmse_m": float(
                    np.sqrt(
                        np.mean(method_error ** 2)
                    )
                ),
                "bias_m": float(
                    np.mean(method_error)
                ),
                "pearson": pearson,
                "sign_agreement": float(np.mean([
                    (
                        record.oracle_robustness >= 0.0
                    )
                    == (
                        getattr(
                            record,
                            robustness_field,
                        ) >= 0.0
                    )
                    for record in method_records
                ])),
                "oracle_crossing_step": oracle_crossing,
                "method_crossing_step": method_crossing,
                "crossing_lag_steps": (
                    int(
                        method_crossing
                        - oracle_crossing
                    )
                    if (
                        oracle_crossing is not None
                        and method_crossing is not None
                    )
                    else None
                ),
            }

        methods = {
            "center": (
                "visual_distance_m",
                "visual_robustness",
            ),
            "surface_p01": (
                "visual_surface_p01_distance_m",
                "visual_surface_p01_robustness",
            ),
            "surface_p05": (
                "visual_surface_p05_distance_m",
                "visual_surface_p05_robustness",
            ),
            "surface_p10": (
                "visual_surface_p10_distance_m",
                "visual_surface_p10_robustness",
            ),
        }

        stats["first_grasp_step"] = first_grasp_step
        stats["oracle_radius_m"] = (
            self.grasp_radius_m
        )
        stats["center_radius_m"] = (
            self.visual_center_radius_m
        )
        stats["surface_radius_m"] = (
            self.surface_radius_m
        )
        stats["visual_grasp_config"] = {
            "near_mode": self.grasp_near_mode,
            "close_threshold_m": self.close_threshold_m,
            "comove_tolerance_m": self.comove_tolerance_m,
            "motion_min_m": self.motion_min_m,
            "displacement_min_m": self.displacement_min_m,
            "hold_steps": self.hold_steps,
            "history_lag_steps": self.grasp_history_lag_steps,
            "confirmation_steps": self.grasp_confirmation_steps,
            "terminal_width_m": self.grasp_terminal_width_m,
            "terminal_comove_slack_m": (
                self.grasp_terminal_comove_slack_m
            ),
            "terminal_motion_slack_m": (
                self.grasp_terminal_motion_slack_m
            ),
            "terminal_confirmation_steps": (
                self.grasp_terminal_confirmation_steps
            ),
        }
        stats["visual_grasp_history_ready_steps"] = int(sum(
            record.visual_grasp_history_ready
            for record in records
        ))
        stats["visual_fresh_steps"] = int(sum(
            record.visual_fresh for record in records
        ))
        stats["visual_held_steps"] = int(sum(
            record.visual_held for record in records
        ))
        stats["visual_grasp_positive_steps"] = int(sum(
            (
                np.isfinite(record.visual_grasp_margin)
                and record.visual_grasp_margin >= 0.0
            )
            for record in records
        ))
        stats["visual_grasp_confirmed"] = bool(any(
            record.visual_grasp_confirmed
            for record in records
        ))
        stats["first_visual_grasp_confirmed_step"] = next(
            (
                record.step
                for record in records
                if record.visual_grasp_confirmed
            ),
            None,
        )
        stats["visual_grasp_confirmation_source"] = next(
            (
                record.visual_grasp_confirmation_source
                for record in records
                if record.visual_grasp_confirmed
            ),
            "none",
        )
        on_valid = [
            record
            for record in records
            if (
                record.visual_on_valid
                and np.isfinite(record.visual_on_margin)
            )
        ]
        on_oracle_valid = [
            record
            for record in on_valid
            if np.isfinite(record.oracle_on_margin)
        ]
        stats["visual_on_config"] = {
            "xy_tolerance_m": self.on_xy_tolerance_m,
            "above_epsilon_m": self.on_above_epsilon_m,
            "surface_tolerance_m": self.on_surface_tolerance_m,
            "ordinary_xy_tolerance_m": (
                self.on_ordinary_xy_tolerance_m
            ),
            "ordinary_xy_uncertainty_m": (
                self.on_ordinary_xy_uncertainty_m
            ),
            "ordinary_completion_xy_uncertainty_m": (
                self.on_ordinary_completion_xy_uncertainty_m
            ),
            "ordinary_effective_xy_tolerance_m": (
                self.on_ordinary_xy_tolerance_m
                + self.on_ordinary_xy_uncertainty_m
            ),
            "ordinary_completion_effective_xy_tolerance_m": (
                self.on_ordinary_xy_tolerance_m
                + self.on_ordinary_completion_xy_uncertainty_m
            ),
            "ordinary_min_above_m": self.on_ordinary_min_above_m,
            "ordinary_surface_tolerance_m": (
                self.on_ordinary_surface_tolerance_m
            ),
            "ordinary_require_release": (
                self.on_ordinary_require_release
            ),
            "ordinary_require_contact": (
                self.on_ordinary_require_contact
            ),
            "ordinary_contact_tolerance_m": (
                self.on_ordinary_contact_tolerance_m
            ),
            "ordinary_confirmation_steps": (
                self.on_ordinary_confirmation_steps
            ),
            "ordinary_near_center_fallback_enabled": (
                self.on_ordinary_near_center_fallback_enabled
            ),
            "ordinary_near_center_fallback_xy_m": (
                self.on_ordinary_near_center_fallback_xy_m
            ),
            "ordinary_near_center_fallback_contact_p01_m": (
                self.on_ordinary_near_center_fallback_contact_p01_m
            ),
            "ordinary_near_center_fallback_confirmation_steps": (
                self.on_ordinary_near_center_fallback_confirmation_steps
            ),
            "site_lower_delta_m": self.on_site_lower_delta_m,
            "site_upper_delta_m": self.on_site_upper_delta_m,
            "confirmation_steps": self.on_confirmation_steps,
            "front_bounds_m": self.on_front_bounds_m,
            "front_confirmation_steps": (
                self.on_front_confirmation_steps
            ),
            "spatial_min_displacement_m": (
                self.on_spatial_min_displacement_m
            ),
            "spatial_stability_tolerance_m": (
                self.on_spatial_stability_tolerance_m
            ),
            "right_min_displacement_m": (
                self.on_right_min_displacement_m
            ),
            "right_released_min_displacement_m": (
                self.on_right_released_min_displacement_m
            ),
            "right_interior_margin_m": self.on_right_interior_margin_m,
            "right_recovery_slack_m": self.on_right_recovery_slack_m,
            "right_recovery_min_displacement_m": (
                self.on_right_recovery_min_displacement_m
            ),
            "right_released_slack_m": self.on_right_released_slack_m,
            "right_require_release": self.on_right_require_release,
            "right_pick_proxy": self.on_right_pick_proxy,
            "right_confirmation_steps": (
                self.on_right_confirmation_steps
            ),
            "attach_width_m": self.on_attach_width_m,
            "attach_confirmation_steps": (
                self.on_attach_confirmation_steps
            ),
            "release_width_m": self.on_release_width_m,
            "release_hold_steps": self.on_release_hold_steps,
            "contact_tolerance_m": self.on_contact_tolerance_m,
        }
        stats["visual_on_valid_steps"] = len(on_valid)
        stats["visual_on_valid_rate"] = (
            len(on_valid) / len(records) if records else 0.0
        )
        stats["visual_on_positive_steps"] = int(sum(
            record.visual_on_margin >= 0.0
            for record in on_valid
        ))
        stats["visual_on_confirmed"] = bool(any(
            record.visual_on_confirmed for record in records
        ))
        stats["visual_on_confirmation_source"] = str(
            self.on_confirmation_sources.get(int(env_id), "none")
        )
        stats["first_visual_on_confirmed_step"] = next(
            (
                record.step
                for record in records
                if record.visual_on_confirmed
            ),
            None,
        )
        stats["first_oracle_on_positive_step"] = next(
            (
                record.step
                for record in records
                if (
                    np.isfinite(record.oracle_on_margin)
                    and record.oracle_on_margin >= 0.0
                )
            ),
            None,
        )
        stats["visual_on_sign_agreement"] = (
            float(np.mean([
                (record.oracle_on_margin >= 0.0)
                == (record.visual_on_margin >= 0.0)
                for record in on_oracle_valid
            ]))
            if on_oracle_valid else None
        )
        stats["visual_on_source_steps"] = {
            source: int(sum(
                record.visual_on_object_source == source
                for record in records
            ))
            for source in (
                "fresh_rgbd",
                "eef_attach",
                "eef_attached",
                "eef_release",
                "eef_release_hold",
            )
        }
        stats["first_visual_on_attached_step"] = next(
            (
                record.step
                for record in records
                if record.visual_on_attached
            ),
            None,
        )
        stats["first_visual_on_released_step"] = next(
            (
                record.step
                for record in records
                if record.visual_on_released
            ),
            None,
        )
        stats["comparison"] = {}

        for method_name, (
            distance_field,
            robustness_field,
        ) in methods.items():
            stats["comparison"][method_name] = {
                "full_episode": method_stats(
                    valid,
                    distance_field,
                    robustness_field,
                ),
                "pregrasp": method_stats(
                    pregrasp_valid,
                    distance_field,
                    robustness_field,
                ),
            }

        prefix.with_suffix(".json").write_text(
            json.dumps(
                stats,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = np.asarray([
            record.step for record in records
        ])
        oracle_distance = np.asarray([
            record.oracle_distance_m
            for record in records
        ])
        visual_distance = np.asarray([
            record.visual_distance_m
            for record in records
        ])
        oracle_rho = np.asarray([
            record.oracle_robustness
            for record in records
        ])
        visual_rho = np.asarray([
            record.visual_robustness
            for record in records
        ])
        surface_p01_distance = np.asarray([
            record.visual_surface_p01_distance_m
            for record in records
        ])
        surface_p05_distance = np.asarray([
            record.visual_surface_p05_distance_m
            for record in records
        ])
        surface_p10_distance = np.asarray([
            record.visual_surface_p10_distance_m
            for record in records
        ])
        surface_p01_rho = np.asarray([
            record.visual_surface_p01_robustness
            for record in records
        ])
        surface_p05_rho = np.asarray([
            record.visual_surface_p05_robustness
            for record in records
        ])
        surface_p10_rho = np.asarray([
            record.visual_surface_p10_robustness
            for record in records
        ])
        near_margin = np.asarray([
            record.visual_near_margin
            for record in records
        ])
        close_margin = np.asarray([
            record.visual_close_margin
            for record in records
        ])
        comove_margin = np.asarray([
            record.visual_comove_margin
            for record in records
        ])
        motion_margin = np.asarray([
            record.visual_motion_margin
            for record in records
        ])
        displacement_margin = np.asarray([
            record.visual_displacement_margin
            for record in records
        ])
        visual_grasp_margin = np.asarray([
            record.visual_grasp_margin
            for record in records
        ])
        oracle_on_margin = np.asarray([
            record.oracle_on_margin for record in records
        ])
        visual_on_xy_margin = np.asarray([
            record.visual_on_xy_margin for record in records
        ])
        visual_on_above_margin = np.asarray([
            record.visual_on_above_margin for record in records
        ])
        visual_on_support_margin = np.asarray([
            record.visual_on_support_margin for record in records
        ])
        visual_on_release_margin = np.asarray([
            record.visual_on_release_margin for record in records
        ])
        visual_on_margin = np.asarray([
            record.visual_on_margin for record in records
        ])
        held_mask = np.asarray([
            record.visual_held for record in records
        ], dtype=bool)

        figure, axes = plt.subplots(
            4,
            1,
            figsize=(10, 13),
            sharex=True,
        )

        axes[0].plot(
            steps,
            oracle_distance,
            label="oracle distance",
            linewidth=2,
        )
        axes[0].plot(
            steps,
            visual_distance,
            label="visual center",
            linewidth=1.5,
        )
        axes[0].plot(
            steps,
            surface_p01_distance,
            label="surface p01",
            linewidth=1.0,
        )
        axes[0].plot(
            steps,
            surface_p05_distance,
            label="surface p05",
            linewidth=1.2,
        )
        axes[0].plot(
            steps,
            surface_p10_distance,
            label="surface p10",
            linewidth=1.0,
        )
        axes[0].axhline(
            self.visual_center_radius_m,
            color="black",
            linestyle="--",
            label="visual center radius",
        )
        axes[0].axhline(
            self.surface_radius_m,
            color="gray",
            linestyle=":",
            label="surface radius",
        )
        axes[0].set_ylabel("distance (m)")
        axes[0].legend(
            fontsize=8,
            ncol=2,
        )
        axes[0].grid(alpha=0.3)

        axes[1].plot(
            steps,
            oracle_rho,
            label="oracle robustness",
            linewidth=2,
        )
        axes[1].plot(
            steps,
            visual_rho,
            label="center robustness",
            linewidth=1.5,
        )
        axes[1].plot(
            steps,
            surface_p01_rho,
            label="surface p01 robustness",
            linewidth=1.0,
        )
        axes[1].plot(
            steps,
            surface_p05_rho,
            label="surface p05 robustness",
            linewidth=1.2,
        )
        axes[1].plot(
            steps,
            surface_p10_rho,
            label="surface p10 robustness",
            linewidth=1.0,
        )
        axes[1].axhline(
            0.0,
            color="black",
            linestyle="--",
        )
        axes[1].set_xlabel("environment step")
        axes[1].set_ylabel("robustness")
        axes[1].legend(
            fontsize=8,
            ncol=2,
        )
        axes[1].grid(alpha=0.3)

        axes[2].plot(
            steps,
            near_margin,
            label=f"near ({self.grasp_near_mode})",
            linewidth=1.2,
        )
        axes[2].plot(
            steps,
            close_margin,
            label="close",
            linewidth=1.2,
        )
        axes[2].plot(
            steps,
            comove_margin,
            label="co-move",
            linewidth=1.2,
        )
        axes[2].plot(
            steps,
            motion_margin,
            label="motion gate",
            linewidth=1.2,
        )
        axes[2].plot(
            steps,
            displacement_margin,
            label="object displacement",
            linewidth=1.2,
        )
        axes[2].plot(
            steps,
            visual_grasp_margin,
            label="combined grasp",
            color="black",
            linewidth=2,
        )
        axes[2].scatter(
            steps[held_mask],
            visual_grasp_margin[held_mask],
            label="held estimate",
            color="orange",
            marker=".",
            s=14,
            zorder=3,
        )
        axes[2].axhline(
            0.0,
            color="black",
            linestyle="--",
        )
        axes[2].set_xlabel("environment step")
        axes[2].set_ylabel("contact-free margin")
        axes[2].legend(fontsize=8, ncol=3)
        axes[2].grid(alpha=0.3)

        axes[3].plot(
            steps,
            oracle_on_margin,
            label="oracle On (diagnostic)",
            linewidth=2,
        )
        axes[3].plot(
            steps,
            visual_on_xy_margin,
            label="visual On: XY",
            linewidth=1.1,
        )
        axes[3].plot(
            steps,
            visual_on_above_margin,
            label="visual On: above",
            linewidth=1.1,
        )
        axes[3].plot(
            steps,
            visual_on_support_margin,
            label="visual On: surface gap",
            linewidth=1.1,
        )
        axes[3].plot(
            steps,
            visual_on_release_margin,
            label="visual On: release",
            linewidth=1.1,
        )
        axes[3].plot(
            steps,
            visual_on_margin,
            label="combined visual On",
            color="black",
            linewidth=2,
        )
        axes[3].axhline(0.0, color="black", linestyle="--")
        axes[3].set_xlabel("environment step")
        axes[3].set_ylabel("On margin")
        axes[3].legend(fontsize=8, ncol=2)
        axes[3].grid(alpha=0.3)

        figure.suptitle(
            f"{task_description}\n"
            f"success={success}, "
            f"valid={len(valid)}/{len(records)}"
        )
        figure.tight_layout()
        figure.savefig(
            prefix.with_suffix(".png"),
            dpi=160,
        )
        plt.close(figure)

        return stats
