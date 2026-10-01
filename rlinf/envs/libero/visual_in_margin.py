"""Vision-only shadow margins for ``In(object, container)``.

Only RGB-D point clouds reconstructed from segmentation masks and the
observable gripper width are accepted.  Simulator sites, object poses,
contacts, joints, and oracle margins are intentionally absent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VisualInConfig:
    min_object_points: int = 20
    min_basket_points: int = 40
    basket_inner_shrink_m: float = 0.015
    below_rim_tolerance_m: float = 0.025
    floor_tolerance_m: float = 0.010
    release_width_threshold_m: float = 0.030
    min_drawer_points: int = 40
    drawer_inner_shrink_m: float = 0.010
    microwave_inner_shrink_m: float = 0.012
    microwave_entry_tolerance_m: float = 0.005
    microwave_front_quantile: float = 0.90
    microwave_min_entry_depth_m: float = 0.170
    microwave_release_max_target_eef_distance_m: float = 0.100
    microwave_occlusion_steps: int = 8
    microwave_release_max_age_steps: int = 30
    microwave_require_release: bool = True
    caddy_inner_shrink_m: float = 0.008
    caddy_back_start_fraction: float = 0.10
    # The desk-caddy's central back compartment occupies roughly 30% of
    # the observed lateral half-span.  This topology prior is diagnostic
    # only and must not affect the canonical caddy containment margin.
    caddy_central_lateral_half_fraction: float = 0.30
    caddy_topology_histogram_bins: int = 32
    caddy_topology_min_bin_points: int = 6
    caddy_topology_outer_exclusion_fraction: float = 0.15
    caddy_topology_search_radius_bins: int = 2
    # Robust object-footprint quantiles for diagnostic containment against
    # the visually inferred caddy divider bounds.  These diagnostics do not
    # alter the canonical center-based caddy margin.
    caddy_footprint_lower_quantile: float = 0.10
    caddy_footprint_upper_quantile: float = 0.90
    caddy_use_object_footprint: bool = False
    # The privileged LIBERO ``In`` predicate tests the object reference
    # point against a container site.  Keep the historical whole-footprint
    # mode as the default, but allow the full-comparison shadow to use the
    # corresponding robust visual-center semantics.
    use_object_footprint: bool = True
    # Release is useful placement evidence, but it is not part of the
    # canonical geometric ``In`` atom.  Preserve the stricter legacy mode by
    # default and make this configurable for audited comparison runs.
    require_release: bool = True


@dataclass(frozen=True)
class VisualInEstimate:
    valid: bool
    reason: str
    margin: float = float("nan")
    inside_xy_margin: float = float("nan")
    below_rim_margin: float = float("nan")
    above_floor_margin: float = float("nan")
    entry_depth_margin: float = float("nan")
    release_margin: float = float("nan")
    caddy_center_longitudinal_margin: float = float("nan")
    caddy_center_lateral_margin: float = float("nan")
    caddy_center_back_offset_m: float = float("nan")
    caddy_center_lateral_offset_m: float = float("nan")
    caddy_back_half_extent_m: float = float("nan")
    caddy_lateral_half_extent_m: float = float("nan")
    caddy_central_lateral_margin_m: float = float("nan")
    caddy_central_back_margin_m: float = float("nan")
    caddy_topology_valid: bool = False
    caddy_topology_reason: str = "not_caddy"
    caddy_topology_confidence: float = float("nan")
    caddy_topology_lateral_low_m: float = float("nan")
    caddy_topology_lateral_high_m: float = float("nan")
    caddy_topology_lateral_margin_m: float = float("nan")
    caddy_topology_back_margin_m: float = float("nan")
    caddy_topology_footprint_valid: bool = False
    caddy_topology_footprint_longitudinal_margin_m: float = float("nan")
    caddy_topology_footprint_lateral_margin_m: float = float("nan")
    caddy_topology_footprint_below_rim_margin_m: float = float("nan")
    caddy_topology_footprint_above_floor_margin_m: float = float("nan")
    caddy_topology_footprint_margin_m: float = float("nan")
    caddy_core_longitudinal_margin: float = float("nan")
    caddy_core_lateral_margin: float = float("nan")
    caddy_full_longitudinal_margin: float = float("nan")
    caddy_full_lateral_margin: float = float("nan")


@dataclass(frozen=True)
class VisualContainerOnEstimate:
    """Visual surrogate for canonical ``On(object, bowl)`` geometry."""

    valid: bool
    reason: str
    margin: float = float("nan")
    center_z_margin: float = float("nan")
    center_xy_distance: float = float("nan")
    center_xy_margin: float = float("nan")
    inside_xy_margin: float = float("nan")
    below_rim_margin: float = float("nan")
    above_floor_margin: float = float("nan")


@dataclass(frozen=True)
class VisualPointContactDiagnostics:
    """Nearest RGB-D surface-distance diagnostics for visual ``On``."""

    valid: bool
    reason: str
    p01_m: float = float("nan")
    p05_m: float = float("nan")
    p10_m: float = float("nan")


@dataclass(frozen=True)
class VisualCaddyTopologyEstimate:
    """Diagnostic central-compartment bounds from visible divider walls."""

    valid: bool
    reason: str
    confidence: float = float("nan")
    lateral_low_m: float = float("nan")
    lateral_high_m: float = float("nan")


def _robust_caddy_footprint_containment_margins(
    *,
    object_back_coordinates: np.ndarray,
    object_lateral_coordinates: np.ndarray,
    object_z_coordinates: np.ndarray,
    longitudinal_low_m: float,
    longitudinal_high_m: float,
    lateral_low_m: float,
    lateral_high_m: float,
    floor_m: float,
    rim_m: float,
    floor_tolerance_m: float,
    rim_tolerance_m: float,
    lower_quantile: float,
    upper_quantile: float,
) -> tuple[float, float, float, float, float]:
    """Measure robust object-footprint containment in caddy-local axes."""
    back = np.asarray(object_back_coordinates, dtype=np.float64).reshape(-1)
    lateral = np.asarray(
        object_lateral_coordinates, dtype=np.float64
    ).reshape(-1)
    height = np.asarray(object_z_coordinates, dtype=np.float64).reshape(-1)
    finite = np.isfinite(back) & np.isfinite(lateral) & np.isfinite(height)
    back = back[finite]
    lateral = lateral[finite]
    height = height[finite]
    if len(back) == 0:
        raise ValueError("empty_caddy_object_footprint")

    lower = float(lower_quantile)
    upper = float(upper_quantile)
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("invalid_caddy_footprint_quantiles")
    bounds = np.asarray([
        longitudinal_low_m,
        longitudinal_high_m,
        lateral_low_m,
        lateral_high_m,
        floor_m,
        rim_m,
        floor_tolerance_m,
        rim_tolerance_m,
    ], dtype=np.float64)
    if not np.all(np.isfinite(bounds)):
        raise ValueError("nonfinite_caddy_footprint_bounds")
    if longitudinal_high_m <= longitudinal_low_m:
        raise ValueError("invalid_caddy_footprint_longitudinal_bounds")
    if lateral_high_m <= lateral_low_m:
        raise ValueError("invalid_caddy_footprint_lateral_bounds")
    if rim_m <= floor_m:
        raise ValueError("invalid_caddy_footprint_vertical_bounds")

    back_low, back_high = np.quantile(back, (lower, upper))
    lateral_low, lateral_high = np.quantile(lateral, (lower, upper))
    z_low, z_high = np.quantile(height, (lower, upper))
    longitudinal_margin = float(min(
        back_low - longitudinal_low_m,
        longitudinal_high_m - back_high,
    ))
    lateral_margin = float(min(
        lateral_low - lateral_low_m,
        lateral_high_m - lateral_high,
    ))
    below_rim_margin = float(rim_m + rim_tolerance_m - z_high)
    above_floor_margin = float(z_low - floor_m + floor_tolerance_m)
    combined_margin = float(min(
        longitudinal_margin,
        lateral_margin,
        below_rim_margin,
        above_floor_margin,
    ))
    return (
        longitudinal_margin,
        lateral_margin,
        below_rim_margin,
        above_floor_margin,
        combined_margin,
    )


def _points(value, name: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name}_shape:{points.shape}")
    return points[np.all(np.isfinite(points), axis=1)]


def _deterministic_point_cap(
    points: np.ndarray,
    max_points: int,
) -> np.ndarray:
    """Bound pairwise work without introducing sampling randomness."""
    if len(points) <= max_points:
        return points
    indices = np.linspace(
        0,
        len(points) - 1,
        num=max_points,
        dtype=np.int64,
    )
    return points[indices]


def _visual_caddy_central_lateral_topology(
    *,
    caddy_back_coordinates: np.ndarray,
    caddy_lateral_coordinates: np.ndarray,
    caddy_z_coordinates: np.ndarray,
    back_start_m: float,
    lateral_low_m: float,
    lateral_high_m: float,
    config: VisualInConfig,
) -> VisualCaddyTopologyEstimate:
    """Locate the central opening from vertical divider evidence.

    A floor contributes many points but almost no vertical span within a
    lateral bin. Divider walls contribute points across a substantial
    height, so their bins rise above the median vertical span. The physical
    central-width prior anchors a local search on each side, preventing
    unrelated mesh edges from becoming compartment boundaries. This estimate
    is diagnostic-only.
    """
    try:
        back = np.asarray(caddy_back_coordinates, dtype=np.float64)
        lateral = np.asarray(caddy_lateral_coordinates, dtype=np.float64)
        height = np.asarray(caddy_z_coordinates, dtype=np.float64)
        if back.ndim != 1 or lateral.ndim != 1 or height.ndim != 1:
            raise ValueError("invalid_caddy_topology_coordinate_shape")
        if not (len(back) == len(lateral) == len(height)):
            raise ValueError("caddy_topology_coordinate_length_mismatch")

        bins = int(config.caddy_topology_histogram_bins)
        min_points = int(config.caddy_topology_min_bin_points)
        exclusion = float(config.caddy_topology_outer_exclusion_fraction)
        search_radius_bins = int(config.caddy_topology_search_radius_bins)
        if bins < 8:
            raise ValueError("insufficient_caddy_topology_bins")
        if min_points < 2:
            raise ValueError("invalid_caddy_topology_min_bin_points")
        if not 0.0 <= exclusion < 0.5:
            raise ValueError("invalid_caddy_topology_outer_exclusion")
        if search_radius_bins < 1:
            raise ValueError("invalid_caddy_topology_search_radius")

        lateral_low = float(lateral_low_m)
        lateral_high = float(lateral_high_m)
        if not lateral_high > lateral_low:
            raise ValueError("invalid_caddy_topology_lateral_range")
        selected = (
            np.isfinite(back)
            & np.isfinite(lateral)
            & np.isfinite(height)
            & (back >= float(back_start_m))
            & (lateral >= lateral_low)
            & (lateral <= lateral_high)
        )
        if int(np.count_nonzero(selected)) < 2 * min_points:
            raise ValueError("insufficient_caddy_topology_points")

        selected_lateral = lateral[selected]
        selected_height = height[selected]
        edges = np.linspace(lateral_low, lateral_high, bins + 1)
        indices = np.searchsorted(
            edges, selected_lateral, side="right"
        ) - 1
        indices = np.clip(indices, 0, bins - 1)
        vertical_spans = np.full(bins, np.nan, dtype=np.float64)
        for index in range(bins):
            values = selected_height[indices == index]
            if len(values) >= min_points:
                low, high = np.quantile(values, (0.10, 0.90))
                vertical_spans[index] = float(high - low)

        finite = np.isfinite(vertical_spans)
        if int(np.count_nonzero(finite)) < 4:
            raise ValueError("insufficient_caddy_topology_bins")
        median_span = float(np.median(vertical_spans[finite]))
        centers = 0.5 * (edges[:-1] + edges[1:])
        lateral_center = 0.5 * (lateral_low + lateral_high)
        lateral_half = 0.5 * (lateral_high - lateral_low)
        central_fraction = float(
            config.caddy_central_lateral_half_fraction
        )
        if not 0.0 < central_fraction <= 1.0:
            raise ValueError("invalid_caddy_central_lateral_half_fraction")
        interior_half = (1.0 - exclusion) * lateral_half
        interior = np.abs(centers - lateral_center) <= interior_half
        expected_half = central_fraction * lateral_half
        expected_left = lateral_center - expected_half
        expected_right = lateral_center + expected_half
        search_radius_m = search_radius_bins * float(edges[1] - edges[0])
        left_candidates = np.flatnonzero(
            finite
            & interior
            & (centers < lateral_center)
            & (np.abs(centers - expected_left) <= search_radius_m)
        )
        right_candidates = np.flatnonzero(
            finite
            & interior
            & (centers > lateral_center)
            & (np.abs(centers - expected_right) <= search_radius_m)
        )
        if len(left_candidates) == 0 or len(right_candidates) == 0:
            raise ValueError("missing_caddy_topology_anchor_neighborhood")

        left_index = int(left_candidates[np.argmax(
            vertical_spans[left_candidates] - median_span
        )])
        right_index = int(right_candidates[np.argmax(
            vertical_spans[right_candidates] - median_span
        )])
        left_prominence = float(vertical_spans[left_index] - median_span)
        right_prominence = float(vertical_spans[right_index] - median_span)
        if left_prominence <= 0.0 or right_prominence <= 0.0:
            raise ValueError("caddy_topology_dividers_not_prominent")

        opening_low = float(edges[left_index + 1])
        opening_high = float(edges[right_index])
        if opening_high <= opening_low:
            raise ValueError("invalid_caddy_topology_opening")
        selected_z05, selected_z95 = np.quantile(
            selected_height, (0.05, 0.95)
        )
        visible_height = float(selected_z95 - selected_z05)
        if visible_height <= 1.0e-9:
            raise ValueError("degenerate_caddy_topology_height")
        left_distance = lateral_center - centers[left_index]
        right_distance = centers[right_index] - lateral_center
        symmetry = float(
            min(left_distance, right_distance)
            / max(left_distance, right_distance)
        )
        prominence = min(left_prominence, right_prominence) / visible_height
        confidence = float(np.clip(prominence * symmetry, 0.0, 1.0))
        return VisualCaddyTopologyEstimate(
            valid=True,
            reason="",
            confidence=confidence,
            lateral_low_m=opening_low,
            lateral_high_m=opening_high,
        )
    except Exception as error:
        return VisualCaddyTopologyEstimate(
            valid=False,
            reason=f"{type(error).__name__}:{error}",
        )


def visual_point_contact_quantiles(
    object_points_world,
    support_points_world,
    *,
    max_points: int = 256,
) -> VisualPointContactDiagnostics:
    """Measure object-to-support nearest-distance quantiles from RGB-D.

    This is deliberately diagnostic-only: it exposes direct visible surface
    proximity but does not change any atomic margin or confirmation state.
    """
    try:
        limit = int(max_points)
        if limit <= 0:
            raise ValueError("invalid_max_points")
        obj = _points(object_points_world, "object_points")
        support = _points(support_points_world, "support_points")
        if len(obj) == 0:
            raise ValueError("missing_object_points")
        if len(support) == 0:
            raise ValueError("missing_support_points")

        obj = _deterministic_point_cap(obj, limit)
        support = _deterministic_point_cap(support, limit)
        deltas = obj[:, None, :] - support[None, :, :]
        nearest = np.sqrt(np.min(
            np.sum(deltas * deltas, axis=2),
            axis=1,
        ))
        p01, p05, p10 = np.quantile(
            nearest,
            (0.01, 0.05, 0.10),
        )
        return VisualPointContactDiagnostics(
            valid=True,
            reason="",
            p01_m=float(p01),
            p05_m=float(p05),
            p10_m=float(p10),
        )
    except Exception as error:
        return VisualPointContactDiagnostics(
            valid=False,
            reason=f"{type(error).__name__}:{error}",
        )


def visual_in_basket_margin(
    object_points_world,
    basket_points_world,
    gripper_width_m: float,
    config: VisualInConfig = VisualInConfig(),
) -> VisualInEstimate:
    """Compute min(inside footprint, below rim, above floor, released)."""
    try:
        obj = _points(object_points_world, "object_points")
        basket = _points(basket_points_world, "basket_points")
        if len(obj) < config.min_object_points:
            raise ValueError("insufficient_object_points")
        if len(basket) < config.min_basket_points:
            raise ValueError("insufficient_basket_points")
        width = float(gripper_width_m)
        if not np.isfinite(width):
            raise ValueError("invalid_gripper_width")

        obj_low, obj_high = np.percentile(obj, (5.0, 95.0), axis=0)
        basket_low, basket_high = np.percentile(
            basket, (5.0, 95.0), axis=0
        )
        obj_center = (obj_low + obj_high) * 0.5
        obj_half_xy = (
            (obj_high[:2] - obj_low[:2]) * 0.5
            if config.use_object_footprint
            else np.zeros(2, dtype=np.float64)
        )
        basket_center_xy = (basket_low[:2] + basket_high[:2]) * 0.5
        basket_inner_half_xy = (
            (basket_high[:2] - basket_low[:2]) * 0.5
            - config.basket_inner_shrink_m
        )
        if np.any(basket_inner_half_xy <= 0.0):
            raise ValueError("basket_opening_too_small")

        inside_xy = float(np.min(
            basket_inner_half_xy
            - np.abs(obj_center[:2] - basket_center_xy)
            - obj_half_xy
        ))
        below_rim = float(
            basket_high[2]
            + config.below_rim_tolerance_m
            - obj_center[2]
        )
        above_floor = float(
            obj_center[2]
            - basket_low[2]
            + config.floor_tolerance_m
        )
        release = float(width - config.release_width_threshold_m)
        return VisualInEstimate(
            valid=True,
            reason="",
            margin=float(min(
                inside_xy,
                below_rim,
                above_floor,
                *([release] if config.require_release else []),
            )),
            inside_xy_margin=inside_xy,
            below_rim_margin=below_rim,
            above_floor_margin=above_floor,
            release_margin=release,
        )
    except Exception as error:
        return VisualInEstimate(
            valid=False,
            reason=f"{type(error).__name__}:{error}",
        )


def visual_container_on_margin(
    object_points_world,
    bowl_points_world,
    *,
    center_xy_tolerance_m: float = 0.03,
    config: VisualInConfig = VisualInConfig(),
) -> VisualContainerOnEstimate:
    """Estimate canonical ``On(object, bowl)`` from RGB-D geometry.

    The center-height term mirrors the privileged ``On`` structure.  A
    center-distance pass or an explicit visual-opening containment pass can
    supply horizontal support evidence; bowl containment also replaces
    unavailable MuJoCo contact.  The observable release event is deliberately
    handled by the monitor's temporal confirmation gate, not by this raw
    semantic margin.
    """
    try:
        obj = _points(object_points_world, "object_points")
        bowl = _points(bowl_points_world, "bowl_points")
        tolerance = float(center_xy_tolerance_m)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("invalid_center_xy_tolerance")

        if len(obj) < config.min_object_points:
            raise ValueError("insufficient_object_points")
        if len(bowl) < config.min_basket_points:
            raise ValueError("insufficient_basket_points")

        obj_low, obj_high = np.percentile(obj, (5.0, 95.0), axis=0)
        bowl_low, bowl_high = np.percentile(
            bowl, (5.0, 95.0), axis=0
        )
        obj_center = (obj_low + obj_high) * 0.5
        bowl_center = (bowl_low + bowl_high) * 0.5
        return visual_container_on_from_bounds(
            object_center_world=obj_center,
            object_xy_q05_m=obj_low[:2],
            object_xy_q95_m=obj_high[:2],
            object_bottom_z_m=float(obj_low[2]),
            object_top_z_m=float(obj_high[2]),
            bowl_center_world=bowl_center,
            bowl_xy_q05_m=bowl_low[:2],
            bowl_xy_q95_m=bowl_high[:2],
            bowl_floor_z_m=float(bowl_low[2]),
            bowl_rim_z_m=float(bowl_high[2]),
            center_xy_tolerance_m=tolerance,
            config=config,
        )
    except Exception as error:
        return VisualContainerOnEstimate(
            valid=False,
            reason=f"{type(error).__name__}:{error}",
        )


def visual_container_on_from_bounds(
    *,
    object_center_world,
    object_xy_q05_m,
    object_xy_q95_m,
    object_bottom_z_m: float,
    object_top_z_m: float,
    bowl_center_world,
    bowl_xy_q05_m,
    bowl_xy_q95_m,
    bowl_floor_z_m: float,
    bowl_rim_z_m: float,
    center_xy_tolerance_m: float = 0.03,
    config: VisualInConfig = VisualInConfig(),
) -> VisualContainerOnEstimate:
    """Apply the hybrid margin to cached visual RGB-D summaries."""
    try:
        obj = np.asarray(object_center_world, dtype=np.float64).reshape(3)
        obj_bottom = float(object_bottom_z_m)
        bowl = np.asarray(bowl_center_world, dtype=np.float64).reshape(3)
        lower = np.asarray(bowl_xy_q05_m, dtype=np.float64).reshape(2)
        upper = np.asarray(bowl_xy_q95_m, dtype=np.float64).reshape(2)
        floor = float(bowl_floor_z_m)
        rim = float(bowl_rim_z_m)
        tolerance = float(center_xy_tolerance_m)
        values = np.concatenate((
            obj,
            bowl,
            lower,
            upper,
            (obj_bottom, floor, rim, tolerance),
        ))
        if not np.all(np.isfinite(values)):
            raise ValueError("nonfinite_visual_bounds")
        if tolerance <= 0.0:
            raise ValueError("invalid_center_xy_tolerance")
        if rim <= floor:
            raise ValueError("invalid_bowl_vertical_bounds")

        inner_lower = lower + float(config.basket_inner_shrink_m)
        inner_upper = upper - float(config.basket_inner_shrink_m)
        if np.any(inner_upper <= inner_lower):
            raise ValueError("basket_opening_too_small")
        inside_xy = float(min(
            np.min(obj[:2] - inner_lower),
            np.min(inner_upper - obj[:2]),
        ))
        center_z = float(obj[2] - bowl[2])
        center_xy_distance = float(np.linalg.norm(
            obj[:2] - bowl[:2]
        ))
        center_xy = float(tolerance - center_xy_distance)
        below_rim = float(
            rim + config.below_rim_tolerance_m - obj_bottom
        )
        above_floor = float(
            obj_bottom - floor + config.floor_tolerance_m
        )
        horizontal = float(max(center_xy, inside_xy))
        return VisualContainerOnEstimate(
            valid=True,
            reason="",
            margin=float(min(
                center_z,
                horizontal,
                below_rim,
                above_floor,
            )),
            center_z_margin=center_z,
            center_xy_distance=center_xy_distance,
            center_xy_margin=center_xy,
            inside_xy_margin=inside_xy,
            below_rim_margin=below_rim,
            above_floor_margin=above_floor,
        )
    except Exception as error:
        return VisualContainerOnEstimate(
            valid=False,
            reason=f"{type(error).__name__}:{error}",
        )


_DRAWER_LEVEL_BANDS = {
    "bottom": (0.04, 0.40),
    "middle": (0.28, 0.72),
    "top": (0.60, 0.96),
}


def visual_in_drawer_margin(
    object_points_world,
    cabinet_points_world,
    gripper_width_m: float,
    *,
    drawer_level: str = "bottom",
    config: VisualInConfig = VisualInConfig(),
) -> VisualInEstimate:
    """Estimate ``In(object, drawer)`` from object/cabinet RGB-D points.

    The visible cabinet point cloud is split into a fixed height band for
    the requested drawer.  The object's robust P05--P95 footprint must lie
    inside that band, below its visual rim, above its visual floor, and the
    gripper must be open.  This is deliberately a shadow estimate: its
    thresholds still require validation against fresh task33 episodes.
    """
    try:
        obj = _points(object_points_world, "object_points")
        cabinet = _points(cabinet_points_world, "cabinet_points")
        if len(obj) < config.min_object_points:
            raise ValueError("insufficient_object_points")
        if len(cabinet) < config.min_drawer_points:
            raise ValueError("insufficient_cabinet_points")
        width = float(gripper_width_m)
        if not np.isfinite(width):
            raise ValueError("invalid_gripper_width")

        level = str(drawer_level).strip().lower()
        if level not in _DRAWER_LEVEL_BANDS:
            raise ValueError(f"unsupported_drawer_level:{level}")

        cabinet_z05, cabinet_z95 = np.quantile(
            cabinet[:, 2], (0.05, 0.95)
        )
        cabinet_height = float(cabinet_z95 - cabinet_z05)
        if not np.isfinite(cabinet_height) or cabinet_height <= 1.0e-3:
            raise ValueError("degenerate_cabinet_height")

        low_fraction, high_fraction = _DRAWER_LEVEL_BANDS[level]
        band_low = cabinet_z05 + low_fraction * cabinet_height
        band_high = cabinet_z05 + high_fraction * cabinet_height
        drawer = cabinet[
            (cabinet[:, 2] >= band_low)
            & (cabinet[:, 2] <= band_high)
        ]
        if len(drawer) < config.min_drawer_points:
            raise ValueError("insufficient_drawer_band_points")

        obj_low, obj_high = np.percentile(obj, (5.0, 95.0), axis=0)
        drawer_low, drawer_high = np.percentile(
            drawer, (5.0, 95.0), axis=0
        )
        obj_center = (obj_low + obj_high) * 0.5
        obj_half_xy = (
            (obj_high[:2] - obj_low[:2]) * 0.5
            if config.use_object_footprint
            else np.zeros(2, dtype=np.float64)
        )
        drawer_center_xy = (drawer_low[:2] + drawer_high[:2]) * 0.5
        drawer_inner_half_xy = (
            (drawer_high[:2] - drawer_low[:2]) * 0.5
            - config.drawer_inner_shrink_m
        )
        if np.any(drawer_inner_half_xy <= 0.0):
            raise ValueError("drawer_opening_too_small")

        inside_xy = float(np.min(
            drawer_inner_half_xy
            - np.abs(obj_center[:2] - drawer_center_xy)
            - obj_half_xy
        ))
        below_rim = float(
            drawer_high[2]
            + config.below_rim_tolerance_m
            - obj_center[2]
        )
        above_floor = float(
            obj_center[2]
            - drawer_low[2]
            + config.floor_tolerance_m
        )
        release = float(width - config.release_width_threshold_m)
        return VisualInEstimate(
            valid=True,
            reason="",
            margin=float(min(
                inside_xy,
                below_rim,
                above_floor,
                *([release] if config.require_release else []),
            )),
            inside_xy_margin=inside_xy,
            below_rim_margin=below_rim,
            above_floor_margin=above_floor,
            release_margin=release,
        )
    except Exception as error:
        return VisualInEstimate(
            valid=False,
            reason=f"{type(error).__name__}:{error}",
        )


def visual_in_microwave_margin(
    object_points_world,
    microwave_points_world,
    gripper_width_m: float,
    config: VisualInConfig = VisualInConfig(),
    *,
    camera_position_world,
) -> VisualInEstimate:
    """Estimate ``In(object, microwave)`` from segmented RGB-D points.

    The microwave segmentation supplies a robust 3-D envelope.  In addition
    to horizontal containment and height, the object center must cross the
    camera-facing front plane of the appliance.  This prevents a mug resting
    in front of the open door from satisfying ``In`` merely because the broad
    appliance envelope contains its world XY position.  No simulator
    geometry, joint, contact, site, object pose, or oracle value is accepted.

    This is deliberately a shadow estimator.  A first task39 rollout must
    validate whether the microwave mask includes enough of its open cavity;
    if it does not, the monitor emits diagnostics instead of using oracle
    geometry as a fallback.
    """
    try:
        obj = _points(object_points_world, "object_points")
        microwave = _points(
            microwave_points_world, "microwave_points"
        )
        if len(obj) < config.min_object_points:
            raise ValueError("insufficient_object_points")
        if len(microwave) < config.min_basket_points:
            raise ValueError("insufficient_microwave_points")
        width = float(gripper_width_m)
        if not np.isfinite(width):
            raise ValueError("invalid_gripper_width")
        camera = np.asarray(camera_position_world, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(camera)):
            raise ValueError("invalid_camera_position")
        front_quantile = float(config.microwave_front_quantile)
        if not 0.5 < front_quantile < 1.0:
            raise ValueError("invalid_microwave_front_quantile")

        obj_low, obj_high = np.percentile(obj, (5.0, 95.0), axis=0)
        box_low, box_high = np.percentile(
            microwave, (2.0, 98.0), axis=0
        )
        obj_center = (obj_low + obj_high) * 0.5
        obj_half_xy = (
            (obj_high[:2] - obj_low[:2]) * 0.5
            if config.use_object_footprint
            else np.zeros(2, dtype=np.float64)
        )
        box_center_xy = (box_low[:2] + box_high[:2]) * 0.5
        inner_half_xy = (
            (box_high[:2] - box_low[:2]) * 0.5
            - config.microwave_inner_shrink_m
        )
        if np.any(inner_half_xy <= 0.0):
            raise ValueError("microwave_envelope_too_small")

        inside_xy = float(np.min(
            inner_half_xy
            - np.abs(obj_center[:2] - box_center_xy)
            - obj_half_xy
        ))
        below_top = float(
            box_high[2]
            + config.below_rim_tolerance_m
            - obj_center[2]
        )
        above_floor = float(
            obj_center[2]
            - box_low[2]
            + config.floor_tolerance_m
        )
        microwave_center_xy = np.median(microwave[:, :2], axis=0)
        camera_axis_xy = camera[:2] - microwave_center_xy
        camera_axis_norm = float(np.linalg.norm(camera_axis_xy))
        if camera_axis_norm <= 1e-6:
            raise ValueError("camera_on_microwave_center")
        camera_axis_xy = camera_axis_xy / camera_axis_norm
        microwave_depth = microwave[:, :2] @ camera_axis_xy
        object_depth = obj[:, :2] @ camera_axis_xy
        front_plane = float(np.quantile(
            microwave_depth, front_quantile
        ))
        object_center_depth = float(np.median(object_depth))
        entry_depth = float(
            front_plane
            + config.microwave_entry_tolerance_m
            - object_center_depth
            - config.microwave_min_entry_depth_m
        )
        release = float(width - config.release_width_threshold_m)
        return VisualInEstimate(
            valid=True,
            reason="",
            margin=float(min(
                inside_xy,
                below_top,
                above_floor,
                entry_depth,
                *([release] if config.require_release else []),
            )),
            inside_xy_margin=inside_xy,
            below_rim_margin=below_top,
            above_floor_margin=above_floor,
            entry_depth_margin=entry_depth,
            release_margin=release,
        )
    except Exception as error:
        return VisualInEstimate(
            valid=False,
            reason=f"{type(error).__name__}:{error}",
        )


def visual_in_caddy_back_margin(
    object_points_world,
    caddy_points_world,
    gripper_width_m: float,
    *,
    camera_position_world,
    config: VisualInConfig = VisualInConfig(),
) -> VisualInEstimate:
    """Estimate containment in the visually defined back caddy compartment.

    ``back`` is the half of the visible caddy footprint farther from the
    calibrated camera.  The canonical margin uses the visual object center,
    matching LIBERO's reference-point ``In`` semantics.  Robust core and full
    footprint margins are exported alongside it for online diagnostics; they
    do not silently replace the atom consumed by plotting, AGM, and reward.
    No simulator site, object pose, contact, joint, or Oracle value is
    accepted.
    """
    try:
        obj = _points(object_points_world, "object_points")
        caddy = _points(caddy_points_world, "caddy_points")
        if len(obj) < config.min_object_points:
            raise ValueError("insufficient_object_points")
        if len(caddy) < config.min_basket_points:
            raise ValueError("insufficient_caddy_points")
        width = float(gripper_width_m)
        camera = np.asarray(camera_position_world, dtype=np.float64).reshape(3)
        if not np.isfinite(width):
            raise ValueError("invalid_gripper_width")
        if not np.all(np.isfinite(camera)):
            raise ValueError("invalid_camera_position")

        caddy_low, caddy_high = np.percentile(caddy, (5.0, 95.0), axis=0)
        caddy_center = (caddy_low + caddy_high) * 0.5
        toward_camera = camera[:2] - caddy_center[:2]
        norm = float(np.linalg.norm(toward_camera))
        if norm <= 1.0e-9:
            raise ValueError("degenerate_caddy_camera_axis")
        back_axis = -toward_camera / norm
        lateral_axis = np.asarray([-back_axis[1], back_axis[0]])

        obj_low, obj_high = np.percentile(obj, (5.0, 95.0), axis=0)
        obj_center = (obj_low + obj_high) * 0.5
        obj_relative_xy = obj[:, :2] - caddy_center[:2]
        caddy_relative_xy = caddy[:, :2] - caddy_center[:2]
        center_relative = obj_center[:2] - caddy_center[:2]
        center_back = float(center_relative @ back_axis)
        center_lat = float(center_relative @ lateral_axis)
        obj_back_core_low, obj_back_core_high = np.percentile(
            obj_relative_xy @ back_axis, (20.0, 80.0)
        )
        obj_lat_core_low, obj_lat_core_high = np.percentile(
            obj_relative_xy @ lateral_axis, (20.0, 80.0)
        )
        obj_back_full_low, obj_back_full_high = np.percentile(
            obj_relative_xy @ back_axis, (5.0, 95.0)
        )
        obj_lat_full_low, obj_lat_full_high = np.percentile(
            obj_relative_xy @ lateral_axis, (5.0, 95.0)
        )
        caddy_back_low, caddy_back_high = np.percentile(
            caddy_relative_xy @ back_axis, (5.0, 95.0)
        )
        caddy_lat_low, caddy_lat_high = np.percentile(
            caddy_relative_xy @ lateral_axis, (5.0, 95.0)
        )
        back_half = 0.5 * float(caddy_back_high - caddy_back_low)
        lateral_half = 0.5 * float(caddy_lat_high - caddy_lat_low)
        if back_half <= config.caddy_inner_shrink_m:
            raise ValueError("caddy_depth_too_small")
        if lateral_half <= config.caddy_inner_shrink_m:
            raise ValueError("caddy_width_too_small")
        central_fraction = float(
            config.caddy_central_lateral_half_fraction
        )
        if not 0.0 < central_fraction <= 1.0:
            raise ValueError("invalid_caddy_central_lateral_half_fraction")
        back_start = (
            config.caddy_back_start_fraction * back_half
        )
        back_end = float(caddy_back_high - config.caddy_inner_shrink_m)
        lateral_low = float(
            caddy_lat_low + config.caddy_inner_shrink_m
        )
        lateral_high = float(
            caddy_lat_high - config.caddy_inner_shrink_m
        )
        center_longitudinal = float(min(
            center_back - back_start,
            back_end - center_back,
        ))
        center_lateral = float(min(
            center_lat - lateral_low,
            lateral_high - center_lat,
        ))
        center_lateral_offset = float(abs(center_lat))
        central_lateral = float(
            central_fraction * lateral_half - center_lateral_offset
        )
        central_back = float(min(
            center_longitudinal,
            central_lateral,
        ))
        topology = _visual_caddy_central_lateral_topology(
            caddy_back_coordinates=caddy_relative_xy @ back_axis,
            caddy_lateral_coordinates=caddy_relative_xy @ lateral_axis,
            caddy_z_coordinates=caddy[:, 2],
            back_start_m=back_start,
            lateral_low_m=lateral_low,
            lateral_high_m=lateral_high,
            config=config,
        )
        topology_lateral = (
            float(min(
                center_lat - topology.lateral_low_m,
                topology.lateral_high_m - center_lat,
            ))
            if topology.valid else float("nan")
        )
        topology_back = (
            float(min(center_longitudinal, topology_lateral))
            if topology.valid else float("nan")
        )
        if topology.valid:
            (
                topology_footprint_longitudinal,
                topology_footprint_lateral,
                topology_footprint_below_rim,
                topology_footprint_above_floor,
                topology_footprint_margin,
            ) = _robust_caddy_footprint_containment_margins(
                object_back_coordinates=obj_relative_xy @ back_axis,
                object_lateral_coordinates=obj_relative_xy @ lateral_axis,
                object_z_coordinates=obj[:, 2],
                longitudinal_low_m=back_start,
                longitudinal_high_m=back_end,
                lateral_low_m=topology.lateral_low_m,
                lateral_high_m=topology.lateral_high_m,
                floor_m=float(caddy_low[2]),
                rim_m=float(caddy_high[2]),
                floor_tolerance_m=float(config.floor_tolerance_m),
                rim_tolerance_m=float(config.below_rim_tolerance_m),
                lower_quantile=float(
                    config.caddy_footprint_lower_quantile
                ),
                upper_quantile=float(
                    config.caddy_footprint_upper_quantile
                ),
            )
        else:
            topology_footprint_longitudinal = float("nan")
            topology_footprint_lateral = float("nan")
            topology_footprint_below_rim = float("nan")
            topology_footprint_above_floor = float("nan")
            topology_footprint_margin = float("nan")
        core_longitudinal = float(min(
            obj_back_core_low - back_start,
            back_end - obj_back_core_high,
        ))
        core_lateral = float(min(
            obj_lat_core_low - lateral_low,
            lateral_high - obj_lat_core_high,
        ))
        full_longitudinal = float(min(
            obj_back_full_low - back_start,
            back_end - obj_back_full_high,
        ))
        full_lateral = float(min(
            obj_lat_full_low - lateral_low,
            lateral_high - obj_lat_full_high,
        ))
        center_inside = float(min(center_longitudinal, center_lateral))
        full_inside = float(min(full_longitudinal, full_lateral))
        inside_xy = (
            full_inside
            if config.caddy_use_object_footprint
            else center_inside
        )
        below_rim = float(
            caddy_high[2]
            + config.below_rim_tolerance_m
            - obj_center[2]
        )
        above_floor = float(
            obj_center[2]
            - caddy_low[2]
            + config.floor_tolerance_m
        )
        release = float(width - config.release_width_threshold_m)
        return VisualInEstimate(
            valid=True,
            reason="",
            margin=float(min(
                inside_xy,
                below_rim,
                above_floor,
                *([release] if config.require_release else []),
            )),
            inside_xy_margin=inside_xy,
            below_rim_margin=below_rim,
            above_floor_margin=above_floor,
            release_margin=release,
            caddy_center_longitudinal_margin=center_longitudinal,
            caddy_center_lateral_margin=center_lateral,
            caddy_center_back_offset_m=center_back,
            caddy_center_lateral_offset_m=center_lateral_offset,
            caddy_back_half_extent_m=back_half,
            caddy_lateral_half_extent_m=lateral_half,
            caddy_central_lateral_margin_m=central_lateral,
            caddy_central_back_margin_m=central_back,
            caddy_topology_valid=bool(topology.valid),
            caddy_topology_reason=str(topology.reason),
            caddy_topology_confidence=float(topology.confidence),
            caddy_topology_lateral_low_m=float(topology.lateral_low_m),
            caddy_topology_lateral_high_m=float(topology.lateral_high_m),
            caddy_topology_lateral_margin_m=topology_lateral,
            caddy_topology_back_margin_m=topology_back,
            caddy_topology_footprint_valid=bool(topology.valid),
            caddy_topology_footprint_longitudinal_margin_m=(
                topology_footprint_longitudinal
            ),
            caddy_topology_footprint_lateral_margin_m=(
                topology_footprint_lateral
            ),
            caddy_topology_footprint_below_rim_margin_m=(
                topology_footprint_below_rim
            ),
            caddy_topology_footprint_above_floor_margin_m=(
                topology_footprint_above_floor
            ),
            caddy_topology_footprint_margin_m=topology_footprint_margin,
            caddy_core_longitudinal_margin=core_longitudinal,
            caddy_core_lateral_margin=core_lateral,
            caddy_full_longitudinal_margin=full_longitudinal,
            caddy_full_lateral_margin=full_lateral,
        )
    except Exception as error:
        return VisualInEstimate(
            valid=False,
            reason=f"{type(error).__name__}:{error}",
        )


def visual_in_microwave_occlusion_margin(
    *,
    last_inside_xy_margin: float,
    last_below_rim_margin: float,
    last_above_floor_margin: float,
    last_entry_depth_margin: float,
    release_evidence_margin: float,
    occlusion_age_steps: int,
    config: VisualInConfig = VisualInConfig(),
) -> VisualInEstimate:
    """Validate a bounded target disappearance at the microwave entrance.

    All inputs are derived from the last fresh RGB-D estimate, observable
    gripper release, and elapsed frames.  This function deliberately has no
    simulator pose, contact, joint, site, or oracle argument.
    """
    try:
        age = int(occlusion_age_steps)
        if age < 1 or age > config.microwave_occlusion_steps:
            raise ValueError(f"occlusion_age_out_of_range:{age}")
        inside = float(
            last_inside_xy_margin
            + config.microwave_entry_tolerance_m
        )
        below = float(last_below_rim_margin)
        above = float(last_above_floor_margin)
        entry = float(last_entry_depth_margin)
        release = float(release_evidence_margin)
        values = np.asarray(
            [inside, below, above, entry, release], dtype=np.float64
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("non_finite_occlusion_evidence")
        margin = float(np.min(values))
        if margin < 0.0:
            raise ValueError(f"not_at_microwave_entry:{margin:.6f}")
        return VisualInEstimate(
            valid=True,
            reason=f"microwave_entry_occlusion:age={age}",
            margin=margin,
            inside_xy_margin=inside,
            below_rim_margin=below,
            above_floor_margin=above,
            entry_depth_margin=entry,
            release_margin=release,
        )
    except Exception as error:
        return VisualInEstimate(
            valid=False,
            reason=f"{type(error).__name__}:{error}",
        )


def visual_release_evidence_is_current(
    *,
    current_step: int,
    release_step: int | None,
    max_age_steps: int,
    already_confirmed: bool = False,
) -> bool:
    """Return whether observable release evidence may affect a new margin."""
    if already_confirmed:
        return True
    if release_step is None or int(max_age_steps) < 0:
        return False
    age = int(current_step) - int(release_step)
    return 0 <= age <= int(max_age_steps)
