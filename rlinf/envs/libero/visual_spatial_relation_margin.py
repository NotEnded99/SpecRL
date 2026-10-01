"""Visual-only margins for named directional tabletop regions.

These predicates are semantic surrogates for simulator-only target sites.
They derive a local frame from the visible anchor and calibrated camera, then
measure signed containment in a region immediately in front of / to the right
of that anchor.  No simulator site, object pose, contact, release flag, or
Oracle margin is accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class VisualSpatialRelationConfig:
    min_clearance_m: float = -0.015
    max_offset_m: float = 0.25
    lateral_padding_m: float = 0.05
    surface_tolerance_m: float = 0.04
    min_anchor_span_m: float = 0.02


@dataclass(frozen=True)
class VisualSpatialRelationEstimate:
    valid: bool
    reason: str
    relation: str
    margin: float
    directional_margin: float
    lateral_margin: float
    surface_margin: float
    directional_offset_m: float
    lateral_offset_m: float
    anchor_direction_half_extent_m: float
    anchor_lateral_half_extent_m: float

    @property
    def satisfied(self) -> bool:
        return bool(self.valid and self.margin >= 0.0)


def _vector(value: Iterable[float], size: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size != size or not np.all(np.isfinite(vector)):
        raise ValueError(f"invalid_{name}")
    return vector


def _unit_xy(vector, name: str) -> np.ndarray:
    xy = _vector(vector, 2, name)
    norm = float(np.linalg.norm(xy))
    if norm <= 1.0e-9:
        raise ValueError(f"degenerate_{name}")
    return xy / norm


def _anchor_corners(lower_xy: np.ndarray, upper_xy: np.ndarray) -> np.ndarray:
    return np.asarray([
        [lower_xy[0], lower_xy[1]],
        [lower_xy[0], upper_xy[1]],
        [upper_xy[0], lower_xy[1]],
        [upper_xy[0], upper_xy[1]],
    ], dtype=np.float64)


def visual_directional_region_margin(
    *,
    relation: str,
    object_center_world: Iterable[float],
    object_bottom_z_m: float,
    anchor_center_world: Iterable[float],
    anchor_lower_xy_world: Iterable[float],
    anchor_upper_xy_world: Iterable[float],
    anchor_top_z_m: float,
    camera_position_world: Iterable[float],
    camera_right_world: Iterable[float],
    config: VisualSpatialRelationConfig = VisualSpatialRelationConfig(),
) -> VisualSpatialRelationEstimate:
    """Return a signed front/right region-containment margin in metres."""
    nan = float("nan")
    invalid = dict(
        valid=False, margin=nan, directional_margin=nan,
        lateral_margin=nan, surface_margin=nan,
        directional_offset_m=nan, lateral_offset_m=nan,
        anchor_direction_half_extent_m=nan,
        anchor_lateral_half_extent_m=nan,
    )
    relation = str(relation).strip().lower()
    if relation not in {"front_of_stove", "right_of_plate"}:
        return VisualSpatialRelationEstimate(
            reason=f"unsupported_relation:{relation}", relation=relation,
            **invalid,
        )
    try:
        obj = _vector(object_center_world, 3, "object_center")
        anchor = _vector(anchor_center_world, 3, "anchor_center")
        lower = _vector(anchor_lower_xy_world, 2, "anchor_lower")
        upper = _vector(anchor_upper_xy_world, 2, "anchor_upper")
        camera = _vector(camera_position_world, 3, "camera_position")
        camera_right = _vector(camera_right_world, 3, "camera_right")[:2]
    except ValueError as error:
        return VisualSpatialRelationEstimate(
            reason=str(error), relation=relation, **invalid,
        )
    if np.any(upper <= lower):
        return VisualSpatialRelationEstimate(
            reason="invalid_anchor_footprint", relation=relation, **invalid,
        )
    if not np.isfinite(object_bottom_z_m) or not np.isfinite(anchor_top_z_m):
        return VisualSpatialRelationEstimate(
            reason="invalid_surface_height", relation=relation, **invalid,
        )

    try:
        toward_camera = _unit_xy(camera[:2] - anchor[:2], "front_axis")
        right = _unit_xy(camera_right, "right_axis")
    except ValueError as error:
        return VisualSpatialRelationEstimate(
            reason=str(error), relation=relation, **invalid,
        )
    if relation == "front_of_stove":
        direction = toward_camera
        lateral = np.asarray([-direction[1], direction[0]])
    else:
        # Remove any component parallel to camera-front so the tabletop frame
        # remains orthogonal even with a slightly tilted calibration.
        right = right - toward_camera * float(np.dot(right, toward_camera))
        try:
            direction = _unit_xy(right, "orthogonal_right_axis")
        except ValueError as error:
            return VisualSpatialRelationEstimate(
                reason=str(error), relation=relation, **invalid,
            )
        lateral = toward_camera

    corners = _anchor_corners(lower, upper) - anchor[:2]
    direction_half = float(np.max(np.abs(corners @ direction)))
    lateral_half = float(np.max(np.abs(corners @ lateral)))
    if min(direction_half, lateral_half) < config.min_anchor_span_m * 0.5:
        return VisualSpatialRelationEstimate(
            reason="anchor_footprint_too_small", relation=relation, **invalid,
        )

    relative = obj[:2] - anchor[:2]
    directional_offset = float(np.dot(relative, direction))
    lateral_offset = float(np.dot(relative, lateral))
    lower_offset = direction_half + config.min_clearance_m
    upper_offset = direction_half + config.max_offset_m
    directional_margin = float(min(
        directional_offset - lower_offset,
        upper_offset - directional_offset,
    ))
    lateral_margin = float(
        lateral_half + config.lateral_padding_m - abs(lateral_offset)
    )
    surface_margin = float(
        config.surface_tolerance_m
        - abs(float(object_bottom_z_m) - float(anchor_top_z_m))
    )
    margin = float(min(directional_margin, lateral_margin, surface_margin))
    return VisualSpatialRelationEstimate(
        valid=True,
        reason="",
        relation=relation,
        margin=margin,
        directional_margin=directional_margin,
        lateral_margin=lateral_margin,
        surface_margin=surface_margin,
        directional_offset_m=directional_offset,
        lateral_offset_m=lateral_offset,
        anchor_direction_half_extent_m=direction_half,
        anchor_lateral_half_extent_m=lateral_half,
    )


__all__ = [
    "VisualSpatialRelationConfig",
    "VisualSpatialRelationEstimate",
    "visual_directional_region_margin",
]

