"""Visual-only margin for closing a drawer that starts open.

The estimator consumes only RGB-D point clouds reconstructed from a YOLOE
cabinet mask.  The reset observation is the open-drawer baseline used by
LIBERO-40 task33.  Simulator joints, sites, contacts, actions, and object
poses are deliberately not accepted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rlinf.envs.libero.visual_open_margin import (
    VisualOpenConfig,
    visual_open_margin,
)


@dataclass(frozen=True)
class VisualCloseConfig:
    """Geometry parameters for reset-open drawer retraction."""

    retraction_threshold_m: float = 0.10
    bound_quantile: float = 0.05
    min_points: int = 40
    extension_axis: str = "y+"
    front_quantile: float = 0.95

    def __post_init__(self) -> None:
        if self.retraction_threshold_m <= 0.0:
            raise ValueError("retraction_threshold_m must be positive")
        if not 0.0 < self.bound_quantile < 0.25:
            raise ValueError("bound_quantile must be in (0, 0.25)")
        if self.min_points < 8:
            raise ValueError("min_points must be at least 8")
        if self.extension_axis not in {"x-", "x+", "y-", "y+"}:
            raise ValueError(
                "extension_axis must be x-, x+, y-, or y+"
            )
        if self.front_quantile not in {
            0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95
        }:
            raise ValueError("front_quantile must be a diagnostic quantile")


@dataclass(frozen=True)
class VisualCloseEstimate:
    valid: bool
    reason: str = ""
    retraction_m: float = float("nan")
    margin: float = float("nan")
    extension_axis: str = ""
    signed_extension_m: float = float("nan")
    reference_shift_m: float = float("nan")
    baseline_band_points: int = 0
    current_band_points: int = 0
    retraction_q05_m: float = float("nan")
    retraction_q10_m: float = float("nan")
    retraction_q25_m: float = float("nan")
    retraction_q50_m: float = float("nan")
    retraction_q75_m: float = float("nan")
    retraction_q90_m: float = float("nan")
    retraction_q95_m: float = float("nan")


_LEVEL_BANDS = {
    "bottom": (0.04, 0.40),
    "middle": (0.28, 0.72),
    "top": (0.60, 0.96),
}
_DIAGNOSTIC_QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def _finite_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return points[np.all(np.isfinite(points), axis=1)]


def _axis_projection(points_xy: np.ndarray, axis: str) -> np.ndarray:
    coordinate = 0 if axis.startswith("x") else 1
    sign = -1.0 if axis.endswith("-") else 1.0
    return sign * np.asarray(points_xy[:, coordinate], dtype=np.float64)


def _quantile_retractions(
    baseline_points_world: np.ndarray,
    current_points_world: np.ndarray,
    *,
    drawer_level: str,
    config: VisualCloseConfig,
) -> tuple[dict[float, float], float, int, int]:
    """Return front-position candidates relative to a fixed top reference."""
    baseline = _finite_points(baseline_points_world)
    current = _finite_points(current_points_world)
    if len(baseline) < config.min_points or len(current) < config.min_points:
        raise ValueError("insufficient_points")
    level = str(drawer_level).strip().lower()
    if level not in _LEVEL_BANDS:
        raise ValueError(f"unsupported_drawer_level:{level}")
    z05, z95 = np.quantile(baseline[:, 2], (0.05, 0.95))
    z_span = float(z95 - z05)
    if not np.isfinite(z_span) or z_span <= 1.0e-3:
        raise ValueError("degenerate_baseline_height")
    low_fraction, high_fraction = _LEVEL_BANDS[level]
    band_low = z05 + low_fraction * z_span
    band_high = z05 + high_fraction * z_span
    baseline_band = baseline[
        (baseline[:, 2] >= band_low) & (baseline[:, 2] <= band_high)
    ]
    current_band = current[
        (current[:, 2] >= band_low) & (current[:, 2] <= band_high)
    ]
    if len(baseline_band) < config.min_points:
        raise ValueError("insufficient_baseline_band_points")
    if len(current_band) < config.min_points:
        raise ValueError("insufficient_current_band_points")

    reference_low = z05 + 0.76 * z_span
    reference_high = z05 + 0.98 * z_span
    baseline_ref = baseline[
        (baseline[:, 2] >= reference_low)
        & (baseline[:, 2] <= reference_high)
    ]
    current_ref = current[
        (current[:, 2] >= reference_low)
        & (current[:, 2] <= reference_high)
    ]
    if (
        len(baseline_ref) >= config.min_points
        and len(current_ref) >= config.min_points
    ):
        shift_xy = (
            np.median(current_ref[:, :2], axis=0)
            - np.median(baseline_ref[:, :2], axis=0)
        )
    else:
        shift_xy = np.zeros(2, dtype=np.float64)
    current_xy = current_band[:, :2] - shift_xy[None, :]
    baseline_projected = _axis_projection(
        baseline_band[:, :2], config.extension_axis
    )
    current_projected = _axis_projection(
        current_xy, config.extension_axis
    )
    values = {
        q: float(
            np.quantile(baseline_projected, q)
            - np.quantile(current_projected, q)
        )
        for q in _DIAGNOSTIC_QUANTILES
    }
    return values, float(np.linalg.norm(shift_xy)), len(baseline_band), len(current_band)


def visual_close_margin(
    open_baseline_points_world: np.ndarray,
    current_points_world: np.ndarray,
    *,
    drawer_level: str = "bottom",
    config: VisualCloseConfig = VisualCloseConfig(),
) -> VisualCloseEstimate:
    """Estimate retraction from the reset-open drawer footprint.

    ``visual_open_margin`` provides the signed footprint change along the
    configured outward axis.  For a drawer that is closing, that signed
    extension becomes negative; its negation is the retraction distance.
    Positive Close margin means retraction exceeds the configured threshold.
    """

    geometry = visual_open_margin(
        open_baseline_points_world,
        current_points_world,
        drawer_level=drawer_level,
        config=VisualOpenConfig(
            displacement_threshold_m=config.retraction_threshold_m,
            bound_quantile=config.bound_quantile,
            min_points=config.min_points,
            extension_axis=config.extension_axis,
        ),
    )
    if not geometry.valid:
        return VisualCloseEstimate(
            valid=False,
            reason=geometry.reason,
            baseline_band_points=geometry.baseline_band_points,
            current_band_points=geometry.current_band_points,
        )

    try:
        candidates, reference_shift, baseline_count, current_count = (
            _quantile_retractions(
                open_baseline_points_world,
                current_points_world,
                drawer_level=drawer_level,
                config=config,
            )
        )
    except ValueError as error:
        return VisualCloseEstimate(valid=False, reason=str(error))

    # Do not clamp the diagnostic candidate: its sign distinguishes closing
    # from outward motion and exposes segmentation drift during calibration.
    retraction = float(candidates[config.front_quantile])
    return VisualCloseEstimate(
        valid=True,
        retraction_m=retraction,
        margin=float(retraction - config.retraction_threshold_m),
        extension_axis=geometry.extension_axis,
        signed_extension_m=geometry.signed_extension_m,
        reference_shift_m=reference_shift,
        baseline_band_points=baseline_count,
        current_band_points=current_count,
        retraction_q05_m=candidates[0.05],
        retraction_q10_m=candidates[0.10],
        retraction_q25_m=candidates[0.25],
        retraction_q50_m=candidates[0.50],
        retraction_q75_m=candidates[0.75],
        retraction_q90_m=candidates[0.90],
        retraction_q95_m=candidates[0.95],
    )


def apply_visual_in_gate(
    raw_close_margin: float,
    *,
    require_in_confirmed: bool,
    visual_in_confirmed: bool,
    blocked_margin: float = -1.0e-6,
) -> float:
    """Suppress a positive Close margin until visual In is confirmed.

    This helper consumes no oracle signal.  Keeping it pure makes the exact
    temporal gate independently testable and auditable.
    """
    raw = float(raw_close_margin)
    blocked = float(blocked_margin)
    if not np.isfinite(raw):
        return raw
    if blocked >= 0.0:
        raise ValueError("blocked_margin must be negative")
    if require_in_confirmed and not visual_in_confirmed:
        return float(min(raw, blocked))
    return raw
