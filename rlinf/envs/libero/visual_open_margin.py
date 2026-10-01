"""Visual-only margin for articulated drawer opening.

The estimator consumes RGB-D point clouds reconstructed from a YOLOE cabinet
mask.  Simulator joints, sites, contacts, and object poses are deliberately not
accepted by this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VisualOpenConfig:
    """Geometry and temporal-independent parameters for visual Open."""

    displacement_threshold_m: float = 0.04
    bound_quantile: float = 0.05
    min_points: int = 40
    extension_axis: str = "auto"

    def __post_init__(self) -> None:
        if self.displacement_threshold_m <= 0.0:
            raise ValueError("displacement_threshold_m must be positive")
        if not 0.0 < self.bound_quantile < 0.25:
            raise ValueError("bound_quantile must be in (0, 0.25)")
        if self.min_points < 8:
            raise ValueError("min_points must be at least 8")
        if self.extension_axis not in {
            "auto", "x-", "x+", "y-", "y+"
        }:
            raise ValueError(
                "extension_axis must be auto, x-, x+, y-, or y+"
            )


@dataclass(frozen=True)
class VisualOpenEstimate:
    valid: bool
    reason: str = ""
    displacement_m: float = float("nan")
    margin: float = float("nan")
    extension_axis: str = ""
    signed_extension_m: float = float("nan")
    extension_x_minus_m: float = float("nan")
    extension_x_plus_m: float = float("nan")
    extension_y_minus_m: float = float("nan")
    extension_y_plus_m: float = float("nan")
    reference_shift_m: float = float("nan")
    baseline_band_points: int = 0
    current_band_points: int = 0


_LEVEL_BANDS = {
    # Fractions of the cabinet's robust vertical extent. Bands overlap
    # slightly because the visible drawer front and its top lip occupy more
    # than one idealized third of the cabinet.
    "bottom": (0.04, 0.40),
    "middle": (0.28, 0.72),
    "top": (0.60, 0.96),
}


def _finite_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return points[np.all(np.isfinite(points), axis=1)]


def _z_band(
    points: np.ndarray,
    *,
    z_low: float,
    z_high: float,
) -> np.ndarray:
    return points[
        (points[:, 2] >= float(z_low))
        & (points[:, 2] <= float(z_high))
    ]


def visual_open_margin(
    baseline_points_world: np.ndarray,
    current_points_world: np.ndarray,
    *,
    drawer_level: str = "middle",
    config: VisualOpenConfig = VisualOpenConfig(),
) -> VisualOpenEstimate:
    """Estimate drawer extension relative to the reset RGB-D observation.

    A robust XY footprint is measured in the requested drawer-height band.
    The current cloud is first aligned by the median of the cabinet's upper
    fixed region, which removes small camera/calibration/detection shifts.
    Opening is the largest outward expansion of the current footprint beyond
    the reset footprint.  Positive margin means the expansion exceeds the
    configured visual threshold.
    """

    level = str(drawer_level).strip().lower()
    if level not in _LEVEL_BANDS:
        return VisualOpenEstimate(
            valid=False,
            reason=f"unsupported_drawer_level:{level}",
        )

    baseline = _finite_points(baseline_points_world)
    current = _finite_points(current_points_world)
    if len(baseline) < config.min_points:
        return VisualOpenEstimate(
            valid=False,
            reason="insufficient_baseline_points",
        )
    if len(current) < config.min_points:
        return VisualOpenEstimate(
            valid=False,
            reason="insufficient_current_points",
        )

    z05, z95 = np.quantile(baseline[:, 2], (0.05, 0.95))
    z_span = float(z95 - z05)
    if not np.isfinite(z_span) or z_span <= 1.0e-3:
        return VisualOpenEstimate(
            valid=False,
            reason="degenerate_baseline_height",
        )

    low_fraction, high_fraction = _LEVEL_BANDS[level]
    band_low = float(z05 + low_fraction * z_span)
    band_high = float(z05 + high_fraction * z_span)
    baseline_band = _z_band(
        baseline, z_low=band_low, z_high=band_high
    )
    current_band = _z_band(
        current, z_low=band_low, z_high=band_high
    )
    if len(baseline_band) < config.min_points:
        return VisualOpenEstimate(
            valid=False,
            reason="insufficient_baseline_band_points",
            baseline_band_points=len(baseline_band),
            current_band_points=len(current_band),
        )
    if len(current_band) < config.min_points:
        return VisualOpenEstimate(
            valid=False,
            reason="insufficient_current_band_points",
            baseline_band_points=len(baseline_band),
            current_band_points=len(current_band),
        )

    # The top 20% of the cabinet remains fixed for the task20 middle drawer.
    # Its median provides a visual-only reference against small whole-mask
    # shifts. If occlusion leaves too few points, no alignment is applied.
    reference_low = float(z05 + 0.76 * z_span)
    reference_high = float(z05 + 0.98 * z_span)
    baseline_reference = _z_band(
        baseline, z_low=reference_low, z_high=reference_high
    )
    current_reference = _z_band(
        current, z_low=reference_low, z_high=reference_high
    )
    if (
        len(baseline_reference) >= config.min_points
        and len(current_reference) >= config.min_points
    ):
        shift_xy = (
            np.median(current_reference[:, :2], axis=0)
            - np.median(baseline_reference[:, :2], axis=0)
        )
    else:
        shift_xy = np.zeros(2, dtype=np.float64)

    aligned_current_xy = current_band[:, :2] - shift_xy[None, :]
    q = float(config.bound_quantile)
    baseline_low = np.quantile(baseline_band[:, :2], q, axis=0)
    baseline_high = np.quantile(
        baseline_band[:, :2], 1.0 - q, axis=0
    )
    current_low = np.quantile(aligned_current_xy, q, axis=0)
    current_high = np.quantile(
        aligned_current_xy, 1.0 - q, axis=0
    )

    signed_extensions = np.asarray([
        baseline_low[0] - current_low[0],
        current_high[0] - baseline_high[0],
        baseline_low[1] - current_low[1],
        current_high[1] - baseline_high[1],
    ])
    labels = ("x-", "x+", "y-", "y+")
    if config.extension_axis == "auto":
        index = int(np.argmax(signed_extensions))
    else:
        index = labels.index(config.extension_axis)
    signed_extension = float(signed_extensions[index])
    displacement = float(max(0.0, signed_extension))
    margin = float(
        displacement - config.displacement_threshold_m
    )
    return VisualOpenEstimate(
        valid=True,
        displacement_m=displacement,
        margin=margin,
        extension_axis=labels[index],
        signed_extension_m=signed_extension,
        extension_x_minus_m=float(signed_extensions[0]),
        extension_x_plus_m=float(signed_extensions[1]),
        extension_y_minus_m=float(signed_extensions[2]),
        extension_y_plus_m=float(signed_extensions[3]),
        reference_shift_m=float(np.linalg.norm(shift_xy)),
        baseline_band_points=len(baseline_band),
        current_band_points=len(current_band),
    )
