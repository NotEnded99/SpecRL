"""Strict visual-only observability probe for LIBERO stove TurnOn.

It measures reset-relative RGB/RGB-D changes inside a stove region fixed from
the first visible frame.  It also exposes a strict visual-only TurnOn margin
from the red burner-ring indicator.  Oracle joint margins are not accepted by
any estimator here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VisualTurnOnConfig:
    min_mask_pixels: int = 40
    roi_padding_fraction: float = 0.12
    grid_rows: int = 3
    grid_cols: int = 4
    diagnostic_interval: int = 5
    blocked_global_change: float = 0.010
    blocked_cell_change: float = 0.020
    red_min_intensity: float = 0.45
    red_excess_threshold: float = 0.18
    red_fraction_threshold: float = 0.010
    confirmation_steps: int = 3

    def __post_init__(self) -> None:
        if self.min_mask_pixels < 1:
            raise ValueError("min_mask_pixels must be positive")
        if not 0.0 <= self.roi_padding_fraction <= 0.5:
            raise ValueError("roi_padding_fraction must be in [0, 0.5]")
        if self.grid_rows < 1 or self.grid_cols < 1:
            raise ValueError("grid dimensions must be positive")
        if self.diagnostic_interval < 1:
            raise ValueError("diagnostic_interval must be positive")
        if not 0.0 <= self.red_min_intensity <= 1.0:
            raise ValueError("red_min_intensity must be in [0, 1]")
        if not 0.0 <= self.red_excess_threshold <= 1.0:
            raise ValueError("red_excess_threshold must be in [0, 1]")
        if not 0.0 < self.red_fraction_threshold < 1.0:
            raise ValueError("red_fraction_threshold must be in (0, 1)")
        if self.confirmation_steps < 1:
            raise ValueError("confirmation_steps must be positive")


@dataclass(frozen=True)
class VisualTurnOnROI:
    valid: bool
    reason: str = ""
    x0: int = 0
    y0: int = 0
    x1: int = 0
    y1: int = 0
    mask_pixels: int = 0


@dataclass(frozen=True)
class VisualTurnOnFeatures:
    valid: bool
    reason: str = ""
    rgb_mae: float = float("nan")
    gray_mae: float = float("nan")
    edge_mae: float = float("nan")
    depth_mae: float = float("nan")
    max_cell_change: float = float("nan")
    max_cell_row: int = -1
    max_cell_col: int = -1


@dataclass(frozen=True)
class VisualTurnOnMargin:
    valid: bool
    reason: str = ""
    red_fraction: float = float("nan")
    margin: float = float("nan")


def visual_turnon_red_margin(
    current_rgb: np.ndarray,
    roi: VisualTurnOnROI,
    *,
    red_min_intensity: float = 0.45,
    red_excess_threshold: float = 0.18,
    red_fraction_threshold: float = 0.010,
) -> VisualTurnOnMargin:
    """Signed TurnOn margin from the red burner ring inside a visual ROI.

    Positive means that the fraction of red-dominant pixels exceeds the
    calibrated threshold.  Inputs are restricted to the current RGB image
    and a fixed ROI obtained from the YOLOE stove mask.
    """
    if not roi.valid:
        return VisualTurnOnMargin(False, reason=roi.reason)
    try:
        image = _as_rgb(current_rgb)
    except ValueError as error:
        return VisualTurnOnMargin(False, reason=str(error))
    crop = image[roi.y0:roi.y1, roi.x0:roi.x1]
    if crop.size == 0:
        return VisualTurnOnMargin(False, reason="empty_stove_roi")
    red_excess = crop[..., 0] - np.maximum(crop[..., 1], crop[..., 2])
    red_pixels = (
        (crop[..., 0] >= float(red_min_intensity))
        & (red_excess >= float(red_excess_threshold))
    )
    fraction = float(np.mean(red_pixels))
    return VisualTurnOnMargin(
        True,
        red_fraction=fraction,
        margin=float(fraction - red_fraction_threshold),
    )


def stove_roi_from_mask(
    mask: np.ndarray,
    *,
    padding_fraction: float = 0.12,
    min_mask_pixels: int = 40,
) -> VisualTurnOnROI:
    """Build one fixed image-space stove ROI from a visual mask."""
    binary = np.asarray(mask, dtype=bool).squeeze()
    if binary.ndim != 2:
        return VisualTurnOnROI(False, reason="mask_must_be_2d")
    ys, xs = np.nonzero(binary)
    if len(xs) < int(min_mask_pixels):
        return VisualTurnOnROI(
            False,
            reason="insufficient_stove_mask_pixels",
            mask_pixels=int(len(xs)),
        )
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    pad_x = int(round((right - left) * float(padding_fraction)))
    pad_y = int(round((bottom - top) * float(padding_fraction)))
    return VisualTurnOnROI(
        True,
        x0=max(0, left - pad_x),
        y0=max(0, top - pad_y),
        x1=min(binary.shape[1], right + pad_x),
        y1=min(binary.shape[0], bottom + pad_y),
        mask_pixels=int(len(xs)),
    )


def _as_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float64)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("image must have shape HxWx3")
    if array.max(initial=0.0) > 1.5:
        array = array / 255.0
    return np.clip(array, 0.0, 1.0)


def _gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(gray)
    return np.sqrt(gx * gx + gy * gy)


def reset_relative_features(
    baseline_rgb: np.ndarray,
    current_rgb: np.ndarray,
    roi: VisualTurnOnROI,
    *,
    baseline_depth: np.ndarray | None = None,
    current_depth: np.ndarray | None = None,
    grid_rows: int = 3,
    grid_cols: int = 4,
) -> VisualTurnOnFeatures:
    """Measure image change without consuming actions, joints, or state."""
    if not roi.valid:
        return VisualTurnOnFeatures(False, reason=roi.reason)
    base = _as_rgb(baseline_rgb)
    current = _as_rgb(current_rgb)
    if base.shape != current.shape:
        return VisualTurnOnFeatures(False, reason="image_shape_changed")
    crop0 = base[roi.y0:roi.y1, roi.x0:roi.x1]
    crop1 = current[roi.y0:roi.y1, roi.x0:roi.x1]
    if crop0.size == 0:
        return VisualTurnOnFeatures(False, reason="empty_stove_roi")
    difference = np.mean(np.abs(crop1 - crop0), axis=2)
    gray0 = np.mean(crop0, axis=2)
    gray1 = np.mean(crop1, axis=2)
    edge_mae = float(np.mean(np.abs(
        _gradient_magnitude(gray1) - _gradient_magnitude(gray0)
    )))

    cell_values: list[tuple[float, int, int]] = []
    y_edges = np.linspace(0, difference.shape[0], grid_rows + 1, dtype=int)
    x_edges = np.linspace(0, difference.shape[1], grid_cols + 1, dtype=int)
    for row in range(grid_rows):
        for col in range(grid_cols):
            cell = difference[
                y_edges[row]:y_edges[row + 1],
                x_edges[col]:x_edges[col + 1],
            ]
            if cell.size:
                cell_values.append((float(np.mean(cell)), row, col))
    max_change, max_row, max_col = max(cell_values, default=(0.0, -1, -1))

    depth_mae = float("nan")
    if baseline_depth is not None and current_depth is not None:
        depth0 = np.asarray(baseline_depth, dtype=np.float64).squeeze()
        depth1 = np.asarray(current_depth, dtype=np.float64).squeeze()
        if depth0.shape == base.shape[:2] and depth1.shape == base.shape[:2]:
            dep0 = depth0[roi.y0:roi.y1, roi.x0:roi.x1]
            dep1 = depth1[roi.y0:roi.y1, roi.x0:roi.x1]
            valid = np.isfinite(dep0) & np.isfinite(dep1)
            if np.any(valid):
                depth_mae = float(np.mean(np.abs(dep1[valid] - dep0[valid])))

    return VisualTurnOnFeatures(
        True,
        rgb_mae=float(np.mean(np.abs(crop1 - crop0))),
        gray_mae=float(np.mean(np.abs(gray1 - gray0))),
        edge_mae=edge_mae,
        depth_mae=depth_mae,
        max_cell_change=max_change,
        max_cell_row=max_row,
        max_cell_col=max_col,
    )


def make_turnon_overlay(
    image_rgb: np.ndarray,
    roi: VisualTurnOnROI,
    *,
    grid_rows: int = 3,
    grid_cols: int = 4,
) -> np.ndarray:
    """Draw the fixed ROI/grid used by the visual observability probe."""
    image = _as_rgb(image_rgb)
    output = np.asarray(np.round(image * 255.0), dtype=np.uint8)
    if not roi.valid:
        return output
    yellow = np.array([255, 220, 0], dtype=np.uint8)
    cyan = np.array([0, 220, 255], dtype=np.uint8)
    x0, y0, x1, y1 = roi.x0, roi.y0, roi.x1, roi.y1
    output[y0:min(y0 + 2, y1), x0:x1] = yellow
    output[max(y1 - 2, y0):y1, x0:x1] = yellow
    output[y0:y1, x0:min(x0 + 2, x1)] = yellow
    output[y0:y1, max(x1 - 2, x0):x1] = yellow
    for row in range(1, grid_rows):
        y = y0 + int(round(row * (y1 - y0) / grid_rows))
        output[max(y - 1, y0):min(y + 1, y1), x0:x1] = cyan
    for col in range(1, grid_cols):
        x = x0 + int(round(col * (x1 - x0) / grid_cols))
        output[y0:y1, max(x - 1, x0):min(x + 1, x1)] = cyan
    return output


__all__ = [
    "VisualTurnOnConfig",
    "VisualTurnOnFeatures",
    "VisualTurnOnMargin",
    "VisualTurnOnROI",
    "make_turnon_overlay",
    "reset_relative_features",
    "stove_roi_from_mask",
    "visual_turnon_red_margin",
]
