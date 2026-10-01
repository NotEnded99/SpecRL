"""Visual Close margin from a reset-locked drawer-panel image ROI.

Only YOLOE cabinet masks are consumed.  The reset mask supplies a fixed ROI;
the current lower-mask edge is aligned by a fixed upper-cabinet reference.
No joint, site, contact, action, object pose, or oracle value is accepted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rlinf.envs.libero.visual_drawer_panel_diagnostic import DrawerPanelROI


@dataclass(frozen=True)
class VisualDrawerCloseConfig:
    retraction_threshold_fraction: float = 0.25
    edge_quantile: float = 0.98
    min_panel_pixels: int = 80
    min_reference_pixels: int = 80
    max_reference_shift_fraction: float = 0.10

    def __post_init__(self) -> None:
        if not 0.0 < self.retraction_threshold_fraction < 1.0:
            raise ValueError("retraction_threshold_fraction must be in (0, 1)")
        if not 0.5 < self.edge_quantile < 1.0:
            raise ValueError("edge_quantile must be in (0.5, 1)")
        if self.min_panel_pixels < 8 or self.min_reference_pixels < 8:
            raise ValueError("minimum pixel counts must be at least 8")
        if not 0.0 < self.max_reference_shift_fraction < 0.5:
            raise ValueError("max_reference_shift_fraction must be in (0, .5)")


@dataclass(frozen=True)
class VisualDrawerCloseEstimate:
    valid: bool
    reason: str = ""
    retraction_fraction: float = float("nan")
    margin: float = float("nan")
    baseline_edge_px: float = float("nan")
    current_edge_px: float = float("nan")
    reference_shift_fraction: float = float("nan")
    baseline_panel_pixels: int = 0
    current_panel_pixels: int = 0


def _reference_center_x(mask: np.ndarray, roi: DrawerPanelROI) -> tuple[float, int]:
    # Use the cabinet above the drawer ROI as the fixed visual reference.
    upper = mask[: roi.y0, roi.x0 : roi.x1]
    ys, xs = np.nonzero(upper)
    del ys
    return (
        float(np.median(xs) + roi.x0) if len(xs) else float("nan"),
        int(len(xs)),
    )


def visual_drawer_close_margin(
    reset_cabinet_mask: np.ndarray,
    current_cabinet_mask: np.ndarray,
    fixed_roi: DrawerPanelROI,
    *,
    config: VisualDrawerCloseConfig = VisualDrawerCloseConfig(),
) -> VisualDrawerCloseEstimate:
    """Return positive margin when the lower cabinet-mask edge retracts."""
    baseline = np.asarray(reset_cabinet_mask, dtype=bool).squeeze()
    current = np.asarray(current_cabinet_mask, dtype=bool).squeeze()
    if baseline.ndim != 2 or current.ndim != 2:
        return VisualDrawerCloseEstimate(False, "mask_must_be_2d")
    if baseline.shape != current.shape:
        return VisualDrawerCloseEstimate(False, "mask_shape_mismatch")
    if not fixed_roi.valid:
        return VisualDrawerCloseEstimate(False, "fixed_roi_invalid")

    x0, y0, x1, y1 = (
        fixed_roi.x0,
        fixed_roi.y0,
        fixed_roi.x1,
        fixed_roi.y1,
    )
    base_panel = baseline[y0:y1, x0:x1]
    curr_panel = current[y0:y1, x0:x1]
    _, base_x = np.nonzero(base_panel)
    _, curr_x = np.nonzero(curr_panel)
    if len(base_x) < config.min_panel_pixels:
        return VisualDrawerCloseEstimate(
            False, "insufficient_baseline_panel_pixels",
            baseline_panel_pixels=len(base_x),
            current_panel_pixels=len(curr_x),
        )
    if len(curr_x) < config.min_panel_pixels:
        return VisualDrawerCloseEstimate(
            False, "insufficient_current_panel_pixels",
            baseline_panel_pixels=len(base_x),
            current_panel_pixels=len(curr_x),
        )

    base_ref, base_ref_count = _reference_center_x(baseline, fixed_roi)
    curr_ref, curr_ref_count = _reference_center_x(current, fixed_roi)
    if base_ref_count < config.min_reference_pixels:
        return VisualDrawerCloseEstimate(False, "insufficient_baseline_reference_pixels")
    if curr_ref_count < config.min_reference_pixels:
        return VisualDrawerCloseEstimate(False, "insufficient_current_reference_pixels")

    width = float(max(1, x1 - x0))
    reference_shift_px = curr_ref - base_ref
    reference_shift_fraction = float(reference_shift_px / width)
    if abs(reference_shift_fraction) > config.max_reference_shift_fraction:
        return VisualDrawerCloseEstimate(
            False,
            "reference_shift_exceeds_gate",
            reference_shift_fraction=reference_shift_fraction,
            baseline_panel_pixels=len(base_x),
            current_panel_pixels=len(curr_x),
        )

    baseline_edge = float(
        np.quantile(base_x + x0, config.edge_quantile)
    )
    current_edge = float(
        np.quantile(curr_x + x0, config.edge_quantile)
        - reference_shift_px
    )
    retraction = float((baseline_edge - current_edge) / width)
    return VisualDrawerCloseEstimate(
        True,
        retraction_fraction=retraction,
        margin=float(retraction - config.retraction_threshold_fraction),
        baseline_edge_px=baseline_edge,
        current_edge_px=current_edge,
        reference_shift_fraction=reference_shift_fraction,
        baseline_panel_pixels=len(base_x),
        current_panel_pixels=len(curr_x),
    )
