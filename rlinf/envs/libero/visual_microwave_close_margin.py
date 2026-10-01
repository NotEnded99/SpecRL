"""Reset-relative visual Close margins for a microwave door.

The primary estimator anchors an expanded RGB door ROI to the reset-time
YOLOE microwave body mask. The legacy RGB-D edge estimator remains a fallback.
Oracle labels and simulator state are never estimator inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def visual_articulated_class(
    articulated_object: str | None,
    task_description: str,
) -> str:
    """Route the visual articulated object without reading simulator state."""
    text = f"{articulated_object or ''} {task_description or ''}".lower()
    if "microwave" in text:
        return "microwave"
    if "white" in text:
        return "white cabinet"
    return "wooden cabinet"


from rlinf.envs.libero.visual_microwave_diagnostic import MicrowaveROI


@dataclass(frozen=True)
class VisualMicrowaveCloseConfig:
    retraction_threshold_fraction: float = 0.12
    closing_sign: int = 1
    min_area_ratio: float = 0.0
    contraction_threshold_fraction: float = 0.0
    contraction_max_area_ratio: float = 0.0
    min_pixels: int = 80
    max_vertical_shift_fraction: float = 0.12
    # Ignore sparse segmentation pixels at both silhouette extremes.
    edge_quantile: float = 0.02
    # Door-pose estimator.  The reset frame supplies a fixed hinge and the
    # fully-open far edge; later frames measure how far that edge retracts.
    door_close_progress_threshold: float = 0.80
    door_vertical_top_fraction: float = 0.05
    door_vertical_bottom_fraction: float = 0.95
    door_vertical_edge_quantile: float = 0.85
    door_min_edge_contrast: float = 0.08
    door_min_relative_edge_strength: float = 0.65
    door_min_open_span_fraction: float = 0.20
    door_hinge_search_start_fraction: float = 0.15
    door_far_edge_search_start_fraction: float = 0.45
    door_edge_search_slack_fraction: float = 0.12
    door_depth_edge_tolerance_px: int = 8
    door_min_relative_depth_edge_strength: float = 0.30
    # The YOLOE microwave mask covers the fixed body more reliably than the
    # articulated door. Expand that body anchor into the known door sweep and
    # measure the dark door-frame silhouette directly from RGB.
    door_search_width_scale: float = 2.10
    door_search_top_fraction: float = 0.06
    door_search_bottom_fraction: float = 0.91
    door_search_min_top_image_fraction: float = 0.33
    door_dark_intensity_threshold: float = 0.20
    door_dark_body_density_fraction: float = 0.40
    door_dark_min_column_pixels: int = 10
    door_dark_min_open_span_fraction: float = 0.15
    door_dark_max_column_gap: int = 3


@dataclass(frozen=True)
class VisualMicrowaveCloseEstimate:
    valid: bool
    reason: str = ""
    retraction_fraction: float = float("nan")
    margin: float = float("nan")
    baseline_edge_px: float = float("nan")
    current_edge_px: float = float("nan")
    reference_shift_fraction: float = float("nan")
    baseline_panel_pixels: int = 0
    current_panel_pixels: int = 0
    area_ratio: float = float("nan")
    area_margin: float = float("nan")
    expansion_margin: float = float("nan")
    contraction_margin: float = float("nan")


def _bounds(
    mask: np.ndarray,
    roi: MicrowaveROI,
    edge_quantile: float,
):
    crop = mask[roi.y0:roi.y1, roi.x0:roi.x1]
    ys, xs = np.nonzero(crop)
    if not len(xs):
        return None
    q = float(edge_quantile)
    return (
        float(np.quantile(xs, q) + roi.x0),
        float(np.quantile(xs, 1.0 - q) + roi.x0),
        float(np.quantile(ys, q) + roi.y0),
        float(np.quantile(ys, 1.0 - q) + roi.y0),
        len(xs),
    )


def visual_microwave_close_margin(
    reset_microwave_mask: np.ndarray,
    current_microwave_mask: np.ndarray,
    fixed_roi: MicrowaveROI,
    *,
    config: VisualMicrowaveCloseConfig = VisualMicrowaveCloseConfig(),
) -> VisualMicrowaveCloseEstimate:
    if config.closing_sign not in {-1, 1}:
        return VisualMicrowaveCloseEstimate(
            False, "closing_sign_must_be_minus_or_plus_one"
        )
    if not 0.0 <= config.edge_quantile < 0.5:
        return VisualMicrowaveCloseEstimate(
            False, "edge_quantile_must_be_in_half_open_unit_interval"
        )
    baseline = np.asarray(reset_microwave_mask, dtype=bool).squeeze()
    current = np.asarray(current_microwave_mask, dtype=bool).squeeze()
    if baseline.ndim != 2 or current.ndim != 2:
        return VisualMicrowaveCloseEstimate(False, "mask_must_be_2d")
    if baseline.shape != current.shape:
        return VisualMicrowaveCloseEstimate(False, "mask_shape_mismatch")
    if not fixed_roi.valid:
        return VisualMicrowaveCloseEstimate(False, "fixed_roi_invalid")
    base = _bounds(baseline, fixed_roi, config.edge_quantile)
    curr = _bounds(current, fixed_roi, config.edge_quantile)
    if base is None or base[4] < config.min_pixels:
        return VisualMicrowaveCloseEstimate(
            False, "insufficient_baseline_pixels"
        )
    if curr is None or curr[4] < config.min_pixels:
        return VisualMicrowaveCloseEstimate(
            False, "insufficient_current_pixels"
        )
    base_width = max(1.0, base[1] - base[0] + 1.0)
    curr_width = max(1.0, curr[1] - curr[0] + 1.0)
    roi_height = float(max(1, fixed_roi.y1 - fixed_roi.y0))
    vertical_shift = (
        0.5 * (curr[2] + curr[3]) - 0.5 * (base[2] + base[3])
    ) / roi_height
    if abs(vertical_shift) > config.max_vertical_shift_fraction:
        return VisualMicrowaveCloseEstimate(
            False,
            "vertical_shift_exceeds_gate",
            reference_shift_fraction=float(vertical_shift),
            baseline_panel_pixels=base[4],
            current_panel_pixels=curr[4],
        )
    retraction = float((base_width - curr_width) / base_width)
    signed_closing = float(config.closing_sign * retraction)
    retraction_margin = float(
        signed_closing - config.retraction_threshold_fraction
    )
    area_ratio = float(curr[4] / base[4])
    area_margin = float(area_ratio - config.min_area_ratio)
    expansion_margin = (
        min(retraction_margin, area_margin)
        if config.min_area_ratio > 0.0
        else retraction_margin
    )
    contraction_enabled = bool(
        config.contraction_threshold_fraction > 0.0
        and config.contraction_max_area_ratio > 0.0
    )
    contraction_margin = (
        min(
            float(
                retraction - config.contraction_threshold_fraction
            ),
            float(config.contraction_max_area_ratio - area_ratio),
        )
        if contraction_enabled
        else float("-inf")
    )
    margin = max(expansion_margin, contraction_margin)
    return VisualMicrowaveCloseEstimate(
        True,
        retraction_fraction=retraction,
        margin=float(margin),
        baseline_edge_px=base_width,
        current_edge_px=curr_width,
        reference_shift_fraction=float(vertical_shift),
        baseline_panel_pixels=base[4],
        current_panel_pixels=curr[4],
        area_ratio=area_ratio,
        area_margin=area_margin,
        expansion_margin=float(expansion_margin),
        contraction_margin=float(contraction_margin),
    )


def _grayscale(image_rgb: np.ndarray) -> np.ndarray | None:
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        return None
    gray = image.astype(np.float64).mean(axis=2)
    if gray.max(initial=0.0) > 1.5:
        gray /= 255.0
    return gray


def microwave_door_search_roi(
    body_roi: MicrowaveROI,
    image_shape: tuple[int, int],
    config: VisualMicrowaveCloseConfig,
) -> MicrowaveROI:
    if not body_roi.valid:
        return MicrowaveROI(False, "fixed_body_roi_invalid")
    height, width = (int(image_shape[0]), int(image_shape[1]))
    body_width = max(1, int(body_roi.x1) - int(body_roi.x0))
    body_height = max(1, int(body_roi.y1) - int(body_roi.y0))
    x0 = max(0, min(width - 1, int(body_roi.x0)))
    x1 = min(
        width,
        x0 + int(round(config.door_search_width_scale * body_width)),
    )
    y0 = max(
        0,
        min(
            height - 1,
            max(
                int(round(
                    config.door_search_min_top_image_fraction * height
                )),
                int(body_roi.y0)
                + int(round(
                    config.door_search_top_fraction * body_height
                )),
            ),
        ),
    )
    y1 = min(
        height,
        int(body_roi.y0)
        + int(round(config.door_search_bottom_fraction * body_height)),
    )
    if x1 - x0 < 8 or y1 - y0 < 8:
        return MicrowaveROI(False, "expanded_door_roi_too_small")
    return MicrowaveROI(
        True,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        mask_pixels=int(body_roi.mask_pixels),
    )


def _anchored_dark_far_column(
    column_counts: np.ndarray,
    hinge_local: float,
    *,
    min_column_pixels: int,
    max_column_gap: int,
) -> float | None:
    columns = np.flatnonzero(column_counts >= min_column_pixels)
    if not len(columns):
        return None
    components: list[tuple[int, int]] = []
    start = previous = int(columns[0])
    for value in columns[1:]:
        column = int(value)
        if column - previous > max_column_gap:
            components.append((start, previous))
            start = column
        previous = column
    components.append((start, previous))
    anchored = [
        end for start, end in components
        if start <= hinge_local + max_column_gap and end >= hinge_local
    ]
    return float(max(anchored)) if anchored else None


def visual_microwave_dark_door_close_margin(
    reset_image_rgb: np.ndarray,
    current_image_rgb: np.ndarray,
    fixed_body_roi: MicrowaveROI,
    *,
    config: VisualMicrowaveCloseConfig = VisualMicrowaveCloseConfig(),
    baseline_mask_pixels: int = 0,
    current_mask_pixels: int = 0,
) -> VisualMicrowaveCloseEstimate:
    """Measure door closure from its dark frame inside a body-anchored ROI."""
    baseline = _grayscale(reset_image_rgb)
    current = _grayscale(current_image_rgb)
    if baseline is None or current is None:
        return VisualMicrowaveCloseEstimate(False, "door_rgb_invalid")
    if baseline.shape != current.shape:
        return VisualMicrowaveCloseEstimate(False, "door_rgb_shape_mismatch")
    fractions = (
        config.door_search_top_fraction,
        config.door_search_bottom_fraction,
        config.door_dark_intensity_threshold,
        config.door_dark_body_density_fraction,
        config.door_dark_min_open_span_fraction,
        config.door_search_min_top_image_fraction,
    )
    if not 0.0 <= fractions[0] < fractions[1] <= 1.0:
        return VisualMicrowaveCloseEstimate(
            False, "door_search_vertical_fraction_invalid"
        )
    if not all(0.0 < value < 1.0 for value in fractions[2:]):
        return VisualMicrowaveCloseEstimate(
            False, "door_dark_fraction_config_out_of_range"
        )
    if config.door_search_width_scale <= 1.0:
        return VisualMicrowaveCloseEstimate(
            False, "door_search_width_scale_must_exceed_one"
        )
    if not 0.0 < config.door_close_progress_threshold < 1.0:
        return VisualMicrowaveCloseEstimate(
            False, "door_close_progress_threshold_out_of_range"
        )
    if config.door_dark_min_column_pixels < 1:
        return VisualMicrowaveCloseEstimate(
            False, "door_dark_min_column_pixels_must_be_positive"
        )
    if config.door_dark_max_column_gap < 0:
        return VisualMicrowaveCloseEstimate(
            False, "door_dark_max_column_gap_must_be_nonnegative"
        )

    roi = microwave_door_search_roi(
        fixed_body_roi, baseline.shape, config
    )
    if not roi.valid:
        return VisualMicrowaveCloseEstimate(False, roi.reason)
    base_crop = baseline[roi.y0:roi.y1, roi.x0:roi.x1]
    curr_crop = current[roi.y0:roi.y1, roi.x0:roi.x1]
    base_counts = np.count_nonzero(
        base_crop <= config.door_dark_intensity_threshold, axis=0
    )
    curr_counts = np.count_nonzero(
        curr_crop <= config.door_dark_intensity_threshold, axis=0
    )
    roi_height = float(max(1, roi.y1 - roi.y0))
    roi_width = float(max(1, roi.x1 - roi.x0))
    body_density_pixels = (
        config.door_dark_body_density_fraction * roi_height
    )
    dense_body = np.flatnonzero(base_counts >= body_density_pixels)
    dense_body = dense_body[dense_body < int(round(0.55 * roi_width))]
    if not len(dense_body):
        return VisualMicrowaveCloseEstimate(
            False, "baseline_dark_body_anchor_unavailable"
        )
    hinge_local = _anchored_dark_far_column(
        base_counts,
        float(np.min(dense_body)),
        min_column_pixels=body_density_pixels,
        max_column_gap=config.door_dark_max_column_gap,
    )
    if hinge_local is None:
        return VisualMicrowaveCloseEstimate(
            False, "baseline_dark_body_anchor_unavailable"
        )
    baseline_far_local = _anchored_dark_far_column(
        base_counts,
        hinge_local,
        min_column_pixels=config.door_dark_min_column_pixels,
        max_column_gap=config.door_dark_max_column_gap,
    )
    if baseline_far_local is None:
        return VisualMicrowaveCloseEstimate(
            False, "baseline_dark_door_edge_unavailable"
        )
    open_span = baseline_far_local - hinge_local
    minimum_span = config.door_dark_min_open_span_fraction * roi_width
    if open_span < minimum_span:
        return VisualMicrowaveCloseEstimate(
            False, "baseline_dark_door_open_span_too_small"
        )
    current_columns = np.flatnonzero(
        curr_counts >= config.door_dark_min_column_pixels
    )
    current_columns = current_columns[
        (current_columns >= hinge_local)
        & (current_columns <= baseline_far_local)
    ]
    current_far_local = (
        float(np.max(current_columns))
        if len(current_columns)
        else hinge_local
    )
    close_progress = float(
        (baseline_far_local - current_far_local) / open_span
    )
    return VisualMicrowaveCloseEstimate(
        True,
        retraction_fraction=close_progress,
        margin=float(
            close_progress - config.door_close_progress_threshold
        ),
        baseline_edge_px=float(roi.x0 + baseline_far_local),
        current_edge_px=float(roi.x0 + current_far_local),
        reference_shift_fraction=0.0,
        baseline_panel_pixels=int(baseline_mask_pixels),
        current_panel_pixels=int(current_mask_pixels),
    )


def _vertical_edge_profile(
    image_rgb: np.ndarray,
    roi: MicrowaveROI,
    config: VisualMicrowaveCloseConfig,
) -> tuple[np.ndarray, np.ndarray] | None:
    gray = _grayscale(image_rgb)
    if gray is None or not roi.valid:
        return None
    height, width = gray.shape
    x0 = max(0, min(width - 1, int(roi.x0)))
    x1 = max(x0 + 1, min(width, int(roi.x1)))
    y0 = max(0, min(height - 1, int(roi.y0)))
    y1 = max(y0 + 1, min(height, int(roi.y1)))
    roi_height = y1 - y0
    inner_y0 = y0 + int(round(
        roi_height * config.door_vertical_top_fraction
    ))
    inner_y1 = y0 + int(round(
        roi_height * config.door_vertical_bottom_fraction
    ))
    inner_y0 = max(y0, min(y1 - 1, inner_y0))
    inner_y1 = max(inner_y0 + 1, min(y1, inner_y1))
    if x1 - x0 < 5 or inner_y1 - inner_y0 < 5:
        return None

    crop = gray[inner_y0:inner_y1, x0:x1]
    gradient = np.abs(crop[:, 2:] - crop[:, :-2])
    profile = np.quantile(
        gradient,
        config.door_vertical_edge_quantile,
        axis=0,
    )
    # A physical edge normally occupies two adjacent gradient columns.  A
    # three-column max filter removes sensitivity to the exact pixel phase.
    padded = np.pad(profile, (1, 1), mode="edge")
    profile = np.maximum.reduce(
        (padded[:-2], padded[1:-1], padded[2:])
    )
    xs = np.arange(x0 + 1, x1 - 1, dtype=np.int64)
    return xs, profile


def _depth_vertical_edge_profile(
    raw_depth: np.ndarray,
    roi: MicrowaveROI,
    config: VisualMicrowaveCloseConfig,
) -> tuple[np.ndarray, np.ndarray] | None:
    depth = np.asarray(raw_depth, dtype=np.float64).squeeze()
    if depth.ndim != 2 or not roi.valid:
        return None
    finite = np.isfinite(depth)
    if not np.any(finite):
        return None
    y0 = max(0, min(depth.shape[0] - 1, int(roi.y0)))
    y1 = max(y0 + 1, min(depth.shape[0], int(roi.y1)))
    x0 = max(0, min(depth.shape[1] - 1, int(roi.x0)))
    x1 = max(x0 + 1, min(depth.shape[1], int(roi.x1)))
    crop = depth[y0:y1, x0:x1]
    crop_finite = np.isfinite(crop)
    if np.count_nonzero(crop_finite) < 16:
        return None
    low, high = np.quantile(crop[crop_finite], (0.02, 0.98))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return None
    normalized = np.zeros_like(depth, dtype=np.float64)
    normalized[finite] = np.clip(
        (depth[finite] - low) / (high - low), 0.0, 1.0
    )
    image = np.repeat(normalized[..., None], 3, axis=2)
    return _vertical_edge_profile(image, roi, config)


def visual_microwave_door_edge_close_margin(
    reset_image_rgb: np.ndarray,
    current_image_rgb: np.ndarray,
    fixed_roi: MicrowaveROI,
    *,
    config: VisualMicrowaveCloseConfig = VisualMicrowaveCloseConfig(),
    baseline_mask_pixels: int = 0,
    current_mask_pixels: int = 0,
    reset_raw_depth: np.ndarray | None = None,
    current_raw_depth: np.ndarray | None = None,
) -> VisualMicrowaveCloseEstimate:
    """Estimate microwave closure from the door far edge, not mask area.

    The camera is fixed in LIBERO.  At reset the open door provides two
    stable vertical edges: its hinge and its far edge.  Closure retracts the
    far edge toward the hinge.  A missing edge is unknown rather than closed,
    which prevents detector dropouts from becoming positive evidence.
    """
    fractions = (
        config.door_vertical_top_fraction,
        config.door_vertical_bottom_fraction,
        config.door_vertical_edge_quantile,
        config.door_min_relative_edge_strength,
        config.door_min_open_span_fraction,
        config.door_hinge_search_start_fraction,
        config.door_far_edge_search_start_fraction,
        config.door_edge_search_slack_fraction,
        config.door_min_relative_depth_edge_strength,
    )
    if not 0.0 < config.door_close_progress_threshold < 1.0:
        return VisualMicrowaveCloseEstimate(
            False, "door_close_progress_threshold_out_of_range"
        )
    if not 0.0 <= config.door_vertical_top_fraction < (
        config.door_vertical_bottom_fraction
    ) <= 1.0:
        return VisualMicrowaveCloseEstimate(
            False, "door_vertical_crop_invalid"
        )
    if not all(0.0 <= value <= 1.0 for value in fractions[2:]):
        return VisualMicrowaveCloseEstimate(
            False, "door_fraction_config_out_of_range"
        )
    if not 0.5 <= config.door_vertical_edge_quantile < 1.0:
        return VisualMicrowaveCloseEstimate(
            False, "door_vertical_edge_quantile_out_of_range"
        )
    if config.door_min_edge_contrast <= 0.0:
        return VisualMicrowaveCloseEstimate(
            False, "door_min_edge_contrast_must_be_positive"
        )
    if config.door_depth_edge_tolerance_px < 0:
        return VisualMicrowaveCloseEstimate(
            False, "door_depth_edge_tolerance_px_must_be_nonnegative"
        )
    if (reset_raw_depth is None) != (current_raw_depth is None):
        return VisualMicrowaveCloseEstimate(
            False, "door_depth_pair_incomplete"
        )
    if not fixed_roi.valid:
        return VisualMicrowaveCloseEstimate(False, "fixed_roi_invalid")

    baseline = _vertical_edge_profile(
        reset_image_rgb, fixed_roi, config
    )
    current = _vertical_edge_profile(
        current_image_rgb, fixed_roi, config
    )
    if baseline is None or current is None:
        return VisualMicrowaveCloseEstimate(
            False, "door_image_or_roi_invalid"
        )
    base_x, base_profile = baseline
    curr_x, curr_profile = current
    if not len(base_x) or not len(curr_x):
        return VisualMicrowaveCloseEstimate(
            False, "door_edge_profile_empty"
        )

    roi_width = float(max(1, fixed_roi.x1 - fixed_roi.x0))
    base_peak = float(np.max(base_profile))
    edge_threshold = max(
        config.door_min_edge_contrast,
        config.door_min_relative_edge_strength * base_peak,
    )
    far_search_start = (
        fixed_roi.x0
        + config.door_far_edge_search_start_fraction * roi_width
    )
    far_candidates = base_x[
        (base_x >= far_search_start)
        & (base_profile >= edge_threshold)
    ]
    if not len(far_candidates):
        return VisualMicrowaveCloseEstimate(
            False, "baseline_door_far_edge_unavailable"
        )
    baseline_far_edge = float(np.max(far_candidates))

    minimum_span = config.door_min_open_span_fraction * roi_width
    hinge_search_start = (
        fixed_roi.x0
        + config.door_hinge_search_start_fraction * roi_width
    )
    hinge_mask = (
        (base_x >= hinge_search_start)
        & (base_x <= baseline_far_edge - minimum_span)
        & (base_profile >= edge_threshold)
    )
    if not np.any(hinge_mask):
        return VisualMicrowaveCloseEstimate(
            False, "baseline_door_hinge_edge_unavailable"
        )
    hinge_indices = np.flatnonzero(hinge_mask)
    hinge_index = hinge_indices[
        int(np.argmax(base_profile[hinge_indices]))
    ]
    hinge_edge = float(base_x[hinge_index])
    open_span = baseline_far_edge - hinge_edge
    if open_span < minimum_span:
        return VisualMicrowaveCloseEstimate(
            False, "baseline_door_open_span_too_small"
        )

    current_search_end = min(
        float(fixed_roi.x1 - 1),
        baseline_far_edge
        + config.door_edge_search_slack_fraction * roi_width,
    )
    current_candidates = curr_x[
        (curr_x >= hinge_edge)
        & (curr_x <= current_search_end)
        & (curr_profile >= edge_threshold)
    ]
    if not len(current_candidates):
        return VisualMicrowaveCloseEstimate(
            False,
            "current_door_far_edge_unavailable",
            baseline_edge_px=baseline_far_edge,
            baseline_panel_pixels=int(baseline_mask_pixels),
            current_panel_pixels=int(current_mask_pixels),
        )
    if reset_raw_depth is not None:
        baseline_depth = _depth_vertical_edge_profile(
            reset_raw_depth, fixed_roi, config
        )
        current_depth = _depth_vertical_edge_profile(
            current_raw_depth, fixed_roi, config
        )
        if baseline_depth is None or current_depth is None:
            return VisualMicrowaveCloseEstimate(
                False, "door_depth_edge_profile_unavailable"
            )
        base_depth_x, base_depth_profile = baseline_depth
        curr_depth_x, curr_depth_profile = current_depth
        tolerance = float(config.door_depth_edge_tolerance_px)
        base_near = np.abs(base_depth_x - baseline_far_edge) <= tolerance
        if not np.any(base_near):
            return VisualMicrowaveCloseEstimate(
                False, "baseline_door_depth_edge_unavailable"
            )
        baseline_depth_strength = float(
            np.max(base_depth_profile[base_near])
        )
        if baseline_depth_strength <= 0.0:
            return VisualMicrowaveCloseEstimate(
                False, "baseline_door_depth_edge_unavailable"
            )
        depth_threshold = (
            config.door_min_relative_depth_edge_strength
            * baseline_depth_strength
        )
        supported_candidates = []
        for candidate in current_candidates:
            near = np.abs(curr_depth_x - float(candidate)) <= tolerance
            if np.any(near) and float(
                np.max(curr_depth_profile[near])
            ) >= depth_threshold:
                supported_candidates.append(float(candidate))
        if not supported_candidates:
            return VisualMicrowaveCloseEstimate(
                False,
                "current_door_depth_edge_unavailable",
                baseline_edge_px=baseline_far_edge,
                baseline_panel_pixels=int(baseline_mask_pixels),
                current_panel_pixels=int(current_mask_pixels),
            )
        current_candidates = np.asarray(
            supported_candidates, dtype=np.float64
        )
    current_far_edge = float(np.max(current_candidates))
    close_progress = float(
        (baseline_far_edge - current_far_edge) / open_span
    )
    margin = float(
        close_progress - config.door_close_progress_threshold
    )
    return VisualMicrowaveCloseEstimate(
        True,
        retraction_fraction=close_progress,
        margin=margin,
        baseline_edge_px=baseline_far_edge,
        current_edge_px=current_far_edge,
        reference_shift_fraction=0.0,
        baseline_panel_pixels=int(baseline_mask_pixels),
        current_panel_pixels=int(current_mask_pixels),
    )
