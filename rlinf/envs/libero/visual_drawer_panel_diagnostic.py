"""Read-only diagnostics for a visually tracked drawer front.

The helpers in this module only operate on an RGB image and a segmentation
mask.  They deliberately do not accept simulator joints, sites, contacts,
actions, or object poses.  The output is for offline inspection and never
enters an STL margin or reward.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DrawerPanelROI:
    valid: bool
    reason: str = ""
    x0: int = 0
    y0: int = 0
    x1: int = 0
    y1: int = 0
    mask_pixels: int = 0
    roi_mask_pixels: int = 0


def drawer_panel_roi(
    mask: np.ndarray,
    *,
    drawer_level: str = "bottom",
    horizontal_padding_fraction: float = 0.05,
) -> DrawerPanelROI:
    """Return an image-space ROI for a drawer front inside a cabinet mask.

    This is intentionally only a proposal region.  It is not treated as a
    drawer-state estimate.  The bottom band is broad so the saved diagnostic
    contains the drawer front, handle, and open-cavity boundary.
    """
    binary = np.asarray(mask, dtype=bool).squeeze()
    if binary.ndim != 2:
        return DrawerPanelROI(False, reason="mask_must_be_2d")
    ys, xs = np.nonzero(binary)
    if len(xs) < 16:
        return DrawerPanelROI(
            False, reason="insufficient_mask_pixels", mask_pixels=len(xs)
        )

    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    width = max(1, right - left)
    height = max(1, bottom - top)
    pad = int(round(horizontal_padding_fraction * width))

    level = str(drawer_level).strip().lower()
    bands = {
        "bottom": (0.52, 1.00),
        "middle": (0.28, 0.76),
        "top": (0.00, 0.52),
    }
    if level not in bands:
        return DrawerPanelROI(
            False,
            reason=f"unsupported_drawer_level:{level}",
            mask_pixels=len(xs),
        )
    low, high = bands[level]
    y0 = top + int(round(low * height))
    y1 = top + int(round(high * height))
    x0 = max(0, left - pad)
    x1 = min(binary.shape[1], right + pad)
    y0 = max(0, min(binary.shape[0] - 1, y0))
    y1 = max(y0 + 1, min(binary.shape[0], y1))
    roi_pixels = int(binary[y0:y1, x0:x1].sum())
    return DrawerPanelROI(
        True,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        mask_pixels=len(xs),
        roi_mask_pixels=roi_pixels,
    )


def make_drawer_panel_overlay(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    roi: DrawerPanelROI,
) -> np.ndarray:
    """Overlay the cabinet mask and proposed drawer-front ROI."""
    image = np.asarray(image_rgb)
    binary = np.asarray(mask, dtype=bool).squeeze()
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_rgb must have shape HxWx3")
    if binary.shape != image.shape[:2]:
        raise ValueError("mask and image dimensions must match")
    output = np.asarray(image, dtype=np.float32).copy()
    if output.max(initial=0.0) <= 1.0:
        output *= 255.0
    cyan = np.array([0.0, 220.0, 255.0], dtype=np.float32)
    output[binary] = 0.60 * output[binary] + 0.40 * cyan

    if roi.valid:
        yellow = np.array([255.0, 220.0, 0.0], dtype=np.float32)
        thickness = 2
        x0, y0, x1, y1 = roi.x0, roi.y0, roi.x1, roi.y1
        output[y0:min(y0 + thickness, y1), x0:x1] = yellow
        output[max(y1 - thickness, y0):y1, x0:x1] = yellow
        output[y0:y1, x0:min(x0 + thickness, x1)] = yellow
        output[y0:y1, max(x1 - thickness, x0):x1] = yellow
    return np.clip(output, 0.0, 255.0).astype(np.uint8)


def make_depth_diagnostic(
    raw_depth: np.ndarray,
    roi: DrawerPanelROI,
) -> np.ndarray:
    """Render a robust grayscale depth view with the fixed ROI in yellow."""
    depth = np.asarray(raw_depth, dtype=np.float64).squeeze()
    if depth.ndim != 2:
        raise ValueError("raw_depth must be two-dimensional")
    finite = np.isfinite(depth)
    output = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if np.any(finite):
        low, high = np.quantile(depth[finite], (0.02, 0.98))
        if high > low:
            scaled = np.clip((depth - low) / (high - low), 0.0, 1.0)
            gray = np.where(finite, 255.0 * scaled, 0.0).astype(np.uint8)
            output[:] = gray[..., None]
    if roi.valid:
        x0, y0, x1, y1 = roi.x0, roi.y0, roi.x1, roi.y1
        yellow = np.array([255, 220, 0], dtype=np.uint8)
        output[y0:min(y0 + 2, y1), x0:x1] = yellow
        output[max(y1 - 2, y0):y1, x0:x1] = yellow
        output[y0:y1, x0:min(x0 + 2, x1)] = yellow
        output[y0:y1, max(x1 - 2, x0):x1] = yellow
    return output
