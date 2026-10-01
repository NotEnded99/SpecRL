"""Vision-only fixed-ROI diagnostics for task39 microwave state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MicrowaveROI:
    valid: bool
    reason: str = ""
    x0: int = 0
    y0: int = 0
    x1: int = 0
    y1: int = 0
    mask_pixels: int = 0


def microwave_roi(mask: np.ndarray, padding_fraction: float = 0.08) -> MicrowaveROI:
    binary = np.asarray(mask, dtype=bool).squeeze()
    if binary.ndim != 2:
        return MicrowaveROI(False, "mask_must_be_2d")
    ys, xs = np.nonzero(binary)
    if len(xs) < 16:
        return MicrowaveROI(
            False, "insufficient_mask_pixels", mask_pixels=len(xs)
        )
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    pad_x = int(round((right - left) * padding_fraction))
    pad_y = int(round((bottom - top) * padding_fraction))
    return MicrowaveROI(
        True,
        x0=max(0, left - pad_x),
        y0=max(0, top - pad_y),
        x1=min(binary.shape[1], right + pad_x),
        y1=min(binary.shape[0], bottom + pad_y),
        mask_pixels=len(xs),
    )


def make_microwave_overlay(
    image_rgb: np.ndarray,
    microwave_mask: np.ndarray,
    mug_mask: np.ndarray,
    roi: MicrowaveROI,
) -> np.ndarray:
    image = np.asarray(image_rgb)
    appliance = np.asarray(microwave_mask, dtype=bool).squeeze()
    mug = np.asarray(mug_mask, dtype=bool).squeeze()
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_rgb must have shape HxWx3")
    if appliance.shape != image.shape[:2] or mug.shape != image.shape[:2]:
        raise ValueError("mask and image dimensions must match")
    output = image.astype(np.float32, copy=True)
    if output.max(initial=0.0) <= 1.0:
        output *= 255.0
    output[appliance] = (
        0.6 * output[appliance] + 0.4 * np.array([0, 220, 255])
    )
    output[mug] = 0.5 * output[mug] + 0.5 * np.array([255, 0, 180])
    if roi.valid:
        x0, y0, x1, y1 = roi.x0, roi.y0, roi.x1, roi.y1
        yellow = np.array([255, 220, 0], dtype=np.float32)
        output[y0:min(y0 + 2, y1), x0:x1] = yellow
        output[max(y1 - 2, y0):y1, x0:x1] = yellow
        output[y0:y1, x0:min(x0 + 2, x1)] = yellow
        output[y0:y1, max(x1 - 2, x0):x1] = yellow
    return np.clip(output, 0, 255).astype(np.uint8)


def make_microwave_depth(
    raw_depth: np.ndarray,
    roi: MicrowaveROI,
) -> np.ndarray:
    depth = np.asarray(raw_depth, dtype=np.float64).squeeze()
    if depth.ndim != 2:
        raise ValueError("raw_depth must be two-dimensional")
    finite = np.isfinite(depth)
    output = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if np.any(finite):
        low, high = np.quantile(depth[finite], (0.02, 0.98))
        if high > low:
            gray = np.where(
                finite,
                255 * np.clip((depth - low) / (high - low), 0, 1),
                0,
            ).astype(np.uint8)
            output[:] = gray[..., None]
    if roi.valid:
        x0, y0, x1, y1 = roi.x0, roi.y0, roi.x1, roi.y1
        yellow = np.array([255, 220, 0], dtype=np.uint8)
        output[y0:min(y0 + 2, y1), x0:x1] = yellow
        output[max(y1 - 2, y0):y1, x0:x1] = yellow
        output[y0:y1, x0:min(x0 + 2, x1)] = yellow
        output[y0:y1, max(x1 - 2, x0):x1] = yellow
    return output
