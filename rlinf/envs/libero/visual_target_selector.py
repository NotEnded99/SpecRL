"""Language-conditioned, vision-only target instance selection.

This module uses only the task description and YOLOE detections lifted to 3-D
from RGB-D. It never reads simulator object poses or oracle state.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


_CLASS_ALIASES = {
    "cookie box": ("cookie box", "cookies"),
    "wooden cabinet": ("wooden cabinet", "storage box"),
    "plate": ("plate",),
    "ramekin": ("ramekin",),
    "stove": ("stove",),
}


def _confidence(item: dict[str, Any]) -> float:
    return float(item["detection"].get("confidence", 0.0))


def _center_world(item: dict[str, Any]) -> np.ndarray:
    return np.asarray(item["estimate"].center_world, dtype=np.float64)


def _mask_geometry(item: dict[str, Any]):
    mask = np.asarray(item["detection"].get("mask"))
    if mask.ndim != 2:
        return None
    rows, cols = np.nonzero(mask > 0)
    if rows.size == 0:
        return None
    centroid = np.array([float(cols.mean()), float(rows.mean())])
    bbox = (int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max()))
    return centroid, bbox, (int(mask.shape[0]), int(mask.shape[1]))


def _class_items(estimated: Iterable[dict[str, Any]], class_name: str):
    aliases = _CLASS_ALIASES.get(class_name, (class_name,))
    return [
        item for item in estimated
        if item["detection"].get("class_name") in aliases
    ]


def _best_reference(estimated: list[dict[str, Any]], class_name: str):
    references = _class_items(estimated, class_name)
    return max(references, key=_confidence) if references else None


def _nearest_xy(candidates, reference):
    reference_xy = _center_world(reference)[:2]
    return min(
        candidates,
        key=lambda item: float(
            np.linalg.norm(_center_world(item)[:2] - reference_xy)
        ),
    )


def _image_bbox_gap(first: dict[str, Any], second: dict[str, Any]) -> float:
    """Return pixel gap between masks' bounding boxes (zero when touching)."""
    first_geometry = _mask_geometry(first)
    second_geometry = _mask_geometry(second)
    if first_geometry is None or second_geometry is None:
        return float("inf")
    _, (ax0, ay0, ax1, ay1), _ = first_geometry
    _, (bx0, by0, bx1, by1), _ = second_geometry
    dx = max(bx0 - ax1, ax0 - bx1, 0)
    dy = max(by0 - ay1, ay0 - by1, 0)
    return float(np.hypot(dx, dy))


def _select_on_ramekin(candidates, estimated):
    """Select the bowl supported by the ramekin using visual geometry.

    When the ramekin is visible, horizontal 3-D proximity is the principal
    cue; vertical ordering and mask proximity break near ties. If YOLOE misses
    the ramekin in the initialization frame, the supported bowl should be the
    elevated bowl, so world-Z supplies a vision-only fallback.
    """
    ramekin = _best_reference(estimated, "ramekin")
    if ramekin is None:
        return (
            max(candidates, key=lambda item: (_center_world(item)[2], _confidence(item))),
            "on_ramekin_highest_z_rgbd",
        )

    reference_center = _center_world(ramekin)

    def score(item):
        center = _center_world(item)
        xy_distance = float(np.linalg.norm(center[:2] - reference_center[:2]))
        # A bowl supported by the ramekin should not lie substantially below
        # its center. Keep this as a soft penalty because RGB-D centers are
        # noisy and class geometry differs.
        below_penalty = max(float(reference_center[2] - center[2]) - 0.02, 0.0)
        image_gap = _image_bbox_gap(item, ramekin)
        if not np.isfinite(image_gap):
            image_gap = 1e3
        return (xy_distance + 4.0 * below_penalty, image_gap, -_confidence(item))

    return min(candidates, key=score), "on_ramekin_rgbd_support"


def _nearest_image_landmark(candidates, landmark_xy):
    scored = []
    for item in candidates:
        geometry = _mask_geometry(item)
        if geometry is not None:
            scored.append((float(np.linalg.norm(geometry[0] - landmark_xy)), item))
    return min(scored, key=lambda pair: pair[0])[1] if scored else None


def _select_between(candidates, estimated):
    plate = _best_reference(estimated, "plate")
    ramekin = _best_reference(estimated, "ramekin")
    if plate is None or ramekin is None:
        return None
    midpoint = (_center_world(plate)[:2] + _center_world(ramekin)[:2]) / 2.0
    return min(candidates, key=lambda item: np.linalg.norm(_center_world(item)[:2] - midpoint))


def _select_table_center(candidates):
    geometries = [g for g in (_mask_geometry(i) for i in candidates) if g is not None]
    if not geometries:
        return None
    height, width = geometries[0][2]
    return _nearest_image_landmark(candidates, np.array([width / 2.0, height / 2.0]))


def _select_cabinet_region(candidates, estimated, *, top_drawer):
    cabinet = _best_reference(estimated, "wooden cabinet")
    if cabinet is None:
        return None
    geometry = _mask_geometry(cabinet)
    if geometry is None:
        return _nearest_xy(candidates, cabinet)
    _, (x0, y0, x1, y1), _ = geometry
    fraction = 0.28 if top_drawer else 0.02
    landmark = np.array([(x0 + x1) / 2.0, y0 + fraction * (y1 - y0)])
    return _nearest_image_landmark(candidates, landmark)


def select_visual_target_candidate(*, candidates, estimated, target_class, task_description):
    if not candidates:
        raise ValueError("candidates must not be empty")
    fallback = max(candidates, key=_confidence)
    if target_class != "black bowl":
        return fallback, "highest_confidence"
    text = " ".join(str(task_description).lower().split())
    if "between the plate and the ramekin" in text:
        selected = _select_between(candidates, estimated)
        if selected is not None:
            return selected, "between_plate_ramekin_xy"

    # This must precede the generic relation table. It remains operational
    # when the ramekin detection is absent, instead of silently reverting to
    # the highest-confidence (often wrong) bowl.
    if "on the ramekin" in text:
        return _select_on_ramekin(candidates, estimated)

    relations = (
        ("next to the ramekin", "ramekin", "next_to_ramekin_xy"),
        ("on the cookie box", "cookie box", "on_cookie_box_xy"),
        ("next to the cookie box", "cookie box", "next_to_cookie_box_xy"),
        ("on the stove", "stove", "on_stove_xy"),
        ("next to the plate", "plate", "next_to_plate_xy"),
    )
    for phrase, reference_class, selector in relations:
        if phrase in text:
            reference = _best_reference(estimated, reference_class)
            if reference is not None:
                return _nearest_xy(candidates, reference), selector
    if "from table center" in text:
        selected = _select_table_center(candidates)
        if selected is not None:
            return selected, "table_center_image"
    if "in the top drawer of the wooden cabinet" in text:
        selected = _select_cabinet_region(candidates, estimated, top_drawer=True)
        if selected is not None:
            return selected, "top_drawer_cabinet_image"
    if "on the wooden cabinet" in text:
        selected = _select_cabinet_region(candidates, estimated, top_drawer=False)
        if selected is not None:
            return selected, "on_cabinet_image"
    return fallback, "highest_confidence"
