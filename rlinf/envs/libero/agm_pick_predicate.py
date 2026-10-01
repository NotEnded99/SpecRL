"""Experimental two-finger grasp predicate for AGM-stage rewards."""

from __future__ import annotations

import numpy as np

from rlinf.algorithms.rewards.stl.stlcg_task_monitor.loader import (
    Episode,
)
from rlinf.algorithms.rewards.stl.stlcg_task_monitor.predicates import (
    PredConfig,
    eef_object_dist,
)


_LEFT_MARKERS = (
    "leftfinger",
    "left_finger",
    "finger_joint1",
)

_RIGHT_MARKERS = (
    "rightfinger",
    "right_finger",
    "finger_joint2",
)


def _matching_body_ids(
    ep: Episode,
    markers: tuple[str, ...],
) -> set[int]:
    result: set[int] = set()

    for body_id, name in zip(ep.body_ids, ep.body_names):
        lowered = str(name).lower()
        if any(marker in lowered for marker in markers):
            result.add(int(body_id))

    return result


def _object_body_ids(ep: Episode, obj_key: str) -> set[int]:
    result: set[int] = set()

    if ep.obj_body_tree:
        for name, body_ids in ep.obj_body_tree.items():
            if obj_key == name or obj_key in name or name in obj_key:
                result.update(int(body_id) for body_id in body_ids)

    if result:
        return result

    for body_id, name in zip(ep.body_ids, ep.body_names):
        name = str(name)
        if obj_key == name or obj_key in name or name in obj_key:
            result.add(int(body_id))

    return result


def two_finger_grasp(
    ep: Episode,
    obj_key: str,
) -> np.ndarray:
    """Match robosuite's two-finger grasp semantics using recorded contacts."""
    left_ids = _matching_body_ids(ep, _LEFT_MARKERS)
    right_ids = _matching_body_ids(ep, _RIGHT_MARKERS)
    object_ids = _object_body_ids(ep, obj_key)

    grasped = np.zeros(ep.T, dtype=bool)

    if not left_ids or not right_ids or not object_ids:
        return grasped

    for step in range(ep.T):
        count = int(ep.ncon[step])
        if count <= 0:
            continue

        body1 = ep.contact_body1[step, :count]
        body2 = ep.contact_body2[step, :count]

        left_contact = bool(
            (
                np.isin(body1, list(left_ids))
                & np.isin(body2, list(object_ids))
            ).any()
            or (
                np.isin(body2, list(left_ids))
                & np.isin(body1, list(object_ids))
            ).any()
        )

        right_contact = bool(
            (
                np.isin(body1, list(right_ids))
                & np.isin(body2, list(object_ids))
            ).any()
            or (
                np.isin(body2, list(right_ids))
                & np.isin(body1, list(object_ids))
            ).any()
        )

        grasped[step] = left_contact and right_contact

    return grasped


def pred_pick_agm(
    ep: Episode,
    obj_key: str,
    cfg: PredConfig,
    success_margin: float = 0.01,
) -> np.ndarray:
    """Negative proximity while approaching; positive after true grasp."""
    proximity = cfg.r_grasp - eef_object_dist(ep, obj_key)
    grasped = two_finger_grasp(ep, obj_key)

    # 接近时始终保持负数，不能仅靠距离宣告抓取成功。
    approach_margin = np.minimum(proximity, -1.0e-6)

    return np.where(
        grasped,
        float(success_margin),
        approach_margin,
    ).astype(np.float64)