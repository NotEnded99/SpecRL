"""
Util for resolving object instances from spatial relations at initial state.

Given a natural-language spatial constraint like "between the plate and the ramekin",
and the initial privileged state (all object positions), find which object instance
best satisfies the constraint.

This enables precise object grounding when multiple instances of the same category
exist, using ONLY initial-state geometry (no trajectory, no motion).

SUPPORTED RELATIONS (from LIBERO benchmarks):
  - between: "between the plate and the ramekin" (libero_spatial)
  - next_to / beside: "next to the cookie box" (libero_spatial)
  - left_of: "on the left", "to the left of the plate" (libero_90)
  - right_of: "on the right", "to the right of the plate" (libero_90)
  - in_front_of / front: "at the front", "to the front of" (libero_90)
  - behind / back: "at the back", "behind" (libero_90)
  - middle: "the middle black bowl", "in the middle" (libero_90)
  - under: "under the cabinet shelf" (libero_90)
  - on_the: "on the stove", "on the cookie box" (libero_spatial)
  - in_the: "in the top drawer" (libero_spatial)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SpatialConstraint:
    """Spatial relation constraint for object grounding."""
    relation: str  # "between", "left_of", "right_of", "in_front_of", "behind", "middle", "under", "next_to", "on_the", "in_the"
    landmarks: List[str]  # reference objects defining the relation
    level: Optional[str] = None  # for "in_the top drawer" - drawer level (top/middle/bottom)


def _euclidean_dist(p1: np.ndarray, p2: np.ndarray) -> float:
    """2D Euclidean distance (xy-plane only)."""
    return float(np.linalg.norm(p1[:2] - p2[:2]))


def _is_between(pos: np.ndarray, pos_a: np.ndarray, pos_b: np.ndarray,
                tol_angle: float = 0.3, tol_dist: float = 0.15) -> float:
    """
    Check if pos is "between" pos_a and pos_b in the xy-plane.

    Returns a continuous score (higher = more "between"):
    - pos should be near the midpoint of A and B
    - pos should be roughly collinear with A and B
    """
    # Work in 2D
    p = pos[:2]
    a = pos_a[:2]
    b = pos_b[:2]

    # Midpoint between A and B
    mid = (a + b) / 2.0

    # Distance from candidate to midpoint
    dist_to_mid = np.linalg.norm(p - mid)

    # Check collinearity: vector (p-a) should be parallel to (b-a)
    vec_pa = p - a
    vec_ba = b - a
    len_ba = np.linalg.norm(vec_ba)
    if len_ba < 1e-6:
        return -1.0

    # Project p onto line AB
    proj_len = np.dot(vec_pa, vec_ba) / len_ba
    proj_ratio = proj_len / len_ba

    # p should be between A (ratio=0) and B (ratio=1)
    # Allow some slack: -0.2 to 1.2
    if proj_ratio < -0.2 or proj_ratio > 1.2:
        return -1.0

    # Perpendicular distance from line AB
    vec_perp = vec_pa - (proj_len / len_ba) * vec_ba
    perp_dist = np.linalg.norm(vec_perp)

    # Score: closer to midpoint and closer to line = higher
    score = -perp_dist - 0.5 * dist_to_mid
    return float(score)


def _is_next_to(pos: np.ndarray, ref_pos: np.ndarray, max_dist: float = 0.15) -> float:
    """
    Check if pos is "next to" ref_pos.

    "next to" means close in xy-plane but not on top of.
    Returns: negative of distance (closer = higher score)
    """
    dist = _euclidean_dist(pos, ref_pos)
    # Prefer some distance (not too close, not too far)
    # Optimal distance: 0.05-0.15m
    optimal_dist = 0.10
    score = -abs(dist - optimal_dist)
    return float(score)


def _is_left_of(pos: np.ndarray, ref_pos: np.ndarray) -> float:
    """Check if pos is to the left of ref_pos (robot frame: +y direction)."""
    return float(pos[1] - ref_pos[1])  # positive if pos_y > ref_y


def _is_right_of(pos: np.ndarray, ref_pos: np.ndarray) -> float:
    """Check if pos is to the right of ref_pos (robot frame: -y direction)."""
    return float(ref_pos[1] - pos[1])


def _is_in_front_of(pos: np.ndarray, ref_pos: np.ndarray) -> float:
    """Check if pos is in front of ref_pos (robot frame: +x direction)."""
    return float(pos[0] - ref_pos[0])


def _is_behind(pos: np.ndarray, ref_pos: np.ndarray) -> float:
    """Check if pos is behind ref_pos (robot frame: -x direction)."""
    return float(ref_pos[0] - pos[0])


def _is_middle_of(pos: np.ndarray, other_positions: List[np.ndarray]) -> float:
    """
    Check if pos is the "middle" of a set of objects.

    For "the middle black bowl", this finds the bowl with median y-coordinate
    (assuming objects are arranged left-to-right).
    """
    if len(other_positions) < 2:
        return 0.0

    # Collect y-coordinates
    y_coords = [p[1] for p in other_positions]
    y_coords.sort()

    # Find median
    median_y = y_coords[len(y_coords) // 2]

    # Score by closeness to median
    score = -abs(pos[1] - median_y)
    return float(score)


def _is_under(pos: np.ndarray, ref_pos: np.ndarray, tol_z: float = 0.1) -> float:
    """Check if pos is under ref_pos (lower z, close in xy)."""
    z_diff = ref_pos[2] - pos[2]
    xy_dist = _euclidean_dist(pos, ref_pos)

    # Must be below AND close in xy
    if z_diff < 0:  # not below
        return -1.0

    # Score: higher z_diff (more under) and lower xy_dist (closer)
    score = z_diff - xy_dist
    return float(score)


def _is_on_the(pos: np.ndarray, ref_pos: np.ndarray, tol_xy: float = 0.08, tol_z: float = 0.1) -> float:
    """
    Check if pos is "on the" ref_pos (on top of a surface/container).

    For "on the stove", "on the cookie box":
    - Should be close in xy-plane
    - Should be slightly above in z
    """
    xy_dist = _euclidean_dist(pos, ref_pos)
    z_diff = pos[2] - ref_pos[2]  # pos should be above ref

    # Must be close in xy
    if xy_dist > tol_xy:
        return float(-xy_dist)

    # Should be above (positive z_diff) but not too high
    if z_diff < 0 or z_diff > tol_z:
        z_score = -abs(z_diff - 0.05)
    else:
        z_score = 0.1 - abs(z_diff - 0.05)  # optimal: 5cm above

    score = -xy_dist + z_score
    return float(score)


def _is_in_the(pos: np.ndarray, ref_pos: np.ndarray, container_type: str = "drawer") -> float:
    """
    Check if pos is "in the" ref_pos (inside a container/drawer).

    For "in the top drawer":
    - Should be close in xy-plane
    - Should be at similar height (inside the drawer volume)
    """
    xy_dist = _euclidean_dist(pos, ref_pos)
    z_diff = abs(pos[2] - ref_pos[2])

    # Should be very close in xy (inside the drawer opening)
    # And at similar height (inside drawer volume)
    score = -xy_dist - z_diff
    return float(score)


def resolve_object_by_spatial_constraint(
    canonical: str,
    constraint: Optional[SpatialConstraint],
    object_positions: Dict[str, np.ndarray],
    body_positions: Optional[Dict[str, np.ndarray]] = None,
) -> Optional[str]:
    """
    Resolve a canonical object name to a specific instance using spatial constraint.

    Args:
        canonical: object category (e.g., "bowl", "black_bowl")
        constraint: spatial constraint (e.g., between A and B)
        object_positions: {obj_key: (3,)} for manipulable objects
        body_positions: {body_name: (3,)} for fixtures (optional)

    Returns:
        best matching obj_key, or None if no match
    """
    # Gather all positions (objects + bodies)
    all_positions = dict(object_positions)
    if body_positions:
        all_positions.update(body_positions)

    # Find candidates matching canonical name
    candidates = []
    for key in object_positions:
        # Match by substring (e.g., "bowl" matches "akita_black_bowl_1")
        base = key.rsplit("_", 1)[0] if key[-1].isdigit() else key
        if canonical in key or canonical in base:
            candidates.append(key)

    if not candidates:
        return None

    # If no constraint, return first candidate (fallback)
    if constraint is None:
        return candidates[0]

    # Resolve landmark positions
    landmark_positions = {}
    for lm in constraint.landmarks:
        # Try to find landmark in all positions
        for key, pos in all_positions.items():
            base = key.rsplit("_", 1)[0] if key[-1].isdigit() else key
            if lm in key or lm in base:
                landmark_positions[lm] = pos
                break

    # Score each candidate by how well it satisfies the constraint
    best_key = None
    best_score = -np.inf

    for cand_key in candidates:
        cand_pos = object_positions[cand_key]
        score = 0.0

        if constraint.relation == "between":
            # Need exactly 2 landmarks
            if len(constraint.landmarks) >= 2:
                lm0, lm1 = constraint.landmarks[0], constraint.landmarks[1]
                pos0 = landmark_positions.get(lm0)
                pos1 = landmark_positions.get(lm1)
                if pos0 is not None and pos1 is not None:
                    score = _is_between(cand_pos, pos0, pos1)
                else:
                    score = -np.inf

        elif constraint.relation in ("next_to", "beside"):
            if constraint.landmarks:
                ref_pos = landmark_positions.get(constraint.landmarks[0])
                if ref_pos is not None:
                    score = _is_next_to(cand_pos, ref_pos)

        elif constraint.relation in ("left_of", "left"):
            if constraint.landmarks:
                ref_pos = landmark_positions.get(constraint.landmarks[0])
                if ref_pos is not None:
                    score = _is_left_of(cand_pos, ref_pos)
            else:
                # "the left bowl" - find the bowl with max y-coordinate
                all_cand_positions = [object_positions[k] for k in candidates]
                y_coords = [p[1] for p in all_cand_positions]
                if cand_pos[1] == max(y_coords):
                    score = 1.0
                else:
                    score = 0.0

        elif constraint.relation in ("right_of", "right"):
            if constraint.landmarks:
                ref_pos = landmark_positions.get(constraint.landmarks[0])
                if ref_pos is not None:
                    score = _is_right_of(cand_pos, ref_pos)
            else:
                # "the right bowl" - find the bowl with min y-coordinate
                all_cand_positions = [object_positions[k] for k in candidates]
                y_coords = [p[1] for p in all_cand_positions]
                if cand_pos[1] == min(y_coords):
                    score = 1.0
                else:
                    score = 0.0

        elif constraint.relation in ("in_front_of", "front"):
            if constraint.landmarks:
                ref_pos = landmark_positions.get(constraint.landmarks[0])
                if ref_pos is not None:
                    score = _is_in_front_of(cand_pos, ref_pos)
            else:
                # "at the front" - find the object with max x-coordinate
                all_cand_positions = [object_positions[k] for k in candidates]
                x_coords = [p[0] for p in all_cand_positions]
                if cand_pos[0] == max(x_coords):
                    score = 1.0
                else:
                    score = 0.0

        elif constraint.relation in ("behind", "back"):
            if constraint.landmarks:
                ref_pos = landmark_positions.get(constraint.landmarks[0])
                if ref_pos is not None:
                    score = _is_behind(cand_pos, ref_pos)
            else:
                # "at the back" - find the object with min x-coordinate
                all_cand_positions = [object_positions[k] for k in candidates]
                x_coords = [p[0] for p in all_cand_positions]
                if cand_pos[0] == min(x_coords):
                    score = 1.0
                else:
                    score = 0.0

        elif constraint.relation == "middle":
            # "the middle black bowl" - find the bowl with median y-coordinate
            all_cand_positions = [object_positions[k] for k in candidates]
            score = _is_middle_of(cand_pos, all_cand_positions)

        elif constraint.relation == "under":
            if constraint.landmarks:
                ref_pos = landmark_positions.get(constraint.landmarks[0])
                if ref_pos is not None:
                    score = _is_under(cand_pos, ref_pos)

        elif constraint.relation == "on_the":
            if constraint.landmarks:
                ref_pos = landmark_positions.get(constraint.landmarks[0])
                if ref_pos is not None:
                    score = _is_on_the(cand_pos, ref_pos)

        elif constraint.relation == "in_the":
            if constraint.landmarks:
                ref_pos = landmark_positions.get(constraint.landmarks[0])
                if ref_pos is not None:
                    score = _is_in_the(cand_pos, ref_pos)

        # Default: no spatial constraint, score = 0
        if score > best_score:
            best_score = score
            best_key = cand_key

    return best_key


def extract_spatial_constraint_from_taskgraph(
    graph: "TaskGraph",  # type: ignore
) -> Optional[SpatialConstraint]:
    """
    Extract spatial constraint from a TaskGraph.

    The TaskGraph.spatial_relation field contains relations like "between",
    "left_of", etc. The reference objects are stored in TaskGraph.reference_object
    or can be inferred from TaskGraph.objects with role=LANDMARK.
    """
    if graph is None:
        return None

    relation = getattr(graph, "spatial_relation", None)
    if relation is None:
        return None

    # Collect landmark objects
    landmarks = []

    # Try reference_object field first
    ref_obj = getattr(graph, "reference_object", None)
    if ref_obj:
        landmarks.append(ref_obj)

    # Also check objects with LANDMARK role
    from .task_graph import ObjectRole
    for obj in getattr(graph, "objects", []):
        if getattr(obj, "role", None) == ObjectRole.LANDMARK:
            landmarks.append(obj.name)

    if not landmarks:
        return None

    # Handle special cases for "the left/right/middle X"
    # These don't need landmarks - the relation itself specifies the position
    if relation in ("left", "right", "middle", "front", "back"):
        return SpatialConstraint(relation=relation, landmarks=[])

    return SpatialConstraint(relation=relation, landmarks=landmarks)


# =====================================================================================================================
# Integration with existing code
# =====================================================================================================================

def resolve_object_key_with_spatial(
    canonical: str,
    ep: "Episode",  # type: ignore
    constraint: Optional[SpatialConstraint] = None,
) -> Optional[str]:
    """
    Enhanced version of resolve_object_key that considers spatial constraints.

    Uses the INITIAL position (step 0) from the Episode to ground objects.
    This works for both offline trajectories and online env (where T=1).

    Args:
        canonical: object category (e.g., "bowl")
        ep: Episode object with object_pos, body_xpos, body_names
        constraint: optional spatial constraint

    Returns:
        best matching object_pos key
    """
    # Extract initial positions (step 0)
    object_positions = {k: v[0] for k, v in ep.object_pos.items()}

    # Build body_positions dict
    body_positions = {}
    if ep.body_xpos is not None and ep.body_names:
        for i, name in enumerate(ep.body_names):
            if i < ep.body_xpos.shape[1]:  # shape is (T, N_body, 3)
                body_positions[name] = ep.body_xpos[0, i, :]

    return resolve_object_by_spatial_constraint(
        canonical=canonical,
        constraint=constraint,
        object_positions=object_positions,
        body_positions=body_positions,
    )
