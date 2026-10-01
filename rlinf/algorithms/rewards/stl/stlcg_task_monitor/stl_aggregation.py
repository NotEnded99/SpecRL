"""Experimental numerical utilities for normalized AGM-stage rewards.

This module was added for the AGM experiment and is not part of the senior's
original STL implementation.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


NORMALIZED_MIN = -1.0
SATISFACTION_BOUNDARY = 0.0
NORMALIZED_MAX = 0.1


def normalize_from_initial_negative(
    margin: float,
    initial_margin: float,
    eps: float = 1e-6,
) -> float:
    """Map an atom margin to the common interval [-1, 0.1].

    For an initially-unsatisfied atom:
      initial margin -> -1
      satisfaction boundary (margin == 0) -> 0
      robustly satisfied margin -> at most 0.1

    An atom that is already satisfied at reset has no negative baseline from
    which to infer a scale. It therefore maps to 0 at the boundary and 0.1
    above the boundary. If it later becomes unsatisfied, it maps
    conservatively to -1.
    """
    margin = float(margin)
    initial_margin = float(initial_margin)

    if not np.isfinite(margin) or not np.isfinite(initial_margin):
        raise ValueError(
            "margin and initial_margin must be finite: "
            f"{margin=}, {initial_margin=}"
        )

    if initial_margin >= -eps:
        if margin > eps:
            return NORMALIZED_MAX
        if margin >= SATISFACTION_BOUNDARY:
            return SATISFACTION_BOUNDARY
        return NORMALIZED_MIN

    # General form for threshold theta=0:
    #   (margin - theta) / (theta - initial_margin)
    normalized = margin / (-initial_margin)
    return float(
        np.clip(normalized, NORMALIZED_MIN, NORMALIZED_MAX)
    )


def normalize_margins_from_initial(
    margins: Sequence[float],
    initial_margins: Sequence[float],
) -> list[float]:
    """Normalize aligned raw margins to [-1, 0.1]."""
    if len(margins) != len(initial_margins):
        raise ValueError(
            "margins and initial_margins must have identical lengths: "
            f"{len(margins)} != {len(initial_margins)}"
        )

    return [
        normalize_from_initial_negative(margin, initial_margin)
        for margin, initial_margin in zip(margins, initial_margins)
    ]


def discounted_progress_reward(
    next_score,
    previous_score,
    gamma: float = 0.99,
):
    """Compute gamma * (Phi(t+1) + 1) - (Phi(t) + 1)."""
    gamma = float(gamma)
    if not np.isfinite(gamma):
        raise ValueError(f"gamma must be finite, got {gamma}.")
    next_value = np.asarray(next_score, dtype=np.float64)
    previous_value = np.asarray(previous_score, dtype=np.float64)
    return gamma * (next_value + 1.0) - (previous_value + 1.0)


def agm_and(normalized_margins: Sequence[float]) -> float:
    """Aggregate normalized margins with AGM conjunction robustness.

    When every atom is strictly satisfied, use AGM's positive geometric
    branch. Otherwise, average only the negative violations. This keeps zero
    as the exact satisfaction boundary and preserves positive headroom up to
    0.1.
    """
    values = np.asarray(normalized_margins, dtype=np.float64)

    if values.size == 0:
        raise ValueError("AGM conjunction requires at least one value.")

    if not np.all(np.isfinite(values)):
        raise ValueError(f"normalized_margins contains NaN/Inf: {values}")

    values = np.clip(values, NORMALIZED_MIN, NORMALIZED_MAX)

    if np.all(values > SATISFACTION_BOUNDARY):
        return float(
            np.prod(1.0 + values) ** (1.0 / values.size) - 1.0
        )

    return float(
        np.minimum(values, SATISFACTION_BOUNDARY).mean()
    )


def stage_agm_score(
    margins: Sequence[float],
    initial_margins: Sequence[float],
) -> float:
    """Normalize a stage's raw margins, then aggregate them with AGM."""
    return agm_and(
        normalize_margins_from_initial(margins, initial_margins)
    )


def atoms_satisfied(
    margins: Sequence[float],
    threshold: float = 0.0,
) -> bool:
    """Return whether every atom margin is at or above the threshold."""
    return bool(margins) and all(
        float(margin) >= float(threshold)
        for margin in margins
    )
