"""goal_flags calibration of each goal atom's zero-point.

LIBERO records per-step binary goal predicates (goal_flags/{on,in,open,close}) —
the simulator's own truth for when a placement/articulated goal holds. My
continuous predicate margin is well-correlated with these but its zero-crossing
is offset by a few mm / a few % because my geometric thresholds don't exactly
match LIBERO's internal regions.

This module fits, per goal_flag key, the offset δ that maximises step-level
agreement of (margin - δ >= 0) with the flag. The offset is then applied as the
atom's GreaterThan threshold, so ρ >= 0 lines up with LIBERO's predicate truth.
It does NOT make the monitor read goal_flags at eval time — goal_flags are only
used once, offline, to pin the zero-point of each continuous margin.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .formula_builder import build_bundle, resolve_steps
from .loader import load_episode
from .nl_parser import parse_task
from .ontology import ObjectOntology
from .predicates import PredConfig


def _onset_offset(per_episode: List[tuple]) -> float:
    """δ = median of the margin at the step goal_flag first flips True, across episodes.

    Pins the atom's zero-crossing to the goal onset: after applying δ, ρ reaches 0
    exactly when LIBERO's predicate first holds. Robust to the heavy class imbalance
    of goal_flags (True only briefly), which a raw accuracy objective would exploit
    by predicting all-negative.
    """
    onset_vals = []
    for margin, flag in per_episode:
        idx = np.where(flag)[0]
        if len(idx) == 0:
            continue
        onset_vals.append(float(margin[idx[0]]))
    if not onset_vals:
        return 0.0
    return float(np.median(onset_vals))


def fit_calibration(paths: List[str], cfg: Optional[PredConfig] = None,
                    onto: Optional[ObjectOntology] = None,
                    max_episodes: int = 200) -> Dict[str, float]:
    """Fit {goal_flag_key: onset offset} over a sample of episodes."""
    cfg = cfg or PredConfig()
    onto = onto or ObjectOntology()
    pools: Dict[str, List[tuple]] = {}     # flag_key -> list of (margin_array, flag_array)

    for path in paths[:max_episodes]:
        try:
            ep = load_episode(path)
            graph = parse_task(str(ep.attrs.get("task_description", "")), onto)
            steps = resolve_steps(graph, ep)
            bundle = build_bundle(steps, ep, cfg)
        except Exception:
            continue
        for a in bundle.atoms:
            if a.flag_key is None or a.flag_key not in ep.goal_flags:
                continue
            flag = ep.goal_flags[a.flag_key]
            n = min(len(a.margin), len(flag))
            pools.setdefault(a.flag_key, []).append((a.margin[:n], flag[:n]))

    return {key: _onset_offset(pairs) for key, pairs in pools.items() if pairs}


def calibration_report(paths: List[str], calib: Dict[str, float],
                       cfg: Optional[PredConfig] = None,
                       onto: Optional[ObjectOntology] = None,
                       max_episodes: int = 200) -> Dict[str, Dict[str, float]]:
    """Per-key onset stats + F1 of (margin-δ>=0) vs flag, before/after."""
    cfg = cfg or PredConfig()
    onto = onto or ObjectOntology()
    pools: Dict[str, List[tuple]] = {}
    for path in paths[:max_episodes]:
        try:
            ep = load_episode(path)
            graph = parse_task(str(ep.attrs.get("task_description", "")), onto)
            steps = resolve_steps(graph, ep)
            bundle = build_bundle(steps, ep, cfg)
        except Exception:
            continue
        for a in bundle.atoms:
            if a.flag_key is None or a.flag_key not in ep.goal_flags:
                continue
            flag = ep.goal_flags[a.flag_key]
            n = min(len(a.margin), len(flag))
            pools.setdefault(a.flag_key, []).append((a.margin[:n], flag[:n]))

    def _f1(m, f, d):
        pred = (m - d) >= 0
        tp = float((pred & f).sum()); fp = float((pred & ~f).sum()); fn = float((~pred & f).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return 2 * p * r / (p + r) if p + r else 0.0

    rep = {}
    for key, pairs in pools.items():
        m = np.concatenate([p[0] for p in pairs]); f = np.concatenate([p[1] for p in pairs])
        d = calib.get(key, 0.0)
        rep[key] = {
            "offset": round(d, 4),
            "n_steps": int(len(m)),
            "f1_before": round(_f1(m, f, 0.0), 3),
            "f1_after": round(_f1(m, f, d), 3),
            "pos_rate": round(float(f.mean()), 4),
        }
    return rep
