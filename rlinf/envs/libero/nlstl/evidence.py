"""Layer 3: unified evidence/state -- the shared contract both branches fill.

After grounding, every branch must hand the *same kind* of state to the shared
predicate code.  That shared state is exactly the privileged dict the existing
LIBERO STL env already builds from ``env.sim`` (see ``venv._extract_privileged_state``)
and reshapes into a ``loader.Episode`` via ``_build_step_episode``.  All atomic
predicates (``goal_atom_margin`` / ``pred_pick_agm``) and the STL runtime key
off ``Episode`` + ``PredConfig``, so anything that produces an ``Episode``-worth
of fields can drive the shared backend unchanged.

This module therefore defines:

* :class:`UnifiedEvidence` -- a thin holder of a ``priv``-shaped dict and the
  lazily-built ``Episode``.
* :func:`evidence_from_priv` -- privileged state extraction (implemented).
* :func:`evidence_from_vision` -- vision state extraction (reserved interface).
* :func:`compute_atom_margins` -- the shared per-atom robustness computation,
  identical to ``libero_env_agm_stage._raw_atom_margin``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional

from rlinf.envs.libero.stl_stage_plan import Atom, canonical_atom


@dataclass
class UnifiedEvidence:
    """Unified state both grounders produce.

    ``priv`` is the privileged-dict shape (object_pos, site_*, contacts, joints,
    ...).  ``episode`` is the T=1 ``loader.Episode`` the predicates consume,
    built lazily on first use so importing this module needs no MuJoCo/libero.
    """

    priv: Dict[str, Any] = field(default_factory=dict)
    source: str = "privileged"
    _episode: Optional[Any] = field(default=None, repr=False)

    def get_episode(self) -> Any:
        if self._episode is None:
            # Local import: _build_step_episode lives in the (heavy) env module.
            from rlinf.envs.libero.libero_env import _build_step_episode

            self._episode = _build_step_episode(self.priv)
        return self._episode


def evidence_from_priv(priv: Mapping[str, Any]) -> UnifiedEvidence:
    """Privileged state extraction: wrap the MuJoCo privileged dict."""
    return UnifiedEvidence(priv=dict(priv), source="privileged")


def evidence_from_vision(
    images: Optional[Mapping[str, Any]] = None,
    detections: Optional[Iterable] = None,
    calibration: Optional[Mapping[str, Any]] = None,
    *,
    gripper_state: Optional[Mapping[str, Any]] = None,
    articulated_state: Optional[Mapping[str, Any]] = None,
) -> UnifiedEvidence:  # pragma: no cover - interface only
    """Reserved vision state extraction.

    A future implementation turns image evidence into the same ``priv``-shaped
    fields the predicates read.  The minimal contract, per atom family:

    * ``pick(o)``        -> ``object_pos[o]`` (3D world centre) + ``eef_pos``;
      plus a vision-derived grasp flag to replace ``two_finger_grasp``.
    * ``on/stack(o,t)``  -> object centre + target reference point (a site /
      surface, not just a body centroid).
    * ``in(o,c)``        -> object centre + container opening frame + half-extents.
    * ``open/close/turn*``-> an articulated-joint progress scalar (needs temporal
      pose estimation; not recoverable from a single mask).

    The existing ``visual_stl_shadow`` / ``visual_stl_monitor`` already produce
    mask+depth -> 3D world centres for the pick/approach term and align its sign
    via ``visual_pick_margin``; that is the natural seed for this function.
    Not implemented here.
    """
    raise NotImplementedError(
        "Vision evidence extraction is reserved (interface only); use "
        "evidence_from_priv for the implemented privileged branch."
    )


def make_pred_config(**overrides) -> Any:
    """Build a ``PredConfig`` (the shared predicate parameter struct)."""
    from rlinf.algorithms.rewards.stl.stlcg_task_monitor.predicates import PredConfig

    return PredConfig(**overrides)


def compute_atom_margins(
    evidence: UnifiedEvidence,
    atoms: Iterable[Atom],
    pred_cfg: Any,
) -> Dict[Atom, float]:
    """Shared per-atom robustness, identical to ``_raw_atom_margin``.

    Reuses the existing predicate functions verbatim (``pred_pick_agm`` for pick,
    ``goal_atom_margin`` for on/in/stack/open/close/turnon/turnoff), so the
    privileged branch yields the same margins -- and therefore the same STL rho
    and reward -- as the current AGM-stage env.
    """
    import numpy as np

    from rlinf.algorithms.rewards.stl.stlcg_task_monitor.loader import (
        resolve_object_key,
    )
    from rlinf.algorithms.rewards.stl.stlcg_task_monitor.predicates import (
        goal_atom_margin,
    )
    from rlinf.envs.libero.agm_pick_predicate import pred_pick_agm

    ep = evidence.get_episode()
    margins: Dict[Atom, float] = {}
    for atom in atoms:
        atom = canonical_atom(atom)
        pred, obj, target = atom
        if pred == "pick":
            object_key = resolve_object_key(obj, ep)
            if object_key is None:
                raise KeyError(f"Cannot resolve pick object {obj!r}.")
            margin = pred_pick_agm(ep, object_key, pred_cfg)
        else:
            margin, _ = goal_atom_margin(ep, pred, obj, target, pred_cfg)
        margins[atom] = float(np.asarray(margin, dtype=np.float64).reshape(-1)[-1])
    return margins
