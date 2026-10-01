"""TaskGraph -> stlcgpp STL formula + stacked signal tensor.

Precomputes each active atomic predicate's per-step margin (numpy, exact), stacks
them into one signal tensor S[T, D] (one column per atom), and wires the stlcgpp
formula with GreaterThan(Predicate(λS:S[:,col]), 0) atoms:

    phi = AND_k  ◇_[0,T] ( grasped_k  ∧  AND_i ◇_[0,tau] goal_{k,i} )

per manipulated object k (grasp-then-goal), conjoined across objects — correct for
multi-target tasks where one gripper grasps each object at a different time.
Articulated goals (open/close/turn) carry no grasp and reduce to ◇_[0,T] goal.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

_STLCGPP = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                       "..", "..", "third_party", "stlcg-plus-plus"))
if _STLCGPP not in sys.path:
    sys.path.insert(0, _STLCGPP)
from stlcgpp.formula import And, Eventually, GreaterThan, Predicate, STLFormula  # noqa: E402

from . import predicates as P
from .loader import Episode, resolve_object_key
from .predicates import PredConfig
from .task_graph import TaskGraph, TaskStep


# ---------------------------------------------------------------------------------------------------------------------
# resolved step (canonical name -> recorded key + target name)
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class ResolvedStep:
    verb: str
    obj_key: Optional[str] = None        # recorded object_pos key for the manipulated obj
    target: Optional[str] = None         # canonical target name (plate/basket/stove/...)
    level: Optional[str] = None
    relation: Optional[str] = None       # into / on_top_of / right_of / ... (on vs in)
    name: str = ""                       # atom display name


# Canonical aliases: NL concept -> recorded-body concept (drawer is part of the cabinet;
# "rack" -> wine_rack). Avoids the loose word-substring that matched wine_rack->wine_bottle.
_ALIAS = {"drawer": "cabinet", "rack": "wine_rack", "ramekin": "ramekin"}


def _resolve_candidates(canonical: str, ep: Episode, include_bodies: bool) -> List[str]:
    """Recorded keys (object_pos and/or body_names) matching a canonical category.
    Full-canonical substring match only (no token splitting, which crossed
    wine_rack <-> wine_bottle). object_pos keys are objects; body_names are fixtures
    (stove/cabinet/wine_rack/microwave)."""
    canon = _ALIAS.get(canonical, canonical)
    cand: List[str] = []
    for k in ep.object_pos:
        if "_to_robot0_eef" in k:
            continue
        base = k.rsplit("_", 1)[0] if k[-1].isdigit() else k
        if canon in k or canon in base:
            cand.append(k)
    if include_bodies:
        for n in ep.body_names:
            if canon in n and n not in cand:
                cand.append(n)
    return cand


def _travel(ep: Episode, key: str) -> float:
    """Max displacement of an object from its start pose (how much it was manipulated)."""
    p = ep.object_pos[key]
    return float(np.max(np.linalg.norm(p - p[0], axis=1)))


def _final_pos(ep: Episode, key: str) -> Optional[np.ndarray]:
    """Final pose of an object_pos key or a body_name."""
    if key in ep.object_pos:
        return ep.object_pos[key][-1]
    bi = ep.body_index(key)
    if bi is not None:
        return ep.body_xpos[-1, bi, :]
    return None


def resolve_steps(graph: TaskGraph, ep: Episode) -> List[ResolvedStep]:
    """Resolve canonical names -> recorded keys.

    OBJECT instances: when a category appears N times, pick the N instances that
    ACTUALLY MOVE the most (largest travel) — this excludes distractor objects that
    are never grasped. Same logical instance (pronoun 'it') reuses the prior key.
    TARGET instances: paired to the object by proximity (the target nearest the
    object's final pose) — avoids guessing which mug is 'white' vs 'yellow'.
    """
    out: List[ResolvedStep] = []
    used: set = set()
    obj_inst_key: Dict[int, str] = {}

    def moving_keys(canonical: str) -> List[str]:
        """Candidate OBJECT instances sorted by travel (most manipulated first)."""
        cands = _resolve_candidates(canonical, ep, include_bodies=False)
        if not cands:
            return []
        movers = sorted(cands, key=lambda k: _travel(ep, k), reverse=True)
        if not movers or _travel(ep, movers[0]) < 0.05:
            return cands
        return movers

    def resolve_object(canonical: Optional[str], inst: Optional[int]) -> Optional[str]:
        if not canonical:
            return None
        if inst is not None and inst in obj_inst_key:
            return obj_inst_key[inst]
        cands = moving_keys(canonical)
        if not cands:
            return canonical            # fixture / landmark -> body-name resolution
        for k in cands:
            if k not in used:
                used.add(k)
                if inst is not None:
                    obj_inst_key[inst] = k
                return k
        k = cands[0]
        if inst is not None:
            obj_inst_key[inst] = k
        return k

    def resolve_target(canonical: Optional[str], obj_key: Optional[str]) -> Optional[str]:
        if not canonical:
            return None
        cands = _resolve_candidates(canonical, ep, include_bodies=True)  # objects + fixtures
        if not cands:
            return canonical
        unused = [k for k in cands if k not in used]
        pool = unused or cands
        if obj_key and obj_key in ep.object_pos:
            final = ep.object_pos[obj_key][-1]
            # nearest by final pose among candidates that have a position
            posd = [(k, _final_pos(ep, k)) for k in pool if _final_pos(ep, k) is not None]
            if posd:
                return min(posd, key=lambda kp: float(np.linalg.norm(kp[1] - final)))[0]
        return pool[0]

    for s in graph.steps:
        obj_key = resolve_object(s.object, s.obj_instance)
        tgt = resolve_target(s.target, obj_key) if s.target else None
        if tgt and tgt in ep.object_pos:
            used.add(tgt)
        nm = f"{s.verb}({obj_key or s.object or ''}{','+tgt if tgt else ''})"
        out.append(ResolvedStep(verb=s.verb, obj_key=obj_key, target=tgt, level=s.level,
                                relation=s.relation, name=nm))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# signal bundle: precomputed margins -> stacked tensor + atom builders
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class AtomCol:
    name: str
    col: int
    margin: np.ndarray        # (T,)
    flag_key: Optional[str] = None   # LIBERO goal_flags key to calibrate its zero-point


@dataclass
class SignalBundle:
    S: torch.Tensor                      # (T, D) float32
    atoms: List[AtomCol] = field(default_factory=list)

    def atom_formula(self, idx: int, threshold: float = 0.0) -> STLFormula:
        """GreaterThan(Predicate(read column), threshold) — margin already precomputed;
        `threshold` is the calibrated zero-point offset for this atom."""
        a = self.atoms[idx]
        col = a.col
        return GreaterThan(Predicate(a.name, lambda S, c=col: S[:, c]), threshold)


def _step_margin(step: ResolvedStep, ep: Episode, cfg: PredConfig) -> Optional[np.ndarray]:
    v = step.verb
    if v == "pick" and step.obj_key:
        return P.pred_pick(ep, step.obj_key, cfg)
    if v == "place" and step.obj_key and step.target:
        # 'into' relation (in/into/inside a container) -> In/contain; else On/ontop
        return P.pred_in(ep, step.obj_key, step.target, cfg) if step.relation == "into" \
            else P.pred_on(ep, step.obj_key, step.target, cfg)
    if v == "push" and step.obj_key and step.target:    # push-to-region ~ placement on a surface
        return P.pred_on(ep, step.obj_key, step.target, cfg)
    if v == "stack" and step.obj_key and step.target:
        return P.pred_stack(ep, step.obj_key, step.target, cfg)
    if v == "insert" and step.obj_key and step.target:
        return P.pred_in(ep, step.obj_key, step.target, cfg)
    if v == "open":
        return P.pred_open(ep, step.level, cfg)
    if v == "close":
        return P.pred_close(ep, step.level, cfg)
    if v == "turn_on":
        return P.pred_turn_on(ep, cfg)
    if v == "turn_off":
        return P.pred_turn_off(ep, cfg)
    return None


def build_bundle(steps: List[ResolvedStep], ep: Episode, cfg: PredConfig) -> SignalBundle:
    """Collect every active atom's margin (one column each).

    A grasp atom 'grasp(<obj_key>)' is created for every manipulated object (one
    that is picked, OR placed/stacked/inserted/pushed — the latter implies an
    implicit grasp even without a 'pick' verb). Goal atoms follow each goal step.
    Each goal atom is tagged with the LIBERO goal_flags key used to calibrate its
    zero-point (on/in for placement, open/close for articulated).
    """
    place_flag = "on" if "on" in ep.goal_flags else ("in" if "in" in ep.goal_flags else None)
    cols: List[Tuple[str, np.ndarray, Optional[str]]] = []   # (name, margin, flag_key)

    # objects that must be grasped (explicit pick, or implicit via a goal step)
    grasp_objs: List[str] = []
    for s in steps:
        if s.verb == "pick" and s.obj_key:
            grasp_objs.append(s.obj_key)
        if s.verb in ("place", "stack", "insert", "push") and s.obj_key:
            grasp_objs.append(s.obj_key)
    seen = set()
    for k in grasp_objs:
        if k in seen:
            continue
        seen.add(k)
        cols.append((f"grasp({k})", P.pred_pick(ep, k, cfg), None))

    # goal atoms (one per non-pick goal step)
    for s in steps:
        if s.verb in ("place", "stack", "insert", "push", "open", "close",
                      "turn_on", "turn_off"):
            m = _step_margin(s, ep, cfg)
            if m is not None:
                flag = {"open": "open", "close": "close"}.get(s.verb)
                if s.verb in ("place", "stack", "insert", "push"):
                    flag = place_flag
                cols.append((s.name, m, flag))

    T = ep.T
    D = len(cols)
    S = np.zeros((T, max(D, 1)), dtype=np.float32)
    atoms: List[AtomCol] = []
    for i, (nm, m, flag) in enumerate(cols):
        S[:, i] = m.astype(np.float32)
        atoms.append(AtomCol(name=nm, col=i, margin=m, flag_key=flag))
    return SignalBundle(S=torch.from_numpy(S), atoms=atoms)


# ---------------------------------------------------------------------------------------------------------------------
# formula
# ---------------------------------------------------------------------------------------------------------------------

def build_formula(steps: List[ResolvedStep], bundle: SignalBundle,
                  tau: int, T_horizon: int,
                  calib: Optional[Dict[str, float]] = None) -> Tuple[STLFormula, Dict[str, int]]:
    """phi = AND_k ◇_[0,T]( grasp_k ∧ AND_i ◇_[0,tau] goal_{k,i} )  (per object),
    plus articulated goals as ◇_[0,T] goal. `calib` maps goal_flag key -> offset;
    each goal atom's zero-point is shifted by its offset (goal_flags calibration)."""
    calib = calib or {}
    name_to_col: Dict[str, int] = {a.name: a.col for a in bundle.atoms}
    col_to_flag: Dict[int, Optional[str]] = {a.col: a.flag_key for a in bundle.atoms}

    def thr_for(nm: str) -> float:
        col = name_to_col.get(nm)
        if col is None:
            return 0.0
        fk = col_to_flag.get(col)
        return calib.get(fk, 0.0) if fk else 0.0

    per_obj_goals: Dict[str, List[str]] = {}     # obj_key -> [goal atom names]
    grasp_for_obj: Dict[str, str] = {}           # obj_key -> grasp atom name
    standalone: List[str] = []                    # articulated/knob goal atom names

    for s in steps:
        if s.verb in ("place", "stack", "insert", "push"):
            k = s.obj_key
            per_obj_goals.setdefault(k, []).append(s.name)
            grasp_for_obj[k] = f"grasp({k})"
        elif s.verb in ("open", "close", "turn_on", "turn_off"):
            standalone.append(s.name)
        elif s.verb == "pick":
            grasp_for_obj[s.obj_key] = f"grasp({s.obj_key})"

    def atom_by_name(nm: str) -> Optional[STLFormula]:
        if nm not in name_to_col:
            return None
        return bundle.atom_formula(name_to_col[nm], threshold=thr_for(nm))

    sub_formulas: List[STLFormula] = []

    # per-object grasp-then-goal
    for k, goals in per_obj_goals.items():
        inner_atoms: List[STLFormula] = []
        gname = grasp_for_obj.get(k)
        if gname and gname in name_to_col:
            inner_atoms.append(bundle.atom_formula(name_to_col[gname]))   # grasp: no calibration
        for gn in goals:
            if gn in name_to_col:
                inner_atoms.append(Eventually(atom_by_name(gn), interval=[0, int(tau)]))
        if not inner_atoms:
            continue
        inner = inner_atoms[0] if len(inner_atoms) == 1 else _chain_and(inner_atoms)
        sub_formulas.append(Eventually(inner, interval=[0, int(T_horizon)]))

    # articulated / standalone goals
    for gn in standalone:
        if gn in name_to_col:
            sub_formulas.append(Eventually(atom_by_name(gn), interval=[0, int(T_horizon)]))

    if not sub_formulas:
        if bundle.atoms:
            return Eventually(bundle.atom_formula(0), interval=[0, int(T_horizon)]), name_to_col
        from stlcgpp.formula import Identity
        return Identity(), name_to_col
    phi = sub_formulas[0] if len(sub_formulas) == 1 else _chain_and(sub_formulas)
    return phi, name_to_col


def _chain_and(fs: List[STLFormula]) -> STLFormula:
    out = fs[0]
    for f in fs[1:]:
        out = And(out, f)
    return out


# =====================================================================================================================
# bddl-driven build (exact goal atoms with site targets) — matches goal_flags bit-for-bit
# =====================================================================================================================

def build_bddl_bundle(ep: Episode, cfg: PredConfig, atoms):
    """Build the signal bundle from EXACT bddl goal atoms (pred, obj, target).

    Each on/in atom yields a grasp(obj) atom + a goal atom realised with the EXACT
    target site (under/in_box) or object (check_ontop); open/turn atoms yield a goal
    atom only. Margins are precomputed; one column per atom."""
    from .bddl_goals import goal_atoms  # noqa (avoid import cycle)
    cols: List[Tuple[str, np.ndarray, Optional[str], Optional[str]]] = []  # name, margin, obj_key, flag_pred
    for (pred, obj, target) in atoms:
        margin, nm = P.goal_atom_margin(ep, pred, obj, target, cfg)
        obj_key = resolve_object_key(obj, ep) if obj else None
        flag = {"on": "on", "in": "in", "open": "open", "close": "close"}.get(pred)
        if pred in ("on", "in") and obj_key:
            cols.append((f"grasp({obj_key})", P.pred_pick(ep, obj_key, cfg), obj_key, None))
        cols.append((nm, margin, obj_key, flag))
    T = ep.T
    S = np.zeros((T, max(len(cols), 1)), dtype=np.float32)
    out_atoms = []
    for i, (nm, m, ok, fl) in enumerate(cols):
        S[:, i] = m.astype(np.float32)
        out_atoms.append(AtomCol(name=nm, col=i, margin=m, flag_key=fl))
    return SignalBundle(S=torch.from_numpy(S), atoms=out_atoms), cols


def build_bddl_formula(ep: Episode, bundle: SignalBundle, cols, tau: int, T_horizon: int):
    """phi = AND_k ◇_[0,T]( grasp_k ∧ ◇_[0,tau] goal_k ) for on/in objects,
    AND ◇_[0,T] goal for open/turn. Goal atoms use the EXACT bddl site targets."""
    name_to_col = {a.name: a.col for a in bundle.atoms}
    per_obj_goals: Dict[str, List[str]] = {}
    grasp_for: Dict[str, str] = {}
    standalone: List[str] = []
    for (nm, m, ok, fl) in cols:
        if nm.startswith("grasp("):
            grasp_for[ok] = nm
    goal_cols = [c for c in cols if not c[0].startswith("grasp(")]
    for (nm, m, ok, fl) in goal_cols:
        if fl in ("on", "in") and ok:
            per_obj_goals.setdefault(ok, []).append(nm)
        else:
            standalone.append(nm)

    def atom(nm):
        return bundle.atom_formula(name_to_col[nm]) if nm in name_to_col else None

    subs: List[STLFormula] = []
    for k, goals in per_obj_goals.items():
        inner = []
        g = grasp_for.get(k)
        if g and g in name_to_col:
            inner.append(bundle.atom_formula(name_to_col[g]))
        for gn in goals:
            ga = atom(gn)
            if ga is not None:
                inner.append(Eventually(ga, interval=[0, int(tau)]))
        if inner:
            subs.append(Eventually(_chain_and(inner) if len(inner) > 1 else inner[0],
                                   interval=[0, int(T_horizon)]))
    for gn in standalone:
        ga = atom(gn)
        if ga is not None:
            subs.append(Eventually(ga, interval=[0, int(T_horizon)]))
    if not subs:
        from stlcgpp.formula import Identity
        return Identity()
    return _chain_and(subs) if len(subs) > 1 else subs[0]
