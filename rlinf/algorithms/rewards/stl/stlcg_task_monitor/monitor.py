"""Online robustness monitor: prefix (live) + suffix (backward) traces of the
task STL over a recorded episode, via stlcgpp.

  * online_trace[t]  = rho(phi, s[0:t+1])  — only past+present; monotone for the
    ◇-achievement structure; the verdict a policy would read live.
  * suffix_trace[t]  = rho(phi, s[t:])     — stlcgpp native backward semantics
    (uses the future; a diagnostic).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from .formula_builder import (SignalBundle, build_bundle, build_formula,
                              resolve_steps)
from .loader import Episode, load_episode
from .nl_parser import parse_task
from .ontology import ObjectOntology
from .predicates import PredConfig
from .task_graph import TaskGraph


# ---------------------------------------------------------------------------------------------------------------------
# traces (self-contained; same stlcgpp semantics as stlcg_online_monitor)
# ---------------------------------------------------------------------------------------------------------------------

def suffix_trace(phi, S: torch.Tensor, approx_method: str = "true",
                 temperature: float = 1.0) -> np.ndarray:
    return phi.robustness_trace(S, approx_method=approx_method,
                                temperature=temperature).detach().cpu().numpy()


def online_prefix_trace(phi_builder, S: torch.Tensor, tau: int, T_horizon: int,
                        approx_method: str = "true", temperature: float = 1.0) -> np.ndarray:
    """At each t: rho(phi, s[0:t+1]). Rebuilds the formula with horizon t (the outer
    ◇ interval must match the prefix length so stlcgpp's Eventually mask aligns)."""
    T = S.shape[0]
    out = np.empty(T, dtype=np.float64)
    for t in range(T):
        phi_t = phi_builder(min(tau, t), t)
        out[t] = float(phi_t.robustness_trace(
            S[: t + 1], approx_method=approx_method, temperature=temperature)[0].item())
    return out


# ---------------------------------------------------------------------------------------------------------------------
# result
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class MonitorResult:
    task_description: str
    template: str
    T: int
    atoms: Dict[str, np.ndarray] = field(default_factory=dict)   # atom name -> (T,) margin
    atom_steps: Dict[str, int] = field(default_factory=dict)     # atom name -> first step >=0
    online_trace: np.ndarray = None
    suffix_trace: np.ndarray = None
    rho_final: float = 0.0
    satisfied: bool = False
    first_sat_step: Optional[int] = None
    approx_method: str = "true"


# ---------------------------------------------------------------------------------------------------------------------
# monitor one episode
# ---------------------------------------------------------------------------------------------------------------------

def monitor_episode(path: str, cfg: Optional[PredConfig] = None, tau: int = 60,
                    approx_method: str = "true", temperature: float = 1.0,
                    onto: Optional[ObjectOntology] = None,
                    graph: Optional[TaskGraph] = None,
                    compute_online: bool = True,
                    calib: Optional[Dict[str, float]] = None,
                    use_bddl: bool = True) -> MonitorResult:
    """Monitor one episode.

    rho_final = rho(phi, whole episode) = suffix_trace[0] (one stlcgpp call, O(T^2)).
    With use_bddl=True the goal atoms come from the EXACT bddl goal_state (site targets
    evaluated via under/in_box) — matches goal_flags bit-for-bit. Falls back to the
    NL-parsed formula if bddl is unavailable.
    """
    cfg = cfg or PredConfig()
    onto = onto or ObjectOntology()
    calib = calib or {}
    ep = load_episode(path)

    atoms_bddl = None
    if use_bddl:
        try:
            from .bddl_goals import goal_atoms
            suite = str(ep.attrs.get("suite", ""))
            tid = int(ep.attrs.get("task_id", -1))
            atoms_bddl = goal_atoms(suite, tid) if suite and tid >= 0 else None
        except Exception:
            atoms_bddl = None

    if atoms_bddl:
        from .formula_builder import build_bddl_bundle, build_bddl_formula
        bundle, cols = build_bddl_bundle(ep, cfg, atoms_bddl)
        phi = build_bddl_formula(ep, bundle, cols, tau=tau, T_horizon=ep.T - 1)
        if compute_online:
            def builder(tau_, T_horizon):
                return build_bddl_formula(ep, bundle, cols, tau=tau_, T_horizon=T_horizon)
            online = online_prefix_trace(builder, bundle.S, tau, ep.T - 1, approx_method, temperature)
            first_sat = next((t for t, r in enumerate(online) if r >= 0.0), None)
        else:
            online = None
            first_sat = None
        task_desc = str(ep.attrs.get("task_description", ""))
        suffix = suffix_trace(phi, bundle.S, approx_method, temperature)
        rho_final = float(suffix[0])
        atoms = {a.name: a.margin for a in bundle.atoms}
        atom_steps = {nm: (int(np.argmax(m >= 0)) if (m >= 0).any() else None) for nm, m in atoms.items()}
        return MonitorResult(task_description=task_desc, template="bddl", T=ep.T,
                             atoms=atoms, atom_steps=atom_steps, online_trace=online,
                             suffix_trace=suffix, rho_final=rho_final,
                             satisfied=bool(rho_final >= -1e-4), first_sat_step=first_sat,
                             approx_method=approx_method)

    if graph is None:
        graph = parse_task(str(ep.attrs.get("task_description", "")), onto)
    steps = resolve_steps(graph, ep)
    bundle = build_bundle(steps, ep, cfg)
    phi, _ = build_formula(steps, bundle, tau=tau, T_horizon=ep.T - 1, calib=calib)
    suffix = suffix_trace(phi, bundle.S, approx_method, temperature)
    rho_final = float(suffix[0])
    if compute_online:
        def builder(tau_, T_horizon):
            return build_formula(steps, bundle, tau=tau_, T_horizon=T_horizon, calib=calib)[0]
        online = online_prefix_trace(builder, bundle.S, tau, ep.T - 1, approx_method, temperature)
        first_sat = next((t for t, r in enumerate(online) if r >= 0.0), None)
    else:
        online = None
        first_sat = None
    atoms = {a.name: a.margin for a in bundle.atoms}
    atom_steps = {nm: (int(np.argmax(m >= 0)) if (m >= 0).any() else None) for nm, m in atoms.items()}
    return MonitorResult(task_description=graph.raw_text, template=graph.template.value,
                         T=ep.T, atoms=atoms, atom_steps=atom_steps, online_trace=online,
                         suffix_trace=suffix, rho_final=rho_final,
                         satisfied=bool(rho_final >= -1e-4), first_sat_step=first_sat,
                         approx_method=approx_method)
