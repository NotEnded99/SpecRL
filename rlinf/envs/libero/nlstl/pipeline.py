"""Layers 4-6: the shared orchestrator -- STL/AGM aggregation -> rho -> reward.

:class:`NLSTLPipeline` wires the layers together for ONE environment:

    symbolic plan (shared)  -- build_symbolic_plan
          |
    grounding (privileged now, vision reserved)  -- Grounder.ground
          |
    unified evidence -> per-atom margins (shared)  -- compute_atom_margins
          |
    STL/AGM temporal aggregation (shared)  -- AGMStageRuntime.update
          |
    task robustness rho (= episode_score) + shaping  -- StageRuntimeState

It reuses the existing :class:`AGMStageRuntime` and predicate functions
unchanged, so for the privileged branch the returned ``rho`` is identical to the
current AGM-stage env's ``_stl_robustness()`` output.  The env's existing
``_calc_step_reward`` consumes that ``rho`` (clip + scale + optional relative
differencing); this module does **not** reimplement reward post-processing, to
avoid any divergence from the live reward path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from rlinf.envs.libero.nlstl.evidence import (
    UnifiedEvidence,
    compute_atom_margins,
    make_pred_config,
)
from rlinf.envs.libero.nlstl.grounding import Grounder, Grounding, PrivilegedGrounder
from rlinf.envs.libero.nlstl.symbolic_plan import SymbolicStagePlan, build_symbolic_plan


@dataclass
class NLSTLPipeline:
    """One-task NL -> STL reward pipeline (shared by both grounding branches).

    Parameters mirror the AGM-stage env knobs that affect the STL formula:
    ``confirmation_steps`` (``agm_stage_confirmation_steps``, default 3) and the
    predicate ``PredConfig`` (e.g. ``r_grasp`` from ``stl_pred_config``).
    """

    pred_cfg: Any = None
    confirmation_steps: int = 3
    grounder: Grounder = field(default_factory=PrivilegedGrounder)

    # Per-task state (populated by setup()).
    symbolic: Optional[SymbolicStagePlan] = None
    grounding: Optional[Grounding] = None
    runtime: Any = None  # AGMStageRuntime
    last_state: Any = None  # most recent StageRuntimeState

    # ------------------------------------------------------------------
    # setup / step
    # ------------------------------------------------------------------
    def setup(
        self,
        description: str,
        evidence: Optional[UnifiedEvidence] = None,
        task_suite_name: str = "libero_40",
        task_id: int = 0,
    ) -> "NLSTLPipeline":
        """Parse NL, ground it, and initialize the STL runtime (call at reset)."""
        from rlinf.envs.libero.agm_stage_runtime import AGMStageRuntime

        if self.pred_cfg is None:
            self.pred_cfg = make_pred_config()

        self.symbolic = build_symbolic_plan(description, task_suite_name, task_id)
        self.grounding = self.grounder.ground(self.symbolic, evidence)

        plan = self.symbolic.concrete_plan  # audited StagePlan (concrete atoms)
        if evidence is not None:
            initial_margins = compute_atom_margins(evidence, plan.atoms, self.pred_cfg)
        else:
            # No live state available (e.g. structural verification): neutral
            # baselines.  AGMStageRuntime only needs one entry per plan atom.
            initial_margins = {atom: 0.0 for atom in plan.atoms}

        self.runtime = AGMStageRuntime(
            plan=plan,
            initial_margins=initial_margins,
            confirmation_steps=self.confirmation_steps,
        )
        self.last_state = None
        return self

    def step(self, evidence: UnifiedEvidence) -> Any:
        """Advance one step: compute margins, aggregate, return StageRuntimeState.

        ``state.episode_score`` is the task robustness ``rho`` in [-1, 0.1] that
        the env's ``_calc_step_reward`` consumes; ``state.shaping_reward`` is the
        per-step delta (kept for logging, matching the AGM env).
        """
        margins = compute_atom_margins(evidence, self.runtime.plan.atoms, self.pred_cfg)
        self.last_state = self.runtime.update(margins)
        return self.last_state

    # ------------------------------------------------------------------
    # convenience accessors
    # ------------------------------------------------------------------
    @property
    def rho(self) -> float:
        """Last task robustness (== episode_score)."""
        if self.last_state is None:
            raise RuntimeError("step() has not been called yet.")
        return float(self.last_state.episode_score)

    @property
    def grounded_atoms(self):
        """Concrete atoms the privileged/vision grounder bound (for inspection)."""
        return self.grounding.atoms if self.grounding else ()
