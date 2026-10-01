"""Independent visual-only AGM runtime with stage-local perception.

Every plan atom is observed and normalized throughout the episode. Only atoms
in a candidate path's current stage are consumed by the AGM gate, so future
observations cannot advance temporal confirmation or complete a stage early.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

from rlinf.algorithms.rewards.stl.stlcg_task_monitor.stl_aggregation import (
    NORMALIZED_MIN,
    SATISFACTION_BOUNDARY,
    agm_and,
    normalize_from_initial_negative,
)
from rlinf.envs.libero.stl_stage_plan import Atom, StagePlan, canonical_atom


@dataclass(frozen=True)
class VisualAtomSample:
    """One visual observation for one canonical audited atom."""

    margin: float
    valid: bool
    reason: str = ""
    source: str = ""


@dataclass(frozen=True)
class VisualAGMStep:
    """Visual task state emitted for plotting and offline comparison."""

    step: int
    ready: bool
    valid: bool
    reason: str
    raw_margins: dict[Atom, float]
    normalized_margins: dict[Atom, float]
    atom_valid: dict[Atom, bool]
    atom_reasons: dict[Atom, str]
    episode_score: float
    active_stage_name: str
    completed_stages: int
    all_complete: bool


@dataclass
class _VisualPathState:
    active_stage_index: int | None = 0
    completed_stages: int = 0
    confirmation_count: int = 0
    stage_anchor: float | None = None
    stage_score: float = NORMALIZED_MIN
    episode_score: float = NORMALIZED_MIN
    all_complete: bool = False


class VisualAGMShadowRuntime:
    """Run an audited stage plan using current-stage visual margins only.

    A normalization baseline is captured when an atom first becomes valid in
    the episode. A positive first observation gets an equal-magnitude negative
    reference. Future atoms are plotted immediately, but can pass only after
    they become active and satisfy the normal confirmation interval.

    Invalid active observations freeze the path. Completed stages remain
    latched while all atom observations continue for plotting.
    """

    def __init__(
        self,
        plan: StagePlan,
        *,
        confirmation_steps: int = 3,
    ) -> None:
        self.plan = plan
        self.confirmation_steps = int(confirmation_steps)
        if self.confirmation_steps < 1:
            raise ValueError("confirmation_steps must be >= 1")

        self._atoms = tuple(canonical_atom(atom) for atom in plan.atoms)
        self._initial_margins: dict[Atom, float] = {}
        self._last_normalized_margins: dict[Atom, float] = {}
        self._path_states = [_VisualPathState() for _ in plan.paths]
        self.trace: list[VisualAGMStep] = []

    @property
    def ready(self) -> bool:
        """Whether at least one active path has a visual baseline."""
        return any(
            state.stage_anchor is not None or state.all_complete
            for state in self._path_states
        )

    @property
    def initial_margins(self) -> dict[Atom, float]:
        return dict(self._initial_margins)

    @property
    def active_atoms(self) -> tuple[Atom, ...]:
        """Union of atoms needed by incomplete paths at this instant."""
        active: list[Atom] = []
        seen: set[Atom] = set()
        for path, state in zip(self.plan.paths, self._path_states):
            if state.all_complete or state.active_stage_index is None:
                continue
            for atom in path.stages[state.active_stage_index].atoms:
                atom = canonical_atom(atom)
                if atom not in seen:
                    seen.add(atom)
                    active.append(atom)
        return tuple(active)

    @property
    def tracked_atoms(self) -> tuple[Atom, ...]:
        """All plan atoms observed for the full episode."""
        return self._atoms

    def _canonical_samples(
        self,
        samples: Mapping[Atom, VisualAtomSample],
    ) -> dict[Atom, VisualAtomSample]:
        canonical = {
            canonical_atom(atom): sample for atom, sample in samples.items()
        }
        unexpected = [atom for atom in canonical if atom not in self._atoms]
        if unexpected:
            raise KeyError(f"samples contain atoms outside the plan: {unexpected}")
        return canonical

    @staticmethod
    def _sample_is_valid(sample: VisualAtomSample | None) -> bool:
        return bool(
            sample is not None
            and sample.valid
            and isfinite(float(sample.margin))
        )

    @staticmethod
    def _negative_reference(margin: float) -> float:
        value = float(margin)
        if value < 0.0:
            return value
        return -max(abs(value), 1.0e-6)

    def _active_stage_atoms(self, path_index: int) -> tuple[Atom, ...]:
        state = self._path_states[path_index]
        if state.all_complete or state.active_stage_index is None:
            return ()
        return tuple(
            canonical_atom(atom)
            for atom in self.plan.paths[path_index]
            .stages[state.active_stage_index]
            .atoms
        )

    def _advance_path(
        self,
        path_index: int,
        current: Mapping[Atom, VisualAtomSample],
    ) -> tuple[bool, dict[Atom, str]]:
        state = self._path_states[path_index]
        if state.all_complete:
            return False, {}

        stage_atoms = self._active_stage_atoms(path_index)
        missing = [
            atom
            for atom in stage_atoms
            if not self._sample_is_valid(current.get(atom))
        ]
        if missing:
            return False, {
                atom: (
                    current[atom].reason if atom in current else "sample_missing"
                )
                for atom in missing
            }

        normalized: list[float] = []
        for atom in stage_atoms:
            raw = float(current[atom].margin)
            if atom not in self._initial_margins:
                self._initial_margins[atom] = self._negative_reference(raw)
            value = float(normalize_from_initial_negative(
                raw,
                self._initial_margins[atom],
            ))
            self._last_normalized_margins[atom] = value
            normalized.append(value)

        raw_stage_score = float(agm_and(normalized))
        if state.stage_anchor is None:
            state.stage_anchor = raw_stage_score

        num_stages = len(self.plan.paths[path_index].stages)
        if all(value >= SATISFACTION_BOUNDARY for value in normalized):
            state.confirmation_count += 1
        else:
            state.confirmation_count = 0

        if state.stage_anchor < SATISFACTION_BOUNDARY:
            stage_progress = raw_stage_score / (-state.stage_anchor)
            stage_progress = float(max(
                NORMALIZED_MIN,
                min(SATISFACTION_BOUNDARY, stage_progress),
            ))
        else:
            stage_progress = (
                SATISFACTION_BOUNDARY
                if raw_stage_score >= SATISFACTION_BOUNDARY
                else NORMALIZED_MIN
            )

        state.stage_score = stage_progress
        stage_number = int(state.active_stage_index or 0) + 1
        score = -(
            num_stages - stage_number
        ) / num_stages + stage_progress / num_stages
        state.episode_score = float(max(
            NORMALIZED_MIN,
            min(SATISFACTION_BOUNDARY, score),
        ))

        if state.confirmation_count < self.confirmation_steps:
            return True, {}

        state.completed_stages += 1
        state.confirmation_count = 0
        next_index = int(state.active_stage_index or 0) + 1
        if next_index >= num_stages:
            state.active_stage_index = None
            state.all_complete = True
            state.stage_score = SATISFACTION_BOUNDARY
            state.episode_score = SATISFACTION_BOUNDARY
        else:
            state.active_stage_index = next_index
            state.stage_anchor = None
            state.stage_score = NORMALIZED_MIN
        return True, {}

    def _best_index(self) -> int:
        return max(
            range(len(self._path_states)),
            key=lambda index: (
                self._path_states[index].all_complete,
                self._path_states[index].episode_score,
                self._path_states[index].completed_stages,
                -index,
            ),
        )

    def observe(
        self,
        step: int,
        samples: Mapping[Atom, VisualAtomSample],
    ) -> VisualAGMStep:
        current = self._canonical_samples(samples)
        atom_valid = {
            atom: self._sample_is_valid(current.get(atom))
            for atom in self._atoms
        }
        atom_reasons = {
            atom: (
                str(current[atom].reason)
                if atom in current else "sample_missing"
            )
            for atom in self._atoms
        }
        raw = {
            atom: (
                float(current[atom].margin)
                if atom_valid[atom]
                else float("nan")
            )
            for atom in self._atoms
        }

        # Establish each atom's one episode-long baseline as soon as its
        # visual channel becomes valid. This same baseline is used by the
        # plotted curve and by AGM if the atom becomes active later.
        for atom in self._atoms:
            if atom_valid[atom] and atom not in self._initial_margins:
                self._initial_margins[atom] = self._negative_reference(
                    float(current[atom].margin)
                )

        updated_paths: list[int] = []
        frozen_reasons: dict[str, dict[Atom, str]] = {}
        for path_index, path in enumerate(self.plan.paths):
            updated, missing = self._advance_path(path_index, current)
            if updated:
                updated_paths.append(path_index)
            elif missing:
                frozen_reasons[path.name] = missing

        normalized: dict[Atom, float] = {}
        for atom in self._atoms:
            # Every atom is normalized for plotting. Only _advance_path reads
            # current-stage values into the gated episode score.
            if atom_valid[atom] and atom in self._initial_margins:
                value = float(normalize_from_initial_negative(
                    float(current[atom].margin),
                    self._initial_margins[atom],
                ))
                self._last_normalized_margins[atom] = value
                normalized[atom] = value
            else:
                normalized[atom] = float("nan")
        all_complete = any(state.all_complete for state in self._path_states)
        best_index = self._best_index()
        best = self._path_states[best_index]
        ready = self.ready
        valid = bool(updated_paths or all_complete)
        episode_score = (
            SATISFACTION_BOUNDARY
            if all_complete
            else (
                max(state.episode_score for state in self._path_states)
                if valid else float("nan")
            )
        )
        if not ready:
            reason = f"visual_active_stage_unavailable:{frozen_reasons}"
        elif not valid:
            reason = f"visual_active_stage_invalid:{frozen_reasons}"
        else:
            reason = ""

        result = VisualAGMStep(
            step=int(step),
            ready=ready,
            valid=valid,
            reason=reason,
            raw_margins=raw,
            normalized_margins=normalized,
            atom_valid=atom_valid,
            atom_reasons=atom_reasons,
            episode_score=float(episode_score),
            active_stage_name=(
                "complete"
                if all_complete
                else self.plan.paths[best_index]
                .stages[int(best.active_stage_index or 0)]
                .name
            ),
            completed_stages=int(best.completed_stages),
            all_complete=all_complete,
        )
        self.trace.append(result)
        return result
