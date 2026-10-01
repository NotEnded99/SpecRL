"""Runtime for normalized AGM rewards with ordered and alternative paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from rlinf.algorithms.rewards.stl.stlcg_task_monitor.stl_aggregation import (
    NORMALIZED_MIN,
    SATISFACTION_BOUNDARY,
    agm_and,
    atoms_satisfied,
    normalize_from_initial_negative,
)
from rlinf.envs.libero.stl_stage_plan import (
    Atom,
    StagePath,
    StagePlan,
)
class MinMaxRuntime:
    """Direct min/max aggregation over normalized plan atom margins.

    Ablation counterpart to :class:`AGMStageRuntime`: identical per-atom
    normalization (reset-time baselines via
    ``normalize_from_initial_negative``) and identical audited plan atom
    set, but the episode score is simply the instantaneous minimum over
    all normalized margins -- no stage state machine, no confirmation
    gates, no candidate-path OR, no anchoring. Exposes the same interface
    as ``AGMStageRuntime`` so the environment code is unchanged.
    """

    def __init__(
        self,
        plan: StagePlan,
        initial_margins: Mapping[Atom, float],
        satisfied_threshold: float = SATISFACTION_BOUNDARY,
    ) -> None:
        self.plan = plan
        self.initial_margins = {
            atom: float(margin)
            for atom, margin in initial_margins.items()
        }
        self.satisfied_threshold = float(satisfied_threshold)

        missing = [
            atom for atom in self.plan.atoms if atom not in self.initial_margins
        ]
        if missing:
            raise KeyError(
                f"Missing initial margins for plan atoms: {missing}"
            )

        self.last_normalized_margins = self.normalize_margins(
            self.initial_margins
        )
        initial_score = min(self.last_normalized_margins.values())
        self._previous_score = initial_score
        self._state = self._make_state(initial_score, 0.0)

    def normalize_margins(
        self,
        current_margins: Mapping[Atom, float],
    ) -> dict[Atom, float]:
        """Normalize every plan atom against its reset-time baseline."""
        missing = [
            atom
            for atom in self.plan.atoms
            if atom not in current_margins
        ]
        if missing:
            raise KeyError(
                f"Missing current margins for plan atoms: {missing}"
            )

        return {
            atom: normalize_from_initial_negative(
                current_margins[atom],
                self.initial_margins[atom],
            )
            for atom in self.plan.atoms
        }

    def update(
        self,
        current_margins: Mapping[Atom, float],
    ) -> StageRuntimeState:
        self.last_normalized_margins = self.normalize_margins(
            current_margins
        )
        score = min(self.last_normalized_margins.values())
        shaping = score - self._previous_score
        self._previous_score = score
        self._state = self._make_state(score, shaping)
        return self._state

    def _make_state(self, score: float, shaping: float) -> StageRuntimeState:
        all_complete = all(
            margin > self.satisfied_threshold
            for margin in self.last_normalized_margins.values()
        )
        return StageRuntimeState(
            active_stage_index=None,
            active_stage_name="min_max",
            completed_stages=0,
            confirmation_count=0,
            stage_score=score,
            episode_score=float(score),
            shaping_reward=float(shaping),
            advanced=False,
            all_complete=all_complete,
            active_path_index=None,
            active_path_name="min_max",
            candidate_scores=(),
        )

@dataclass(frozen=True)
class StageRuntimeState:
    """Result after one environment step."""

    active_stage_index: Optional[int]
    active_stage_name: str
    completed_stages: int
    confirmation_count: int
    stage_score: float
    episode_score: float
    shaping_reward: float
    advanced: bool
    all_complete: bool
    active_path_index: Optional[int] = None
    active_path_name: str = ""
    candidate_scores: tuple[float, ...] = ()


@dataclass(frozen=True)
class _PathState:
    active_stage_index: Optional[int]
    active_stage_name: str
    completed_stages: int
    confirmation_count: int
    stage_score: float
    episode_score: float
    advanced: bool
    all_complete: bool


class _LinearPathRuntime:
    """State machine for one legal linear candidate path."""

    def __init__(
        self,
        path: StagePath,
        initial_margins: Mapping[Atom, float],
        confirmation_steps: int,
        satisfied_threshold: float,
        normalize: bool = True,
    ) -> None:
        self.path = path
        self.initial_margins = initial_margins
        self._normalize = bool(normalize)
        self.normalized_initial_margins = {
            atom: (
                normalize_from_initial_negative(margin, margin)
                if self._normalize
                else float(margin)
            )
            for atom, margin in initial_margins.items()
        }
        self.confirmation_steps = confirmation_steps
        self.satisfied_threshold = satisfied_threshold
        self.reset()

    def reset(self) -> _PathState:
        self._active_stage_index = 0
        self._completed_stages = 0
        self._confirmation_count = 0
        self._all_complete = False

        initial_stage_score = self._score_stage(
            self._active_stage_index,
            self.normalized_initial_margins,
        )
        self._stage_anchor = initial_stage_score
        self._episode_score = NORMALIZED_MIN
        self._last_stage_score = NORMALIZED_MIN
        return self._make_state(advanced=False)

    def _stage_progress_score(self, stage_score: float) -> float:
        """Rebase the active stage to the paper-defined interval [-1, 0].

        Progress made before the stage becomes active is absorbed into the
        entry anchor and therefore earns no retrospective reward. A stage
        that is already satisfied on entry receives the fixed score 0 while
        its confirmation counter advances independently.
        """
        if self._stage_anchor < SATISFACTION_BOUNDARY:
            progress = stage_score / (-self._stage_anchor)
            return float(max(
                NORMALIZED_MIN,
                min(SATISFACTION_BOUNDARY, progress),
            ))

        if stage_score >= SATISFACTION_BOUNDARY:
            return SATISFACTION_BOUNDARY
        return NORMALIZED_MIN

    def update(
        self,
        normalized_margins: Mapping[Atom, float],
    ) -> _PathState:
        if self._all_complete:
            return self._make_state(advanced=False)

        raw_stage_score = self._score_stage(
            self._active_stage_index,
            normalized_margins,
        )
        # 路径尚未完全通过门控时，分数必须严格小于0。
        normalized_stage_margins = self._stage_margins(
            self._active_stage_index,
            normalized_margins,
        )
        stage_done = atoms_satisfied(
            normalized_stage_margins,
            threshold=self.satisfied_threshold,
        )

        if stage_done:
            self._confirmation_count += 1
        else:
            self._confirmation_count = 0

        stage_progress = self._stage_progress_score(raw_stage_score)
        self._last_stage_score = stage_progress

        num_stages = len(self.path.stages)
        stage_number = self._active_stage_index + 1
        score = -(
            num_stages - stage_number
        ) / num_stages + stage_progress / num_stages
        self._episode_score = float(
            max(
                NORMALIZED_MIN,
                min(SATISFACTION_BOUNDARY, score),
            )
        )

        advanced = False

        if self._confirmation_count >= self.confirmation_steps:
            self._completed_stages += 1
            self._active_stage_index += 1
            self._confirmation_count = 0
            advanced = True

            if self._active_stage_index >= len(self.path.stages):
                # The final gate was stably satisfied. A completed
                # specification receives the paper-defined zero endpoint.
                self._all_complete = True
                self._episode_score = SATISFACTION_BOUNDARY
                self._last_stage_score = SATISFACTION_BOUNDARY
            else:
                # Rebase the newly active stage at the transition. Its lower
                # interval boundary equals the preceding stage's upper bound.
                self._stage_anchor = self._score_stage(
                    self._active_stage_index,
                    normalized_margins,
                )
                self._last_stage_score = NORMALIZED_MIN

        return self._make_state(advanced=advanced)

    def _stage_margins(
        self,
        stage_index: int,
        margins: Mapping[Atom, float],
    ) -> list[float]:
        stage = self.path.stages[stage_index]
        missing = [atom for atom in stage.atoms if atom not in margins]
        if missing:
            raise KeyError(
                f"Missing margins for path={self.path.name}, "
                f"stage={stage.name}: {missing}"
            )
        return [float(margins[atom]) for atom in stage.atoms]

    def _score_stage(
        self,
        stage_index: int,
        margins: Mapping[Atom, float],
    ) -> float:
        normalized = self._stage_margins(
            stage_index,
            margins,
        )
        return agm_and(normalized)

    def _make_state(self, advanced: bool) -> _PathState:
        if self._all_complete:
            return _PathState(
                active_stage_index=None,
                active_stage_name="complete",
                completed_stages=self._completed_stages,
                confirmation_count=0,
                stage_score=SATISFACTION_BOUNDARY,
                episode_score=SATISFACTION_BOUNDARY,
                advanced=advanced,
                all_complete=True,
            )

        stage = self.path.stages[self._active_stage_index]
        return _PathState(
            active_stage_index=self._active_stage_index,
            active_stage_name=stage.name,
            completed_stages=self._completed_stages,
            confirmation_count=self._confirmation_count,
            stage_score=self._last_stage_score,
            episode_score=self._episode_score,
            advanced=advanced,
            all_complete=False,
        )


class AGMStageRuntime:
    """Evaluate all legal task paths and accept any completed specification.

    Each path has an independent temporal gate. The exposed episode potential
    is the maximum candidate-path potential, corresponding to logical OR over
    legal specifications. Therefore a two-object task may execute object 1
    first or object 2 first without receiving a false ordering penalty.
    """

    def __init__(
        self,
        plan: StagePlan,
        initial_margins: Mapping[Atom, float],
        confirmation_steps: int = 3,
        satisfied_threshold: float = SATISFACTION_BOUNDARY,
        normalize: bool = True,
    ) -> None:
        if confirmation_steps < 1:
            raise ValueError(
                "confirmation_steps must be >= 1, "
                f"got {confirmation_steps}."
            )

        self.plan = plan
        self.initial_margins = {
            atom: float(margin)
            for atom, margin in initial_margins.items()
        }
        self.confirmation_steps = int(confirmation_steps)
        self.satisfied_threshold = float(satisfied_threshold)
        self._normalize = bool(normalize)

        missing = [
            atom
            for atom in self.plan.atoms
            if atom not in self.initial_margins
        ]
        if missing:
            raise KeyError(
                f"Missing initial margins for plan atoms: {missing}"
            )

        self._paths = tuple(
            _LinearPathRuntime(
                path=path,
                initial_margins=self.initial_margins,
                confirmation_steps=self.confirmation_steps,
                satisfied_threshold=self.satisfied_threshold,
                normalize=self._normalize,
            )
            for path in self.plan.paths
        )
        self.reset()

    def reset(self) -> StageRuntimeState:
        states = tuple(path.reset() for path in self._paths)
        self._previous_score = NORMALIZED_MIN
        self.last_normalized_margins = self.normalize_margins(
            self.initial_margins
        )
        return self._combine(states, shaping_reward=0.0)

    def normalize_margins(
        self,
        current_margins: Mapping[Atom, float],
    ) -> dict[Atom, float]:
        """Normalize every plan atom for logging and downstream inspection.

        With normalize=False (ablation), returns the raw margins unchanged;
        downstream scores then live on the raw scale clipped by agm_and.
        """
        missing = [
            atom
            for atom in self.plan.atoms
            if atom not in current_margins
        ]
        if missing:
            raise KeyError(
                f"Missing current margins for plan atoms: {missing}"
            )

        if not self._normalize:
            return {
                atom: float(current_margins[atom])
                for atom in self.plan.atoms
            }

        return {
            atom: normalize_from_initial_negative(
                current_margins[atom],
                self.initial_margins[atom],
            )
            for atom in self.plan.atoms
        }

    def update(
        self,
        current_margins: Mapping[Atom, float],
    ) -> StageRuntimeState:
        self.last_normalized_margins = self.normalize_margins(
            current_margins
        )
        states = tuple(
            path.update(self.last_normalized_margins)
            for path in self._paths
        )

        episode_score = max(state.episode_score for state in states)
        shaping_reward = episode_score - self._previous_score
        self._previous_score = episode_score

        return self._combine(
            states,
            shaping_reward=shaping_reward,
        )

    def _combine(
        self,
        states: tuple[_PathState, ...],
        shaping_reward: float,
    ) -> StageRuntimeState:
        # Prefer a completed path; otherwise prefer greater potential, then
        # more completed gates, while keeping path order deterministic on ties.
        best_index = max(
            range(len(states)),
            key=lambda index: (
                states[index].all_complete,
                states[index].episode_score,
                states[index].completed_stages,
                -index,
            ),
        )
        best = states[best_index]
        all_complete = any(state.all_complete for state in states)
        episode_score = max(state.episode_score for state in states)

        if all_complete:
            episode_score = SATISFACTION_BOUNDARY

        return StageRuntimeState(
            active_stage_index=(
                None if all_complete else best.active_stage_index
            ),
            active_stage_name=(
                "complete" if all_complete else best.active_stage_name
            ),
            completed_stages=best.completed_stages,
            confirmation_count=(
                0 if all_complete else best.confirmation_count
            ),
            stage_score=(
                SATISFACTION_BOUNDARY if all_complete else best.stage_score
            ),
            episode_score=float(episode_score),
            shaping_reward=float(shaping_reward),
            advanced=any(state.advanced for state in states),
            all_complete=all_complete,
            active_path_index=(None if all_complete else best_index),
            active_path_name=(
                self.plan.paths[best_index].name
                if not all_complete
                else self.plan.paths[best_index].name
            ),
            candidate_scores=tuple(
                float(state.episode_score) for state in states
            ),
        )
