"""Additive pure-visual LIBERO AGM environment for PPO training.

This module has a distinct name so the senior's ``libero_env.py`` and
``libero_env_agm_stage.py`` remain byte-for-byte untouched.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import json
import numpy as np
import os
import re
from rlinf.envs.libero.agm_stage_runtime import AGMStageRuntime
from rlinf.envs.libero.libero_env import (
    LiberoEnv as BaseLiberoEnv,
    OnlineRobustnessTracker,
    _GEOMETRIC_PREDS,
    _STLEnvState,
    _build_step_episode,
    _normalize_goal_atoms,
    _stl_atom_name,
    get_bddl_goal_atoms,
    goal_atom_margin,
    resolve_object_key,
)
from rlinf.envs.libero.stl_stage_plan import (
    Atom,
    StagePlan,
    build_stage_plan,
)
from rlinf.envs.libero.visual_yoloe_rollout_exporter import (
    VisualYoloeRolloutExporter,
)
from rlinf.algorithms.rewards.stl.stlcg_task_monitor.predicates import (
    contact_force_on_object,
    eef_object_dist,
    gripper_width,
)
from rlinf.algorithms.rewards.stl.stlcg_task_monitor.stl_aggregation import (
    discounted_progress_reward,
)
from rlinf.envs.libero.agm_pick_predicate import (
    pred_pick_agm,
    two_finger_grasp,
)
from rlinf.envs.libero.visual_stl_shadow import (
    VisualSTLShadowMonitor,
)
from rlinf.envs.libero.visual_in_margin import VisualInConfig
from rlinf.envs.libero.visual_open_margin import VisualOpenConfig
from rlinf.envs.libero.visual_close_margin import VisualCloseConfig
from rlinf.envs.libero.visual_drawer_close_margin import (
    VisualDrawerCloseConfig,
)
from rlinf.envs.libero.visual_turnon_observer import VisualTurnOnConfig
from rlinf.envs.libero.visual_microwave_close_margin import (
    VisualMicrowaveCloseConfig,
)
from rlinf.envs.libero.visual_agm_shadow_runtime import (
    VisualAGMShadowRuntime,
    VisualAtomSample,
)
from rlinf.envs.libero.visual_stl_comparison_plot import (
    STLComparisonFrame,
    atom_label,
    plot_stl_comparison,
    write_stl_comparison_csv,
)
from rlinf.envs.libero.visual_venv import (
    ReconfigureSubprocEnv as VisualReconfigureSubprocEnv,
)
from rlinf.envs.libero.utils import (
    distribute_reset_state_ids_round_robin,
)
from rlinf.envs.utils import to_tensor


_VISUAL_COLD_START_SCORE = -1.0
_VISUAL_SITE_CONTACT_SLACK_M = 0.007
_VISUAL_SITE_CONTACT_RELEASE_WINDOW_STEPS = 5
_VISUAL_SITE_CONTACT_CONFIRMATION_STEPS = 3
_VISUAL_MULTI_IN_MAX_DISPLACEMENT_M = 0.04


def _one_shot_visual_completion_bonus(
    visual_complete,
    bonus_paid,
    *,
    bonus_value: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pay once when the pure-visual AGM runtime reaches all_complete."""
    complete = np.asarray(visual_complete, dtype=bool)
    paid = np.asarray(bonus_paid, dtype=bool)
    if complete.shape != paid.shape:
        raise ValueError(
            "visual_complete and bonus_paid must have the same shape"
        )
    value = float(bonus_value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(
            "visual completion bonus must be finite and nonnegative"
        )
    newly_successful = complete & ~paid
    bonus = newly_successful.astype(np.float64) * value
    return bonus, paid | complete, complete


def _visual_telemetry_json_value(value):
    """Convert diagnostics-only visual values to strict JSON primitives."""
    if isinstance(value, np.ndarray):
        return [_visual_telemetry_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _visual_telemetry_json_value(value.item())
    if isinstance(value, dict):
        return {
            str(key): _visual_telemetry_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_visual_telemetry_json_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _visual_site_contact_sequence_complete(records) -> bool:
    """Validate a visible site placement near its observed release event."""
    release_step = next(
        (
            int(record.step)
            for record in records
            if bool(record.visual_on_released)
        ),
        None,
    )
    if release_step is None:
        return False
    positive_count = 0
    for record in records:
        age = int(record.step) - release_step
        if age < 0:
            continue
        if age > _VISUAL_SITE_CONTACT_RELEASE_WINDOW_STEPS:
            break
        values = (
            float(record.visual_on_margin),
            float(record.visual_on_contact_margin),
        )
        positive = bool(
            record.visual_on_valid
            and record.visual_on_object_source == "fresh_rgbd"
            and record.visual_on_contact_valid
            and np.all(np.isfinite(values))
            and values[0] >= 0.0
            and values[1] >= -_VISUAL_SITE_CONTACT_SLACK_M
        )
        positive_count = positive_count + 1 if positive else 0
        if positive_count >= _VISUAL_SITE_CONTACT_CONFIRMATION_STEPS:
            return True
    return False


def _visual_in_target_center(record) -> np.ndarray | None:
    center = np.asarray(
        [
            record.target_center_world_x_m,
            record.target_center_world_y_m,
            record.target_center_world_z_m,
        ],
        dtype=np.float64,
    )
    return center if np.all(np.isfinite(center)) else None


def _visual_multi_in_group_safe(record_groups) -> bool:
    """Revalidate a multi-object container and reject displaced relations."""
    confirmed = []
    for records in record_groups:
        if not records:
            return False
        current = records[-1]
        current_center = _visual_in_target_center(current)
        current_geometry = np.asarray(
            [
                current.inside_xy_margin,
                current.below_rim_margin,
                current.above_floor_margin,
            ],
            dtype=np.float64,
        )
        if not bool(
            current.valid
            and current.visual_fresh
            and current.visual_in_confirmed
            and current_center is not None
            and np.all(np.isfinite(current_geometry))
            and np.min(current_geometry) >= 0.0
        ):
            return False
        first = next(
            (
                record
                for record in records
                if record.valid
                and record.visual_fresh
                and record.visual_in_confirmed
                and _visual_in_target_center(record) is not None
            ),
            None,
        )
        if first is None:
            return False
        confirmed.append(
            (
                int(first.step),
                _visual_in_target_center(first),
                current_center,
            )
        )

    # The final relation may settle during its own confirmation. Earlier
    # relations must remain where they were visually established.
    confirmed.sort(key=lambda item: item[0])
    return all(
        float(np.linalg.norm(current - initial))
        <= _VISUAL_MULTI_IN_MAX_DISPLACEMENT_M
        for _, initial, current in confirmed[:-1]
    )


def _visual_shadow_atom_matches_preference(
    atom: Atom,
    preferred_on_target: str | None,
) -> bool:
    """Opt-in diagnostic routing without changing canonical AGM stages.

    The empty/default preference preserves the historical first-atom shadow
    route.  A non-empty value selects one exact canonical On target solely for
    visual diagnostics; privileged margin and reward evaluation still iterate
    over every original goal atom unchanged.
    """
    preferred = str(preferred_on_target or "").strip()
    if not preferred:
        return True
    pred, _, target = atom
    return pred == "on" and str(target or "") == preferred


def _causal_median_margin(
    history: deque[float],
    margin: float,
    window: int,
) -> float:
    """Append one finite margin and return its causal trailing median."""
    if window < 1:
        raise ValueError("temporal median window must be positive")
    if history.maxlen != window:
        raise ValueError("temporal median history has the wrong maxlen")
    history.append(float(margin))
    return float(np.median(np.asarray(history, dtype=np.float64)))


def _stabilize_margin_jump(
    state: dict[str, float | int],
    margin: float,
    *,
    max_jump: float,
    confirmation_steps: int,
    consistency_tolerance: float,
) -> tuple[float, bool]:
    """Hold an isolated raw-margin jump until a new level repeats.

    This operates after the raw causal median and before EMA/AGM normalization.
    """
    value = float(margin)
    if not np.isfinite(value):
        raise ValueError("visual margin must be finite")
    if max_jump < 0.0:
        raise ValueError("visual max jump must be nonnegative")
    if confirmation_steps < 1:
        raise ValueError("visual jump confirmation steps must be positive")
    if consistency_tolerance < 0.0:
        raise ValueError(
            "visual jump consistency tolerance must be nonnegative"
        )

    accepted = float(state.get("accepted", float("nan")))
    if not np.isfinite(accepted):
        state.pop("pending", None)
        state.pop("pending_count", None)
        state["accepted"] = value
        return value, False

    if abs(value - accepted) <= max_jump:
        state.pop("pending", None)
        state.pop("pending_count", None)
        state["accepted"] = value
        return value, False

    pending = float(state.get("pending", float("nan")))
    if (
        np.isfinite(pending)
        and abs(value - pending) <= consistency_tolerance
    ):
        pending_count = int(state.get("pending_count", 1)) + 1
        pending = (
            pending * float(pending_count - 1) + value
        ) / float(pending_count)
    else:
        pending = value
        pending_count = 1

    if pending_count >= confirmation_steps:
        state.pop("pending", None)
        state.pop("pending_count", None)
        state["accepted"] = pending
        return float(pending), False

    state["pending"] = pending
    state["pending_count"] = pending_count
    return accepted, True


def _causal_ema_margin(
    state: dict[str, float | int],
    margin: float,
    alpha: float,
) -> float:
    """Update a causal EMA using only the current and previous values."""
    value = float(margin)
    if not np.isfinite(value):
        raise ValueError("EMA margin must be finite")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("EMA alpha must be in (0, 1]")
    previous = float(state.get("ema", float("nan")))
    output = (
        value
        if not np.isfinite(previous)
        else alpha * value + (1.0 - alpha) * previous
    )
    state["ema"] = float(output)
    return float(output)


def _interleave_filtered_eval_reset_state_ids(
    task_ids: list[int],
    trial_id_bins: list[int],
    cumsum_trial_id_bins: np.ndarray,
) -> np.ndarray:
    """Interleave selected tasks so the first eval batch is diverse."""
    selected: list[int] = []
    for raw_task_id in task_ids:
        task_id = int(raw_task_id)
        if task_id not in selected:
            selected.append(task_id)

    if not selected:
        raise ValueError("task_id_filter must contain at least one task")

    reset_state_ids: list[int] = []
    max_trials = max(int(trial_id_bins[task_id]) for task_id in selected)
    for trial_id in range(max_trials):
        for task_id in selected:
            if trial_id >= int(trial_id_bins[task_id]):
                continue
            task_start = (
                int(cumsum_trial_id_bins[task_id - 1])
                if task_id > 0 else 0
            )
            reset_state_ids.append(task_start + trial_id)

    return np.asarray(reset_state_ids, dtype=np.int64)


class LiberoEnv(BaseLiberoEnv):
    """Pure-visual reward environment selected only by the visual launcher."""

    def get_reset_state_ids_all(self):
        """Keep filtered eval tasks interleaved without changing the base env."""
        if not self.is_eval or self.task_id_filter is None:
            return super().get_reset_state_ids_all()

        reset_state_ids = _interleave_filtered_eval_reset_state_ids(
            self.task_id_filter,
            self.trial_id_bins,
            self.cumsum_trial_id_bins,
        )
        return distribute_reset_state_ids_round_robin(
            reset_state_ids,
            self.total_num_processes,
        )

    def _init_env(self):
        self.env = VisualReconfigureSubprocEnv(self.get_env_fns())

    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
    ):
        super().__init__(
            cfg,
            num_envs,
            seed_offset,
            total_num_processes,
            worker_info,
        )
        self._visual_yoloe_rollout_exporter = None
        if bool(cfg.get("visual_stl_export_yoloe_dataset", False)):
            export_errors = []
            export_dir = str(
                cfg.get("visual_stl_export_yoloe_dir", "")
            ).strip()
            if not bool(cfg.get("is_eval", False)):
                export_errors.append("is_eval must be true")
            if not self._uses_visual_agm_reward():
                export_errors.append("agm_reward_source must be visual")
            if not export_dir:
                export_errors.append(
                    "visual_stl_export_yoloe_dir must be nonempty"
                )
            if export_errors:
                raise ValueError(
                    "Invalid diagnostics-only YOLOE rollout export: "
                    + "; ".join(export_errors)
                )
            self._visual_yoloe_rollout_exporter = (
                VisualYoloeRolloutExporter(
                    output_root=export_dir,
                    interval=int(
                        cfg.get("visual_stl_export_yoloe_interval", 5)
                    ),
                    cameras=tuple(cfg.get(
                        "visual_stl_export_yoloe_cameras",
                        ["agentview", "robot0_eye_in_hand"],
                    )),
                    task_ids=tuple(cfg.get(
                        "visual_stl_export_yoloe_task_ids", [35]
                    )),
                    required_class_names=tuple(cfg.get(
                        "visual_stl_export_yoloe_required_class_names",
                        ["black book", "desk caddy"],
                    )),
                    val_every=int(
                        cfg.get("visual_stl_export_yoloe_val_every", 5)
                    ),
                    min_mask_pixels=int(cfg.get(
                        "visual_stl_export_yoloe_min_mask_pixels", 20
                    )),
                    min_contour_area=float(cfg.get(
                        "visual_stl_export_yoloe_min_contour_area", 20.0
                    )),
                )
            )
        self._visual_temporal_median_window = int(
            cfg.get("visual_stl_temporal_median_window", 5)
        )
        self._visual_temporal_ema_alpha = float(
            cfg.get("visual_stl_temporal_ema_alpha", 0.30)
        )
        self._visual_temporal_max_jump = float(
            cfg.get("visual_stl_temporal_max_jump", 0.05)
        )
        self._visual_temporal_jump_confirmation_steps = int(
            cfg.get(
                "visual_stl_temporal_jump_confirmation_steps", 2
            )
        )
        self._visual_temporal_jump_consistency = float(
            cfg.get("visual_stl_temporal_jump_consistency", 0.025)
        )
        if self._uses_visual_agm_reward():
            errors = []
            if not self._stl_enabled:
                errors.append("stl_reward must be true")
            if not self._visual_stl_enabled():
                errors.append("visual_stl_monitor must be true")
            if float(cfg.get("reward_coef", 0.0)) != 0.0:
                errors.append("reward_coef must be 0.0")
            if not bool(cfg.get("ignore_terminations", False)):
                errors.append("ignore_terminations must be true")
            if bool(cfg.get("use_step_penalty", False)):
                errors.append("use_step_penalty must be false")
            success_bonus = float(cfg.get(
                "visual_stl_completion_bonus", 1.0
            ))
            if not np.isfinite(success_bonus) or success_bonus < 0.0:
                errors.append(
                    "visual_stl_completion_bonus must be finite and "
                    "nonnegative"
                )
            self._visual_completion_bonus_value = success_bonus
            if self._visual_temporal_median_window < 1:
                errors.append(
                    "visual_stl_temporal_median_window must be positive"
                )
            if not 0.0 < self._visual_temporal_ema_alpha <= 1.0:
                errors.append(
                    "visual_stl_temporal_ema_alpha must be in (0, 1]"
                )
            if self._visual_temporal_max_jump < 0.0:
                errors.append(
                    "visual_stl_temporal_max_jump must be nonnegative"
                )
            if self._visual_temporal_jump_confirmation_steps < 1:
                errors.append(
                    "visual_stl_temporal_jump_confirmation_steps must be "
                    "positive"
                )
            if self._visual_temporal_jump_consistency < 0.0:
                errors.append(
                    "visual_stl_temporal_jump_consistency must be nonnegative"
                )
            if (
                self._visual_oracle_comparison_enabled()
                and not bool(cfg.get("is_eval", False))
            ):
                errors.append(
                    "visual_stl_oracle_comparison is diagnostics-only and "
                    "requires is_eval=true"
                )
            if errors:
                raise ValueError(
                    "Invalid pure-visual AGM reward configuration: "
                    + "; ".join(errors)
                )

    def _agm_reward_source(self) -> str:
        source = str(
            self.cfg.get("agm_reward_source", "privileged")
        ).strip().lower()
        if source not in {"privileged", "visual"}:
            raise ValueError(
                "agm_reward_source must be 'privileged' or 'visual', "
                f"got {source!r}"
            )
        return source

    def _uses_visual_agm_reward(self) -> bool:
        return self._agm_reward_source() == "visual"

    def _setup_visual_yoloe_rollout_export(self, ids) -> None:
        """Register reset observations for the opt-in offline exporter."""
        exporter = getattr(self, "_visual_yoloe_rollout_exporter", None)
        if exporter is None or not ids:
            return
        mappings = self.env.get_instance_id_to_name(id=ids)
        for position, raw_env_id in enumerate(ids):
            env_id = int(raw_env_id)
            mapping = mappings[position] if position < len(mappings) else {}
            if not mapping:
                raise RuntimeError(
                    "YOLOE rollout export could not read the simulator "
                    f"instance mapping for env {env_id}"
                )
            exporter.start_episode(
                env_id=env_id,
                task_id=int(np.asarray(self.task_ids).reshape(-1)[env_id]),
                trial_id=int(np.asarray(self.trial_ids).reshape(-1)[env_id]),
                instance_id_to_name=mapping,
                raw_obs=(
                    self.current_raw_obs[env_id]
                    if self.current_raw_obs is not None else None
                ),
            )

    def _record_visual_yoloe_rollout_export(self) -> None:
        """Persist sampled frames; exported labels never feed runtime logic."""
        exporter = getattr(self, "_visual_yoloe_rollout_exporter", None)
        if exporter is None or self.current_raw_obs is None:
            return
        elapsed = np.asarray(self._elapsed_steps).reshape(-1)
        for env_id in range(self.num_envs):
            exporter.record_step(
                env_id=env_id,
                step=int(elapsed[env_id]),
                raw_obs=self.current_raw_obs[env_id],
            )

    def _visual_oracle_comparison_enabled(self) -> bool:
        return bool(
            self.cfg.get("visual_stl_oracle_comparison", False)
        )

    def _oracle_comparison_setup_states(self, ids) -> dict[int, object]:
        """Read privileged state only for explicitly enabled eval plots."""
        if not self._visual_oracle_comparison_enabled():
            return {}
        try:
            privileged = self.env.get_privileged_state(id=ids)
            return {
                int(raw_env_id): privileged[position]
                for position, raw_env_id in enumerate(ids)
                if position < len(privileged)
            }
        except Exception as exc:
            print(
                "[VISUAL_ORACLE_COMPARISON_SETUP_ERROR] "
                f"error={exc!r}",
                flush=True,
            )
            return {}

    def _oracle_comparison_step_states(self) -> list[object | None]:
        """Read eval-only oracle frames; callers never use them as reward."""
        if not self._visual_oracle_comparison_enabled():
            return [None] * self.num_envs
        try:
            return list(self.env.get_privileged_state())
        except Exception as exc:
            print(
                "[VISUAL_ORACLE_COMPARISON_ERROR] "
                f"error={exc!r}",
                flush=True,
            )
            return [None] * self.num_envs

    def _setup_oracle_comparison_runtime(
        self,
        env_id: int,
        plan: StagePlan,
        confirmation_steps: int,
        privileged,
    ) -> None:
        """Construct the diagnostic oracle runtime outside visual setup."""
        if privileged is None:
            return
        oracle_ep0 = _build_step_episode(privileged)
        oracle_initial = {
            atom: self._raw_atom_margin(oracle_ep0, atom)
            for atom in plan.atoms
        }
        self._visual_oracle_comparison_runtimes[env_id] = AGMStageRuntime(
            plan=plan,
            initial_margins=oracle_initial,
            confirmation_steps=confirmation_steps,
        )

    def _filter_visual_samples(
        self,
        env_id: int,
        samples: dict[Atom, VisualAtomSample],
    ) -> dict[Atom, VisualAtomSample]:
        """Smooth valid atomic margins before they enter visual AGM."""
        histories = self._visual_margin_histories.setdefault(env_id, {})
        invalid_streaks = self._visual_margin_invalid_streaks.setdefault(
            env_id, {}
        )
        filter_states = self._visual_filter_states.setdefault(
            env_id, {}
        )
        return self._filter_visual_sample_map(
            samples,
            histories,
            invalid_streaks,
            filter_states,
        )

    def _filter_visual_sample_map(
        self,
        samples: dict[Atom, VisualAtomSample],
        histories: dict[Atom, deque],
        invalid_streaks: dict[Atom, int],
        filter_states: dict[Atom, dict[str, float | int]],
    ) -> dict[Atom, VisualAtomSample]:
        """Median, jump-confirm, and EMA raw margins before normalization."""
        window = int(self._visual_temporal_median_window)

        filtered: dict[Atom, VisualAtomSample] = {}

        for atom, sample in samples.items():
            margin = float(sample.margin)
            if not sample.valid or not np.isfinite(margin):
                streak = int(invalid_streaks.get(atom, 0)) + 1
                invalid_streaks[atom] = streak
                if streak >= window:
                    histories.pop(atom, None)
                filtered[atom] = sample
                continue

            invalid_streaks[atom] = 0
            atom_history = histories.setdefault(
                atom, deque(maxlen=window)
            )
            median_margin = _causal_median_margin(
                atom_history,
                margin,
                window,
            )
            atom_filter_state = filter_states.setdefault(atom, {})
            stable_margin, held_jump = _stabilize_margin_jump(
                atom_filter_state,
                median_margin,
                max_jump=self._visual_temporal_max_jump,
                confirmation_steps=(
                    self._visual_temporal_jump_confirmation_steps
                ),
                consistency_tolerance=(
                    self._visual_temporal_jump_consistency
                ),
            )
            ema_margin = _causal_ema_margin(
                atom_filter_state,
                stable_margin,
                self._visual_temporal_ema_alpha,
            )
            source = str(sample.source or "")
            filter_source = (
                f"temporal_median_w{window}"
                f"|temporal_ema_a{self._visual_temporal_ema_alpha:g}"
            )
            if held_jump:
                filter_source = (
                    "temporal_jump_hold|" + filter_source
                )
            filtered[atom] = VisualAtomSample(
                margin=ema_margin,
                valid=True,
                reason=sample.reason,
                source=(
                    f"{source}|{filter_source}"
                    if source else filter_source
                ),
            )

        return filtered

    def _visual_stl_yoloe_url(self) -> str:
        configured = str(
            self.cfg.get(
                "visual_stl_yoloe_url",
                "http://127.0.0.1:8010",
            )
        )
        if not bool(self.cfg.get("visual_stl_yoloe_per_gpu", False)):
            return configured

        worker_rank = int(
            getattr(self.worker_info, "accelerator_rank", -1)
        )
        if worker_rank < 0:
            available = list(
                getattr(
                    self.worker_info,
                    "available_accelerators",
                    [],
                )
            )
            if len(available) == 1:
                worker_rank = int(available[0])
        if worker_rank < 0:
            raise RuntimeError(
                "Cannot route YOLOE per GPU: EnvWorker has no accelerator rank"
            )

        host = str(
            self.cfg.get("visual_stl_yoloe_host", "127.0.0.1")
        )
        base_port = int(
            self.cfg.get("visual_stl_yoloe_base_port", 8010)
        )
        return f"http://{host}:{base_port + worker_rank}"

    def _raw_atom_margin(self, ep, atom: Atom) -> float:
        """Evaluate one exported atom from the current privileged state."""
        pred, obj, target = atom

        if pred == "pick":
            if obj is None or resolve_object_key is None:
                raise ValueError(f"Cannot evaluate pick atom: {atom}")

            object_key = resolve_object_key(obj, ep)
            if object_key is None:
                raise KeyError(f"Cannot resolve pick object {obj!r}.")

            margin = pred_pick_agm(
                ep,
                object_key,
                self._pred_cfg,
            )
        else:
            margin, _ = goal_atom_margin(
                ep,
                pred,
                obj,
                target,
                self._pred_cfg,
            )

        return float(
            np.asarray(margin, dtype=np.float64).reshape(-1)[-1]
        )

    def _visual_stl_enabled(self) -> bool:
        return bool(
            self.cfg.get("visual_stl_monitor", False)
        )

    def _setup_visual_stl_shadow(self, ids) -> None:
        """Reset traces and cache one calibration per worker."""
        if not self._visual_stl_enabled():
            return

        if not hasattr(self, "_visual_stl_shadow"):
            self._visual_stl_shadow = (
                VisualSTLShadowMonitor(
                    yoloe_url=self._visual_stl_yoloe_url(),
                    yoloe_timeout_s=float(
                        self.cfg.get(
                            "visual_stl_yoloe_timeout_s",
                            10.0,
                        )
                    ),
                    yoloe_batch_enabled=bool(
                        self.cfg.get(
                            "visual_stl_yoloe_batch_enabled",
                            True,
                        )
                    ),
                    yoloe_batch_max_size=int(
                        self.cfg.get(
                            "visual_stl_yoloe_batch_max_size",
                            32,
                        )
                    ),
                    yoloe_raw_transport=bool(
                        self.cfg.get(
                            "visual_stl_yoloe_raw_transport",
                            False,
                        )
                    ),
                    camera_name=str(
                        self.cfg.get(
                            "visual_stl_camera_name",
                            "agentview",
                        )
                    ),
                    fallback_camera_name=self.cfg.get(
                        "visual_stl_fallback_camera_name",
                        None,
                    ),
                    fallback_min_confidence=self.cfg.get(
                        "visual_stl_fallback_min_confidence",
                        None,
                    ),
                    mask_to_depth_orientation=str(
                        self.cfg.get(
                            "visual_stl_mask_to_depth_orientation",
                            "rot180",
                        )
                    ),
                    grasp_radius_m=float(
                        self.cfg.get(
                            "stl_pred_config",
                            {},
                        ).get("r_grasp", 0.10)
                    ),
                    surface_radius_m=float(
                        self.cfg.get(
                            "visual_stl_surface_radius_m",
                            0.05,
                        )
                    ),
                    visual_center_radius_m=(
                        self.cfg.get(
                            "visual_stl_center_radius_m",
                            None,
                        )
                    ),
                    min_confidence=float(
                        self.cfg.get(
                            "visual_stl_min_confidence",
                            0.25,
                        )
                    ),
                    close_threshold_m=float(
                        self.cfg.get(
                            "visual_stl_close_threshold_m",
                            0.08,
                        )
                    ),
                    comove_tolerance_m=float(
                        self.cfg.get(
                            "visual_stl_comove_tolerance_m",
                            0.015,
                        )
                    ),
                    motion_min_m=float(
                        self.cfg.get(
                            "visual_stl_motion_min_m",
                            0.005,
                        )
                    ),
                    displacement_min_m=float(
                        self.cfg.get(
                            "visual_stl_displacement_min_m",
                            0.010,
                        )
                    ),
                    hold_steps=int(
                        self.cfg.get(
                            "visual_stl_hold_steps",
                            5,
                        )
                    ),
                    grasp_near_mode=str(
                        self.cfg.get(
                            "visual_stl_grasp_near_mode",
                            "center",
                        )
                    ),
                    grasp_history_lag_steps=int(
                        self.cfg.get(
                            "visual_stl_grasp_history_lag_steps",
                            3,
                        )
                    ),
                    grasp_confirmation_steps=int(
                        self.cfg.get(
                            "visual_stl_grasp_confirmation_steps",
                            3,
                        )
                    ),
                    grasp_terminal_width_m=float(
                        self.cfg.get(
                            "visual_stl_grasp_terminal_width_m",
                            0.050,
                        )
                    ),
                    grasp_terminal_comove_slack_m=float(
                        self.cfg.get(
                            "visual_stl_grasp_terminal_comove_slack_m",
                            0.001,
                        )
                    ),
                    grasp_terminal_motion_slack_m=float(
                        self.cfg.get(
                            "visual_stl_grasp_terminal_motion_slack_m",
                            0.0005,
                        )
                    ),
                    grasp_terminal_confirmation_steps=int(
                        self.cfg.get(
                            "visual_stl_grasp_terminal_confirmation_steps",
                            2,
                        )
                    ),
                    on_xy_tolerance_m=float(
                        self.cfg.get(
                            "visual_stl_on_xy_tolerance_m",
                            0.05,
                        )
                    ),
                    on_above_epsilon_m=float(
                        self.cfg.get(
                            "visual_stl_on_above_epsilon_m",
                            0.0,
                        )
                    ),
                    on_surface_tolerance_m=float(
                        self.cfg.get(
                            "visual_stl_on_surface_tolerance_m",
                            0.03,
                        )
                    ),
                    on_ordinary_xy_tolerance_m=float(
                        self.cfg.get(
                            "visual_stl_on_ordinary_xy_tolerance_m",
                            0.030,
                        )
                    ),
                    on_ordinary_xy_uncertainty_m=float(
                        self.cfg.get(
                            "visual_stl_on_ordinary_xy_uncertainty_m",
                            0.014,
                        )
                    ),
                    on_ordinary_min_above_m=float(
                        self.cfg.get(
                            "visual_stl_on_ordinary_min_above_m",
                            -0.005,
                        )
                    ),
                    on_ordinary_surface_tolerance_m=float(
                        self.cfg.get(
                            "visual_stl_on_ordinary_surface_tolerance_m",
                            0.03,
                        )
                    ),
                    on_ordinary_require_release=bool(
                        self.cfg.get(
                            "visual_stl_on_ordinary_require_release",
                            False,
                        )
                    ),
                    on_ordinary_require_contact=bool(
                        self.cfg.get(
                            "visual_stl_on_ordinary_require_contact",
                            False,
                        )
                    ),
                    on_ordinary_contact_tolerance_m=float(
                        self.cfg.get(
                            "visual_stl_on_ordinary_contact_tolerance_m",
                            0.01,
                        )
                    ),
                    on_ordinary_confirmation_steps=int(
                        self.cfg.get(
                            "visual_stl_on_ordinary_confirmation_steps",
                            5,
                        )
                    ),
                    on_confirmation_steps=int(
                        self.cfg.get(
                            "visual_stl_on_confirmation_steps",
                            3,
                        )
                    ),
                    on_site_lower_delta_m=float(
                        self.cfg.get(
                            "visual_stl_on_site_lower_delta_m",
                            -0.20,
                        )
                    ),
                    on_site_upper_delta_m=float(
                        self.cfg.get(
                            "visual_stl_on_site_upper_delta_m",
                            0.10,
                        )
                    ),
                    on_front_x_lower_m=float(
                        self.cfg.get(
                            "visual_stl_on_front_x_lower_m", 0.12
                        )
                    ),
                    on_front_x_upper_m=float(
                        self.cfg.get(
                            "visual_stl_on_front_x_upper_m", 0.24
                        )
                    ),
                    on_front_y_lower_m=float(
                        self.cfg.get(
                            "visual_stl_on_front_y_lower_m", -0.04
                        )
                    ),
                    on_front_y_upper_m=float(
                        self.cfg.get(
                            "visual_stl_on_front_y_upper_m", 0.05
                        )
                    ),
                    on_front_z_lower_m=float(
                        self.cfg.get(
                            "visual_stl_on_front_z_lower_m", -0.03
                        )
                    ),
                    on_front_z_upper_m=float(
                        self.cfg.get(
                            "visual_stl_on_front_z_upper_m", 0.03
                        )
                    ),
                    on_front_confirmation_steps=int(
                        self.cfg.get(
                            "visual_stl_on_front_confirmation_steps", 7
                        )
                    ),
                    on_spatial_min_displacement_m=float(
                        self.cfg.get(
                            "visual_stl_on_spatial_min_displacement_m",
                            0.03,
                        )
                    ),
                    on_spatial_stability_tolerance_m=float(
                        self.cfg.get(
                            "visual_stl_on_spatial_stability_tolerance_m",
                            0.003,
                        )
                    ),
                    on_right_min_displacement_m=float(
                        self.cfg.get(
                            "visual_stl_on_right_min_displacement_m",
                            0.135,
                        )
                    ),
                    on_right_released_min_displacement_m=float(
                        self.cfg.get(
                            "visual_stl_on_right_released_min_displacement_m",
                            0.10,
                        )
                    ),
                    on_right_interior_margin_m=float(
                        self.cfg.get(
                            "visual_stl_on_right_interior_margin_m",
                            0.015,
                        )
                    ),
                    on_right_recovery_slack_m=float(
                        self.cfg.get(
                            "visual_stl_on_right_recovery_slack_m",
                            0.015,
                        )
                    ),
                    on_right_recovery_min_displacement_m=float(
                        self.cfg.get(
                            "visual_stl_on_right_recovery_min_displacement_m",
                            0.16,
                        )
                    ),
                    on_right_released_slack_m=float(
                        self.cfg.get(
                            "visual_stl_on_right_released_slack_m",
                            0.02,
                        )
                    ),
                    on_right_release_max_age_steps=int(
                        self.cfg.get(
                            "visual_stl_on_right_release_max_age_steps",
                            30,
                        )
                    ),
                    on_right_lateral_padding_m=float(
                        self.cfg.get(
                            "visual_stl_on_right_lateral_padding_m",
                            0.01,
                        )
                    ),
                    on_right_max_offset_m=float(
                        self.cfg.get(
                            "visual_stl_on_right_max_offset_m",
                            0.06,
                        )
                    ),
                    on_right_require_release=bool(
                        self.cfg.get(
                            "visual_stl_on_right_require_release", False
                        )
                    ),
                    on_right_pick_proxy=bool(
                        self.cfg.get(
                            "visual_stl_on_right_pick_proxy", True
                        )
                    ),
                    on_right_confirmation_steps=int(
                        self.cfg.get(
                            "visual_stl_on_right_confirmation_steps", 5
                        )
                    ),
                    on_attach_width_m=float(
                        self.cfg.get(
                            "visual_stl_on_attach_width_m",
                            0.05,
                        )
                    ),
                    on_attach_confirmation_steps=int(
                        self.cfg.get(
                            "visual_stl_on_attach_confirmation_steps",
                            2,
                        )
                    ),
                    on_release_width_m=float(
                        self.cfg.get(
                            "visual_stl_on_release_width_m",
                            0.06,
                        )
                    ),
                    on_release_hold_steps=int(
                        self.cfg.get(
                            "visual_stl_on_release_hold_steps",
                            30,
                        )
                    ),
                    on_support_cache_steps=int(
                        self.cfg.get(
                            "visual_stl_on_support_cache_steps",
                            0,
                        )
                    ),
                    on_require_fresh_after_release=bool(
                        self.cfg.get(
                            "visual_stl_on_require_fresh_after_release",
                            False,
                        )
                    ),
                    on_contact_tolerance_m=float(
                        self.cfg.get(
                            "visual_stl_on_contact_tolerance_m",
                            0.0012,
                        )
                    ),
                    in_enabled=bool(
                        self.cfg.get("visual_stl_in_monitor", False)
                    ),
                    in_confirmation_steps=int(
                        self.cfg.get(
                            "visual_stl_in_confirmation_steps", 3
                        )
                    ),
                    in_reacquire_gate_m=float(
                        self.cfg.get(
                            "visual_stl_in_reacquire_gate_m", 0.30
                        )
                    ),
                    in_release_history_steps=int(
                        self.cfg.get(
                            "visual_stl_in_release_history_steps", 5
                        )
                    ),
                    in_release_delta_m=float(
                        self.cfg.get(
                            "visual_stl_in_release_delta_m", 0.003
                        )
                    ),
                    in_positive_hold_steps=int(
                        self.cfg.get(
                            "visual_stl_in_positive_hold_steps", 1
                        )
                    ),
                    in_caddy_release_separation_m=float(
                        self.cfg.get(
                            "visual_stl_in_caddy_release_separation_m",
                            0.10,
                        )
                    ),
                    in_caddy_release_confirmation_steps=int(
                        self.cfg.get(
                            "visual_stl_in_caddy_release_confirmation_steps",
                            3,
                        )
                    ),
                    in_caddy_release_max_age_steps=int(
                        self.cfg.get(
                            "visual_stl_in_caddy_release_max_age_steps",
                            60,
                        )
                    ),
                    in_caddy_fresh_entry_tolerance_m=float(
                        self.cfg.get(
                            "visual_stl_in_caddy_fresh_entry_tolerance_m",
                            0.03,
                        )
                    ),
                    in_caddy_opening_tolerance_m=float(
                        self.cfg.get(
                            "visual_stl_in_caddy_opening_tolerance_m",
                            0.001,
                        )
                    ),
                    in_caddy_fresh_min_core_lateral_margin_m=float(
                        self.cfg.get(
                            "visual_stl_in_caddy_fresh_min_core_lateral_margin_m",
                            0.04,
                        )
                    ),
                    in_caddy_fresh_release_max_age_steps=int(
                        self.cfg.get(
                            "visual_stl_in_caddy_fresh_release_max_age_steps",
                            90,
                        )
                    ),
                    in_caddy_max_target_displacement_m=float(
                        self.cfg.get(
                            "visual_stl_in_caddy_max_target_displacement_m",
                            0.04,
                        )
                    ),
                    in_caddy_settled_confirmation_age_steps=int(
                        self.cfg.get(
                            "visual_stl_in_caddy_settled_confirmation_age_steps",
                            60,
                        )
                    ),
                    in_caddy_latched_occlusion_hold_steps=int(
                        self.cfg.get(
                            "visual_stl_in_caddy_latched_occlusion_hold_steps",
                            12,
                        )
                    ),
                    in_caddy_proxy_entry_tolerance_m=float(
                        self.cfg.get(
                            "visual_stl_in_caddy_proxy_entry_tolerance_m",
                            0.03,
                        )
                    ),
                    in_caddy_proxy_opening_confirmation_steps=int(
                        self.cfg.get(
                            "visual_stl_in_caddy_proxy_opening_confirmation_steps",
                            2,
                        )
                    ),
                    in_caddy_proxy_release_max_age_steps=int(
                        self.cfg.get(
                            "visual_stl_in_caddy_proxy_release_max_age_steps",
                            90,
                        )
                    ),
                    in_caddy_unreleased_ceiling_m=float(
                        self.cfg.get(
                            "visual_stl_in_caddy_unreleased_ceiling_m",
                            0.001,
                        )
                    ),
                    in_caddy_contradiction_margin_m=float(
                        self.cfg.get(
                            "visual_stl_in_caddy_contradiction_margin_m",
                            0.03,
                        )
                    ),
                    in_caddy_contradiction_confirmation_steps=int(
                        self.cfg.get(
                            "visual_stl_in_caddy_contradiction_confirmation_steps",
                            3,
                        )
                    ),
                    in_config=VisualInConfig(
                        basket_inner_shrink_m=float(
                            self.cfg.get(
                                "visual_stl_in_inner_shrink_m", 0.015
                            )
                        ),
                        below_rim_tolerance_m=float(
                            self.cfg.get(
                                "visual_stl_in_below_rim_tolerance_m",
                                0.025,
                            )
                        ),
                        floor_tolerance_m=float(
                            self.cfg.get(
                                "visual_stl_in_floor_tolerance_m", 0.010
                            )
                        ),
                        release_width_threshold_m=float(
                            self.cfg.get(
                                "visual_stl_in_release_width_m", 0.030
                            )
                        ),
                        min_drawer_points=int(
                            self.cfg.get(
                                "visual_stl_in_min_drawer_points", 40
                            )
                        ),
                        drawer_inner_shrink_m=float(
                            self.cfg.get(
                                "visual_stl_in_drawer_inner_shrink_m",
                                0.010,
                            )
                        ),
                        caddy_inner_shrink_m=float(
                            self.cfg.get(
                                "visual_stl_in_caddy_inner_shrink_m",
                                0.008,
                            )
                        ),
                        caddy_back_start_fraction=float(
                            self.cfg.get(
                                "visual_stl_in_caddy_back_start_fraction",
                                0.10,
                            )
                        ),
                        caddy_use_object_footprint=bool(
                            self.cfg.get(
                                "visual_stl_in_caddy_use_object_footprint",
                                False,
                            )
                        ),
                        use_object_footprint=bool(
                            self.cfg.get(
                                "visual_stl_in_use_object_footprint",
                                True,
                            )
                        ),
                        require_release=bool(
                            self.cfg.get(
                                "visual_stl_in_require_release",
                                True,
                            )
                        ),
                        microwave_inner_shrink_m=float(
                            self.cfg.get(
                                "visual_stl_in_microwave_inner_shrink_m", 0.012
                            )
                        ),
                        microwave_entry_tolerance_m=float(
                            self.cfg.get(
                                "visual_stl_in_microwave_entry_tolerance_m",
                                0.005,
                            )
                        ),
                        microwave_front_quantile=float(
                            self.cfg.get(
                                "visual_stl_in_microwave_front_quantile",
                                0.90,
                            )
                        ),
                        microwave_min_entry_depth_m=float(
                            self.cfg.get(
                                "visual_stl_in_microwave_min_entry_depth_m",
                                0.170,
                            )
                        ),
                        microwave_release_max_target_eef_distance_m=float(
                            self.cfg.get(
                                "visual_stl_in_microwave_release_max_target_eef_distance_m",
                                0.100,
                            )
                        ),
                        microwave_occlusion_steps=int(
                            self.cfg.get(
                                "visual_stl_in_microwave_occlusion_steps", 8
                            )
                        ),
                        microwave_release_max_age_steps=int(
                            self.cfg.get(
                                "visual_stl_in_microwave_release_max_age_steps",
                                30,
                            )
                        ),
                        microwave_require_release=bool(
                            self.cfg.get(
                                "visual_stl_in_microwave_require_release",
                                True,
                            )
                        ),
                    ),
                    open_enabled=bool(
                        self.cfg.get("visual_stl_open_monitor", False)
                    ),
                    open_confirmation_steps=int(
                        self.cfg.get(
                            "visual_stl_open_confirmation_steps", 3
                        )
                    ),
                    open_hold_steps=int(
                        self.cfg.get("visual_stl_open_hold_steps", 3)
                    ),
                    open_config=VisualOpenConfig(
                        displacement_threshold_m=float(
                            self.cfg.get(
                                "visual_stl_open_displacement_threshold_m",
                                0.04,
                            )
                        ),
                        bound_quantile=float(
                            self.cfg.get(
                                "visual_stl_open_bound_quantile", 0.05
                            )
                        ),
                        min_points=int(
                            self.cfg.get(
                                "visual_stl_open_min_points", 40
                            )
                        ),
                        extension_axis=str(
                            self.cfg.get(
                                "visual_stl_open_extension_axis", "auto"
                            )
                        ),
                    ),
                    close_enabled=bool(
                        self.cfg.get("visual_stl_close_monitor", False)
                    ),
                    close_require_in_confirmed=bool(
                        self.cfg.get(
                            "visual_stl_close_require_in_confirmed",
                            False,
                        )
                    ),
                    close_confirmation_steps=int(
                        self.cfg.get(
                            "visual_stl_close_confirmation_steps", 3
                        )
                    ),
                    close_hold_steps=int(
                        self.cfg.get("visual_stl_close_hold_steps", 3)
                    ),
                    close_config=VisualCloseConfig(
                        retraction_threshold_m=float(
                            self.cfg.get(
                                "visual_stl_close_retraction_threshold_m",
                                0.10,
                            )
                        ),
                        bound_quantile=float(
                            self.cfg.get(
                                "visual_stl_close_bound_quantile", 0.05
                            )
                        ),
                        min_points=int(
                            self.cfg.get(
                                "visual_stl_close_min_points", 40
                            )
                        ),
                        extension_axis=str(
                            self.cfg.get(
                                "visual_stl_close_extension_axis", "y+"
                            )
                        ),
                        front_quantile=float(
                            self.cfg.get(
                                "visual_stl_close_front_quantile", 0.95
                            )
                        ),
                    ),
                    close_panel_diagnostic=bool(
                        self.cfg.get(
                            "visual_stl_close_panel_diagnostic", False
                        )
                    ),
                    close_panel_diagnostic_interval=int(
                        self.cfg.get(
                            "visual_stl_close_panel_diagnostic_interval",
                            10,
                        )
                    ),
                    close_mode=str(
                        self.cfg.get(
                            "visual_stl_close_mode",
                            "legacy_pointcloud",
                        )
                    ),
                    close_panel_config=VisualDrawerCloseConfig(
                        retraction_threshold_fraction=float(
                            self.cfg.get(
                                "visual_stl_close_panel_threshold_fraction",
                                0.25,
                            )
                        ),
                        edge_quantile=float(
                            self.cfg.get(
                                "visual_stl_close_panel_edge_quantile",
                                0.98,
                            )
                        ),
                        min_panel_pixels=int(
                            self.cfg.get(
                                "visual_stl_close_panel_min_pixels", 80
                            )
                        ),
                        min_reference_pixels=int(
                            self.cfg.get(
                                "visual_stl_close_panel_min_reference_pixels",
                                80,
                            )
                        ),
                        max_reference_shift_fraction=float(
                            self.cfg.get(
                                "visual_stl_close_panel_max_reference_shift_fraction",
                                0.10,
                            )
                        ),
                    ),
                    microwave_close_config=VisualMicrowaveCloseConfig(
                        retraction_threshold_fraction=float(
                            self.cfg.get(
                                "visual_stl_close_microwave_threshold_fraction", 0.12
                            )
                        ),
                        closing_sign=int(
                            self.cfg.get(
                                "visual_stl_close_microwave_closing_sign", 1
                            )
                        ),
                        min_area_ratio=float(
                            self.cfg.get(
                                "visual_stl_close_microwave_min_area_ratio",
                                0.0,
                            )
                        ),
                        contraction_threshold_fraction=float(
                            self.cfg.get(
                                "visual_stl_close_microwave_contraction_threshold_fraction",
                                0.0,
                            )
                        ),
                        contraction_max_area_ratio=float(
                            self.cfg.get(
                                "visual_stl_close_microwave_contraction_max_area_ratio",
                                0.0,
                            )
                        ),
                        min_pixels=int(
                            self.cfg.get(
                                "visual_stl_close_microwave_min_pixels", 80
                            )
                        ),
                        max_vertical_shift_fraction=float(
                            self.cfg.get(
                                "visual_stl_close_microwave_max_vertical_shift_fraction", 0.12
                            )
                        ),
                        edge_quantile=float(
                            self.cfg.get(
                                "visual_stl_close_microwave_edge_quantile",
                                0.02,
                            )
                        ),
                        door_close_progress_threshold=float(
                            self.cfg.get(
                                "visual_stl_close_microwave_door_progress_threshold",
                                0.65,
                            )
                        ),
                        door_vertical_edge_quantile=float(
                            self.cfg.get(
                                "visual_stl_close_microwave_door_edge_quantile",
                                0.85,
                            )
                        ),
                        door_min_edge_contrast=float(
                            self.cfg.get(
                                "visual_stl_close_microwave_door_min_edge_contrast",
                                0.08,
                            )
                        ),
                        door_min_relative_edge_strength=float(
                            self.cfg.get(
                                "visual_stl_close_microwave_door_min_relative_strength",
                                0.65,
                            )
                        ),
                        door_min_open_span_fraction=float(
                            self.cfg.get(
                                "visual_stl_close_microwave_door_min_open_span_fraction",
                                0.20,
                            )
                        ),
                        door_depth_edge_tolerance_px=int(
                            self.cfg.get(
                                "visual_stl_close_microwave_door_depth_tolerance_px",
                                8,
                            )
                        ),
                        door_min_relative_depth_edge_strength=float(
                            self.cfg.get(
                                "visual_stl_close_microwave_door_min_relative_depth_strength",
                                0.30,
                            )
                        ),
                        door_search_width_scale=float(
                            self.cfg.get(
                                "visual_stl_close_microwave_door_search_width_scale",
                                2.10,
                            )
                        ),
                        door_search_top_fraction=float(
                            self.cfg.get(
                                "visual_stl_close_microwave_door_search_top_fraction",
                                0.06,
                            )
                        ),
                        door_search_bottom_fraction=float(
                            self.cfg.get(
                                "visual_stl_close_microwave_door_search_bottom_fraction",
                                0.91,
                            )
                        ),
                        door_search_min_top_image_fraction=float(
                            self.cfg.get(
                                "visual_stl_close_microwave_door_search_min_top_image_fraction",
                                0.33,
                            )
                        ),
                        door_dark_intensity_threshold=float(
                            self.cfg.get(
                                "visual_stl_close_microwave_door_dark_intensity_threshold",
                                0.20,
                            )
                        ),
                        door_dark_body_density_fraction=float(
                            self.cfg.get(
                                "visual_stl_close_microwave_door_dark_body_density_fraction",
                                0.40,
                            )
                        ),
                        door_dark_min_column_pixels=int(
                            self.cfg.get(
                                "visual_stl_close_microwave_door_dark_min_column_pixels",
                                10,
                            )
                        ),
                        door_dark_min_open_span_fraction=float(
                            self.cfg.get(
                                "visual_stl_close_microwave_door_dark_min_open_span_fraction",
                                0.15,
                            )
                        ),
                        door_dark_max_column_gap=int(
                            self.cfg.get(
                                "visual_stl_close_microwave_door_dark_max_column_gap",
                                3,
                            )
                        ),
                    ),
                    close_microwave_settled_progress_threshold=float(
                        self.cfg.get(
                            "visual_stl_close_microwave_settled_progress_threshold",
                            0.75,
                        )
                    ),
                    close_microwave_settled_confirmation_steps=int(
                        self.cfg.get(
                            "visual_stl_close_microwave_settled_confirmation_steps",
                            30,
                        )
                    ),
                    turnon_enabled=bool(
                        self.cfg.get("visual_stl_turnon_monitor", False)
                    ),
                    turnon_config=VisualTurnOnConfig(
                        min_mask_pixels=int(self.cfg.get(
                            "visual_stl_turnon_min_mask_pixels", 40
                        )),
                        roi_padding_fraction=float(self.cfg.get(
                            "visual_stl_turnon_roi_padding_fraction", 0.12
                        )),
                        grid_rows=int(self.cfg.get(
                            "visual_stl_turnon_grid_rows", 3
                        )),
                        grid_cols=int(self.cfg.get(
                            "visual_stl_turnon_grid_cols", 4
                        )),
                        diagnostic_interval=int(self.cfg.get(
                            "visual_stl_turnon_diagnostic_interval", 5
                        )),
                        red_min_intensity=float(self.cfg.get(
                            "visual_stl_turnon_red_min_intensity", 0.45
                        )),
                        red_excess_threshold=float(self.cfg.get(
                            "visual_stl_turnon_red_excess_threshold", 0.18
                        )),
                        red_fraction_threshold=float(self.cfg.get(
                            "visual_stl_turnon_red_fraction_threshold", 0.010
                        )),
                        confirmation_steps=int(self.cfg.get(
                            "visual_stl_turnon_confirmation_steps", 3
                        )),
                    ),
                )
            )

        camera_name = str(
            self.cfg.get(
                "visual_stl_camera_name",
                "agentview",
            )
        )
        height = int(
            self.cfg.init_params.get(
                "camera_heights",
                256,
            )
        )
        width = int(
            self.cfg.init_params.get(
                "camera_widths",
                256,
            )
        )

        calibrations = (
            self.env.get_camera_calibration(
                camera_name=camera_name,
                camera_height=height,
                camera_width=width,
                id=ids,
            )
        )

        fallback_camera_name = self.cfg.get(
            "visual_stl_fallback_camera_name",
            None,
        )
        if fallback_camera_name:
            fallback_calibrations = (
                self.env.get_camera_calibration(
                    camera_name=str(fallback_camera_name),
                    camera_height=height,
                    camera_width=width,
                    id=ids,
                )
            )
        else:
            fallback_calibrations = [None] * len(ids)

        for position, raw_env_id in enumerate(ids):
            env_id = int(raw_env_id)
            calibration = calibrations[position]
            self._visual_stl_shadow.reset_env(
                env_id,
                calibration,
                fallback_calibration=(
                    fallback_calibrations[position]
                ),
                raw_obs=(
                    self.current_raw_obs[env_id]
                    if self.current_raw_obs is not None
                    else None
                ),
            )

            if not calibration.get("valid", False):
                print(
                    "[VISUAL_STL_CALIBRATION_ERROR] "
                    f"env={env_id} "
                    f"reason={calibration.get('reason', '')}",
                    flush=True,
                )

            fallback_calibration = (
                fallback_calibrations[position]
            )
            if (
                fallback_camera_name
                and (
                    fallback_calibration is None
                    or not fallback_calibration.get("valid", False)
                )
            ):
                print(
                    "[VISUAL_STL_CALIBRATION_ERROR] "
                    f"env={env_id} "
                    f"camera={fallback_camera_name} "
                    "reason="
                    f"{(fallback_calibration or {}).get('reason', '')}",
                    flush=True,
                )

    def _record_visual_stl_step(
        self,
        *,
        env_id: int,
        grasp_obj: str,
        priv: dict | None = None,
        ep=None,
        object_key: str | None = None,
        oracle_pick_margin: float = float("nan"),
        on_target_object: str | None = None,
        oracle_on_margin: float = float("nan"),
        in_target_object: str | None = None,
        oracle_in_margin: float = float("nan"),
        visual_state_id: int | None = None,
    ):
        """Record one visual estimate; never affect reward."""
        if (
            not self._visual_stl_enabled()
            or not hasattr(self, "_visual_stl_shadow")
        ):
            return None

        try:
            raw_obs = self.current_raw_obs[env_id]

            oracle_center = None
            oracle_distance = float("nan")
            grasped = False
            object_positions = (
                (priv or {}).get("object_pos", {}) or {}
            )
            if object_key is not None:
                oracle_center = object_positions.get(
                    object_key
                )

            if oracle_center is None and object_key is not None:
                for name, center in object_positions.items():
                    name = str(name)
                    if (
                        name == object_key
                        or name in object_key
                        or object_key in name
                    ):
                        oracle_center = center
                        break

            try:
                if object_key is None:
                    raise KeyError("oracle object key unavailable")
                oracle_distance = float(
                    np.asarray(
                        eef_object_dist(ep, object_key),
                        dtype=np.float64,
                    ).reshape(-1)[-1]
                )
                grasped = bool(
                    np.asarray(
                        two_finger_grasp(
                            ep,
                            object_key,
                        )
                    ).reshape(-1)[-1]
                )
            except Exception:
                pass
            step = int(
                np.asarray(
                    self._elapsed_steps
                ).reshape(-1)[env_id]
            )

            record = self._visual_stl_shadow.evaluate(
                env_id=(
                    int(visual_state_id)
                    if visual_state_id is not None else env_id
                ),
                step=step,
                task_id=int(self.task_ids[env_id]),
                task_description=(
                    self._task_description_for_env(
                        env_id
                    )
                ),
                target_object=grasp_obj,
                raw_obs=raw_obs,
                oracle_distance_m=oracle_distance,
                oracle_pick_margin=float(
                    oracle_pick_margin
                ),
                on_target_object=on_target_object,
                oracle_on_margin=float(oracle_on_margin),
                in_target_object=in_target_object,
                oracle_in_margin=float(oracle_in_margin),
                oracle_center=oracle_center,
                two_finger_grasp=grasped,
            )

            debug_interval = int(
                self.cfg.get(
                    "visual_stl_debug_interval",
                    20,
                )
            )

            if (
                debug_interval > 0
                and step % debug_interval == 0
            ):
                print(
                    "[VISUAL_STL] "
                    f"env={env_id} step={step} "
                    f"target={object_key} "
                    f"valid={record.valid} "
                    f"oracle_rho="
                    f"{record.oracle_robustness:+.4f} "
                    f"visual_rho="
                    f"{record.visual_robustness:+.4f} "
                    f"grasp_rho="
                    f"{record.visual_grasp_margin:+.4f} "
                    f"grasp_confirmed="
                    f"{record.visual_grasp_confirmed} "
                    f"grasp_source="
                    f"{record.visual_grasp_confirmation_source} "
                    f"on_rho={record.visual_on_margin:+.4f} "
                    f"on_confirmed={record.visual_on_confirmed} "
                    f"reason={record.reason}",
                    flush=True,
                )

            return record

        except Exception as error:
            print(
                "[VISUAL_STL_ERROR] "
                f"env={env_id} "
                f"error={type(error).__name__}: "
                f"{error}",
                flush=True,
            )
            return None

    def _record_visual_open_step(
        self,
        *,
        env_id: int,
        atom: Atom,
        oracle_open_margin: float,
        visual_state_id: int | None = None,
    ):
        """Record a canonical Open atom without changing reward."""
        if (
            not self._visual_stl_enabled()
            or not hasattr(self, "_visual_stl_shadow")
            or not bool(self.cfg.get("visual_stl_open_monitor", False))
        ):
            return None
        pred, obj, target = atom
        if str(pred).lower() != "open":
            return None
        try:
            env_id = int(env_id)
            step = int(
                np.asarray(self._elapsed_steps).reshape(-1)[env_id]
            )
            record = self._visual_stl_shadow.evaluate_open(
                env_id=(
                    int(visual_state_id)
                    if visual_state_id is not None else env_id
                ),
                step=step,
                task_id=int(self.task_ids[env_id]),
                task_description=self._task_description_for_env(env_id),
                articulated_object=str(obj or target or "cabinet"),
                raw_obs=self.current_raw_obs[env_id],
                oracle_open_margin=float(oracle_open_margin),
            )
            debug_interval = int(
                self.cfg.get("visual_stl_debug_interval", 20)
            )
            if (
                record is not None
                and debug_interval > 0
                and step % debug_interval == 0
            ):
                print(
                    "[VISUAL_OPEN] "
                    f"env={env_id} step={step} "
                    f"valid={record.valid} "
                    f"displacement={record.displacement_m:+.4f} "
                    f"visual_rho={record.visual_open_margin:+.4f} "
                    f"oracle_rho={record.oracle_open_margin:+.4f} "
                    f"confirmed={record.visual_open_confirmed} "
                    f"reason={record.reason}",
                    flush=True,
                )
            return record
        except Exception as error:
            print(
                "[VISUAL_OPEN_ERROR] "
                f"env={env_id} error={type(error).__name__}: {error}",
                flush=True,
            )
            return None

    def _record_visual_close_step(
        self,
        *,
        env_id: int,
        atom: Atom,
        oracle_close_margin: float,
        visual_state_id: int | None = None,
    ):
        """Record canonical Close from RGB-D without changing reward."""
        if (
            not self._visual_stl_enabled()
            or not hasattr(self, "_visual_stl_shadow")
            or not bool(self.cfg.get("visual_stl_close_monitor", False))
        ):
            return None
        pred, obj, target = atom
        if str(pred).lower() != "close":
            return None
        try:
            env_id = int(env_id)
            step = int(
                np.asarray(self._elapsed_steps).reshape(-1)[env_id]
            )
            record = self._visual_stl_shadow.evaluate_close(
                env_id=(
                    int(visual_state_id)
                    if visual_state_id is not None else env_id
                ),
                step=step,
                task_id=int(self.task_ids[env_id]),
                task_description=self._task_description_for_env(env_id),
                articulated_object=str(obj or target or "cabinet"),
                raw_obs=self.current_raw_obs[env_id],
                oracle_close_margin=float(oracle_close_margin),
            )
            debug_interval = int(
                self.cfg.get("visual_stl_debug_interval", 20)
            )
            if (
                record is not None
                and debug_interval > 0
                and step % debug_interval == 0
            ):
                print(
                    "[VISUAL_CLOSE] "
                    f"env={env_id} step={step} valid={record.valid} "
                    f"retraction={record.retraction_m:+.4f} "
                    f"visual_rho={record.visual_close_margin:+.4f} "
                    f"oracle_rho={record.oracle_close_margin:+.4f} "
                    f"confirmed={record.visual_close_confirmed} "
                    f"reason={record.reason}",
                    flush=True,
                )
            return record
        except Exception as error:
            print(
                "[VISUAL_CLOSE_ERROR] "
                f"env={env_id} error={type(error).__name__}: {error}",
                flush=True,
            )
            return None

    def _record_visual_turnon_step(
        self,
        *,
        env_id: int,
        atom: Atom,
        oracle_turnon_margin: float,
        visual_state_id: int | None = None,
    ):
        """Record a strict visual-only TurnOn observability trace."""
        if (
            not self._visual_stl_enabled()
            or not hasattr(self, "_visual_stl_shadow")
            or not bool(self.cfg.get("visual_stl_turnon_monitor", False))
        ):
            return None
        pred, obj, target = atom
        if str(pred).lower() not in {"turnon", "turn_on"}:
            return None
        try:
            env_id = int(env_id)
            step = int(np.asarray(self._elapsed_steps).reshape(-1)[env_id])
            record = self._visual_stl_shadow.evaluate_turnon(
                env_id=(
                    int(visual_state_id)
                    if visual_state_id is not None else env_id
                ),
                step=step,
                task_id=int(self.task_ids[env_id]),
                articulated_object=str(obj or target or "stove"),
                raw_obs=self.current_raw_obs[env_id],
                oracle_turnon_margin=float(oracle_turnon_margin),
            )
            debug_interval = int(
                self.cfg.get("visual_stl_debug_interval", 20)
            )
            if record is not None and debug_interval > 0 and step % debug_interval == 0:
                print(
                    "[VISUAL_TURNON] "
                    f"env={env_id} step={step} valid={record.valid} "
                    f"rgb_mae={record.rgb_mae:+.5f} "
                    f"cell={record.max_cell_change:+.5f} "
                    f"red={record.red_activation_fraction:+.5f} "
                    f"visual_rho={record.visual_turnon_margin:+.5f} "
                    f"confirmed={record.visual_turnon_confirmed} "
                    f"oracle_rho={record.oracle_turnon_margin:+.4f} "
                    f"reason={record.reason}",
                    flush=True,
                )
            return record
        except Exception as error:
            print(
                "[VISUAL_TURNON_ERROR] "
                f"env={env_id} error={type(error).__name__}: {error}",
                flush=True,
            )
            return None

    @staticmethod
    def _visual_atom_sample(
        record,
        margin_field: str,
        *,
        valid_field: str = "valid",
        reason_field: str = "reason",
        source_field: str | None = None,
    ) -> VisualAtomSample:
        """Convert one visual record without consulting oracle fields."""
        if record is None:
            return VisualAtomSample(
                margin=float("nan"),
                valid=False,
                reason="visual_record_unavailable",
            )
        margin = float(getattr(record, margin_field, float("nan")))
        # Export the continuous geometric margin. Confirmation is owned by
        # VisualAGMShadowRuntime, so a detector latch must not quantize a
        # smooth predicate to +/- epsilon or create a one-frame score jump.
        return VisualAtomSample(
            margin=margin,
            valid=bool(getattr(record, valid_field, False))
            and bool(np.isfinite(margin)),
            reason=str(getattr(record, reason_field, "")),
            source=(
                str(getattr(record, source_field, ""))
                if source_field else ""
            ),
        )

    @staticmethod
    def _visual_on_requires_confirmation(geometry_mode: str) -> bool:
        """Whether an On estimator owns confirmation before AGM export."""
        return str(geometry_mode) in {
            "ordinary_surface",
            "spatial_relation:front_of_stove",
            "spatial_relation:right_of_plate",
        }

    def _latest_visual_in_sample(
        self,
        visual_state_id: int,
    ) -> VisualAtomSample:
        records = getattr(
            getattr(self, "_visual_stl_shadow", None),
            "in_records",
            {},
        ).get(int(visual_state_id), [])
        return self._visual_atom_sample(
            records[-1] if records else None,
            "visual_in_margin",
        )

    def _dump_visual_stl_episode(
        self,
        *,
        env_id: int,
        out_path: str,
        task_description: str,
        success: bool,
    ) -> None:
        """Save CSV, JSON and comparison PNG."""
        if (
            not self._visual_stl_enabled()
            or not hasattr(self, "_visual_stl_shadow")
        ):
            return

        try:
            directory = os.path.dirname(out_path)
            stem = os.path.splitext(
                os.path.basename(out_path)
            )[0]
            state_ids = getattr(
                self, "_visual_atom_state_ids", {}
            ).get(env_id, {})
            atom_stats = {}
            if bool(
                self.cfg.get(
                    "visual_stl_save_atom_diagnostics",
                    False,
                )
            ):
                for atom, state_id in state_ids.items():
                    atom_suffix = "_".join(
                        str(value or "none")
                        .replace("/", "_")
                        .replace(" ", "_")
                        for value in atom
                    )
                    atom_stem = stem.replace(
                        "stl_plot_",
                        f"visual_atom_{atom_suffix}_",
                        1,
                    )
                    atom_prefix = os.path.join(directory, atom_stem)
                    atom_stats[atom_suffix] = (
                        self._visual_stl_shadow.dump_episode(
                            env_id=state_id,
                            output_prefix=atom_prefix,
                            task_description=task_description,
                            success=success,
                        )
                    )
                    print(
                        "[VISUAL_ATOM_DUMP] "
                        f"env={env_id} atom={atom} "
                        f"prefix={atom_prefix} "
                        f"stats={atom_stats[atom_suffix]}",
                        flush=True,
                    )

            comparison_stem = stem.replace(
                "stl_plot_", "stl_comparison_", 1
            )
            comparison_prefix = os.path.join(
                directory, comparison_stem
            )
            frames = getattr(
                self, "_stl_comparison_frames", {}
            ).get(env_id, [])
            plan = getattr(self, "_agm_stage_plans", {}).get(env_id)
            comparison_png = None
            comparison_csv = None
            comparison_metadata = None
            if frames and plan is not None:
                comparison_png = plot_stl_comparison(
                    frames,
                    plan,
                    f"{comparison_prefix}.png",
                    title=(
                        f"task {int(self.task_ids[env_id])}: "
                        f"{task_description}"
                    ),
                )
                comparison_csv = write_stl_comparison_csv(
                    frames,
                    plan,
                    f"{comparison_prefix}.csv",
                )
                episode_match = re.search(r"ep(\d+)", stem)
                episode_number = (
                    int(episode_match.group(1))
                    if episode_match is not None else None
                )
                trial_id = int(
                    np.asarray(self.trial_ids).reshape(-1)[env_id]
                )
                video_index = (
                    episode_number - 1
                    if episode_number is not None else None
                )
                video_cfg = self.cfg.get("video_cfg", {}) or {}
                video_base_dir = str(
                    video_cfg.get("video_base_dir", "")
                )
                # RecordVideo names directories from the environment seed,
                # not the task-local LIBERO trial id.  With one env per
                # worker this is seed_offset; env_id keeps this correct for
                # workers that host more than one environment.
                video_seed = int(self.seed) + int(env_id)
                video_relative_path = (
                    f"seed_{video_seed}/{video_index}.mp4"
                    if video_index is not None else ""
                )
                video_path = (
                    os.path.join(video_base_dir, video_relative_path)
                    if video_base_dir and video_relative_path else ""
                )
                comparison_metadata = f"{comparison_prefix}.metadata.json"
                with open(
                    comparison_metadata, "w", encoding="utf-8"
                ) as stream:
                    json.dump(
                        {
                            "task_id": int(self.task_ids[env_id]),
                            "trial_id": trial_id,
                            "env_id": int(env_id),
                            "seed_offset": int(self.seed_offset),
                            "video_seed": video_seed,
                            "episode_number": episode_number,
                            "worker_pid": int(os.getpid()),
                            "task_description": str(task_description),
                            "success": bool(success),
                            "comparison_csv": str(comparison_csv),
                            "comparison_png": str(comparison_png),
                            "video_path": video_path,
                            "video_relative_path": video_relative_path,
                            "stage_paths": [
                                {
                                    "name": str(path.name),
                                    "stages": [
                                        {
                                            "name": str(stage.name),
                                            "atoms": [
                                                atom_label(atom)
                                                for atom in stage.atoms
                                            ],
                                        }
                                        for stage in path.stages
                                    ],
                                }
                                for path in plan.paths
                            ],
                        },
                        stream,
                        indent=2,
                        ensure_ascii=False,
                    )

            stats = {
                "visual_atoms": len(state_ids),
                "comparison_frames": len(frames),
                "comparison_png": str(comparison_png or ""),
                "comparison_csv": str(comparison_csv or ""),
                "comparison_metadata": str(
                    comparison_metadata or ""
                ),
            }

            print(
                "[VISUAL_STL_DUMP] "
                f"env={env_id} "
                f"prefix={comparison_prefix} "
                f"stats={stats}",
                flush=True,
            )

        except Exception as error:
            print(
                "[VISUAL_STL_DUMP_ERROR] "
                f"env={env_id} "
                f"error={type(error).__name__}: "
                f"{error}",
                flush=True,
            )

    def _task_description_for_env(self, env_id: int) -> str:
        if (
            hasattr(self, "task_descriptions")
            and env_id < len(self.task_descriptions)
        ):
            return str(self.task_descriptions[env_id])

        task_id = int(self.task_ids[env_id])
        return str(self.task_suite.get_task(task_id).language)

    def _ensure_agm_state_maps(self) -> None:
        if not hasattr(self, "_agm_stage_runtimes"):
            self._agm_stage_runtimes = {}
        if not hasattr(self, "_agm_stage_plans"):
            self._agm_stage_plans = {}
        if not hasattr(self, "_agm_initial_atom_margins"):
            self._agm_initial_atom_margins = {}
        if not hasattr(self, "_agm_initial_goal_margins"):
            self._agm_initial_goal_margins = {}
        if not hasattr(self, "_agm_initial_grasp_margins"):
            self._agm_initial_grasp_margins = {}
        if not hasattr(self, "_agm_last_raw_margins"):
            self._agm_last_raw_margins = {}
        if not hasattr(self, "_visual_atom_state_ids"):
            self._visual_atom_state_ids = {}
        if not hasattr(self, "_visual_active_atoms"):
            self._visual_active_atoms = {}
        if not hasattr(self, "_visual_initialized_state_ids"):
            self._visual_initialized_state_ids = {}
        if not hasattr(self, "_visual_agm_runtimes"):
            self._visual_agm_runtimes = {}
        if not hasattr(self, "_visual_reward_last_score"):
            self._visual_reward_last_score = {}
        if not hasattr(self, "_visual_completion_bonus_paid"):
            self._visual_completion_bonus_paid = {}
        if not hasattr(self, "_visual_success_current"):
            self._visual_success_current = {}
        if not hasattr(self, "_visual_completion_bonus_trace"):
            self._visual_completion_bonus_trace = {}
        if not hasattr(self, "_visual_completion_bonus_snapshots"):
            self._visual_completion_bonus_snapshots = {}
        if not hasattr(self, "_stl_comparison_frames"):
            self._stl_comparison_frames = {}
        if not hasattr(self, "_visual_margin_histories"):
            self._visual_margin_histories = {}
        if not hasattr(self, "_visual_margin_invalid_streaks"):
            self._visual_margin_invalid_streaks = {}
        if not hasattr(self, "_visual_filter_states"):
            self._visual_filter_states = {}
        if not hasattr(self, "_visual_plot_ready_seen"):
            self._visual_plot_ready_seen = {}
        if not hasattr(self, "_visual_oracle_comparison_runtimes"):
            self._visual_oracle_comparison_runtimes = {}
        if not hasattr(self, "_visual_oracle_comparison_errors"):
            self._visual_oracle_comparison_errors = set()
        if not hasattr(self, "_visual_raw_rgb_frames"):
            self._visual_raw_rgb_frames = {}
        if not hasattr(self, "_visual_raw_rgb_steps"):
            self._visual_raw_rgb_steps = {}

    def _reset_visual_atom_states(
        self,
        env_id: int,
        plan: StagePlan,
    ) -> None:
        state_ids: dict[Atom, int] = {}
        picked_objects = {
            atom[1]
            for atom in plan.atoms
            if atom[0] == "pick" and atom[1] is not None
        }
        object_ids: dict[str, int] = {}
        for atom_index, atom in enumerate(plan.atoms):
            object_name = atom[1]
            if (
                atom[0] in {"pick", "on", "in"}
                and object_name in picked_objects
            ):
                state_ids[atom] = object_ids.setdefault(
                    str(object_name),
                    (env_id + 1) * 1000 + atom_index + 1,
                )
            else:
                state_ids[atom] = (env_id + 1) * 1000 + atom_index + 1
        self._visual_atom_state_ids[env_id] = state_ids
        self._visual_active_atoms[env_id] = set()
        self._visual_initialized_state_ids[env_id] = set()

    def _initialize_visual_atom_states(self, env_id: int) -> None:
        """Initialize every unique atom estimator at episode start."""
        visual_monitor = getattr(self, "_visual_stl_shadow", None)
        if visual_monitor is None:
            return
        calibration = visual_monitor.calibrations.get(env_id)
        fallback_calibration = visual_monitor.fallback_calibrations.get(
            env_id
        )
        initialized = self._visual_initialized_state_ids.setdefault(
            env_id, set()
        )
        for state_id in dict.fromkeys(
            self._visual_atom_state_ids.get(env_id, {}).values()
        ):
            visual_monitor.reset_env(
                state_id,
                calibration,
                fallback_calibration=fallback_calibration,
                raw_obs=(
                    self.current_raw_obs[env_id]
                    if self.current_raw_obs is not None else None
                ),
            )
            initialized.add(state_id)

    def _activate_visual_atoms(
        self,
        env_id: int,
        atoms,
    ) -> None:
        """Reset estimator state exactly when an atom enters an active stage."""
        active = {tuple(atom) for atom in atoms}
        previous = self._visual_active_atoms.get(env_id, set())
        newly_active = active - previous
        self._visual_active_atoms[env_id] = active
        if not newly_active:
            return

        visual_monitor = getattr(self, "_visual_stl_shadow", None)
        if visual_monitor is None:
            return
        calibration = visual_monitor.calibrations.get(env_id)
        fallback_calibration = visual_monitor.fallback_calibrations.get(env_id)
        state_ids = self._visual_atom_state_ids.get(env_id, {})
        initialized = self._visual_initialized_state_ids.setdefault(
            env_id, set()
        )
        for atom in newly_active:
            state_id = state_ids.get(atom)
            if state_id is None:
                raise KeyError(f"No visual state id for active atom {atom}")
            if state_id in initialized:
                visual_monitor.begin_atom_stage(state_id, atom[0])
                continue
            visual_monitor.reset_env(
                state_id,
                calibration,
                fallback_calibration=fallback_calibration,
                raw_obs=(
                    self.current_raw_obs[env_id]
                    if self.current_raw_obs is not None else None
                ),
            )
            initialized.add(state_id)

    def _setup_pure_visual_agm(self, ids) -> None:
        """Build the reward plan from task specification, never simulator state."""
        if not ids:
            return
        if get_bddl_goal_atoms is None:
            raise RuntimeError(
                "Pure-visual AGM requires the static BDDL goal loader"
            )

        self._ensure_agm_state_maps()
        if (
            self._stl_plot_dir
            and not getattr(self, "_agm_worker_plot_dir_ready", False)
        ):
            self._stl_plot_dir = os.path.join(
                str(self._stl_plot_dir),
                f"worker_pid_{os.getpid()}",
            )
            os.makedirs(self._stl_plot_dir, exist_ok=True)
            self._agm_worker_plot_dir_ready = True
        self._setup_visual_stl_shadow(ids)
        confirmation_steps = int(
            self.cfg.get("agm_stage_confirmation_steps", 3)
        )
        oracle_privileged = self._oracle_comparison_setup_states(ids)

        for raw_env_id in ids:
            env_id = int(raw_env_id)
            self._clear_agm_env(env_id)
            try:
                task_id = int(self.task_ids[env_id])
                description = self._task_description_for_env(env_id)
                goal_atoms = _normalize_goal_atoms(
                    get_bddl_goal_atoms(
                        self.cfg.task_suite_name,
                        task_id,
                    )
                    or []
                )
                if not goal_atoms:
                    raise ValueError(
                        f"No BDDL goal atoms for task {task_id}"
                    )
                grasp_objs = [
                    obj if pred in _GEOMETRIC_PREDS else None
                    for pred, obj, _target in goal_atoms
                ]
                plan = build_stage_plan(
                    task_suite_name=self.cfg.task_suite_name,
                    task_id=task_id,
                    goal_atoms=goal_atoms,
                    task_description=description,
                )

                self._stl_states[env_id] = _STLEnvState(
                    valid=True,
                    goal_atoms=goal_atoms,
                    grasp_objs=grasp_objs,
                    tracker=OnlineRobustnessTracker(
                        self._stl_tau,
                        len(goal_atoms),
                    ),
                )
                self._agm_stage_plans[env_id] = plan
                self._visual_agm_runtimes[env_id] = (
                    VisualAGMShadowRuntime(
                        plan=plan,
                        confirmation_steps=confirmation_steps,
                    )
                )
                self._visual_reward_last_score[env_id] = (
                    _VISUAL_COLD_START_SCORE
                )
                clipped_cold_start = float(_VISUAL_COLD_START_SCORE)
                if (
                    self._stl_clip_lower is not None
                    or self._stl_clip_upper is not None
                ):
                    clipped_cold_start = float(np.clip(
                        clipped_cold_start,
                        self._stl_clip_lower,
                        self._stl_clip_upper,
                    ))
                # Store Phi(t), not a reward, for discounted progress shaping.
                self.prev_step_reward[env_id] = clipped_cold_start
                self._visual_completion_bonus_paid[env_id] = False
                self._visual_success_current[env_id] = False
                self._visual_completion_bonus_trace[env_id] = []
                self._visual_completion_bonus_snapshots[env_id] = None
                self._stl_comparison_frames[env_id] = []
                self._visual_raw_rgb_frames[env_id] = []
                self._visual_raw_rgb_steps[env_id] = []
                self._visual_plot_ready_seen[env_id] = False
                self._reset_visual_atom_states(env_id, plan)
                self._initialize_visual_atom_states(env_id)
                self._activate_visual_atoms(
                    env_id,
                    self._visual_agm_runtimes[env_id].active_atoms,
                )

                self._setup_oracle_comparison_runtime(
                    env_id,
                    plan,
                    confirmation_steps,
                    oracle_privileged.get(env_id),
                )

                if getattr(self, "_stl_dbg", False):
                    print(
                        "[VISUAL_AGM_SETUP] "
                        f"env={env_id} task={task_id} "
                        f"atoms={goal_atoms} "
                        f"paths={[path.name for path in plan.paths]} "
                        f"yoloe={self._visual_stl_yoloe_url()}",
                        flush=True,
                    )
            except Exception:
                self._clear_agm_env(env_id)
                self._stl_states[env_id] = _STLEnvState(valid=False)
                raise
        self._setup_visual_yoloe_rollout_export(ids)

    def _stl_setup(self, ids) -> None:
        """Run base setup, then initialize candidate-path AGM runtimes."""
        if self._uses_visual_agm_reward():
            self._setup_pure_visual_agm(ids)
            return

        super()._stl_setup(ids)
        # 每个 Ray worker 使用独立目录，防止同名图片互相覆盖。
        if (
            self._stl_plot_dir
            and not getattr(self, "_agm_worker_plot_dir_ready", False)
        ):
            self._stl_plot_dir = os.path.join(
                str(self._stl_plot_dir),
                f"worker_pid_{os.getpid()}",
            )
            os.makedirs(self._stl_plot_dir, exist_ok=True)
            self._agm_worker_plot_dir_ready = True


        if ids is not None and len(ids) > 0:
            try:
                self._setup_visual_stl_shadow(ids)
            except Exception as error:
                print(
                    "[VISUAL_STL_SETUP_ERROR] "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )

        if ids is None or len(ids) == 0 or goal_atom_margin is None:
            return

        self._ensure_agm_state_maps()
        if not hasattr(self, "_agm_stage_runtimes"):
            self._agm_stage_runtimes: dict[int, AGMStageRuntime] = {}
        if not hasattr(self, "_agm_stage_plans"):
            self._agm_stage_plans: dict[int, StagePlan] = {}
        if not hasattr(self, "_agm_initial_atom_margins"):
            self._agm_initial_atom_margins: dict[
                int,
                dict[Atom, float],
            ] = {}
        if not hasattr(self, "_agm_initial_goal_margins"):
            self._agm_initial_goal_margins: dict[int, list[float]] = {}
        if not hasattr(self, "_agm_initial_grasp_margins"):
            self._agm_initial_grasp_margins: dict[
                int,
                list[Optional[float]],
            ] = {}
        if not hasattr(self, "_agm_last_raw_margins"):
            self._agm_last_raw_margins: dict[
                int,
                dict[str, float],
            ] = {}
        if not hasattr(self, "_visual_atom_state_ids"):
            self._visual_atom_state_ids: dict[
                int, dict[Atom, int]
            ] = {}
        if not hasattr(self, "_visual_agm_runtimes"):
            self._visual_agm_runtimes: dict[
                int, VisualAGMShadowRuntime
            ] = {}
        if not hasattr(self, "_stl_comparison_frames"):
            self._stl_comparison_frames: dict[
                int, list[STLComparisonFrame]
            ] = {}

        confirmation_steps = int(
            self.cfg.get("agm_stage_confirmation_steps", 3)
        )
        priv_list = self.env.get_privileged_state(id=ids)

        for position, raw_env_id in enumerate(ids):
            env_id = int(raw_env_id)
            st = self._stl_states[env_id]
            priv = (
                priv_list[position]
                if position < len(priv_list)
                else None
            )

            if st is None or not st.valid or priv is None:
                self._clear_agm_env(env_id)
                continue

            try:
                task_id = int(self.task_ids[env_id])
                description = self._task_description_for_env(env_id)
                plan = build_stage_plan(
                    task_suite_name=self.cfg.task_suite_name,
                    task_id=task_id,
                    goal_atoms=st.goal_atoms,
                    task_description=description,
                )

                ep0 = _build_step_episode(priv)
                initial_by_atom = {
                    atom: self._raw_atom_margin(ep0, atom)
                    for atom in plan.atoms
                }

                runtime = AGMStageRuntime(
                    plan=plan,
                    initial_margins=initial_by_atom,
                    confirmation_steps=confirmation_steps,
                )

                self._agm_stage_runtimes[env_id] = runtime
                self._agm_stage_plans[env_id] = plan
                self._agm_initial_atom_margins[env_id] = initial_by_atom

                if self._visual_stl_enabled():
                    self._visual_agm_runtimes[env_id] = (
                        VisualAGMShadowRuntime(
                            plan=plan,
                            confirmation_steps=confirmation_steps,
                        )
                    )
                    self._stl_comparison_frames[env_id] = []
                    state_ids: dict[Atom, int] = {}
                    visual_monitor = getattr(
                        self, "_visual_stl_shadow", None
                    )
                    if visual_monitor is not None:
                        calibration = visual_monitor.calibrations.get(env_id)
                        fallback_calibration = (
                            visual_monitor.fallback_calibrations.get(env_id)
                        )
                        for atom_index, atom in enumerate(plan.atoms):
                            if atom[0] == "pick":
                                continue
                            state_id = (
                                (env_id + 1) * 1000 + atom_index + 1
                            )
                            state_ids[atom] = state_id
                            visual_monitor.reset_env(
                                state_id,
                                calibration,
                                fallback_calibration=fallback_calibration,
                                raw_obs=(
                                    self.current_raw_obs[env_id]
                                    if self.current_raw_obs is not None
                                    else None
                                ),
                            )
                    self._visual_atom_state_ids[env_id] = state_ids

                goal_initial: list[float] = []
                grasp_initial: list[Optional[float]] = []

                for atom, grasp_obj in zip(
                    st.goal_atoms,
                    st.grasp_objs,
                ):
                    goal_initial.append(
                        self._raw_atom_margin(ep0, atom)
                    )

                    if grasp_obj is None:
                        grasp_initial.append(None)
                    else:
                        grasp_initial.append(
                            self._raw_atom_margin(
                                ep0,
                                ("pick", grasp_obj, None),
                            )
                        )

                self._agm_initial_goal_margins[env_id] = goal_initial
                self._agm_initial_grasp_margins[env_id] = (
                    grasp_initial
                )

                if getattr(self, "_stl_dbg", False):
                    print(
                        "[AGM_STAGE_DBG] "
                        f"env={env_id} task={task_id} "
                        f"audited={plan.audited} "
                        f"paths={[path.name for path in plan.paths]} "
                        f"reason={plan.reason}",
                        flush=True,
                    )

            except Exception as exc:
                # Preserve senior's base STL implementation if the experiment
                # cannot initialize for this environment.
                self._clear_agm_env(env_id)
                if getattr(self, "_stl_dbg", False):
                    print(
                        f"[AGM_STAGE_DBG] setup failed for env "
                        f"{env_id}: {exc!r}",
                        flush=True,
                    )

    def _clear_agm_env(self, env_id: int) -> None:
        self._agm_stage_runtimes.pop(env_id, None)
        self._agm_stage_plans.pop(env_id, None)
        self._agm_initial_atom_margins.pop(env_id, None)
        self._agm_last_raw_margins.pop(env_id, None)
        if hasattr(self, "_visual_agm_runtimes"):
            self._visual_agm_runtimes.pop(env_id, None)
        if hasattr(self, "_visual_atom_state_ids"):
            self._visual_atom_state_ids.pop(env_id, None)
        if hasattr(self, "_visual_active_atoms"):
            self._visual_active_atoms.pop(env_id, None)
        if hasattr(self, "_visual_initialized_state_ids"):
            self._visual_initialized_state_ids.pop(env_id, None)
        if hasattr(self, "_visual_reward_last_score"):
            self._visual_reward_last_score.pop(env_id, None)
        if hasattr(self, "_visual_completion_bonus_paid"):
            self._visual_completion_bonus_paid.pop(env_id, None)
        if hasattr(self, "_visual_success_current"):
            self._visual_success_current.pop(env_id, None)
        if hasattr(self, "_visual_completion_bonus_trace"):
            self._visual_completion_bonus_trace.pop(env_id, None)
        if hasattr(self, "_visual_completion_bonus_snapshots"):
            self._visual_completion_bonus_snapshots.pop(env_id, None)
        if hasattr(self, "_stl_comparison_frames"):
            self._stl_comparison_frames.pop(env_id, None)
        if hasattr(self, "_visual_margin_histories"):
            self._visual_margin_histories.pop(env_id, None)
        if hasattr(self, "_visual_margin_invalid_streaks"):
            self._visual_margin_invalid_streaks.pop(env_id, None)
        if hasattr(self, "_visual_filter_states"):
            self._visual_filter_states.pop(env_id, None)
        if hasattr(self, "_visual_plot_ready_seen"):
            self._visual_plot_ready_seen.pop(env_id, None)
        if hasattr(self, "_visual_oracle_comparison_runtimes"):
            self._visual_oracle_comparison_runtimes.pop(env_id, None)
        if hasattr(self, "_visual_oracle_comparison_errors"):
            self._visual_oracle_comparison_errors.discard(env_id)
        if hasattr(self, "_visual_raw_rgb_frames"):
            self._visual_raw_rgb_frames.pop(env_id, None)
        if hasattr(self, "_visual_raw_rgb_steps"):
            self._visual_raw_rgb_steps.pop(env_id, None)
        self._agm_initial_goal_margins[env_id] = []
        self._agm_initial_grasp_margins[env_id] = []

    def _record_visual_raw_rgb_frame(self, env_id: int, step: int) -> None:
        """Buffer exact upright YOLOE RGB inputs without drawing overlays."""
        if not bool(self.cfg.get("visual_stl_export_raw_rgb", False)):
            return
        stride = max(1, int(self.cfg.get("visual_stl_raw_rgb_stride", 1)))
        if (int(step) - 1) % stride:
            return
        if self.current_raw_obs is None:
            raise RuntimeError("raw RGB export requires current_raw_obs")
        raw_obs = self.current_raw_obs[int(env_id)]
        primary_name = str(self.cfg.get("visual_stl_camera_name", "agentview"))
        fallback_name = self.cfg.get("visual_stl_fallback_camera_name", None)

        def upright(camera_name: str):
            key = f"{camera_name}_image"
            if key not in raw_obs:
                raise KeyError(f"raw RGB camera is unavailable: {key}")
            return np.ascontiguousarray(
                np.asarray(raw_obs[key], dtype=np.uint8)[::-1, ::-1]
            )

        frame = {
            "primary": upright(primary_name),
            "fallback": (
                upright(str(fallback_name))
                if fallback_name else None
            ),
            "primary_camera": primary_name,
            "fallback_camera": str(fallback_name or ""),
        }
        self._visual_raw_rgb_frames.setdefault(int(env_id), []).append(frame)
        self._visual_raw_rgb_steps.setdefault(int(env_id), []).append(int(step))

    def _dump_visual_raw_rgb_episode(
        self,
        *,
        env_id: int,
        episode_stem: str,
        task_id: int,
        trial_id: int,
    ) -> str | None:
        """Write one lossless NPZ containing the exact detector input arrays."""
        if not bool(self.cfg.get("visual_stl_export_raw_rgb", False)):
            return None
        frames = self._visual_raw_rgb_frames.get(int(env_id), [])
        steps = self._visual_raw_rgb_steps.get(int(env_id), [])
        if not frames or len(frames) != len(steps):
            raise RuntimeError(
                "raw RGB export buffer is empty or misaligned: "
                f"frames={len(frames)} steps={len(steps)}"
            )
        suffix = (
            episode_stem[len("stl_plot_"):]
            if episode_stem.startswith("stl_plot_") else episode_stem
        )
        path = os.path.join(self._stl_plot_dir, f"raw_rgb_{suffix}.npz")
        primary = np.stack([item["primary"] for item in frames]).astype(
            np.uint8, copy=False
        )
        fallback_frames = [item["fallback"] for item in frames]
        fallback = (
            np.stack(fallback_frames).astype(np.uint8, copy=False)
            if fallback_frames[0] is not None
            else np.empty((0,), dtype=np.uint8)
        )
        np.savez_compressed(
            path,
            steps=np.asarray(steps, dtype=np.int32),
            primary_rgb=primary,
            fallback_rgb=fallback,
            primary_camera=np.asarray(frames[0]["primary_camera"]),
            fallback_camera=np.asarray(frames[0]["fallback_camera"]),
            task_id=np.asarray(int(task_id), dtype=np.int32),
            trial_id=np.asarray(int(trial_id), dtype=np.int32),
            env_id=np.asarray(int(env_id), dtype=np.int32),
        )
        self._visual_raw_rgb_frames[int(env_id)] = []
        self._visual_raw_rgb_steps[int(env_id)] = []
        print(
            "[VISUAL_RAW_RGB_DUMP] "
            f"env={env_id} task={task_id} trial={trial_id} "
            f"frames={len(steps)} path={path}",
            flush=True,
        )
        return path

    def _plot_stl_episode(
        self,
        out_path,
        st,
        task_desc,
        success,
    ) -> None:
        """Use mathtext-safe display names without changing saved atom names."""
        original_atom_margins = st.atom_margins

        try:
            st.atom_margins = {
                name.replace("_", "-"): trace
                for name, trace in original_atom_margins.items()
            }

            super()._plot_stl_episode(
                out_path,
                st,
                task_desc,
                success,
            )
        finally:
            st.atom_margins = original_atom_margins


    def _dump_episode_plots(self, ids) -> None:
        """Save every AGM trace once and expose plotting failures."""
        if not hasattr(self, "_agm_last_plotted_stl_state"):
            self._agm_last_plotted_stl_state = {}

        for raw_env_id in ids:
            env_id = int(raw_env_id)
            st = (
                self._stl_states[env_id]
                if env_id < len(self._stl_states)
                else None
            )

            if st is None or not st.rho_trace:
                continue

            if self._agm_last_plotted_stl_state.get(env_id) is st:
                continue

            try:
                task_id = int(self.task_ids[env_id])
                task_desc = self.task_suite.get_task(task_id).language
                success = bool(self.success_once[env_id])

                self._stl_plot_counter += 1
                ep_str = f"ep{self._stl_plot_counter:04d}"

                out_path = os.path.join(
                    self._stl_plot_dir,
                    (
                        f"stl_plot_{ep_str}_env{env_id:02d}_"
                        f"task{task_id:02d}_"
                        f"{'SUCCESS' if success else 'fail'}.png"
                    ),
                )

                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                stem = os.path.splitext(os.path.basename(out_path))[0]
                data_path = os.path.join(
                    self._stl_plot_dir,
                    stem.replace("stl_plot_", "stl_data_", 1) + ".npz",
                )
                np.savez(
                    data_path,
                    rho=np.asarray(st.rho_trace, dtype=np.float64),
                    shaping=np.asarray(st.shaping_trace, dtype=np.float64),
                    reward=np.asarray(st.reward_trace, dtype=np.float64),
                    visual_completion_bonus=np.asarray(
                        self._visual_completion_bonus_trace.get(env_id, []),
                        dtype=np.float64,
                    ),
                    atom_names=list(st.atom_margins.keys()),
                    **{
                        f"atom_{name}": np.asarray(values, dtype=np.float64)
                        for name, values in st.atom_margins.items()
                    },
                )
                self._plot_stl_episode(
                    out_path,
                    st,
                    task_desc,
                    success,
                )

                self._dump_visual_stl_episode(
                    env_id=env_id,
                    out_path=out_path,
                    task_description=task_desc,
                    success=success,
                )
                self._dump_visual_raw_rgb_episode(
                    env_id=env_id,
                    episode_stem=stem,
                    task_id=task_id,
                    trial_id=int(
                        np.asarray(self.trial_ids).reshape(-1)[env_id]
                    ),
                )

            except Exception as exc:
                print(
                    f"[AGM_PLOT_ERROR] env={env_id} "
                    f"task={getattr(self, 'task_ids', [None])[env_id]} "
                    f"error={exc!r}",
                    flush=True,
                )
                continue

            self._agm_last_plotted_stl_state[env_id] = st

    def _collect_visual_samples_for_atoms(
        self,
        env_id: int,
        atoms,
        visual_state_ids: dict[Atom, int],
    ) -> dict[Atom, VisualAtomSample]:
        """Evaluate one explicit atom set against one isolated state map."""
        samples: dict[Atom, VisualAtomSample] = {}
        object_atoms: dict[int, list[Atom]] = {}

        for atom in atoms:
            pred = atom[0]
            visual_state_id = visual_state_ids.get(atom)
            if pred == "open":
                record = self._record_visual_open_step(
                    env_id=env_id,
                    atom=atom,
                    oracle_open_margin=float("nan"),
                    visual_state_id=visual_state_id,
                )
                samples[atom] = self._visual_atom_sample(
                    record,
                    "visual_open_margin",
                )
                continue
            elif pred == "close":
                record = self._record_visual_close_step(
                    env_id=env_id,
                    atom=atom,
                    oracle_close_margin=float("nan"),
                    visual_state_id=visual_state_id,
                )
                samples[atom] = self._visual_atom_sample(
                    record,
                    "visual_close_margin",
                )
                continue
            elif pred in {"turnon", "turn_on"}:
                record = self._record_visual_turnon_step(
                    env_id=env_id,
                    atom=atom,
                    oracle_turnon_margin=float("nan"),
                    visual_state_id=visual_state_id,
                )
                samples[atom] = self._visual_atom_sample(
                    record,
                    "visual_turnon_margin",
                )
                continue

            if pred not in {"pick", "on", "in"} or atom[1] is None:
                samples[atom] = VisualAtomSample(
                    margin=float("nan"),
                    valid=False,
                    reason=f"unsupported_visual_atom:{atom}",
                )
                continue

            if visual_state_id is None:
                samples[atom] = VisualAtomSample(
                    margin=float("nan"),
                    valid=False,
                    reason=f"visual_state_unavailable:{atom}",
                )
                continue
            object_atoms.setdefault(int(visual_state_id), []).append(atom)

        # Pick/On/In atoms for the same manipulated object intentionally share
        # estimator state. Evaluate that object once per frame and export all
        # entered predicate margins from the same RGB-D result.
        for visual_state_id, atoms in object_atoms.items():
            grasp_obj = str(atoms[0][1])
            on_atom = next((atom for atom in atoms if atom[0] == "on"), None)
            in_atom = next((atom for atom in atoms if atom[0] == "in"), None)
            record = self._record_visual_stl_step(
                env_id=env_id,
                grasp_obj=grasp_obj,
                object_key=grasp_obj,
                on_target_object=(on_atom[2] if on_atom is not None else None),
                in_target_object=(in_atom[2] if in_atom is not None else None),
                visual_state_id=visual_state_id,
            )
            for atom in atoms:
                pred = atom[0]
                if pred == "pick":
                    samples[atom] = self._visual_atom_sample(
                        record,
                        "visual_grasp_margin",
                    )
                elif pred == "on":
                    samples[atom] = self._visual_atom_sample(
                        record,
                        "visual_on_margin",
                        valid_field="visual_on_valid",
                        reason_field="visual_on_reason",
                        source_field="visual_on_object_source",
                    )
                elif pred == "in":
                    samples[atom] = self._latest_visual_in_sample(
                        visual_state_id
                    )

        return samples

    def _collect_pure_visual_samples(
        self,
        env_id: int,
        st: _STLEnvState,
    ) -> dict[Atom, VisualAtomSample]:
        """Evaluate every plan atom; AGM gates only the active stage."""
        runtime = self._visual_agm_runtimes.get(env_id)
        if runtime is None:
            return {}
        return self._collect_visual_samples_for_atoms(
            env_id,
            tuple(runtime.tracked_atoms),
            self._visual_atom_state_ids.get(env_id, {}),
        )

    def _append_visual_oracle_comparison(
        self,
        *,
        env_id: int,
        step: int,
        privileged,
        visual_step,
        visual_score: float,
    ) -> None:
        """Record eval-only oracle diagnostics without returning a reward."""
        oracle_runtime = self._visual_oracle_comparison_runtimes.get(env_id)
        if oracle_runtime is None or privileged is None:
            return

        try:
            episode = _build_step_episode(privileged)
            current = {
                atom: self._raw_atom_margin(episode, atom)
                for atom in oracle_runtime.plan.atoms
            }
            oracle_state = oracle_runtime.update(current)
            self._stl_comparison_frames.setdefault(env_id, []).append(
                STLComparisonFrame(
                    step=int(step),
                    privileged_normalized=dict(
                        oracle_runtime.last_normalized_margins
                    ),
                    visual_normalized=dict(visual_step.normalized_margins),
                    privileged_score=float(oracle_state.episode_score),
                    visual_score=float(visual_score),
                    visual_valid=bool(visual_step.valid),
                    visual_raw=dict(visual_step.raw_margins),
                    visual_atom_valid=dict(visual_step.atom_valid),
                    visual_atom_reasons=dict(visual_step.atom_reasons),
                    privileged_stage=str(oracle_state.active_stage_name),
                    visual_stage=str(visual_step.active_stage_name),
                    visual_reason=str(visual_step.reason),
                )
            )
        except Exception as exc:
            if env_id not in self._visual_oracle_comparison_errors:
                self._visual_oracle_comparison_errors.add(env_id)
                print(
                    "[VISUAL_ORACLE_COMPARISON_ERROR] "
                    f"env={env_id} error={exc!r}",
                    flush=True,
                )

    def _prepare_visual_yoloe_batch(self) -> None:
        """Batch primary-camera detector work for this EnvGroup step."""
        monitor = getattr(self, "_visual_stl_shadow", None)
        if monitor is None or self.current_raw_obs is None:
            return
        camera_name = str(
            self.cfg.get("visual_stl_camera_name", "agentview")
        )
        image_key = f"{camera_name}_image"
        images = []
        for env_id in range(self.num_envs):
            raw_obs = self.current_raw_obs[env_id]
            if raw_obs is None or image_key not in raw_obs:
                continue
            images.append(np.ascontiguousarray(
                np.asarray(raw_obs[image_key], dtype=np.uint8)[::-1, ::-1]
            ))
        try:
            monitor.prepare_inference_batch(images)
            fallback_name = self.cfg.get(
                "visual_stl_fallback_camera_name", None
            )
            if fallback_name:
                fallback_key = f"{fallback_name}_image"
                fallback_requests = []
                for env_id in range(self.num_envs):
                    raw_obs = self.current_raw_obs[env_id]
                    runtime = self._visual_agm_runtimes.get(env_id)
                    if (
                        raw_obs is None
                        or image_key not in raw_obs
                        or fallback_key not in raw_obs
                        or runtime is None
                        or env_id not in monitor.fallback_calibrations
                    ):
                        continue
                    target_class = monitor.target_class_for_atoms(
                        runtime.tracked_atoms
                    )
                    if target_class is None:
                        continue
                    primary_rgb = np.ascontiguousarray(
                        np.asarray(
                            raw_obs[image_key], dtype=np.uint8
                        )[::-1, ::-1]
                    )
                    fallback_rgb = np.ascontiguousarray(
                        np.asarray(
                            raw_obs[fallback_key], dtype=np.uint8
                        )[::-1, ::-1]
                    )
                    fallback_requests.append((
                        primary_rgb,
                        fallback_rgb,
                        target_class,
                    ))
                monitor.prepare_missing_target_fallback_batch(
                    fallback_requests
                )
        except Exception as exc:
            # Preserve the established single-image path when a mixed-version
            # service or malformed batch rejects the optimization.
            monitor.prepare_inference_batch(())
            print(
                "[YOLOE_BATCH_PREFETCH_ERROR] "
                f"images={len(images)} error={exc!r}",
                flush=True,
            )

    def _pure_visual_stl_robustness(self) -> np.ndarray:
        """Return finite visual AGM potentials without reading simulator state."""
        rho_out = np.zeros(self.num_envs, dtype=np.float64)
        oracle_privileged = self._oracle_comparison_step_states()
        self._prepare_visual_yoloe_batch()
        for env_id in range(self.num_envs):
            last_score = float(self._visual_reward_last_score.get(
                env_id,
                _VISUAL_COLD_START_SCORE,
            ))
            rho_out[env_id] = last_score
            st = (
                self._stl_states[env_id]
                if env_id < len(self._stl_states)
                else None
            )
            runtime = self._visual_agm_runtimes.get(env_id)
            if st is None or not st.valid or runtime is None:
                raise RuntimeError(
                    "Pure-visual AGM was not initialized for "
                    f"env {env_id}; refusing zero-reward training"
                )

            try:
                step = int(
                    np.asarray(self._elapsed_steps).reshape(-1)[env_id]
                )
                self._record_visual_raw_rgb_frame(env_id, step)
                self._activate_visual_atoms(env_id, runtime.active_atoms)
                samples = self._filter_visual_samples(
                    env_id,
                    self._collect_pure_visual_samples(env_id, st),
                )
                visual_step = runtime.observe(
                    step=step,
                    samples=samples,
                )
                # A stage transition changes the active atom set. Reset the
                # newly entered estimator now; its first sample is next frame.
                self._activate_visual_atoms(env_id, runtime.active_atoms)
                if visual_step.valid and np.isfinite(
                    visual_step.episode_score
                ):
                    last_score = float(visual_step.episode_score)
                    self._visual_reward_last_score[env_id] = last_score
                rho_out[env_id] = last_score
                st.prev_rho = last_score

                plot_ready = bool(
                    self._visual_plot_ready_seen.get(env_id, False)
                )
                if visual_step.ready:
                    plot_ready = True
                    self._visual_plot_ready_seen[env_id] = True
                display_score = last_score if plot_ready else float("nan")
                self._append_visual_oracle_comparison(
                    env_id=env_id,
                    step=step,
                    privileged=(
                        oracle_privileged[env_id]
                        if env_id < len(oracle_privileged) else None
                    ),
                    visual_step=visual_step,
                    visual_score=display_score,
                )

                step_margins = {
                    _stl_atom_name(atom): float(
                        visual_step.normalized_margins.get(
                            atom,
                            float("nan"),
                        )
                    )
                    for atom in runtime.plan.atoms
                }
                st.last_margins = step_margins
                if self._stl_plot_dir:
                    st.rho_trace.append(display_score)
                    st.shaping_trace.append(0.0)
                    for name, margin in step_margins.items():
                        st.atom_margins.setdefault(name, []).append(
                            margin
                        )

                if (
                    getattr(self, "_stl_dbg", False)
                    and not visual_step.valid
                ):
                    print(
                        "[VISUAL_AGM_FREEZE] "
                        f"env={env_id} step={step} "
                        f"score={last_score:+.4f} "
                        f"reason={visual_step.reason}",
                        flush=True,
                    )
            except Exception as exc:
                if getattr(self, "_stl_dbg", False):
                    print(
                        "[VISUAL_AGM_ERROR] "
                        f"env={env_id} score_frozen={last_score:+.4f} "
                        f"error={exc!r}",
                        flush=True,
                    )

        return rho_out

    def _visual_completion_trigger_snapshot(self, env_id: int) -> dict:
        """Freeze pure-visual relation evidence at the first bonus step."""
        env_id = int(env_id)
        runtime = self._visual_agm_runtimes.get(env_id)
        monitor = getattr(self, "_visual_stl_shadow", None)
        state_ids = self._visual_atom_state_ids.get(env_id, {})
        if runtime is None or monitor is None or not runtime.trace:
            raise RuntimeError(
                "visual completion trigger snapshot requires an active runtime"
            )
        trigger_state = runtime.trace[-1]
        atoms = []
        for atom in runtime.plan.atoms:
            if atom[0] not in {"on", "in"}:
                continue
            state_id = state_ids.get(atom)
            records = ()
            if state_id is not None:
                records = (
                    monitor.records.get(int(state_id), ())
                    if atom[0] == "on"
                    else monitor.in_records.get(int(state_id), ())
                )
            record = records[-1] if records else None
            record_fields = (
                {
                    key: value
                    for key, value in vars(record).items()
                    if "oracle" not in key.lower()
                }
                if record is not None
                else None
            )
            atoms.append({
                "atom": list(atom),
                "state_id": int(state_id) if state_id is not None else None,
                "record": _visual_telemetry_json_value(record_fields),
            })
        runtime_fields = {
            key: value
            for key, value in vars(trigger_state).items()
            if "oracle" not in key.lower()
        }
        return {
            "schema_version": 1,
            "step": int(trigger_state.step),
            "reward_source": "visual_agm_completion",
            "completion_gate": (
                "visual_agm_all_complete_and_relation_safe"
            ),
            "runtime": _visual_telemetry_json_value(runtime_fields),
            "relation_atoms": atoms,
            "privileged_fields_included": False,
        }

    def _visual_relation_atoms_safe_for_completion(self, env_id: int) -> bool:
        """Require visual relation latches plus conservative final evidence."""
        runtime = self._visual_agm_runtimes.get(int(env_id))
        monitor = getattr(self, "_visual_stl_shadow", None)
        if runtime is None or monitor is None:
            return False
        state_ids = self._visual_atom_state_ids.get(int(env_id), {})
        in_record_groups = {}
        for atom in runtime.plan.atoms:
            if atom[0] not in {"on", "in"}:
                continue
            state_id = state_ids.get(atom)
            if state_id is None:
                return False
            state_id = int(state_id)
            if atom[0] == "on":
                if not bool(monitor.on_confirmed.get(state_id, False)):
                    return False
                records = monitor.records.get(state_id, ())
                if not records:
                    return False
                latest = records[-1]
                if (
                    latest.visual_on_geometry_mode == "site_surface"
                    and str(latest.on_target_class).replace("_", " ")
                    == "wine rack"
                    and not _visual_site_contact_sequence_complete(records)
                ):
                    return False
                continue

            records = monitor.in_records.get(state_id, ())
            in_record_groups.setdefault(atom[2], []).append(records)

        for record_groups in in_record_groups.values():
            if len(record_groups) >= 2 and any(
                not records for records in record_groups
            ):
                return False
            if (
                len(record_groups) >= 2
                and all(
                    records[-1].container_geometry == "basket"
                    for records in record_groups
                )
                and not _visual_multi_in_group_safe(record_groups)
            ):
                return False
        return True

    def _calc_step_reward(self, terminations):
        if not self._uses_visual_agm_reward():
            return super()._calc_step_reward(terminations)

        rho = self._pure_visual_stl_robustness()
        visual_complete = np.asarray(
            [
                bool(
                    self._visual_agm_runtimes[env_id].trace
                    and self._visual_agm_runtimes[env_id]
                    .trace[-1].all_complete
                    and self._visual_relation_atoms_safe_for_completion(
                        env_id
                    )
                )
                for env_id in range(self.num_envs)
            ],
            dtype=bool,
        )
        if self._stl_clip_lower is not None or self._stl_clip_upper is not None:
            rho = np.clip(rho, self._stl_clip_lower, self._stl_clip_upper)
        if not np.all(np.isfinite(rho)):
            raise RuntimeError("Pure-visual AGM produced a non-finite reward")

        paid = np.asarray(
            [
                self._visual_completion_bonus_paid.get(env_id, False)
                for env_id in range(self.num_envs)
            ],
            dtype=bool,
        )
        success_bonus, paid, visual_success = (
            _one_shot_visual_completion_bonus(
                visual_complete,
                paid,
                bonus_value=float(self._visual_completion_bonus_value),
            )
        )
        for env_id in range(self.num_envs):
            if (
                float(success_bonus[env_id]) > 0.0
                and self._visual_completion_bonus_snapshots.get(env_id)
                is None
            ):
                self._visual_completion_bonus_snapshots[env_id] = (
                    self._visual_completion_trigger_snapshot(env_id)
                )
            self._visual_completion_bonus_paid[env_id] = bool(
                paid[env_id]
            )
            self._visual_success_current[env_id] = bool(
                visual_success[env_id]
            )

        if self.use_rel_reward:
            step_reward = self._stl_reward_scale * discounted_progress_reward(
                rho,
                self.prev_step_reward,
                gamma=self._stl_gamma,
            )
            self.prev_step_reward = np.asarray(rho, dtype=np.float64)
        else:
            step_reward = self._stl_reward_scale * rho

        # Add after progress shaping so the visual one-shot bonus is not
        # subtracted on the following step. Environment truth never enters
        # this reward branch; it is consumed only by metric recording.
        step_reward = np.asarray(step_reward, dtype=np.float64) + success_bonus

        if self._stl_plot_dir:
            for env_id, value in enumerate(step_reward):
                st = self._stl_states[env_id]
                if st is not None and st.valid:
                    st.reward_trace.append(float(value))
                    self._visual_completion_bonus_trace[env_id].append(
                        float(success_bonus[env_id])
                    )
        return step_reward

    def _visual_success_mask(self) -> np.ndarray:
        return np.asarray(
            [
                self._visual_success_current.get(env_id, False)
                for env_id in range(self.num_envs)
            ],
            dtype=bool,
        )

    def _init_metrics(self):
        """Initialize oracle primary metrics and visual audit counters."""
        super()._init_metrics()
        self.oracle_success_once = np.zeros(self.num_envs, dtype=bool)
        self.visual_success_once = np.zeros(self.num_envs, dtype=bool)
        self._oracle_success_current = np.zeros(self.num_envs, dtype=bool)
        self._root_cause_episode_ordinal = np.zeros(
            self.num_envs, dtype=np.int64
        )
        self._root_cause_emitted_ordinal = np.full(
            self.num_envs, -1, dtype=np.int64
        )

    def _reset_metrics(self, env_idx=None):
        """Reset both metric namespaces without changing reward state."""
        super()._reset_metrics(env_idx)
        if not hasattr(self, "oracle_success_once"):
            self.oracle_success_once = np.zeros(self.num_envs, dtype=bool)
            self._oracle_success_current = np.zeros(
                self.num_envs, dtype=bool
            )
            self._root_cause_episode_ordinal = np.zeros(
                self.num_envs, dtype=np.int64
            )
            self._root_cause_emitted_ordinal = np.full(
                self.num_envs, -1, dtype=np.int64
            )
        if not hasattr(self, "visual_success_once"):
            self.visual_success_once = np.zeros(
                self.num_envs, dtype=bool
            )

        if env_idx is None:
            ids = np.arange(self.num_envs, dtype=np.int64)
        else:
            ids = np.asarray(env_idx, dtype=np.int64).reshape(-1)
        self.oracle_success_once[ids] = False
        self.visual_success_once[ids] = False
        self._oracle_success_current[ids] = False
        if hasattr(self, "_visual_completion_bonus_snapshots"):
            for raw_env_id in ids:
                self._visual_completion_bonus_snapshots[
                    int(raw_env_id)
                ] = None
        self._root_cause_episode_ordinal[ids] += 1

    def _root_cause_audit_enabled(self) -> bool:
        return bool(self.cfg.get("visual_stl_root_cause_audit", False))

    def _append_root_cause_episode_metrics(self, infos):
        """Expose explicit oracle and visual metric namespaces.

        The base ``success_once`` and explicit ``oracle_success_once`` use
        simulator truth. Visual completion is the sparse reward source and is
        retained separately for reward-alignment diagnostics.
        """
        if not self._root_cause_audit_enabled():
            return infos

        visual_once = np.asarray(self.visual_success_once, dtype=bool)
        oracle_once = np.asarray(self.oracle_success_once, dtype=bool)
        visual_current = self._visual_success_mask()
        oracle_current = np.asarray(
            self._oracle_success_current, dtype=bool
        )
        episode = infos.setdefault("episode", {})
        audit = {
            "oracle_success_once": oracle_once.copy(),
            "visual_success_once": visual_once.copy(),
            "visual_completion_bonus_paid": np.asarray(
                [
                    self._visual_completion_bonus_paid.get(
                        env_id, False
                    )
                    for env_id in range(self.num_envs)
                ],
                dtype=bool,
            ),
            "visual_oracle_match_once": (visual_once == oracle_once),
            "visual_false_positive_once": visual_once & ~oracle_once,
            "visual_false_negative_once": oracle_once & ~visual_once,
            "oracle_success_at_end": oracle_current.copy(),
            "visual_success_at_end": visual_current.copy(),
            "visual_oracle_match_at_end": (
                visual_current == oracle_current
            ),
            "visual_false_positive_at_end": (
                visual_current & ~oracle_current
            ),
            "visual_false_negative_at_end": (
                oracle_current & ~visual_current
            ),
        }
        for key, value in audit.items():
            episode[key] = to_tensor(value)
        return infos

    def _write_root_cause_audit_rows(self, done_mask) -> None:
        """Write one diagnostics-only JSON record per completed episode."""
        if not self._root_cause_audit_enabled():
            return
        output_dir = str(
            self.cfg.get("visual_stl_root_cause_audit_dir", "")
        ).strip()
        if not output_dir:
            raise RuntimeError(
                "visual_stl_root_cause_audit=true requires "
                "visual_stl_root_cause_audit_dir"
            )

        ids = np.flatnonzero(np.asarray(done_mask, dtype=bool))
        if ids.size == 0:
            return
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir, f"pid_{os.getpid()}.jsonl"
        )
        rollout_epoch = max(1, int(self.cfg.get("rollout_epoch", 1)))
        task_ids = np.asarray(self.task_ids).reshape(-1)
        trial_ids = np.asarray(self.trial_ids).reshape(-1)
        reset_ids = np.asarray(self.reset_state_ids).reshape(-1)
        elapsed = np.asarray(self.elapsed_steps).reshape(-1)
        visual_return = np.asarray(self.returns).reshape(-1)
        visual_once = np.asarray(self.visual_success_once, dtype=bool)
        oracle_once = np.asarray(self.oracle_success_once, dtype=bool)
        visual_current = self._visual_success_mask()
        oracle_current = np.asarray(
            self._oracle_success_current, dtype=bool
        )

        with open(output_path, "a", encoding="utf-8") as handle:
            for raw_env_id in ids:
                env_id = int(raw_env_id)
                ordinal = int(self._root_cause_episode_ordinal[env_id])
                if self._root_cause_emitted_ordinal[env_id] == ordinal:
                    continue
                self._root_cause_emitted_ordinal[env_id] = ordinal
                inferred_step = (max(ordinal, 1) - 1) // rollout_epoch + 1
                row = {
                    "schema_version": 4,
                    "pid": os.getpid(),
                    "env_id": env_id,
                    "episode_ordinal": ordinal,
                    "inferred_global_step": inferred_step,
                    "rollout_within_step": (
                        (max(ordinal, 1) - 1) % rollout_epoch + 1
                    ),
                    "task_id": int(task_ids[env_id]),
                    "trial_id": int(trial_ids[env_id]),
                    "reset_state_id": int(reset_ids[env_id]),
                    "elapsed_steps": int(elapsed[env_id]),
                    "success_once_source": "environment_truth",
                    "reward_bonus_source": "visual_agm_completion",
                    "visual_completion_gate": (
                        "visual_agm_all_complete_and_relation_safe"
                    ),
                    "success_once": bool(oracle_once[env_id]),
                    "return": float(visual_return[env_id]),
                    "visual_return": float(visual_return[env_id]),
                    "oracle_success_once": bool(oracle_once[env_id]),
                    "visual_success_once": bool(visual_once[env_id]),
                    "visual_completion_bonus_paid": bool(
                        self._visual_completion_bonus_paid.get(
                            env_id, False
                        )
                    ),
                    "visual_completion_bonus_trigger_step": (
                        self._visual_completion_bonus_snapshots.get(
                            env_id, {}
                        ).get("step")
                        if self._visual_completion_bonus_snapshots.get(
                            env_id
                        ) is not None
                        else None
                    ),
                    "visual_completion_bonus_trigger": (
                        self._visual_completion_bonus_snapshots.get(env_id)
                    ),
                    "visual_oracle_match_once": bool(
                        visual_once[env_id] == oracle_once[env_id]
                    ),
                    "visual_false_positive_once": bool(
                        visual_once[env_id] and not oracle_once[env_id]
                    ),
                    "visual_false_negative_once": bool(
                        oracle_once[env_id] and not visual_once[env_id]
                    ),
                    "oracle_success_at_end": bool(
                        oracle_current[env_id]
                    ),
                    "visual_success_at_end": bool(
                        visual_current[env_id]
                    ),
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()

    def _record_metrics(self, step_reward, terminations, infos):
        if not self._uses_visual_agm_reward():
            return super()._record_metrics(
                step_reward,
                terminations,
                infos,
            )
        oracle_current = np.asarray(terminations, dtype=bool).copy()
        visual_current = self._visual_success_mask()
        self._oracle_success_current = oracle_current
        self.oracle_success_once = (
            np.asarray(self.oracle_success_once, dtype=bool)
            | oracle_current
        )
        self.visual_success_once = (
            np.asarray(self.visual_success_once, dtype=bool)
            | visual_current
        )
        infos = super()._record_metrics(
            step_reward,
            oracle_current,
            infos,
        )
        return self._append_root_cause_episode_metrics(infos)

    def _handle_eval_auto_reset(self, dones, final_obs, infos):
        if self._uses_visual_agm_reward():
            infos.setdefault("episode", {})["success_at_end"] = to_tensor(
                np.asarray(self._oracle_success_current, dtype=bool)
            )
            self._append_root_cause_episode_metrics(infos)
            self._write_root_cause_audit_rows(dones)
        return super()._handle_eval_auto_reset(dones, final_obs, infos)

    def step(self, actions=None, auto_reset=True):
        result = super().step(actions=actions, auto_reset=auto_reset)
        self._record_visual_yoloe_rollout_export()
        if not self._uses_visual_agm_reward():
            return result

        obs, reward, terminations, truncations, infos = result
        if "final_info" not in infos:
            self._write_root_cause_audit_rows(
                np.asarray(terminations, dtype=bool)
                | np.asarray(truncations, dtype=bool)
            )
        # Keep the generic metric namespace tied to simulator truth. Visual
        # completion remains available only under explicit visual_* keys.
        if "final_info" not in infos and "episode" in infos:
            infos["episode"]["success_at_end"] = to_tensor(
                np.asarray(self._oracle_success_current, dtype=bool)
            )
        return obs, reward, terminations, truncations, infos

    def _stl_robustness(self) -> np.ndarray:
        """Return the selected candidate-path potential in [-1, 0]."""
        if self._uses_visual_agm_reward():
            return self._pure_visual_stl_robustness()

        priv_list = self.env.get_privileged_state()
        rho_out = np.zeros(self.num_envs, dtype=np.float64)

        for env_id, priv in enumerate(priv_list):
            st = (
                self._stl_states[env_id]
                if env_id < len(self._stl_states)
                else None
            )

            if (
                st is None
                or not st.valid
                or st.tracker is None
                or priv is None
            ):
                continue

            try:
                ep = _build_step_episode(priv)
                if (
                    bool(self.cfg.get("agm_debug_gripper_bodies", False))
                    and not getattr(self, "_agm_printed_gripper_bodies", False)
                ):
                    candidates = [
                        (int(body_id), str(name))
                        for body_id, name in zip(ep.body_ids, ep.body_names)
                        if any(
                            keyword in str(name).lower()
                            for keyword in ("gripper", "finger", "hand")
                        )
                    ]
                    print(
                        f"[AGM_GRIPPER_BODIES] {candidates}",
                        flush=True,
                    )
                    self._agm_printed_gripper_bodies = True
                grasp_margins: list[Optional[float]] = []
                goal_margins: list[float] = []
                step_margins: dict[str, float] = {}
                current_by_atom: dict[Atom, float] = {}
                visual_samples: dict[Atom, VisualAtomSample] = {}
                visual_state_ids = getattr(
                    self, "_visual_atom_state_ids", {}
                ).get(env_id, {})

                # Retain the base environment's atom records and fallback data.
                for atom, grasp_obj in zip(
                    st.goal_atoms,
                    st.grasp_objs,
                ):
                    goal_value = self._raw_atom_margin(ep, atom)
                    goal_margins.append(goal_value)
                    current_by_atom[atom] = goal_value
                    step_margins[_stl_atom_name(atom)] = goal_value

                    if atom[0] == "open" and self._visual_stl_enabled():
                        record = self._record_visual_open_step(
                            env_id=env_id,
                            atom=atom,
                            oracle_open_margin=goal_value,
                            visual_state_id=visual_state_ids.get(atom),
                        )
                        visual_samples[atom] = self._visual_atom_sample(
                            record,
                            "visual_open_margin",
                        )

                    if atom[0] == "close" and self._visual_stl_enabled():
                        record = self._record_visual_close_step(
                            env_id=env_id,
                            atom=atom,
                            oracle_close_margin=goal_value,
                            visual_state_id=visual_state_ids.get(atom),
                        )
                        visual_samples[atom] = self._visual_atom_sample(
                            record,
                            "visual_close_margin",
                        )

                    if (
                        atom[0] in {"turnon", "turn_on"}
                        and self._visual_stl_enabled()
                    ):
                        record = self._record_visual_turnon_step(
                            env_id=env_id,
                            atom=atom,
                            oracle_turnon_margin=goal_value,
                            visual_state_id=visual_state_ids.get(atom),
                        )
                        visual_samples[atom] = self._visual_atom_sample(
                            record,
                            "visual_turnon_margin",
                        )

                    if grasp_obj is None:
                        grasp_margins.append(None)
                        continue

                    pick_atom: Atom = ("pick", grasp_obj, None)
                    pick_value = self._raw_atom_margin(ep, pick_atom)

                    if self._visual_stl_enabled():
                        visual_state_id = visual_state_ids.get(atom)
                        record = self._record_visual_stl_step(
                            env_id=env_id,
                            priv=priv,
                            ep=ep,
                            grasp_obj=grasp_obj,
                            object_key=resolve_object_key(
                                grasp_obj,
                                ep,
                            ),
                            oracle_pick_margin=pick_value,
                            on_target_object=(
                                atom[2]
                                if atom[0] == "on"
                                else None
                            ),
                            oracle_on_margin=(
                                goal_value
                                if atom[0] == "on"
                                else float("nan")
                            ),
                            in_target_object=(
                                atom[2]
                                if atom[0] == "in"
                                else None
                            ),
                            oracle_in_margin=(
                                goal_value
                                if atom[0] == "in"
                                else float("nan")
                            ),
                            visual_state_id=visual_state_id,
                        )
                        visual_samples.setdefault(
                            pick_atom,
                            self._visual_atom_sample(
                                record,
                                "visual_grasp_margin",
                            ),
                        )
                        if atom[0] == "on":
                            visual_samples[atom] = (
                                self._visual_atom_sample(
                                    record,
                                    "visual_on_margin",
                                    valid_field="visual_on_valid",
                                    reason_field="visual_on_reason",
                                    source_field=(
                                        "visual_on_object_source"
                                    ),
                                )
                            )
                        elif atom[0] == "in" and visual_state_id is not None:
                            visual_samples[atom] = (
                                self._latest_visual_in_sample(
                                    visual_state_id
                                )
                            )

                    if (
                        bool(self.cfg.get("agm_debug_pick_components", False))
                        and int(self.task_ids[env_id]) == 6
                    ):
                        object_key = resolve_object_key(grasp_obj, ep)
                        two_finger = bool(
                            np.asarray(
                                two_finger_grasp(ep, object_key)
                            ).reshape(-1)[-1]
                        )
                        force_margin = float(
                            np.asarray(
                                contact_force_on_object(ep, object_key)
                                - self._pred_cfg.f_eps
                            ).reshape(-1)[-1]
                        )
                        closure_margin = float(
                            np.asarray(
                                self._pred_cfg.q_g_max - gripper_width(ep)
                            ).reshape(-1)[-1]
                        )
                        proximity_margin = float(
                            np.asarray(
                                self._pred_cfg.r_grasp
                                - eef_object_dist(ep, object_key)
                            ).reshape(-1)[-1]
                        )

                        print(
                            "[PICK_COMPONENTS] "
                            f"env={env_id} object={grasp_obj} "
                            f"force={force_margin:+.4f} "
                            f"closure={closure_margin:+.4f} "
                            f"proximity={proximity_margin:+.4f} "
                            f"pick={pick_value:+.4f} "
                            f"two_finger={two_finger}",
                            flush=True,
                        )
                    grasp_margins.append(pick_value)
                    current_by_atom[pick_atom] = pick_value
                    step_margins[
                        f"pick({grasp_obj})"
                    ] = pick_value

                runtime = self._agm_stage_runtimes.get(env_id)

                if runtime is None:
                    # Safe fallback to the unmodified senior implementation.
                    rho_t = st.tracker.step(
                        grasp_margins,
                        goal_margins,
                    )
                    shaping_value = 0.0
                else:
                    for atom in runtime.plan.atoms:
                        if atom not in current_by_atom:
                            value = self._raw_atom_margin(ep, atom)
                            current_by_atom[atom] = value
                            step_margins[
                                _stl_atom_name(atom)
                            ] = value

                    self._agm_last_raw_margins[env_id] = dict(
                        step_margins
                    )
                    runtime_state = runtime.update(current_by_atom)
                    normalized_by_atom = (
                        runtime.last_normalized_margins
                    )
                    step_margins = {
                        _stl_atom_name(atom): value
                        for atom, value in normalized_by_atom.items()
                    }

                    rho_t = runtime_state.episode_score
                    shaping_value = runtime_state.shaping_reward

                    visual_runtime = getattr(
                        self, "_visual_agm_runtimes", {}
                    ).get(env_id)
                    if visual_runtime is not None:
                        step = int(
                            np.asarray(self._elapsed_steps)
                            .reshape(-1)[env_id]
                        )
                        visual_samples = self._filter_visual_samples(
                            env_id,
                            visual_samples,
                        )
                        visual_step = visual_runtime.observe(
                            step=step,
                            samples=visual_samples,
                        )
                        getattr(
                            self, "_stl_comparison_frames", {}
                        ).setdefault(env_id, []).append(
                            STLComparisonFrame(
                                step=step,
                                privileged_normalized=dict(
                                    normalized_by_atom
                                ),
                                visual_normalized=dict(
                                    visual_step.normalized_margins
                                ),
                                privileged_score=float(
                                    runtime_state.episode_score
                                ),
                                visual_score=float(
                                    visual_step.episode_score
                                ),
                                visual_valid=bool(visual_step.valid),
                                visual_raw=dict(visual_step.raw_margins),
                                visual_atom_valid=dict(
                                    visual_step.atom_valid
                                ),
                                visual_atom_reasons=dict(
                                    visual_step.atom_reasons
                                ),
                                privileged_stage=str(
                                    runtime_state.active_stage_name
                                ),
                                visual_stage=str(
                                    visual_step.active_stage_name
                                ),
                                visual_reason=str(visual_step.reason),
                            )
                        )

                    if (
                        getattr(self, "_stl_dbg", False)
                        and runtime_state.advanced
                    ):
                        print(
                            "[AGM_STAGE_TRANSITION] "
                            f"env={env_id} "
                            f"task={int(self.task_ids[env_id])} "
                            f"path={runtime_state.active_path_name} "
                            f"next_stage={runtime_state.active_stage_name} "
                            f"completed={runtime_state.completed_stages} "
                            f"all_complete={runtime_state.all_complete} "
                            f"score={runtime_state.episode_score:+.4f} "
                            f"candidates={runtime_state.candidate_scores}",
                            flush=True,
                        )

                rho_out[env_id] = float(rho_t)
                st.prev_rho = float(rho_t)

                if self._stl_plot_dir:
                    st.rho_trace.append(float(rho_t))
                    st.shaping_trace.append(float(shaping_value))
                    st.last_margins = step_margins
                    for name, margin in step_margins.items():
                        st.atom_margins.setdefault(name, []).append(
                            margin
                        )

            except Exception as exc:
                if getattr(self, "_stl_dbg", False):
                    print(
                        f"[AGM_STAGE_DBG] robustness failed for env "
                        f"{env_id}: {exc!r}",
                        flush=True,
                    )

        return rho_out
