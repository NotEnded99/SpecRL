"""Experimental LIBERO environment for normalized AGM-stage rewards.

The original ``rlinf/envs/libero/libero_env.py`` remains unchanged.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import os
from rlinf.envs.libero.agm_stage_runtime import AGMStageRuntime, MinMaxRuntime
from rlinf.envs.libero.libero_env import (
    LiberoEnv as BaseLiberoEnv,
    _build_step_episode,
    _stl_atom_name,
    goal_atom_margin,
    resolve_object_key,
)
from rlinf.envs.libero.stl_stage_plan import (
    Atom,
    StagePlan,
    build_stage_plan,
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

class LiberoEnv(BaseLiberoEnv):
    """Experimental overlay using normalized AGM and temporal gates."""

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

    def _task_description_for_env(self, env_id: int) -> str:
        if (
            hasattr(self, "task_descriptions")
            and env_id < len(self.task_descriptions)
        ):
            return str(self.task_descriptions[env_id])

        task_id = int(self.task_ids[env_id])
        return str(self.task_suite.get_task(task_id).language)

    def _stl_setup(self, ids) -> None:
        """Run base setup, then initialize candidate-path AGM runtimes."""
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


        if ids is None or len(ids) == 0 or goal_atom_margin is None:
            return

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

        confirmation_steps = int(
            self.cfg.get("agm_stage_confirmation_steps", 3)
        )
        normalize_margins = bool(
            self.cfg.get("agm_normalize_margins", True)
        )
        aggregation = str(self.cfg.get("agm_aggregation", "agm")).lower()
        if aggregation not in ("agm", "min_max"):
            raise ValueError(
                f"agm_aggregation must be 'agm' or 'min_max', got: {aggregation}"
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

                if aggregation == "min_max":
                    runtime = MinMaxRuntime(
                        plan=plan,
                        initial_margins=initial_by_atom,
                    )
                else:
                    runtime = AGMStageRuntime(
                        plan=plan,
                        initial_margins=initial_by_atom,
                        confirmation_steps=confirmation_steps,
                        normalize=normalize_margins,
                    )

                self._agm_stage_runtimes[env_id] = runtime
                self._agm_stage_plans[env_id] = plan
                self._agm_initial_atom_margins[env_id] = initial_by_atom
                # ``prev_step_reward`` stores Phi(t) for the discounted
                # progress-shaping formula. Every staged path starts at -1.
                self.prev_step_reward[env_id] = -1.0

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
        self._agm_initial_goal_margins[env_id] = []
        self._agm_initial_grasp_margins[env_id] = []

    # NOTE: underscore escaping for mathtext now lives in the base env's
    # _plot_stl_episode, so no override is needed here.

    def _calc_step_reward(self, terminations):
        """Return discounted specification progress plus a one-shot +1."""
        if not getattr(self, "_stl_enabled", False) or not self.use_rel_reward:
            return super()._calc_step_reward(terminations)

        phi_next = self._stl_robustness()
        if self._stl_clip_lower is not None or self._stl_clip_upper is not None:
            phi_next = np.clip(
                phi_next,
                self._stl_clip_lower,
                self._stl_clip_upper,
            )

        phi_previous = np.asarray(self.prev_step_reward, dtype=np.float64)
        progress_reward = self._stl_reward_scale * discounted_progress_reward(
            phi_next,
            phi_previous,
            gamma=self._stl_gamma,
        )
        self.prev_step_reward = np.asarray(phi_next, dtype=np.float64)

        completed_now = np.asarray(terminations, dtype=bool) & ~np.asarray(
            self.success_once,
            dtype=bool,
        )
        completion_reward = (
            float(self.cfg.reward_coef) * completed_now.astype(np.float64)
        )
        step_penalty = -1.0 if self.use_step_penalty else 0.0
        reward = progress_reward + completion_reward + step_penalty

        if self._stl_plot_dir:
            for env_id, value in enumerate(reward):
                st = self._stl_states[env_id]
                if st is not None and st.valid:
                    st.reward_trace.append(float(value))
        return reward

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
                self._plot_stl_episode(
                    out_path,
                    st,
                    task_desc,
                    success,
                )

                # Save raw data as NPZ (same format as the base env, consumed
                # by plot_stl_from_npz.py). Atom margins remain normalized to
                # [-1, 0.1], while the staged task potential lives in [-1, 0].
                npz_path = os.path.join(
                    self._stl_plot_dir,
                    (
                        f"stl_data_{ep_str}_env{env_id:02d}_"
                        f"task{task_id:02d}_"
                        f"{'SUCCESS' if success else 'fail'}.npz"
                    ),
                )
                os.makedirs(os.path.dirname(npz_path), exist_ok=True)
                np.savez(
                    npz_path,
                    rho=np.asarray(st.rho_trace, dtype=np.float64),
                    shaping=np.asarray(st.shaping_trace, dtype=np.float64),
                    reward=np.asarray(st.reward_trace, dtype=np.float64),
                    atom_names=list(st.atom_margins.keys()),
                    **{
                        f"atom_{nm}": np.asarray(
                            st.atom_margins[nm], dtype=np.float64
                        )
                        for nm in st.atom_margins
                    },
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

    def _stl_robustness(self) -> np.ndarray:
        """Return the selected candidate-path potential in [-1, 0]."""
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

                # Retain the base environment's atom records and fallback data.
                for atom, grasp_obj in zip(
                    st.goal_atoms,
                    st.grasp_objs,
                ):
                    goal_value = self._raw_atom_margin(ep, atom)
                    goal_margins.append(goal_value)
                    current_by_atom[atom] = goal_value
                    step_margins[_stl_atom_name(atom)] = goal_value

                    if grasp_obj is None:
                        grasp_margins.append(None)
                        continue

                    pick_atom: Atom = ("pick", grasp_obj, None)
                    pick_value = self._raw_atom_margin(ep, pick_atom)
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
