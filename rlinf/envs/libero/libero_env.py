# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import glob
import importlib
import os
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import torch
from omegaconf.omegaconf import OmegaConf

from rlinf.envs.libero.utils import (
    build_interleaved_eval_reset_state_ids,
    distribute_reset_state_ids_round_robin,
    get_benchmark_overridden,
    get_libero_image,
    get_libero_type,
    get_libero_wrist_image,
    quat2axisangle,
    record_completed_episode_task_stats,
)
from rlinf.envs.libero.venv import ReconfigureSubprocEnv
from rlinf.envs.utils import list_of_dict_to_dict_of_list, to_tensor
from rlinf.utils.logging import get_logger

logger = get_logger()

# --- STL engine (optional, embedded in LiberoEnv) ---
try:
    from rlinf.algorithms.rewards.stl.stlcg_task_monitor.predicates import (  # noqa: WPS433
        PredConfig,
        goal_atom_margin,
        pred_pick,
    )
    from rlinf.algorithms.rewards.stl.stlcg_task_monitor.loader import (  # noqa: WPS433
        Episode,
        JointInfo,
        resolve_object_key,
    )
    from rlinf.algorithms.rewards.stl.stlcg_task_monitor.nl_grounder import (  # noqa: WPS433
        ground_task_state,
    )
    from rlinf.algorithms.rewards.stl.stlcg_task_monitor.bddl_goals import (  # noqa: WPS433
        goal_atoms as get_bddl_goal_atoms,
    )

    _STL_OK = True
    _STL_ERR: Optional[str] = None
except Exception as _e:  # pragma: no cover - import guard
    _STL_OK = False
    _STL_ERR = repr(_e)
    PredConfig = None  # type: ignore[assignment]
    Episode = None  # type: ignore[assignment]
    JointInfo = None  # type: ignore[assignment]
    goal_atom_margin = None  # type: ignore[assignment]
    pred_pick = None  # type: ignore[assignment]
    resolve_object_key = None  # type: ignore[assignment]
    ground_task_state = None  # type: ignore[assignment]
    get_bddl_goal_atoms = None  # type: ignore[assignment]

# --- STL helpers (copied from libero_env_stl_reward.py) ---

_GEOMETRIC_PREDS = {"on", "in", "stack"}
_ARTICULATED_PREDS = {"open", "close", "turnon", "turnoff"}


class OnlineRobustnessTracker:
    """Incremental ``rho(phi, s[0:t+1])`` for ``phi = AND_k EV(grasp_k AND EV_[0,tau] goal_k)``."""

    def __init__(self, tau: int, num_sub: int):
        self.tau = max(0, int(tau))
        self.num_sub = int(num_sub)
        self.grasp_buf: List[List[Optional[float]]] = [[] for _ in range(self.num_sub)]
        self.goal_buf: List[List[float]] = [[] for _ in range(self.num_sub)]
        self.inner_max: List[deque] = [deque() for _ in range(self.num_sub)]
        self.frozen_max: List[float] = [-1e18] * self.num_sub
        self.t = -1

    def step(self, grasp_margins: List[Optional[float]], goal_margins: List[float]) -> float:
        self.t += 1
        t = self.t
        tau = self.tau
        best_overall = np.inf
        for k in range(self.num_sub):
            gr = grasp_margins[k]
            g = float(goal_margins[k])
            gbuf = self.grasp_buf[k]
            gbuf.append(gr)
            self.goal_buf[k].append(g)
            im_deq = self.inner_max[k]
            cutoff = t - tau
            while im_deq and im_deq[0][0] < cutoff:
                s, im = im_deq.popleft()
                psi = min(gbuf[s], im) if gbuf[s] is not None else im
                if psi > self.frozen_max[k]:
                    self.frozen_max[k] = psi
            for idx in range(len(im_deq)):
                s, im = im_deq[idx]
                if g > im:
                    im_deq[idx] = (s, g)
            im_deq.append((t, g))
            best = self.frozen_max[k]
            for s, im in im_deq:
                gs = gbuf[s]
                psi = min(gs, im) if gs is not None else im
                if psi > best:
                    best = psi
            if best < best_overall:
                best_overall = best
        if self.num_sub == 0 or not np.isfinite(best_overall):
            return 0.0
        return float(best_overall)

    def reset(self) -> None:
        self.grasp_buf = [[] for _ in range(self.num_sub)]
        self.goal_buf = [[] for _ in range(self.num_sub)]
        self.inner_max = [deque() for _ in range(self.num_sub)]
        self.frozen_max = [-1e18] * self.num_sub
        self.t = -1


def _normalize_goal_atoms(raw_atoms: list) -> List[Tuple[str, Optional[str], Optional[str]]]:
    out: List[Tuple[str, Optional[str], Optional[str]]] = []
    for atom in raw_atoms:
        if not atom:
            continue
        pred = str(atom[0]).strip().lower()
        args = [str(x) for x in atom[1:] if x not in (None, "")]
        pred = {"ontop": "on", "inside": "in", "turn_on": "turnon", "turn_off": "turnoff"}.get(pred, pred)
        if pred in _GEOMETRIC_PREDS:
            obj = args[0] if len(args) >= 1 else None
            target = args[1] if len(args) >= 2 else None
            out.append((pred, obj, target))
        elif pred in _ARTICULATED_PREDS:
            out.append((pred, None, args[0] if len(args) >= 1 else None))
    return out


def _stl_atom_name(atom: Tuple[str, Optional[str], Optional[str]]) -> str:
    pred, obj, target = atom
    if obj is None:
        return f"{pred}({target})"
    return f"{pred}({obj},{target})"


@dataclass
class _STLEnvState:
    valid: bool = False
    goal_atoms: List[Tuple[str, Optional[str], Optional[str]]] = field(default_factory=list)
    grasp_objs: List[Optional[str]] = field(default_factory=list)
    tracker: Optional[OnlineRobustnessTracker] = None
    prev_rho: float = 0.0
    atom_margins: Dict[str, List[float]] = field(default_factory=dict)
    last_margins: Dict[str, float] = field(default_factory=dict)
    rho_trace: List[float] = field(default_factory=list)
    shaping_trace: List[float] = field(default_factory=list)
    reward_trace: List[float] = field(default_factory=list)


def _build_step_episode(priv: dict):
    body_xpos = np.asarray(priv["body_xpos"], dtype=np.float64).reshape(-1, 3)
    nbody = body_xpos.shape[0]
    object_pos = {k: np.asarray(v, dtype=np.float64).reshape(1, 3) for k, v in priv["object_pos"].items()}
    site_xpos = np.asarray(priv["site_xpos"], dtype=np.float64).reshape(-1, 3)
    site_xmat = np.asarray(priv["site_xmat"], dtype=np.float64).reshape(-1, 3, 3)
    site_sizes = np.asarray(priv["site_sizes"], dtype=np.float64).reshape(-1, 3)
    nsite = site_xpos.shape[0]
    joint_qpos = np.asarray(priv["joint_qpos"], dtype=np.float64).reshape(-1)
    nj = joint_qpos.shape[0]
    gripper = np.asarray(priv["gripper_qpos"], dtype=np.float64).reshape(2)
    joints = [
        JointInfo(name=j["name"], obj=j["obj"], col=int(j["col"]), addr=int(j["addr"]))
        for j in priv["joints"]
    ]
    # eef_pos from privileged state (for pred_pick proximity term)
    eef_pos = np.asarray(priv.get("eef_pos", np.zeros(3, dtype=np.float64)), dtype=np.float64).reshape(1, 3)
    return Episode(
        path="", T=1,
        object_pos=object_pos, eef_pos=eef_pos,
        gripper_qpos=gripper.reshape(1, 2),
        body_xpos=body_xpos.reshape(1, nbody, 3),
        site_xpos=site_xpos.reshape(1, nsite, 3) if nsite else np.zeros((1, 0, 3)),
        site_xmat=site_xmat.reshape(1, nsite, 3, 3) if nsite else np.zeros((1, 0, 3, 3)),
        joint_qpos=joint_qpos.reshape(1, nj) if nj else np.zeros((1, 0)),
        priv_gripper_qpos=gripper.reshape(1, 2),
        contact_body1=np.asarray(priv["contact_body1"], dtype=np.int32).reshape(1, -1),
        contact_body2=np.asarray(priv["contact_body2"], dtype=np.int32).reshape(1, -1),
        contact_dist=np.asarray(priv["contact_dist"], dtype=np.float32).reshape(1, -1),
        contact_force=np.asarray(priv["contact_force"], dtype=np.float32).reshape(1, -1),
        ncon=np.array([int(priv["ncon"])], dtype=np.int32),
        body_names=list(priv["body_names"]), body_ids=np.arange(nbody, dtype=np.int64),
        obj_body_tree=dict(priv["obj_body_tree"]),
        site_names=list(priv["site_names"]), site_sizes=site_sizes,
        joints=joints, articulated=list(priv["articulated"]),
    )

libero_type = get_libero_type()

if libero_type in ["pro", "plus"]:
    sys.path[:] = [p for p in sys.path if "opt/libero" not in p]
    LIBERO_PKG_NAME = f"libero{libero_type}"
    LIBERO_MAIN_MODULE_PATH = f"{LIBERO_PKG_NAME}.{LIBERO_PKG_NAME}"
    try:
        real_libero_pkg = importlib.import_module(LIBERO_PKG_NAME)
        real_libero_core = importlib.import_module(LIBERO_MAIN_MODULE_PATH)

        try:
            real_libero_benchmark = importlib.import_module(
                f"{LIBERO_MAIN_MODULE_PATH}.benchmark"
            )
        except ImportError:
            real_libero_benchmark = importlib.import_module(
                f"{LIBERO_PKG_NAME}.benchmark"
            )

        try:
            real_libero_envs = importlib.import_module(
                f"{LIBERO_MAIN_MODULE_PATH}.envs"
            )
        except ImportError:
            real_libero_envs = importlib.import_module(f"{LIBERO_PKG_NAME}.envs")

        sys.modules["libero"] = real_libero_pkg
        sys.modules["libero.libero"] = real_libero_core
        sys.modules["libero.libero.benchmark"] = real_libero_benchmark
        sys.modules["libero.libero.envs"] = real_libero_envs
    except ImportError as e:
        print(
            f"[Main Process Routing Error] Failed to import '{LIBERO_MAIN_MODULE_PATH}'. Error: {e}"
        )

if libero_type == "pro":
    from liberopro.liberopro.benchmark import Benchmark
elif libero_type == "plus":
    from liberoplus.liberoplus.benchmark import Benchmark
else:
    from libero.libero.benchmark import Benchmark


class LiberoEnv(gym.Env):
    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        self.seed_offset = seed_offset
        self.cfg = cfg
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info

        if seed_offset == 0:
            self._log_evaluation_mode()
        self.seed = self.cfg.seed + seed_offset
        self._is_start = True
        self.num_envs = num_envs
        self.group_size = self.cfg.group_size
        self.num_group = self.num_envs // self.group_size
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.specific_reset_id = cfg.get("specific_reset_id", None)
        self.task_id_filter = cfg.get("task_id_filter", None)
        if self.task_id_filter is not None:
            self.task_id_filter = list(self.task_id_filter)

        self.ignore_terminations = cfg.ignore_terminations
        self.auto_reset = cfg.auto_reset
        self.is_eval = cfg.get("is_eval", False)

        self._generator = np.random.default_rng(seed=self.seed)
        self._generator_ordered = np.random.default_rng(seed=0)
        self.start_idx = 0

        self.task_suite: Benchmark = get_benchmark_overridden(cfg.task_suite_name)()

        self._compute_total_num_group_envs()
        self.reset_state_ids_all = self.get_reset_state_ids_all()
        if self.is_eval:
            pool = self.reset_state_ids_all[self.seed_offset]
            self._eval_reset_pool = pool[pool >= 0].copy()
        else:
            self._eval_reset_pool = np.array([], dtype=np.int64)
        self.update_reset_state_ids()
        self._init_task_and_trial_ids()
        self._init_env()

        self.prev_step_reward = np.zeros(self.num_envs)
        self.use_rel_reward = cfg.use_rel_reward
        self.use_step_penalty = getattr(cfg, "use_step_penalty", False)

        self._init_metrics()
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)

        self.video_cfg = cfg.video_cfg
        self.current_raw_obs = None

        # --- STL reward (embedded in LiberoEnv) ---
        self._stl_enabled = bool(cfg.get("stl_reward", False)) and _STL_OK
        if self._stl_enabled:
            self._stl_tau = int(cfg.get("stl_tau", 60))
            self._stl_reward_scale = float(cfg.get("stl_reward_scale", 1.0))
            self._stl_gamma = float(cfg.get("stl_gamma", 0.99))
            # Clip on the STL rho term ONLY (the sparse task reward is never clipped).
            # Supports asymmetric bounds via stl_clip_min / stl_clip_max; falls back to
            # the symmetric stl_clip for backward compatibility. None = no bound on that side.
            clip = cfg.get("stl_clip", 10.0)
            sym_clip = float(clip) if clip is not None else None
            cmin = cfg.get("stl_clip_min", None)
            cmax = cfg.get("stl_clip_max", None)
            self._stl_clip_lower = (
                float(cmin) if cmin is not None
                else (-sym_clip if sym_clip is not None else None)
            )
            self._stl_clip_upper = (
                float(cmax) if cmax is not None
                else (sym_clip if sym_clip is not None else None)
            )
            pred_overrides = cfg.get("stl_pred_config", {}) or {}
            self._pred_cfg = PredConfig(**pred_overrides) if _STL_OK else None
            self._stl_warned = False
            self._stl_plot_dir = cfg.get("stl_plot_dir", None) or None
            self._stl_save_plots = bool(cfg.get("stl_save_plots", False))  # flag to control saving
            self._stl_plot_counter = 0
            self._stl_states: List[Optional[_STLEnvState]] = [None] * self.num_envs
            # Safety net: when the NL parse disagrees with the env's BDDL ground
            # truth, use BDDL so a misparse can never corrupt the STL reward.
            self._stl_fallback_bddl = bool(cfg.get("stl_fallback_bddl", True))
            self._stl_dbg = int(os.environ.get("STL_DBG", "0"))
            if self._stl_dbg:
                print(f"[STL_DBG] LiberoEnv.__init__ STL enabled, tau={self._stl_tau} "
                      f"scale={self._stl_reward_scale} num_envs={self.num_envs}", flush=True)
        else:
            self._stl_states = [None] * self.num_envs

    def _log_evaluation_mode(self):
        """Log the LIBERO evaluation mode banner (rank 0 env worker only)."""
        libero_type = get_libero_type()
        if libero_type == "pro":
            perturbation = os.environ.get("LIBERO_PERTURBATION", "all")
            logger.info(f"Evaluation Mode: LIBERO-PRO | Perturbation: {perturbation}")
        elif libero_type == "plus":
            suffix = os.environ.get("LIBERO_SUFFIX", "all")
            logger.info(f"Evaluation Mode: LIBERO-PLUS | Suffix: {suffix}")
        else:
            logger.info("Evaluation Mode: Standard LIBERO")

    def _init_env(self):
        env_fns = self.get_env_fns()
        self.env = ReconfigureSubprocEnv(env_fns)

    def get_env_fns(self):
        env_fn_params = self.get_env_fn_params()
        env_fns = []

        current_type_val = get_libero_type()

        for env_fn_param in env_fn_params:

            def env_fn(param=env_fn_param, _type_val=current_type_val):
                os.environ["LIBERO_TYPE"] = _type_val
                seed = param.pop("seed")

                if _type_val in ["pro", "plus"]:
                    sys.path[:] = [p for p in sys.path if "opt/libero" not in p]

                    pkg_name = f"libero{_type_val}"
                    core_name = f"{pkg_name}.{pkg_name}"

                    try:
                        real_pkg = importlib.import_module(pkg_name)
                        real_core = importlib.import_module(core_name)
                        real_bench = importlib.import_module(f"{core_name}.benchmark")
                        real_envs = importlib.import_module(f"{core_name}.envs")

                        sys.modules["libero"] = real_pkg
                        sys.modules["libero.libero"] = real_core
                        sys.modules["libero.libero.benchmark"] = real_bench
                        sys.modules["libero.libero.envs"] = real_envs

                        loaded_path = os.path.dirname(real_core.__file__)
                        os.environ["LIBERO_ASSET_ROOT"] = os.path.join(
                            loaded_path, "assets"
                        )
                        os.environ["LIBERO_BDDL_PATH"] = os.path.join(
                            loaded_path, "bddl_files"
                        )
                        os.environ["LIBERO_INIT_STATES_PATH"] = os.path.join(
                            loaded_path, "init_files"
                        )

                        WorkerEnv = real_envs.OffScreenRenderEnv

                    except ImportError as e:
                        print(f"[Worker Env Error] {e}")
                        raise e
                else:
                    from libero.libero.envs import OffScreenRenderEnv as WorkerEnv

                env = WorkerEnv(**param)
                env.seed(seed)
                return env

            env_fns.append(env_fn)
        return env_fns

    def get_env_fn_params(self, env_idx=None):
        env_fn_params = []
        base_env_args = OmegaConf.to_container(self.cfg.init_params, resolve=True)

        variant = os.environ.get(
            "LIBERO_TYPE",
            self.cfg.get("libero_variant", "standard")
            if hasattr(self.cfg, "get")
            else "standard",
        )
        raw_suffix = os.environ.get(
            "LIBERO_SUFFIX",
            os.environ.get(
                "LIBERO_PERTURBATION",
                self.cfg.get("perturbation_suffix", None)
                if hasattr(self.cfg, "get")
                else None,
            ),
        )
        if variant == "pro":
            import liberopro.liberopro as l_pro

            bddl_root = l_pro.get_libero_path("bddl_files")
        elif variant == "plus":
            import liberoplus.liberoplus as l_plus

            bddl_root = l_plus.get_libero_path("bddl_files")
        else:
            from libero.libero import get_libero_path

            bddl_root = get_libero_path("bddl_files")

        suite_name = self.cfg.task_suite_name.lower()
        suite_keyword = suite_name.replace("libero_", "").strip()

        task_descriptions = []
        if env_idx is None:
            env_idx = np.arange(self.num_envs)

        for env_id in range(self.num_envs):
            if env_id not in env_idx:
                task_descriptions.append(
                    self.task_descriptions[env_id]
                    if hasattr(self, "task_descriptions")
                    else ""
                )
                continue

            task = self.task_suite.get_task(self.task_ids[env_id])
            folder_name = task.problem_folder
            file_name = task.bddl_file
            original_path = os.path.join(bddl_root, folder_name, file_name)

            final_path = original_path

            if variant == "pro":
                pro_suffix = raw_suffix.replace(".bddl", "") if raw_suffix else None

                valid_perts = ["_lan", "_object", "_swap", "_task"]
                if pro_suffix == "all":
                    filter_perts = valid_perts
                elif pro_suffix is not None:
                    # Map bare name (e.g. "task") to directory suffix (e.g. "_task")
                    normalized = (
                        f"_{pro_suffix}"
                        if not pro_suffix.startswith("_")
                        else pro_suffix
                    )
                    filter_perts = [normalized] if normalized in valid_perts else []
                else:
                    filter_perts = []

                if filter_perts:
                    all_sub_dirs = [
                        d
                        for d in os.listdir(bddl_root)
                        if os.path.isdir(os.path.join(bddl_root, d))
                        and suite_keyword in d
                        and any(d.endswith(pert) for pert in filter_perts)
                    ]

                    core_task_name = file_name.replace(".bddl", "")
                    all_candidates = []

                    for sub_dir in all_sub_dirs:
                        target_dir_path = os.path.join(bddl_root, sub_dir)
                        matches = [
                            os.path.join(target_dir_path, f)
                            for f in os.listdir(target_dir_path)
                            if core_task_name in f and f.endswith(".bddl")
                        ]
                        all_candidates.extend(matches)

                    if all_candidates:
                        all_candidates.sort()
                        if self.is_eval:
                            idx_offset = (
                                list(env_idx).index(env_id) if env_id in env_idx else 0
                            )
                            final_path = all_candidates[
                                (self.seed + idx_offset) % len(all_candidates)
                            ]
                        else:
                            final_path = self._generator.choice(all_candidates)

            elif variant == "plus":
                plus_suffix = raw_suffix.replace(".bddl", "") if raw_suffix else None

                valid_perts = [
                    "_light",
                    "_language",
                    "_table",
                    "_add",
                    "_tb",
                    "_sample",
                    "_level",
                ]
                if plus_suffix == "all":
                    filter_perts = valid_perts
                elif plus_suffix is not None:
                    normalized = (
                        f"_{plus_suffix}"
                        if not plus_suffix.startswith("_")
                        else plus_suffix
                    )
                    filter_perts = [normalized] if normalized in valid_perts else []
                else:
                    filter_perts = []

                if filter_perts:
                    clean_name = file_name.replace(".bddl", "")
                    for marker in valid_perts:
                        if marker in clean_name:
                            clean_name = clean_name.split(marker)[0]
                            break

                    suite_pattern = folder_name.replace("_", "").lower()
                    all_dirs = [
                        d
                        for d in os.listdir(bddl_root)
                        if os.path.isdir(os.path.join(bddl_root, d))
                    ]
                    search_dirs = [
                        os.path.join(bddl_root, d)
                        for d in all_dirs
                        if suite_pattern in d.lower().replace("_", "")
                    ]

                    if not search_dirs:
                        search_dirs = [os.path.join(bddl_root, folder_name)]

                    all_candidates = []
                    for target_dir in search_dirs:
                        matches = [
                            f
                            for f in glob.glob(os.path.join(target_dir, "*.bddl"))
                            if clean_name in os.path.basename(f)
                            and any(
                                pert in os.path.basename(f) for pert in filter_perts
                            )
                        ]
                        all_candidates.extend(matches)

                    if all_candidates:
                        all_candidates.sort()
                        if self.is_eval:
                            idx_offset = (
                                list(env_idx).index(env_id) if env_id in env_idx else 0
                            )
                            final_path = all_candidates[
                                (self.seed + idx_offset) % len(all_candidates)
                            ]
                        else:
                            final_path = self._generator.choice(all_candidates)

            env_fn_params.append(
                {
                    **base_env_args,
                    "bddl_file_name": final_path,
                    "seed": self.seed,
                }
            )
            task_descriptions.append(task.language)

        self.task_descriptions = task_descriptions
        return env_fn_params

    def _compute_total_num_group_envs(self):
        self.total_num_group_envs = 0
        self.trial_id_bins = []
        for task_id in range(self.task_suite.get_num_tasks()):
            task_num_trials = len(self.task_suite.get_task_init_states(task_id))
            self.trial_id_bins.append(task_num_trials)
            self.total_num_group_envs += task_num_trials
        self.cumsum_trial_id_bins = np.cumsum(self.trial_id_bins)

        if self.task_id_filter is not None:
            num_tasks = len(self.trial_id_bins)
            validated_tids = []
            for tid in self.task_id_filter:
                if not isinstance(tid, (int, np.integer)):
                    raise ValueError(
                        f"task_id_filter must contain ints, got "
                        f"{type(tid).__name__}: {tid}"
                    )
                tid_int = int(tid)
                if tid_int < 0 or tid_int >= num_tasks:
                    raise ValueError(
                        f"task_id {tid_int} in task_id_filter is out of range "
                        f"[0, {num_tasks - 1}]"
                    )
                validated_tids.append(tid_int)
            validated_tids = sorted(set(validated_tids))

            self._valid_reset_state_ids = []
            for tid in validated_tids:
                start = self.cumsum_trial_id_bins[tid - 1] if tid > 0 else 0
                end = self.cumsum_trial_id_bins[tid]
                self._valid_reset_state_ids.extend(range(start, end))
            self._valid_reset_state_ids = np.array(self._valid_reset_state_ids)
        else:
            self._valid_reset_state_ids = None

    def update_reset_state_ids(self):
        if self.is_eval or self.cfg.use_ordered_reset_state_ids:
            reset_state_ids = self._get_ordered_reset_state_ids(self.num_group)
        else:
            reset_state_ids = self._get_random_reset_state_ids(self.num_group)
        self.reset_state_ids = reset_state_ids.repeat(self.group_size)

    def _init_task_and_trial_ids(self):
        self.task_ids, self.trial_ids = (
            self._get_task_and_trial_ids_from_reset_state_ids(self.reset_state_ids)
        )

    def _get_random_reset_state_ids(self, num_reset_states):
        if self.specific_reset_id is not None:
            reset_state_ids = self.specific_reset_id * np.ones(
                (num_reset_states,), dtype=int
            )
        elif self._valid_reset_state_ids is not None:
            indices = self._generator.integers(
                low=0, high=len(self._valid_reset_state_ids), size=(num_reset_states,)
            )
            reset_state_ids = self._valid_reset_state_ids[indices]
        else:
            reset_state_ids = self._generator.integers(
                low=0, high=self.total_num_group_envs, size=(num_reset_states,)
            )
        return reset_state_ids

    def get_reset_state_ids_all(self):
        if self.is_eval:
            if self._valid_reset_state_ids is not None:
                reset_state_ids = self._valid_reset_state_ids.copy()
            else:
                reset_state_ids = build_interleaved_eval_reset_state_ids(
                    self.trial_id_bins, self.cumsum_trial_id_bins
                )
            return distribute_reset_state_ids_round_robin(
                reset_state_ids, self.total_num_processes
            )

        if self._valid_reset_state_ids is not None:
            reset_state_ids = self._valid_reset_state_ids.copy()
        else:
            reset_state_ids = np.arange(self.total_num_group_envs)

        self._generator_ordered.shuffle(reset_state_ids)

        # Ensure we have enough IDs for all processes by tiling if needed
        if len(reset_state_ids) < self.total_num_processes:
            repeats = (self.total_num_processes // len(reset_state_ids)) + 1
            reset_state_ids = np.tile(reset_state_ids, repeats)

        valid_size = len(reset_state_ids) - (
            len(reset_state_ids) % self.total_num_processes
        )
        reset_state_ids = reset_state_ids[:valid_size]
        reset_state_ids = reset_state_ids.reshape(self.total_num_processes, -1)
        return reset_state_ids

    def _get_ordered_reset_state_ids(self, num_reset_states):
        if self.specific_reset_id is not None:
            return self.specific_reset_id * np.ones((num_reset_states,), dtype=int)

        if self.is_eval:
            pool = self._eval_reset_pool
            if self.start_idx >= len(pool):
                return np.full((num_reset_states,), -1, dtype=np.int64)
            end = min(self.start_idx + num_reset_states, len(pool))
            n_valid = end - self.start_idx
            result = np.full((num_reset_states,), -1, dtype=np.int64)
            if n_valid > 0:
                result[:n_valid] = pool[self.start_idx : end]
            self.start_idx = end
            return result

        if self.start_idx + num_reset_states > len(self.reset_state_ids_all[0]):
            self.reset_state_ids_all = self.get_reset_state_ids_all()
            self.start_idx = 0
        reset_state_ids = self.reset_state_ids_all[self.seed_offset][
            self.start_idx : self.start_idx + num_reset_states
        ]
        self.start_idx = self.start_idx + num_reset_states
        return reset_state_ids

    def _get_task_and_trial_ids_from_reset_state_ids(self, reset_state_ids):
        task_ids = []
        trial_ids = []
        # get task id and trial id from reset state ids
        for reset_state_id in reset_state_ids:
            start_pivot = 0
            for task_id, end_pivot in enumerate(self.cumsum_trial_id_bins):
                if reset_state_id < end_pivot and reset_state_id >= start_pivot:
                    task_ids.append(task_id)
                    trial_ids.append(reset_state_id - start_pivot)
                    break
                start_pivot = end_pivot

        return np.array(task_ids), np.array(trial_ids)

    def _get_reset_states(self, env_idx):
        if env_idx is None:
            env_idx = np.arange(self.num_envs)
        init_state = [
            self.task_suite.get_task_init_states(self.task_ids[env_id])[
                self.trial_ids[env_id]
            ]
            for env_id in env_idx
        ]
        return init_state

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    @property
    def info_logging_keys(self):
        return []

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    def _init_metrics(self):
        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.fail_once = np.zeros(self.num_envs, dtype=bool)
        self.returns = np.zeros(self.num_envs)
        self.success_episode_len = np.zeros(self.num_envs, dtype=np.int32)
        self._task_success_stats: dict[int, dict[str, int]] = {}
        self._eval_seen_trials: set[tuple[int, int]] = set()

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = np.zeros(self.num_envs, dtype=bool)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self.success_once[mask] = False
            self.fail_once[mask] = False
            self.returns[mask] = 0
            self.success_episode_len[mask] = 0
            self._elapsed_steps[env_idx] = 0
        else:
            self.prev_step_reward[:] = 0
            self.success_once[:] = False
            self.fail_once[:] = False
            self.returns[:] = 0.0
            self.success_episode_len[:] = 0
            self._elapsed_steps[:] = 0

    def _record_metrics(self, step_reward, terminations, infos):
        episode_info = {}
        # Only accumulate returns while not yet succeeded
        self.returns += step_reward * (~self.success_once)
        # Record episode_len at first success
        new_success_mask = terminations & ~self.success_once
        if new_success_mask.any():
            self.success_episode_len[new_success_mask] = self.elapsed_steps[
                new_success_mask
            ]

        self.success_once = self.success_once | terminations

        # DEBUG: print success_once at each step
        # if os.environ.get("DEBUG_TERM", "0") == "1":
        #     print(f"[DEBUG_METRICS] step={self._elapsed_steps[0]:.0f} "
        #           f"new_success={new_success_mask[:3].tolist()} "
        #           f"success_once={self.success_once[:3].tolist()} "
        #           f"terminations={terminations[:3].tolist()}",
        #           flush=True)

        episode_info["success_once"] = self.success_once.copy()
        episode_info["return"] = self.returns.copy()
        episode_info["episode_len"] = self.elapsed_steps.copy()

        # Use success episode_len for reward if already succeeded, else current elapsed
        episode_len_for_reward = np.where(
            self.success_once, self.success_episode_len, self.elapsed_steps
        )
        episode_info["reward"] = episode_info["return"] / np.maximum(
            episode_len_for_reward, 1
        )
        infos["episode"] = to_tensor(episode_info)
        return infos

    def _extract_image_and_state(self, obs):
        return {
            "full_image": get_libero_image(obs),
            "wrist_image": get_libero_wrist_image(obs),
            "state": np.concatenate(
                [
                    obs["robot0_eef_pos"],
                    quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                ]
            ),
        }

    def _wrap_obs(self, obs_list):
        images_and_states_list = []
        for obs in obs_list:
            images_and_states = self._extract_image_and_state(obs)
            images_and_states_list.append(images_and_states)

        images_and_states = to_tensor(
            list_of_dict_to_dict_of_list(images_and_states_list)
        )

        full_image_tensor = torch.stack(
            [value.clone() for value in images_and_states["full_image"]]
        )
        wrist_image_tensor = torch.stack(
            [value.clone() for value in images_and_states["wrist_image"]]
        )

        states = images_and_states["state"]

        obs = {
            "main_images": full_image_tensor,
            "wrist_images": wrist_image_tensor,
            "states": states,
            "task_descriptions": self.task_descriptions,
        }
        return obs

    def _reconfigure(self, reset_state_ids, env_idx):
        reconfig_env_idx = []
        task_ids, trial_ids = self._get_task_and_trial_ids_from_reset_state_ids(
            reset_state_ids
        )
        for j, env_id in enumerate(env_idx):
            task_changed = self.task_ids[env_id] != task_ids[j]
            self.task_ids[env_id] = task_ids[j]
            self.trial_ids[env_id] = trial_ids[j]
            if task_changed or not self.is_eval:
                reconfig_env_idx.append(env_id)
        if reconfig_env_idx:
            env_fn_params = self.get_env_fn_params(reconfig_env_idx)
            self.env.reconfigure_env_fns(env_fn_params, reconfig_env_idx)
        self.env.seed(self.seed * len(env_idx))
        self.env.reset(id=env_idx)
        variant = os.environ.get(
            "LIBERO_TYPE",
            self.cfg.get("libero_variant", "standard")
            if hasattr(self.cfg, "get")
            else "standard",
        )
        if variant != "plus":
            init_state = self._get_reset_states(env_idx=env_idx)
            self.env.set_init_state(init_state=init_state, id=env_idx)

    def reset(
        self,
        env_idx: Optional[Union[int, list[int], np.ndarray]] = None,
        reset_state_ids=None,
    ):
        if env_idx is None:
            env_idx = np.arange(self.num_envs)

        if self.is_start:
            if self.is_eval:
                self._task_success_stats = {}
                self._eval_seen_trials = set()
                self.start_idx = 0
                pool = self.reset_state_ids_all[self.seed_offset]
                self._eval_reset_pool = pool[pool >= 0].copy()
                self.update_reset_state_ids()
            reset_state_ids = (
                self.reset_state_ids if self.use_fixed_reset_state_ids else None
            )
            self._is_start = False

        if reset_state_ids is None:
            num_reset_states = len(env_idx)
            reset_state_ids = self._get_random_reset_state_ids(num_reset_states)

        self._reconfigure(reset_state_ids, env_idx)
        for _ in range(15):
            zero_actions = np.zeros((len(env_idx), 7))
            if self.cfg.reset_gripper_open:
                zero_actions[:, -1] = -1
            raw_obs, _reward, terminations, info_lists = self.env.step(
                zero_actions, env_idx
            )
        if self.current_raw_obs is None:
            self.current_raw_obs = [None] * self.num_envs
        for i, idx in enumerate(env_idx):
            self.current_raw_obs[idx] = raw_obs[i]

        obs = self._wrap_obs(self.current_raw_obs)
        self._reset_metrics(env_idx)
        infos = {}

        # STL: setup after reset (fetch goal atoms via pipe)
        if self._stl_enabled:
            try:
                ids = list(env_idx) if env_idx is not None else list(range(self.num_envs))
                self._stl_setup(ids)
            except Exception:
                pass

        return obs, infos

    def step(self, actions=None, auto_reset=True):
        """Step the environment with the given actions."""
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        self._elapsed_steps += 1
        raw_obs, _reward, terminations, info_lists = self.env.step(actions)
        self.current_raw_obs = raw_obs
        infos = list_of_dict_to_dict_of_list(info_lists)
        truncations = self.elapsed_steps >= self.cfg.max_episode_steps
        obs = self._wrap_obs(raw_obs)

        step_reward = self._calc_step_reward(terminations)

        # DEBUG: print terminations at each step (first 3 envs)
        # if os.environ.get("DEBUG_TERM", "0") == "1":
        #     print(f"[DEBUG_STEP] step={self._elapsed_steps[0]:.0f} terminations={terminations[:3].tolist()} "
        #           f"truncations={truncations[:3].tolist()} reward={np.asarray(step_reward).reshape(-1)[:3].tolist()}",
        #           flush=True)

        infos = self._record_metrics(step_reward, terminations, infos)
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = to_tensor(terminations)
            terminations[:] = False

        dones = terminations | truncations
        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            obs, infos, _ = self._handle_auto_reset(dones, obs, infos)

        # STL: dump plots at episode end when episode completes
        if getattr(self, "_stl_enabled", False) and getattr(self, "_stl_plot_dir", None) and getattr(self, "_stl_save_plots", False):
            if dones.any():
                try:
                    env_idx_done = np.arange(0, self.num_envs)[dones]
                    # dump BEFORE any auto_reset clears the state
                    self._dump_episode_plots(env_idx_done)
                except Exception:
                    pass
        return (
            obs,
            to_tensor(step_reward),
            to_tensor(terminations),
            to_tensor(truncations),
            infos,
        )

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_step, action_dim]
        chunk_size = chunk_actions.shape[1]
        obs_list = []
        infos_list = []

        chunk_rewards = []

        raw_chunk_terminations = []
        raw_chunk_truncations = []
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                actions, auto_reset=False
            )
            obs_list.append(extracted_obs)
            infos_list.append(infos)

            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)  # [num_envs, chunk_steps]
        raw_chunk_terminations = torch.stack(
            raw_chunk_terminations, dim=1
        )  # [num_envs, chunk_steps]
        raw_chunk_truncations = torch.stack(
            raw_chunk_truncations, dim=1
        )  # [num_envs, chunk_steps]

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        # eval_count_mask: per-env bool, True if this completion counts toward eval metrics.
        eval_count_mask = None
        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1], eval_count_mask = self._handle_auto_reset(
                past_dones.cpu().numpy(), obs_list[-1], infos_list[-1]
            )

        if self.auto_reset or self.ignore_terminations:
            chunk_terminations = torch.zeros_like(raw_chunk_terminations)
            chunk_terminations[:, -1] = past_terminations

            chunk_truncations = torch.zeros_like(raw_chunk_truncations)
            chunk_truncations[:, -1] = past_truncations

            if eval_count_mask is not None:
                eval_count_mask = torch.tensor(
                    eval_count_mask,
                    dtype=torch.bool,
                    device=past_terminations.device,
                )
                chunk_terminations[:, -1] &= eval_count_mask
                chunk_truncations[:, -1] &= eval_count_mask
        else:
            chunk_terminations = raw_chunk_terminations.clone()
            chunk_truncations = raw_chunk_truncations.clone()
        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def _handle_auto_reset(self, dones, _final_obs, infos):
        if self.is_eval:
            return self._handle_eval_auto_reset(dones, _final_obs, infos)
        obs, infos = self._handle_train_auto_reset(dones, _final_obs, infos)
        return obs, infos, None

    def _handle_eval_auto_reset(self, dones, _final_obs, infos):
        # STL: dump plots for finished episodes before reset
        if getattr(self, "_stl_enabled", False) and getattr(self, "_stl_plot_dir", None) and getattr(self, "_stl_save_plots", False):
            try:
                env_idx_done = np.arange(0, self.num_envs)[dones]
                self._dump_episode_plots(env_idx_done)
            except Exception:
                pass

        final_obs = copy.deepcopy(_final_obs)
        env_idx = np.arange(0, self.num_envs)[dones]
        final_info = copy.deepcopy(infos)

        count_mask = record_completed_episode_task_stats(
            env_idx,
            final_info,
            self.task_ids,
            self.trial_ids,
            self.num_envs,
            self._eval_seen_trials,
            self._task_success_stats,
        )

        new_reset_state_ids = self._get_ordered_reset_state_ids(len(env_idx))
        valid_mask = new_reset_state_ids >= 0
        env_to_reset = env_idx[valid_mask]
        if len(env_to_reset) > 0:
            self.reset_state_ids[env_to_reset] = new_reset_state_ids[valid_mask]
            obs, infos = self.reset(
                env_idx=env_to_reset,
                reset_state_ids=self.reset_state_ids[env_to_reset],
            )
        else:
            obs = _final_obs
            infos = {}

        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = np.asarray(dones, dtype=bool) & count_mask
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs, infos, count_mask

    def _handle_train_auto_reset(self, dones, _final_obs, infos):
        final_obs = copy.deepcopy(_final_obs)
        env_idx = np.arange(0, self.num_envs)[dones]
        final_info = copy.deepcopy(infos)

        if self.use_fixed_reset_state_ids:
            self.update_reset_state_ids()
            obs, infos = self.reset(
                env_idx=env_idx,
                reset_state_ids=self.reset_state_ids[env_idx],
            )
        else:
            obs, infos = self.reset(env_idx=env_idx, reset_state_ids=None)

        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = np.asarray(dones, dtype=bool)
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs, infos

    def _calc_step_reward(self, terminations):
        # Base sparse reward: success bonus = reward_coef * terminations
        step_penalty = -1 if self.use_step_penalty else 0
        termination_bonus = self.cfg.reward_coef * terminations
        reward = step_penalty + termination_bonus

        # Add STL robustness as reward if enabled
        # r_t = reward_coef * terminations + stl_reward * rho_t
        # where rho_t is the prefix STL robustness (task-level margin)
        if getattr(self, "_stl_enabled", False):
            try:
                rho = self._stl_robustness()  # (num_envs,) current rho_t
                if self._stl_clip_lower is not None or self._stl_clip_upper is not None:
                    rho = np.clip(rho, self._stl_clip_lower, self._stl_clip_upper)
                reward = np.asarray(reward, dtype=np.float64) + self._stl_reward_scale * rho
                # record total reward trace for plotting
                if getattr(self, "_stl_plot_dir", None):
                    for i in range(self.num_envs):
                        st = self._stl_states[i] if i < len(self._stl_states) else None
                        if st is not None and st.valid:
                            st.reward_trace.append(float(reward[i]))
            except Exception:
                pass

        # NOTE: only the STL rho term is clipped (above); the sparse task reward
        # (reward_coef * terminations) is left unrestricted.

        if self.use_rel_reward:
            reward_diff = reward - self.prev_step_reward
            self.prev_step_reward = reward
            return reward_diff
        else:
            return reward

    # ------------------------------------------------------------------
    # STL: privileged state, goal atoms, shaping computation
    # ------------------------------------------------------------------

    def _stl_robustness(self) -> np.ndarray:
        """Return the current prefix STL robustness rho_t for each env.

        This is the dense task-level margin used as reward signal:
            rho_t = rho(phi, s[0:t+1])
        where phi = AND_k EV(grasp_k AND EV_[0,tau] goal_k).

        Returns (num_envs,) float64 array.
        """
        priv_list = self.env.get_privileged_state()
        rho_out = np.zeros(self.num_envs, dtype=np.float64)
        for i, priv in enumerate(priv_list):
            st = self._stl_states[i] if i < len(self._stl_states) else None
            if st is None or not st.valid or st.tracker is None or priv is None:
                continue
            try:
                ep = _build_step_episode(priv)
                grasp_margins: List[Optional[float]] = []
                goal_margins: List[float] = []
                step_margins: Dict[str, float] = {}
                for (pred, obj, target), gobj in zip(st.goal_atoms, st.grasp_objs):
                    gm, _name = goal_atom_margin(ep, pred, obj, target, self._pred_cfg)
                    gmv = float(np.asarray(gm, dtype=np.float64).reshape(-1)[-1])
                    goal_margins.append(gmv)
                    step_margins[_stl_atom_name((pred, obj, target))] = gmv
                    if gobj is not None:
                        gk = resolve_object_key(gobj, ep)
                        if gk is not None:
                            pm = pred_pick(ep, gk, self._pred_cfg)
                            pmv = float(np.asarray(pm, dtype=np.float64).reshape(-1)[-1])
                            grasp_margins.append(pmv)
                            step_margins[f"pick({gobj})"] = pmv
                        else:
                            grasp_margins.append(None)
                    else:
                        grasp_margins.append(None)
                rho_t = st.tracker.step(grasp_margins, goal_margins)
                rho_out[i] = rho_t
                st.prev_rho = rho_t
                if self._stl_plot_dir:
                    st.rho_trace.append(rho_t)
                    # shaping for backward compatibility (not used in reward)
                    st.shaping_trace.append(0.0)
                    # record per-atom margins
                    st.last_margins = step_margins
                    for nm, mv in step_margins.items():
                        st.atom_margins.setdefault(nm, []).append(mv)
            except Exception:
                continue
        return rho_out

    def _stl_setup(self, ids) -> None:
        """Initialize STL state for each environment in ids using NL parser."""
        if not ids:
            return
        if not hasattr(self, "env") or self.env is None:
            return
        # Get initial privileged state for all requested envs
        priv_list = self.env.get_privileged_state(id=ids)

        for j, i in enumerate(ids):
            task_desc = self.task_descriptions[i]
            priv = priv_list[j] if j < len(priv_list) else None

            if not task_desc or priv is None:
                self._stl_states[i] = _STLEnvState(
                    valid=False, goal_atoms=[], grasp_objs=[],
                    tracker=None, prev_rho=0.0, atom_margins={},
                    last_margins={}, rho_trace=[], shaping_trace=[], reward_trace=[],
                )
                continue

            try:
                # Ground NL -> goal_atoms + grasp_objs against the live state.
                # Closed-set grounder: names are only selected from the scene
                # (object_pos keys U site_names); nothing is invented.
                goal_atoms, grasp_objs = ground_task_state(task_desc, priv)
                goal_atoms = _normalize_goal_atoms(goal_atoms)
                grasp_objs = [
                    obj if pred in _GEOMETRIC_PREDS else None
                    for pred, obj, _target in goal_atoms
                ]
                # Safety net: if the env's BDDL ground truth is available and the
                # NL parse disagrees (or parsed nothing), fall back to BDDL so a
                # misparse can never corrupt the STL reward during training.
                if self._stl_fallback_bddl and get_bddl_goal_atoms is not None:
                    bddl_atoms = None
                    try:
                        tid = self.task_ids[i] if getattr(self, "task_ids", None) is not None else None
                        if tid is not None:
                            bddl_atoms = get_bddl_goal_atoms(self.cfg.task_suite_name, tid)
                    except Exception:
                        bddl_atoms = None
                    # if bddl_atoms and set(goal_atoms) != set(bddl_atoms):
                    #     if self._stl_dbg:
                    #         print(f"[STL_DBG] env {i}: NL parse mismatch -> "
                    #               f"falling back to BDDL {bddl_atoms} "
                    #               f"(parse was {goal_atoms})", flush=True)
                    #     goal_atoms = [(p, o, t) for (p, o, t) in bddl_atoms]
                    #     grasp_objs = [o if p in _GEOMETRIC_PREDS else None
                    #                   for (p, o, t) in goal_atoms]
                    
                    if bddl_atoms:
                        normalized_bddl_atoms = _normalize_goal_atoms(bddl_atoms)

                        if (
                            normalized_bddl_atoms
                            and set(goal_atoms) != set(normalized_bddl_atoms)
                        ):
                            if self._stl_dbg:
                                print(f"[STL_DBG] env {i}: NL parse mismatch -> "
                                    f"falling back to BDDL {bddl_atoms} "
                                    f"(parse was {goal_atoms})", flush=True)

                            goal_atoms = normalized_bddl_atoms
                            grasp_objs = [
                                obj if pred in _GEOMETRIC_PREDS else None
                                for pred, obj, _target in goal_atoms
                            ]
                                            
                valid = bool(goal_atoms)
                st = _STLEnvState(
                    valid=valid,
                    goal_atoms=goal_atoms,
                    grasp_objs=grasp_objs,
                    tracker=OnlineRobustnessTracker(self._stl_tau, len(goal_atoms)) if valid else None,
                    prev_rho=0.0,
                    atom_margins={},
                    last_margins={},
                    rho_trace=[],
                    shaping_trace=[],
                    reward_trace=[],
                )
                self._stl_states[i] = st

                if self._stl_dbg:
                    print(f"[STL_DBG] LiberoEnv._stl_setup env {i}: "
                          f"valid={st.valid} atoms={st.goal_atoms[:3]} "
                          f"from NL='{task_desc[:60]}'", flush=True)

            except Exception:
                if self._stl_dbg:
                    print(f"[STL_DBG] LiberoEnv._stl_setup FAILED env {i}: "
                          f"NL='{task_desc[:60]}'", flush=True)
                self._stl_states[i] = _STLEnvState(
                    valid=False, goal_atoms=[], grasp_objs=[],
                    tracker=None, prev_rho=0.0, atom_margins={},
                    last_margins={}, rho_trace=[], shaping_trace=[], reward_trace=[],
                )

    # ------------------------------------------------------------------
    # STL: online resolve — NL category names → instance names
    # ------------------------------------------------------------------

    def _stl_shaping(self) -> np.ndarray:
        priv_list = self.env.get_privileged_state()
        shaping = np.zeros(self.num_envs, dtype=np.float64)
        for i, priv in enumerate(priv_list):
            st = self._stl_states[i] if i < len(self._stl_states) else None
            if st is None or not st.valid or st.tracker is None or priv is None:
                continue
            try:
                ep = _build_step_episode(priv)
                grasp_margins: List[Optional[float]] = []
                goal_margins: List[float] = []
                step_margins: Dict[str, float] = {}
                for (pred, obj, target), gobj in zip(st.goal_atoms, st.grasp_objs):
                    gm, _name = goal_atom_margin(ep, pred, obj, target, self._pred_cfg)
                    gmv = float(np.asarray(gm, dtype=np.float64).reshape(-1)[-1])
                    goal_margins.append(gmv)
                    step_margins[_stl_atom_name((pred, obj, target))] = gmv
                    if gobj is not None:
                        gk = resolve_object_key(gobj, ep)
                        if gk is not None:
                            pm = pred_pick(ep, gk, self._pred_cfg)
                            pmv = float(np.asarray(pm, dtype=np.float64).reshape(-1)[-1])
                            grasp_margins.append(pmv)
                            step_margins[f"pick({gobj})"] = pmv
                        else:
                            grasp_margins.append(None)
                    else:
                        grasp_margins.append(None)
                rho_t = st.tracker.step(grasp_margins, goal_margins)
                shaping[i] = self._stl_gamma * rho_t - st.prev_rho
                st.prev_rho = rho_t
                if self._stl_plot_dir:
                    st.rho_trace.append(rho_t)
                    st.shaping_trace.append(shaping[i])
                    # record per-atom margins
                    st.last_margins = step_margins
                    for nm, mv in step_margins.items():
                        st.atom_margins.setdefault(nm, []).append(mv)
            except Exception:
                continue
        return shaping

    # ------------------------------------------------------------------
    # STL: per-episode plotting
    # ------------------------------------------------------------------

    def _dump_episode_plots(self, ids) -> None:
        """Save per-episode STL data (NPZ) and robustness PNG for each just-finished env in ids."""
        import os

        for i in ids:
            st = self._stl_states[i] if i < len(self._stl_states) else None
            if st is None or not st.rho_trace:
                continue
            try:
                task_desc = ""
                task_id = -1
                if getattr(self, "task_ids", None) is not None and i < len(self.task_ids):
                    try:
                        tid = int(self.task_ids[i])
                        task_id = tid
                        task_desc = self.task_suite.get_task(tid).language
                    except Exception:
                        task_desc = ""
                success = bool(self.success_once[i]) if getattr(self, "success_once", None) is not None else False
                self._stl_plot_counter += 1
                ep_str = f"ep{self._stl_plot_counter:04d}"

                # Save raw data as NPZ (format consumed by plot_stl_from_npz.py)
                npz_path = os.path.join(
                    self._stl_plot_dir,
                    f"stl_data_{ep_str}_task{task_id:02d}_{'SUCCESS' if success else 'fail'}.npz",
                )
                os.makedirs(os.path.dirname(npz_path), exist_ok=True)
                np.savez(
                    npz_path,
                    rho=np.asarray(st.rho_trace, dtype=np.float64),
                    shaping=np.asarray(st.shaping_trace, dtype=np.float64),
                    reward=np.asarray(st.reward_trace, dtype=np.float64),
                    atom_names=list(st.atom_margins.keys()),
                    **{f"atom_{nm}": np.asarray(st.atom_margins[nm], dtype=np.float64)
                       for nm in st.atom_margins},
                )

                # Save PNG plot
                out_path = os.path.join(
                    self._stl_plot_dir,
                    f"stl_plot_{ep_str}_task{task_id:02d}_{'SUCCESS' if success else 'fail'}.png",
                )
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                self._plot_stl_episode(out_path, st, task_desc, success)
            except Exception as exc:
                # npz was already written; surface the plotting failure so
                # missing PNGs are visible instead of silently dropped.
                print(
                    f"[STL_PLOT_ERROR] task={task_id} ep={ep_str} error={exc!r}",
                    flush=True,
                )
                continue

    def _plot_stl_episode(self, out_path: str, st: "_STLEnvState", task_desc: str, success: bool) -> None:
        import os

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rho = np.asarray(st.rho_trace, dtype=np.float64)
        T = len(rho)
        t = np.arange(T)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 9), sharex=True,
                                            gridspec_kw={"height_ratios": [1.0, 1.0, 0.8]})

        # panel 1: per-atom robustness margins
        ax1.axhline(0.0, color="k", lw=0.8, alpha=0.5)
        cmap = plt.get_cmap("tab10")
        li = 0
        for nm in sorted(st.atom_margins.keys()):
            tr = np.asarray(st.atom_margins[nm], dtype=np.float64)
            if len(tr) != T:
                continue
            is_pick = nm.startswith("pick(")
            # Escape underscores for mathtext: raw names like
            # on(moka_pot_1,flat_stove_1_cook_region) trigger
            # "Double subscript" ParseSyntaxException in the legend.
            nm_escaped = nm.replace("_", r"\_")
            ax1.plot(t, tr, color=cmap(li % 10), lw=1.7, ls=":" if is_pick else "-",
                     label=rf"$\mu_{{{nm_escaped}}}(t)$")
            on = int(np.where(tr >= 0.0)[0][0]) if (tr >= 0.0).any() else None
            if on is not None:
                ax1.axvline(on, color=cmap(li % 10), ls=":", lw=0.8, alpha=0.5)
            li += 1
        ax1.set_ylabel("per-atom margin\n(>0 satisfied)")
        ax1.set_title("Per-atom predicate robustness  μ(t)")
        ax1.legend(loc="best", fontsize=7.5, ncol=2)
        ax1.grid(alpha=0.25)

        # panel 2: whole-task prefix robustness ρ
        ax2.axhline(0.0, color="k", lw=0.9, alpha=0.6)
        ax2.plot(t, rho, color="crimson", lw=2.2, label=r"online $\rho(\phi,\,s[0:t+1])$")
        ax2.set_title(rf"$\rho_{{final}}$={rho[-1]:+.4f} ({'SATISFIED' if rho[-1] >= 0 else 'VIOLATED'})"
                      rf"   $\tau$={self._stl_tau}   (prefix STL robustness)")
        ax2.set_ylabel("task robustness\nmargin")
        ax2.legend(loc="best", fontsize=9)
        ax2.grid(alpha=0.25)

        # panel 3: reward (sparse bonus + STL shaping = total)
        ax3.axhline(0.0, color="k", lw=0.8, alpha=0.5)
        if st.shaping_trace:
            ax3.plot(t, np.asarray(st.shaping_trace), color="tab:purple", lw=1.3, ls="--",
                     label="STL shaping (γ·ρ_t − ρ_{t-1})")
        if st.reward_trace:
            ax3.plot(t, np.asarray(st.reward_trace), color="black", lw=1.8, label="total reward r_t")
        ax3.set_title(f"Reward  (sparse success bonus + stl_reward_scale · shaping)")
        ax3.set_ylabel("reward")
        ax3.set_xlabel("env step t")
        ax3.legend(loc="best", fontsize=9)
        ax3.grid(alpha=0.25)
        ax3.set_xlim(0, max(T - 1, 0))

        fig.suptitle(f"{task_desc}    success={success}  steps={T}", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
