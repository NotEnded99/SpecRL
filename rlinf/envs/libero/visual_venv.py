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

import multiprocessing
import warnings
from multiprocessing import connection
from typing import Any, Callable, Optional, Union

import gym
import numpy as np

from rlinf.envs.libero.utils import get_libero_type
from rlinf.envs.venv import (
    BaseVectorEnv,
    CloudpickleWrapper,
    EnvWorker,
    ShArray,
    SubprocEnvWorker,
    SubprocVectorEnv,
    _setup_buf,
)

# ---------------------------------------------------------------------------
# Dynamic Module Import Logic for Libero Pro / Plus
# ---------------------------------------------------------------------------
libero_type = get_libero_type()

if libero_type == "pro":
    try:
        from liberopro.liberopro.envs import OffScreenRenderEnv
    except ImportError as e:
        print(
            f"[Venv] Warning: LIBERO_TYPE=pro but import failed ({e}). Falling back to standard libero..."
        )
        from libero.libero.envs import OffScreenRenderEnv

elif libero_type == "plus":
    try:
        from liberoplus.liberoplus.envs import OffScreenRenderEnv
    except ImportError as e:
        print(
            f"[Venv] Warning: LIBERO_TYPE=plus but import failed ({e}). Falling back to standard libero..."
        )
        from libero.libero.envs import OffScreenRenderEnv

else:
    try:
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError:
        try:
            from liberopro.liberopro.envs import OffScreenRenderEnv
        except ImportError:
            try:
                from liberoplus.liberoplus.envs import OffScreenRenderEnv
            except ImportError:
                raise ImportError(
                    "Could not import OffScreenRenderEnv from libero, liberopro, or liberoplus."
                )


def _decode_mj_names(names) -> list[str]:
    """mujoco-py name arrays come as numpy arrays of bytes/str -> list[str]."""
    out = []
    for n in list(names):
        if isinstance(n, (bytes, np.bytes_)):
            out.append(n.decode("utf-8", "ignore"))
        else:
            out.append(str(n))
    return out


def _extract_instance_id_to_name(env) -> dict[int, str]:
    """Return robosuite's instance-segmentation ID mapping as plain data."""
    domain = getattr(env, "env", env)
    instances = getattr(getattr(domain, "model", None), "instances_to_ids", None)
    if not instances:
        return {}
    return {
        index + 1: str(name)
        for index, name in enumerate(instances.keys())
    }


def _extract_privileged_state(env, max_con: int = 128) -> dict:
    """Collect ONE state's privileged MuJoCo quantities needed by the 8 LIBERO
    atomic-predicate margins (pick / on / stack / in / open / close / turn_on /
    turn_off). Adapted from openvla/run_libero_eval_fast_v2._get_privileged_state
    but reads everything straight from ``env.sim`` (no obs dependency) and exposes
    ALL bodies so fixtures (stove/cabinet/...) are resolvable too.

    Best-effort: NEVER raises. On any sub-failure returns a minimal valid dict so a
    faulty privileged read never breaks a rollout — the caller falls back to the
    sparse success reward.
    """
    zero = {
        "body_xpos": np.zeros((0, 3), dtype=np.float64),
        "body_names": [],
        "eef_pos": np.zeros(3, dtype=np.float64),
        "site_xpos": np.zeros((0, 3), dtype=np.float64),
        "site_xmat": np.zeros((0, 3, 3), dtype=np.float64),
        "site_names": [],
        "site_sizes": np.zeros((0, 3), dtype=np.float64),
        "joint_qpos": np.zeros(0, dtype=np.float64),
        "joints": [],  # list[dict{name,obj,col,addr}]
        "articulated": [],  # list[name, n, open_rng, close_rng, turnon_rng, turnoff_rng]
        "gripper_qpos": np.zeros(2, dtype=np.float64),
        "contact_body1": np.full(max_con, -1, dtype=np.int32),
        "contact_body2": np.full(max_con, -1, dtype=np.int32),
        "contact_dist": np.zeros(max_con, dtype=np.float32),
        "contact_force": np.zeros(max_con, dtype=np.float32),
        "ncon": 0,
        "object_pos": {},  # name -> (3,) world pos for manipulable objects
        "obj_body_tree": {},  # object name -> [mujoco body ids of its subtree]
    }
    try:
        sim = env.sim
    except Exception:
        return zero
    domain = getattr(env, "env", env)

    # ---- all body poses + names (objects AND fixtures) ----
    # IMPORTANT: sim.data.body_xpos is a *view* into MuJoCo's pybind11-backed
    # memory; np.asarray(...) returns that view unchanged. Returning the view in
    # the dict (then pickling it across the worker pipe) lets the underlying
    # pybind11 instance be deallocated wrongly later -> worker SIGSEGV
    # ("pybind11_object_dealloc(): Tried to deallocate unregistered instance!").
    # np.array(...) forces a real COPY and breaks that reference. Every sim.data
    # array read below MUST copy.
    try:
        zero["body_names"] = _decode_mj_names(sim.model.body_names)
        zero["body_xpos"] = np.array(sim.data.body_xpos, dtype=np.float64).reshape(-1, 3)
    except Exception:
        pass

    # ---- eef position (end-effector, for pick proximity term) ----
    try:
        # Try to get eef_pos from the robot's eef site/body
        eef_site_names = [n for n in _decode_mj_names(sim.model.site_names) if "eef" in n.lower() or "endeffector" in n.lower() or "grip_site" in n.lower()]
        if eef_site_names:
            zero["eef_pos"] = np.array(sim.data.get_site_xpos(eef_site_names[0]), dtype=np.float64).reshape(3)
        else:
            # Fallback: try to find the robot0 eef body
            for i, n in enumerate(zero.get("body_names", [])):
                if "gripper" in n.lower() and "base" not in n.lower():
                    zero["eef_pos"] = np.array(sim.data.body_xpos[i], dtype=np.float64).reshape(3)
                    break
    except Exception:
        pass

    # ---- manipulable object world positions ----
    try:
        for name, bid in (getattr(domain, "obj_body_id", {}) or {}).items():
            try:
                zero["object_pos"][str(name)] = (
                    np.array(sim.data.body_xpos[int(bid)], dtype=np.float64).reshape(3)
                )
            except Exception:
                pass
    except Exception:
        pass

    # ---- container sites (contain()/under() AABB) ----
    try:
        sites = getattr(domain, "object_sites_dict", {}) or {}
        sn = list(sites.keys())
        if sn:
            zero["site_names"] = [str(n) for n in sn]
            zero["site_xpos"] = np.stack(
                [np.array(sim.data.get_site_xpos(str(n)), dtype=np.float64).reshape(3) for n in sn]
            )
            zero["site_xmat"] = np.stack(
                [np.array(sim.data.get_site_xmat(str(n)), dtype=np.float64).reshape(3, 3) for n in sn]
            )
            zero["site_sizes"] = np.stack(
                [np.array(sites[n].size, dtype=np.float64).reshape(3) for n in sn]
            )
    except Exception:
        pass

    # ---- articulated joint qpos + per-verb design ranges ----
    try:
        osd = getattr(domain, "object_states_dict", {}) or {}
        jvals: list[float] = []
        jinfo: list[dict] = []
        art: list = []
        for name, st in osd.items():
            try:
                js = st.get_joint_state()
            except Exception:
                js = []
            if not (hasattr(js, "__len__") and len(js) > 0):
                continue
            try:
                obj = domain.get_object(name)
            except Exception:
                obj = None
            try:
                joints = list(obj.joints) if obj is not None else [None] * len(js)
            except Exception:
                joints = [None] * len(js)
            for k, q in enumerate(js):
                jn = joints[k] if k < len(joints) else None
                try:
                    addr = int(sim.model.get_joint_qpos_addr(jn)) if jn is not None else -1
                except Exception:
                    addr = -1
                jvals.append(float(q))
                jinfo.append({"name": str(jn), "obj": str(name), "col": len(jvals) - 1, "addr": addr})

            def _rng(key):
                try:
                    r = obj.object_properties["articulation"][key]
                    return [float(r[0]), float(r[1])]
                except Exception:
                    return None

            if obj is not None:
                art.append(
                    [
                        str(name),
                        len(js),
                        _rng("default_open_ranges"),
                        _rng("default_close_ranges"),
                        _rng("default_turnon_ranges"),
                        _rng("default_turnoff_ranges"),
                    ]
                )
        zero["joint_qpos"] = np.asarray(jvals, dtype=np.float64).reshape(-1)
        zero["joints"] = jinfo
        zero["articulated"] = art
    except Exception:
        pass

    # ---- gripper finger qpos (pick closure margin) from sim directly ----
    try:
        gq: list[float] = []
        for jn in ("robot0_left_finger_joint", "robot0_right_finger_joint"):
            try:
                a = sim.model.get_joint_qpos_addr(jn)
                gq.append(float(sim.data.qpos[a]))
            except Exception:
                pass
        if len(gq) != 2:
            jnames = _decode_mj_names(sim.model.joint_names)
            ids = [i for i, n in enumerate(jnames) if "finger" in n.lower()]
            if len(ids) >= 2:
                addrs = [int(sim.model.jnt_qposadr[i]) for i in ids[:2]]
                gq = [float(sim.data.qpos[a]) for a in addrs]
        if len(gq) == 2:
            zero["gripper_qpos"] = np.asarray(gq, dtype=np.float64)
    except Exception:
        pass

    # ---- full MuJoCo contact list (padded) ----
    try:
        gbid = sim.model.geom_bodyid
        m = min(int(sim.data.ncon), max_con)
        contacts = sim.data.contact
        efc_force = sim.data.efc_force
        cb1 = np.full(max_con, -1, dtype=np.int32)
        cb2 = np.full(max_con, -1, dtype=np.int32)
        cd = np.zeros(max_con, dtype=np.float32)
        cf = np.zeros(max_con, dtype=np.float32)
        for i in range(m):
            c = contacts[i]
            cb1[i] = int(gbid[c.geom1])
            cb2[i] = int(gbid[c.geom2])
            cd[i] = float(c.dist)  # <0 = penetrating
            ea = int(c.efc_address)
            cf[i] = abs(float(efc_force[ea])) if ea >= 0 else 0.0
        zero["contact_body1"] = cb1
        zero["contact_body2"] = cb2
        zero["contact_dist"] = cd
        zero["contact_force"] = cf
        zero["ncon"] = m
    except Exception:
        pass

    # ---- per-object body subtrees (bit-exact contact predicate) ----
    # Constant per task (changes only on reconfigure, which builds a fresh env) —
    # memoize on the env object so we don't pay the O(nbody^2) walk every step.
    try:
        cached = getattr(env, "_stl_obj_body_tree", None)
        if cached is None:
            obid = getattr(domain, "obj_body_id", {}) or {}
            parent = sim.model.body_parentid
            nbody = len(parent)

            def _descendants(root: int):
                res = {int(root)}
                changed = True
                while changed:
                    changed = False
                    for b in range(nbody):
                        if b in res:
                            continue
                        if int(parent[b]) in res:
                            res.add(b)
                            changed = True
                return sorted(int(b) for b in res)

            cached = {str(name): _descendants(int(rid)) for name, rid in obid.items()}
            try:
                env._stl_obj_body_tree = cached
            except Exception:
                pass
        zero["obj_body_tree"] = cached
    except Exception:
        pass

    return zero


def _get_goal_atoms(env) -> list:
    """Return the BDDL goal atoms of the current task, read from the live env's
    parsed_problem['goal_state']. Returns the raw picklable list; the caller
    normalizes the tuple shape. Best-effort: returns [] on any failure."""
    try:
        domain = getattr(env, "env", env)
        gp = getattr(domain, "parsed_problem", None)
        if gp is None:
            gp = getattr(domain, "problem", None)
        gs = gp.get("goal_state", []) if isinstance(gp, dict) else []
        # coerce to plain python tuples (mujoco/libero may return nested arrays)
        out = []
        for a in gs:
            try:
                out.append(_atom_to_tuple(a))
            except Exception:
                continue
        return out
    except Exception:
        return []


def _atom_to_tuple(a):
    """Flatten a libero goal atom to a plain python tuple of strings."""
    parts = []
    if isinstance(a, (str, bytes)):
        return (str(a),)
    for x in list(a):
        if isinstance(x, (bytes, np.bytes_)):
            parts.append(x.decode("utf-8", "ignore"))
        elif isinstance(x, (list, tuple, np.ndarray)):
            parts.extend(str(y) for y in list(x))
        else:
            parts.append(str(x))
    return tuple(parts)


gym_old_venv_step_type = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
gym_new_venv_step_type = tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]
warnings.simplefilter("once", DeprecationWarning)


def _extract_camera_calibration(
    env,
    camera_name: str = "agentview",
    camera_height: int = 256,
    camera_width: int = 256,
) -> dict:
    """Return copied camera parameters safe for worker IPC."""
    result = {
        "valid": False,
        "camera_name": str(camera_name),
        "camera_height": int(camera_height),
        "camera_width": int(camera_width),
        "intrinsic": np.eye(3, dtype=np.float64),
        "camera_to_world": np.eye(4, dtype=np.float64),
        "near_m": float("nan"),
        "far_m": float("nan"),
        "reason": "",
    }
    try:
        from robosuite.utils import camera_utils

        sim = env.sim
        extent = float(sim.model.stat.extent)
        result.update({
            "valid": True,
            "intrinsic": np.array(
                camera_utils.get_camera_intrinsic_matrix(
                    sim,
                    camera_name,
                    int(camera_height),
                    int(camera_width),
                ),
                dtype=np.float64,
                copy=True,
            ),
            "camera_to_world": np.array(
                camera_utils.get_camera_extrinsic_matrix(
                    sim,
                    camera_name,
                ),
                dtype=np.float64,
                copy=True,
            ),
            "near_m": float(sim.model.vis.map.znear) * extent,
            "far_m": float(sim.model.vis.map.zfar) * extent,
        })
    except Exception as error:
        result["reason"] = f"{type(error).__name__}: {error}"
    return result


def _worker(
    parent: connection.Connection,
    p: connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
    obs_bufs: Optional[Union[dict, tuple, ShArray]] = None,
) -> None:
    def _encode_obs(
        obs: Union[dict, tuple, np.ndarray], buffer: Union[dict, tuple, ShArray]
    ) -> None:
        if isinstance(obs, np.ndarray) and isinstance(buffer, ShArray):
            buffer.save(obs)
        elif isinstance(obs, tuple) and isinstance(buffer, tuple):
            for o, b in zip(obs, buffer):
                _encode_obs(o, b)
        elif isinstance(obs, dict) and isinstance(buffer, dict):
            for k in obs.keys():
                _encode_obs(obs[k], buffer[k])
        return None

    parent.close()
    env = env_fn_wrapper.data()
    try:
        while True:
            try:
                cmd, data = p.recv()
            except EOFError:  # the pipe has been closed
                p.close()
                break
            if cmd == "step":
                env_return = env.step(data)
                if obs_bufs is not None:
                    _encode_obs(env_return[0], obs_bufs)
                    env_return = (None, *env_return[1:])
                p.send(env_return)
            elif cmd == "reset":
                retval = env.reset(**data)
                reset_returns_info = (
                    isinstance(retval, (tuple, list))
                    and len(retval) == 2
                    and isinstance(retval[1], dict)
                )
                if reset_returns_info:
                    obs, info = retval
                else:
                    obs = retval
                if obs_bufs is not None:
                    _encode_obs(obs, obs_bufs)
                    obs = None
                if reset_returns_info:
                    p.send((obs, info))
                else:
                    p.send(obs)
            elif cmd == "close":
                p.send(env.close())
                p.close()
                break
            elif cmd == "render":
                p.send(env.render(**data) if hasattr(env, "render") else None)
            elif cmd == "seed":
                if hasattr(env, "seed"):
                    p.send(env.seed(data))
                else:
                    env.reset(seed=data)
                    p.send(None)
            elif cmd == "getattr":
                p.send(getattr(env, data) if hasattr(env, data) else None)
            elif cmd == "setattr":
                setattr(env.unwrapped, data["key"], data["value"])
            elif cmd == "check_success":
                p.send(env.check_success())
            elif cmd == "get_segmentation_of_interest":
                p.send(env.get_segmentation_of_interest(data))
            elif cmd == "get_sim_state":
                p.send(env.get_sim_state())
            elif cmd == "get_privileged_state":
                p.send(_extract_privileged_state(env))
            elif cmd == "get_instance_id_to_name":
                p.send(_extract_instance_id_to_name(env))
            elif cmd == "get_camera_calibration":
                p.send(
                    _extract_camera_calibration(
                        env,
                        **(data or {}),
                    )
                )
            elif cmd == "get_goal_atoms":
                p.send(_get_goal_atoms(env))
            elif cmd == "set_init_state":
                obs = env.set_init_state(data)
                p.send(obs)
            elif cmd == "reconfigure":
                env.close()
                seed = data.pop("seed")
                env = OffScreenRenderEnv(**data)
                env.seed(seed)
                p.send(None)
            else:
                p.close()
                raise NotImplementedError
    except KeyboardInterrupt:
        p.close()


class ReconfigureSubprocEnvWorker(SubprocEnvWorker):
    def __init__(self, env_fn: Callable[[], gym.Env], share_memory: bool = False):
        ctx = multiprocessing.get_context("spawn")
        self.parent_remote, self.child_remote = ctx.Pipe()
        self.share_memory = share_memory
        self.buffer: Optional[Union[dict, tuple, ShArray]] = None
        if self.share_memory:
            dummy = env_fn()
            obs_space = dummy.observation_space
            dummy.close()
            del dummy
            self.buffer = _setup_buf(obs_space)
        args = (
            self.parent_remote,
            self.child_remote,
            CloudpickleWrapper(env_fn),
            self.buffer,
        )
        self.process = ctx.Process(target=_worker, args=args, daemon=True)
        self.process.start()
        self.child_remote.close()
        EnvWorker.__init__(self, env_fn)

    def reconfigure_env_fn(self, env_fn_param):
        self.parent_remote.send(["reconfigure", env_fn_param])
        return self.parent_remote.recv()

    def get_privileged_state(self):
        self.parent_remote.send(["get_privileged_state", None])
        return self.parent_remote.recv()

    def get_instance_id_to_name(self):
        self.parent_remote.send(["get_instance_id_to_name", None])
        return self.parent_remote.recv()

    def get_camera_calibration(
        self,
        camera_name="agentview",
        camera_height=256,
        camera_width=256,
    ):
        self.parent_remote.send([
            "get_camera_calibration",
            {
                "camera_name": camera_name,
                "camera_height": int(camera_height),
                "camera_width": int(camera_width),
            },
        ])
        return self.parent_remote.recv()

    def get_goal_atoms(self):
        self.parent_remote.send(["get_goal_atoms", None])
        return self.parent_remote.recv()


class ReconfigureSubprocEnv(SubprocVectorEnv):
    def __init__(self, env_fns: list[Callable[[], gym.Env]], **kwargs: Any) -> None:
        def worker_fn(fn: Callable[[], gym.Env]) -> ReconfigureSubprocEnvWorker:
            return ReconfigureSubprocEnvWorker(fn, share_memory=False)

        BaseVectorEnv.__init__(self, env_fns, worker_fn, **kwargs)

    def reconfigure_env_fns(self, env_fns, id=None):
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)

        for j, i in enumerate(id):
            self.workers[i].reconfigure_env_fn(env_fns[j])

    def get_privileged_state(self, id=None):
        """Per-env privileged MuJoCo state for STL predicate margins. Returns a
        list of dicts (one per env in ``id``, default all)."""
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        return [self.workers[i].get_privileged_state() for i in id]

    def get_instance_id_to_name(self, id=None):
        """Fetch copied instance-segmentation maps from selected workers."""
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        return [self.workers[i].get_instance_id_to_name() for i in id]

    def get_camera_calibration(
        self,
        camera_name="agentview",
        camera_height=256,
        camera_width=256,
        id=None,
    ):
        """Fetch copied camera calibration from selected workers."""
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        return [
            self.workers[i].get_camera_calibration(
                camera_name,
                camera_height,
                camera_width,
            )
            for i in id
        ]

    def get_goal_atoms(self, id=None):
        """Per-env BDDL goal atoms (list of plain-string tuples)."""
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        return [self.workers[i].get_goal_atoms() for i in id]
