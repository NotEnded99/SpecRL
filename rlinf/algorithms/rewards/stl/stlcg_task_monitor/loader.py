"""Episode loader: time-aligned states/forces + decoded privileged metadata.

The privileged HDF5 group already records everything needed to compute the 8
atomic predicates EXACTLY (no proxies):
  * body_names / body_ids           -> contact body id  -> object name
  * site_names + site_sizes + xmat  -> container contain-region AABB (In/Stack)
  * joint_names + joint_obj_names + joint_qpos_addr + articulated_info
                                     -> articulated joint column + theta ranges
                                       (open/close/turn_on/turn_off)

String-list attrs are stored as a JSON char/string blob; numeric ones as arrays.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np


# ---------------------------------------------------------------------------------------------------------------------
# attr decoding
# ---------------------------------------------------------------------------------------------------------------------

def _decode_str_list(v) -> List[str]:
    """Decode an attr that is a JSON string list or a variable-length string array."""
    if v is None:
        return []
    s = str(v)
    # JSON form: '["a","b"]'
    if s.lstrip().startswith("["):
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return [str(x) for x in obj]
        except Exception:
            pass
    # numpy array of bytes/str
    try:
        arr = np.asarray(v).ravel()
        out = []
        for x in arr:
            if isinstance(x, (bytes, np.bytes_)):
                out.append(x.decode("utf-8", "ignore"))
            else:
                out.append(str(x))
        # if it char-decomposed a single string, rejoin
        if out and all(len(o) <= 1 for o in out):
            return ["".join(out)]
        return out
    except Exception:
        return []


def _decode_articulated(v) -> List[list]:
    """articulated_info: JSON list of [name, n, open_rng, close_rng, turnon_rng, turnoff_rng]."""
    if v is None:
        return []
    try:
        obj = json.loads(str(v))
        return obj if isinstance(obj, list) else []
    except Exception:
        # already a numpy array / list of lists
        try:
            return np.asarray(v).tolist()
        except Exception:
            return []


# ---------------------------------------------------------------------------------------------------------------------
# loaded episode
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class JointInfo:
    name: str                 # mujoco joint name, e.g. "wooden_cabinet_1_middle_level"
    obj: str                  # owning body name, e.g. "wooden_cabinet_1"
    col: int                  # column index into joint_qpos (T, N_joint)
    addr: int                 # qpos address (-1 if unknown)


@dataclass
class Episode:
    path: str
    T: int
    # states (time-aligned, length T)
    object_pos: Dict[str, np.ndarray] = field(default_factory=dict)   # name -> (T,3)
    eef_pos: np.ndarray = None                                         # (T,3)
    gripper_qpos: np.ndarray = None                                    # (T,2)
    # forces
    arm_contact_force: Optional[np.ndarray] = None                     # (T,)
    # LIBERO per-step goal predicate truth (binary), used to calibrate atom zero-points
    goal_flags: Dict[str, np.ndarray] = field(default_factory=dict)    # key -> (T,) bool
    # privileged (time-aligned, length T+1 -> cropped to T)
    body_xpos: np.ndarray = None                                       # (T, N_body, 3)
    site_xpos: np.ndarray = None                                       # (T, N_site, 3)
    site_xmat: np.ndarray = None                                       # (T, N_site, 3, 3)
    joint_qpos: np.ndarray = None                                      # (T, N_joint)
    priv_gripper_qpos: np.ndarray = None                               # (T,2)
    contact_body1: np.ndarray = None                                   # (T, MAX_CON)
    contact_body2: np.ndarray = None                                   # (T, MAX_CON)
    contact_dist: np.ndarray = None                                    # (T, MAX_CON)
    contact_force: np.ndarray = None                                   # (T, MAX_CON)
    ncon: np.ndarray = None                                            # (T,)
    # decoded metadata
    body_names: List[str] = field(default_factory=list)
    body_ids: np.ndarray = None                                        # (N_body,)
    obj_body_tree: Dict[str, List[int]] = field(default_factory=dict)  # object -> all its body ids
    site_names: List[str] = field(default_factory=list)
    site_sizes: np.ndarray = None                                      # (N_site, 3)
    joints: List[JointInfo] = field(default_factory=list)
    articulated: List[list] = field(default_factory=list)              # articulated_info
    # attrs
    attrs: dict = field(default_factory=dict)

    # ---- name lookups -------------------------------------------------------
    def body_index(self, name_substr: str) -> Optional[int]:
        """Index into body_xpos / body_ids for the body whose name contains substring."""
        for i, n in enumerate(self.body_names):
            if name_substr in n:
                return i
        return None

    def body_id_of(self, name_substr: str) -> Optional[int]:
        idx = self.body_index(name_substr)
        return None if idx is None else int(self.body_ids[idx])

    def site_index(self, name_substr: str) -> Optional[int]:
        for i, n in enumerate(self.site_names):
            if name_substr in n:
                return i
        return None

    def joints_for(self, obj_substr: str) -> List[JointInfo]:
        return [j for j in self.joints if obj_substr in j.obj]

    def joint_by_level(self, level: Optional[str]) -> Optional[JointInfo]:
        """Pick the joint matching a top/middle/bottom level; fall back to the
        joint that travels the most during the episode."""
        if level is not None:
            for j in self.joints:
                if level in j.name.lower():
                    return j
        # fallback: most-travelled joint
        if self.joint_qpos is None or self.joint_qpos.shape[1] == 0 or not self.joints:
            return None
        spans = self.joint_qpos.max(0) - self.joint_qpos.min(0)
        return self.joints[int(np.argmax(spans))]


# ---------------------------------------------------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------------------------------------------------

def _crop_to_T(arr: Optional[np.ndarray], T: int) -> Optional[np.ndarray]:
    if arr is None:
        return None
    return arr[:T]


def load_episode(path: str) -> Episode:
    with h5py.File(path, "r") as f:
        st = f["states"]
        object_pos = {k[:-4]: st[k][:].astype(np.float64)
                      for k in st.keys() if k.endswith("_pos") and not k.startswith("robot0")}
        eef_pos = st["robot0_eef_pos"][:].astype(np.float64)
        gripper_qpos = st["robot0_gripper_qpos"][:].astype(np.float64)
        arm_cf = None
        if "forces" in f and "arm_contact_force" in f["forces"]:
            arm_cf = f["forces/arm_contact_force"][:].astype(np.float64)
        goal_flags = {}
        if "goal_flags" in f:
            for k in f["goal_flags"].keys():
                goal_flags[k] = f["goal_flags"][k][:].astype(bool)

        priv = f["privileged"]
        body_xpos = priv["body_xpos"][:].astype(np.float64)
        site_xpos = priv["site_xpos"][:].astype(np.float64)
        site_xmat = priv["site_xmat"][:].astype(np.float64)
        joint_qpos = priv["joint_qpos"][:].astype(np.float64)
        pgq = priv["gripper_qpos"][:].astype(np.float64)
        cb1 = priv["contact_body1"][:].astype(np.int32)
        cb2 = priv["contact_body2"][:].astype(np.int32)
        cd = priv["contact_dist"][:].astype(np.float32)
        cf = priv["contact_force"][:].astype(np.float32)
        ncon = priv["ncon"][:].astype(np.int32)

        body_names = _decode_str_list(priv.attrs.get("body_names"))
        body_ids = np.asarray(priv.attrs.get("body_ids", []), dtype=np.int64)
        try:
            obj_body_tree = json.loads(priv.attrs.get("obj_body_tree", "{}"))
            obj_body_tree = {k: [int(b) for b in v] for k, v in obj_body_tree.items()}
        except Exception:
            obj_body_tree = {}
        site_names = _decode_str_list(priv.attrs.get("site_names"))
        site_sizes = np.asarray(priv.attrs.get("site_sizes", []), dtype=np.float64).reshape(-1, 3)
        joint_names = _decode_str_list(priv.attrs.get("joint_names"))
        joint_obj_names = _decode_str_list(priv.attrs.get("joint_obj_names"))
        joint_qpos_addr = np.asarray(priv.attrs.get("joint_qpos_addr", []), dtype=np.int64)
        articulated = _decode_articulated(priv.attrs.get("articulated_info"))

        attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}

    # time length: STATE arrays (object_pos/eef/body_xpos/site_*/joint_qpos/contacts)
    # have n_states = n_steps + 1 (initial + one per step); per-STEP arrays (forces,
    # goal_flags) have n_steps. Keep ALL states — the goal is achieved at the LAST
    # state, which cropping to the step length would drop (causing false negatives).
    state_lengths = [eef_pos.shape[0], gripper_qpos.shape[0], body_xpos.shape[0]]
    state_lengths += [v.shape[0] for v in object_pos.values()]
    T = int(min(state_lengths))

    object_pos = {k: v[:T] for k, v in object_pos.items()}
    eef_pos = eef_pos[:T]
    gripper_qpos = gripper_qpos[:T]
    if arm_cf is not None:
        arm_cf = arm_cf[: min(T, arm_cf.shape[0])]
    # goal_flags are step-indexed (len n_steps = T-1): goal_flags[i] is the goal
    # evaluated AFTER step i, i.e. at STATE i+1. Keep them at their own length.
    goal_flags = {k: v[: v.shape[0]] for k, v in goal_flags.items()}
    body_xpos = body_xpos[:T]
    site_xpos = site_xpos[:T]
    site_xmat = site_xmat[:T]
    joint_qpos = joint_qpos[:T]
    pgq = pgq[:T]
    cb1, cb2, cd, cf, ncon = cb1[:T], cb2[:T], cd[:T], cf[:T], ncon[:T]

    # joint infos (align names to joint_qpos columns)
    joints: List[JointInfo] = []
    n_j = joint_qpos.shape[1] if joint_qpos.ndim == 2 else 0
    for col in range(n_j):
        nm = joint_names[col] if col < len(joint_names) else f"joint_{col}"
        ob = joint_obj_names[col] if col < len(joint_obj_names) else ""
        ad = int(joint_qpos_addr[col]) if col < len(joint_qpos_addr) else -1
        joints.append(JointInfo(name=nm, obj=ob, col=col, addr=ad))

    return Episode(
        path=path, T=T, object_pos=object_pos, eef_pos=eef_pos, gripper_qpos=gripper_qpos,
        arm_contact_force=arm_cf, goal_flags=goal_flags,
        body_xpos=body_xpos, site_xpos=site_xpos, site_xmat=site_xmat,
        joint_qpos=joint_qpos, priv_gripper_qpos=pgq, contact_body1=cb1, contact_body2=cb2,
        contact_dist=cd, contact_force=cf, ncon=ncon, body_names=body_names, body_ids=body_ids,
        obj_body_tree=obj_body_tree, site_names=site_names, site_sizes=site_sizes, joints=joints, articulated=articulated,
        attrs=attrs,
    )


# ---------------------------------------------------------------------------------------------------------------------
# object-name -> recorded-key resolution
# ---------------------------------------------------------------------------------------------------------------------

def resolve_object_key(canonical: str, ep: Episode) -> Optional[str]:
    """Map a canonical category (e.g. 'bowl', 'alphabet_soup') to a recorded
    object_pos key (e.g. 'akita_black_bowl_1', 'alphabet_soup_1')."""
    # exact / substring against recorded object_pos keys (strip the _N suffix)
    cand = []
    for k in ep.object_pos:
        base = k.rsplit("_", 1)[0] if k[-1].isdigit() else k
        cand.append((k, base))
    # 1) canonical token appears in key
    for k, base in cand:
        if canonical in k or canonical in base:
            return k
    # 2) category word-substring (e.g. 'soup' matches 'alphabet_soup')
    canon_words = canonical.replace("_", " ").split()
    for kw in canon_words:
        for k, base in cand:
            if kw in k:
                return k
    return None
