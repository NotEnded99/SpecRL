"""The 8 atomic predicates, realized EXACTLY from recorded privileged metadata.

Each builder returns a per-step continuous robustness margin (T,) numpy array
(>0 = predicate satisfied). All quantities come from the episode's privileged
group (contacts, container contain-region sites, articulated joints) — no proxy
heuristics, no environment/MuJoCo needed.

  pick(o)     = min( contact_force(o,gripper) - f_eps , q_g_max - gripper_width ,
                      r_grasp - dist(eef, o) )                      [continuous force sum]
  On(o,t)     = min( z_o - z_t - eps_z , dxy_tol - dxy , contact_radius - dist3d(o,t) )
  Stack(o,t)  = min( contact_margin(o,t) , contain_margin(o,t) , z_o - z_t )
  In(o,c)     = min( contact_margin(o,c) , contain_margin(o,c) )
  open(a)     = norm_travel(a) - theta_art            [articulated, data-driven]
  close(a)    = norm_travel(a) - theta_art
  turn_on(s)  = norm_travel(s) - theta_art
  turn_off(s) = norm_travel(s) - theta_art

Articulated margin is the joint's travel from its start, normalized by its full
observed range, minus a threshold — continuous and correct for both open (closed
->open) and close (open->closed) since both move away from the start value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .loader import Episode, JointInfo, resolve_object_key


# ---------------------------------------------------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class PredConfig:
    # pick
    f_eps: float = 0.05           # N; small so the proximity term carries the approach gradient
    q_g_max: float = 0.08         # m; fully-open gripper width
    r_grasp: float = 0.07         # m; grasp-reach radius (eef-to-object-centre); large enough
                                  #      that big/thin objects (book, mug) whose centre is far from
                                  #      the grip point don't get vetoed when force+closure are positive
    # On / contact_margin  — LIBERO check_ontop uses a GLOBAL 0.03 xy threshold
    # (see libero/envs/object_states/base_object_states.py: ||pos_xy diff|| < 0.03);
    # not per-scene. Matching it removes the zero-point offset vs goal_flags/on.
    eps_z: float = 0.0            # m; LIBERO ontop requires obj_z >= target_z (dz >= 0)
    dxy_tol: float = 0.03         # m; LIBERO's exact ontop xy constant
    contact_radius: float = 0.06  # m; proximity margin (fallback only, not used by On)
    # articulated
    theta_art: float = 0.75       # joint must reach 75% of its DESIGN travel to count as achieved
    articulated_abs_gate: float = 0.05   # m/rad absolute travel gate when no design range is recorded
    art_relax: float = 0.1        # fraction of the range width to relax the near-boundary toward
                                  #      rest: LIBERO's actual goal trigger sits a few % inside the
                                  #      recorded design range (drawer -0.14 vs trigger -0.136,
                                  #      stove 0.5 vs 0.494); matters for turn_on/turn_off which
                                  #      have no goal_flag to calibrate against.


# ---------------------------------------------------------------------------------------------------------------------
# raw feature helpers
# ---------------------------------------------------------------------------------------------------------------------

def gripper_width(ep: Episode) -> np.ndarray:
    return ep.priv_gripper_qpos[:, 0] - ep.priv_gripper_qpos[:, 1]


def eef_object_dist(ep: Episode, obj_key: str) -> np.ndarray:
    o = ep.object_pos[obj_key]
    return np.linalg.norm(ep.eef_pos - o, axis=1)


def contact_force_on_object(ep: Episode, obj_key: str) -> np.ndarray:
    """Continuous: sum of contact normal force on object obj's body at each step.

    Contacts store raw mujoco body ids; obj's body id is found via body_names/body_ids.
    During a grasp this is dominated by the gripper, so it realises
    contact_force(o, gripper) without needing the gripper's (unrecorded) body ids.
    """
    obj_body_id = _key_to_body_id(ep, obj_key)
    if obj_body_id is None:
        return np.zeros(ep.T, dtype=np.float64)
    T, MAX = ep.contact_body1.shape
    out = np.zeros(T, dtype=np.float64)
    for t in range(T):
        n = int(ep.ncon[t])
        if n <= 0:
            continue
        b1 = ep.contact_body1[t, :n]
        b2 = ep.contact_body2[t, :n]
        f = ep.contact_force[t, :n]
        mask = (b1 == obj_body_id) | (b2 == obj_body_id)
        out[t] = float(f[mask].sum())
    return out


def _key_to_body_id(ep: Episode, obj_key: str) -> Optional[int]:
    """obj_key (e.g. 'akita_black_bowl_1') -> its mujoco body id via body_names."""
    # exact name match first
    for i, n in enumerate(ep.body_names):
        if n == obj_key:
            return int(ep.body_ids[i])
    # substring fallback: obj_key appears inside the body name
    for i, n in enumerate(ep.body_names):
        if obj_key in n or n in obj_key:
            return int(ep.body_ids[i])
    # obj_body_tree fallback: check if obj_key is a key in obj_body_tree
    if ep.obj_body_tree:
        for name, bids in ep.obj_body_tree.items():
            if obj_key in name or name in obj_key:
                return int(bids[0]) if bids else None
    return None


def target_pos(ep: Episode, target_name: str) -> Optional[np.ndarray]:
    """Reference point (T,3) for measuring 'on / against' a target.

    * manipulable object (plate, bowl, ...): its recorded object_pos centre.
    * fixture (stove, cabinet, table, ...): prefer a representative SITE on its
      functional surface (cook_region / top_region / *_region) — the body origin
      sits at the base, so using it would put a bowl-on-cooktop ~0.2 m 'above'.
      Falls back to body_xpos if no such site exists.
    """
    # 1) recorded object pose?
    key = _resolve_in_object_pos(ep, target_name)
    if key is not None:
        return ep.object_pos[key]
    # 2) fixture: best-matching surface site
    si = _surface_site_index(ep, target_name)
    if si is not None and si < ep.site_xpos.shape[1]:
        return ep.site_xpos[:, si, :]
    # 3) fixture body pose
    bi = ep.body_index(target_name)
    if bi is not None:
        return ep.body_xpos[:, bi, :]
    return None


def _target_body_name(ep: Episode, target_name: str) -> Optional[str]:
    """The recorded body name (e.g. 'flat_stove_1') for a canonical target word."""
    for n in ep.body_names:
        if target_name in n:
            return n
    # also accept object_pos keys (objects not in body_names list still have poses)
    for k in ep.object_pos:
        if target_name in k:
            return k
    return None


def _surface_site_index(ep: Episode, target_name: str) -> Optional[int]:
    """A site ON the target's own body (cook_region / top_region). Excludes table
    init regions by requiring the site name to start with the target's body name."""
    body = _target_body_name(ep, target_name)
    if body is None:
        return None
    cands = [i for i, n in enumerate(ep.site_names) if n.startswith(body)]
    if not cands:
        return None
    for kw in ("cook", "top", "contain"):
        for i in cands:
            if kw in ep.site_names[i]:
                return i
    return cands[0]


def _resolve_in_object_pos(ep: Episode, name: str) -> Optional[str]:
    for k in ep.object_pos:
        base = k.rsplit("_", 1)[0] if k[-1].isdigit() else k
        if name in k or name in base:
            return k
    for kw in name.replace("_", " ").split():
        for k in ep.object_pos:
            if kw in k:
                return k
    return None


def contact_margin(ep: Episode, obj_key: str, target_name: str, cfg: PredConfig) -> np.ndarray:
    """contact_margin(o, x): continuous margin, positive when o touches/rests on x.

    Uses the real MuJoCo contact list (contact_dist between o's and x's bodies):
    a direct contact gives margin = -min(dist) (>0 when penetrating/touching).
    When there is no direct contact it falls back to a proximity margin against the
    target's contain-region centre (containers) or its body pose (surfaces), so the
    approach phase still gets a continuous gradient.
    """
    o = ep.object_pos[obj_key]
    obj_body = _key_to_body_id(ep, obj_key)
    tgt_idx = ep.body_index(target_name)
    tgt_body = int(ep.body_ids[tgt_idx]) if tgt_idx is not None and ep.body_ids is not None else None

    # proximity fallback when no direct contact:
    #   * container (own contain-region site): over-opening margin min(hx-|lx|,hy-|ly|)
    #     — positive whenever o is above the opening; ignores the z offset of a tall
    #     jar sitting inside (its centre is above the region centre);
    #   * surface (plate/stove): contact_radius - dist3d(o, target_pos).
    si = _contain_site_index(ep, target_name)
    if si is not None and si < ep.site_xpos.shape[1] and ep.site_sizes is not None \
            and si < ep.site_sizes.shape[0]:
        local = _region_local(ep, obj_key, si)
        half = ep.site_sizes[si]
        cm = np.minimum(half[0] - np.abs(local[:, 0]), half[1] - np.abs(local[:, 1]))
    else:
        ref = target_pos(ep, target_name)
        if ref is None:
            return np.full(ep.T, -cfg.contact_radius, dtype=np.float64)
        cm = cfg.contact_radius - np.linalg.norm(o - ref, axis=1)

    if obj_body is None or tgt_body is None:
        return cm
    for t in range(ep.T):
        n = int(ep.ncon[t])
        if n <= 0:
            continue
        b1 = ep.contact_body1[t, :n]; b2 = ep.contact_body2[t, :n]; d = ep.contact_dist[t, :n]
        mask = ((b1 == obj_body) & (b2 == tgt_body)) | ((b1 == tgt_body) & (b2 == obj_body))
        if mask.any():
            cm[t] = -float(d[mask].min())     # touching/penetrating -> >=0
    return cm


def _region_local(ep: Episode, obj_key: str, si: int) -> np.ndarray:
    """Object pose in the contain-region site frame (T,3)."""
    o = ep.object_pos[obj_key]
    center = ep.site_xpos[:, si, :]
    R = ep.site_xmat[:, si, :, :]
    return np.einsum("tij,tj->ti", R.transpose(0, 2, 1), o - center)


def _site_index_exact(ep: Episode, site_name: str) -> Optional[int]:
    for i, n in enumerate(ep.site_names):
        if n == site_name:
            return i
    return None


def site_under_margin(ep: Episode, obj_key: str, site_name: str) -> np.ndarray:
    """EXACT replica of LIBERO SiteObject.under() (the check_ontop for site targets),
    as a continuous margin. under(): with delta = R@(obj-site), obj is 'on top' iff
    |delta_xy| < site.size_xy AND site.size_z-0.005 < delta_z < site.size_z+0.10.
    Used for 'On X <site>' goals (bowl on stove cook_region, bottle on cabinet top_side).
    """
    si = _site_index_exact(ep, site_name)
    if si is None or ep.site_sizes is None or si >= ep.site_sizes.shape[0]:
        return np.full(ep.T, -1.0, dtype=np.float64)
    o = ep.object_pos[obj_key]
    center = ep.site_xpos[:, si, :]
    R = ep.site_xmat[:, si, :, :]
    size = ep.site_sizes[si]
    delta = np.einsum("tij,tj->ti", R, o - center)              # obj in site frame
    xy = np.minimum(size[0] - np.abs(delta[:, 0]), size[1] - np.abs(delta[:, 1]))
    zlo = delta[:, 2] - (size[2] - 0.005)
    zhi = (size[2] + 0.10) - delta[:, 2]
    return np.minimum(xy, np.minimum(zlo, zhi))


def goal_atom_margin(ep: Episode, pred: str, obj: Optional[str], target: Optional[str],
                     cfg: PredConfig) -> Tuple[np.ndarray, str]:
    """Realise ONE bddl goal atom's predicate, dispatching on whether the target is a
    SITE (under/in_box) or an OBJECT (check_ontop). Returns (margin, atom_name)."""
    obj_key = resolve_object_key(obj, ep) if obj else None
    if pred in ("open", "close"):
        # target is a region site whose parent is the articulated body
        j = ep.joint_by_level(_level_from_site(target)) if target else ep.joint_by_level(None)
        m = articulated_margin(ep, j, pred, cfg)
        return m, f"{pred}({target or j})"
    if pred in ("turnon", "turnoff"):
        j = _rotary_joint(ep)
        verb = "turn_on" if pred == "turnon" else "turn_off"
        return articulated_margin(ep, j, verb, cfg), f"{pred}({target})"
    if obj_key is None:
        return np.full(ep.T, -1.0, dtype=np.float64), f"{pred}({obj},{target})"
    if pred == "in":
        # in(obj, site) -> in_box against the exact site
        return _in_box_site(ep, obj_key, target), f"in({obj_key},{target})"
    if pred == "on":
        # on(obj, site) -> site under(); on(obj, object) -> check_ontop (0.03+contact)
        si = _site_index_exact(ep, target) if target else None
        if si is not None:
            return site_under_margin(ep, obj_key, target), f"on({obj_key},{target})"
        return pred_on(ep, obj_key, target, cfg), f"on({obj_key},{target})"
    return np.full(ep.T, -1.0, dtype=np.float64), f"{pred}({obj},{target})"


def _level_from_site(site_name: Optional[str]) -> Optional[str]:
    if not site_name:
        return None
    for lv in ("top", "middle", "bottom"):
        if lv in site_name:
            return lv
    return None


def _in_box_site(ep: Episode, obj_key: str, site_name: Optional[str]) -> np.ndarray:
    """in_box against one exact site (LIBERO SiteObject.in_box)."""
    si = _site_index_exact(ep, site_name) if site_name else None
    if si is None or ep.site_sizes is None or si >= ep.site_sizes.shape[0]:
        return np.full(ep.T, -1.0, dtype=np.float64)
    o = ep.object_pos[obj_key]
    center = ep.site_xpos[:, si, :]
    R = ep.site_xmat[:, si, :, :]
    size = ep.site_sizes[si]
    total = np.abs(np.einsum("tij,j->ti", R, size))
    low = o - (center - total); low[:, 2] += 0.01
    high = (center + total) - o
    return np.minimum(low.min(axis=1), high.min(axis=1))


def _all_contain_sites(ep: Episode, container_name: str) -> List[int]:
    """All contain-region sites of a container (e.g. a cabinet has top/middle/bottom
    drawer regions); the object may be inside any of them."""
    canon = _target_body_name(ep, container_name) or container_name
    cands = [i for i, n in enumerate(ep.site_names) if n.startswith(canon)
             and ("region" in n or "contain" in n)]
    return cands


def contain_margin(ep: Episode, obj_key: str, container_name: str) -> np.ndarray:
    """Containment margin = max over the container's region sites (the object is inside
    whichever drawer/opening it's in). xy strict, z permissive (tall jars). Non-site
    containers fall back to xy proximity."""
    sis = _all_contain_sites(ep, container_name)
    sis = [s for s in sis if ep.site_sizes is not None and s < ep.site_sizes.shape[0]
           and s < ep.site_xpos.shape[1]]
    if not sis:
        tp = target_pos(ep, container_name)
        if tp is None:
            return np.zeros(ep.T, dtype=np.float64)
        o = ep.object_pos[obj_key]
        return 0.10 - np.linalg.norm(o[:, :2] - tp[:, :2], axis=1)
    o = ep.object_pos[obj_key]
    best = np.full(ep.T, -1e9, dtype=np.float64)
    for si in sis:
        half = ep.site_sizes[si]
        local = _region_local(ep, obj_key, si)
        m = np.minimum(np.minimum(half[0] - np.abs(local[:, 0]),
                                  half[1] - np.abs(local[:, 1])),
                       half[2] + local[:, 2])
        best = np.maximum(best, m)
    return best
    si = _contain_site_index(ep, container_name)
    if si is None or ep.site_sizes is None or si >= ep.site_sizes.shape[0]:
        # no region site -> degrade to a wide xy proximity around the container pose
        tp = target_pos(ep, container_name)
        if tp is None:
            return np.zeros(ep.T, dtype=np.float64)
        o = ep.object_pos[obj_key]
        dxy = np.linalg.norm(o[:, :2] - tp[:, :2], axis=1)
        return 0.10 - dxy
    o = ep.object_pos[obj_key]
    half = ep.site_sizes[si]                             # (3,)
    local = _region_local(ep, obj_key, si)
    # xy: strict containment within the opening; z: permissive upward (a tall jar's
    # center sits above the rim when correctly placed inside), only penalize if the
    # object drops below the receptacle floor (lz < -rz).
    return np.minimum(np.minimum(half[0] - np.abs(local[:, 0]),
                                 half[1] - np.abs(local[:, 1])),
                      half[2] + local[:, 2])


def _contain_site_index(ep: Episode, container_name: str) -> Optional[int]:
    """The container's own receptacle site (e.g. 'basket_1_contain_region',
    'wine_rack_1_top_region'). Requires the site to start with the container's
    recorded body name so table/floor init regions are not picked."""
    body = _target_body_name(ep, container_name)
    if body is None:
        return None
    cands = [i for i, n in enumerate(ep.site_names) if n.startswith(body)
             and ("region" in n or "contain" in n)]
    if not cands:
        return None
    for kw in ("contain", "top", "middle"):
        for i in cands:
            if kw in ep.site_names[i]:
                return i
    return cands[0]


# ---------------------------------------------------------------------------------------------------------------------
# articulated (open / close / turn_on / turn_off) — unified data-driven margin
# ---------------------------------------------------------------------------------------------------------------------

def _articulated_range(ep: Episode, obj_name: str, verb: str) -> Optional[List[float]]:
    """The design joint range ([r0,r1]) for verb on object, from articulated_info.

    articulated_info rows: [name, n_joints, open_rng, close_rng, turnon_rng, turnoff_rng].
    """
    key = {"open": 2, "close": 3, "turn_on": 4, "turn_off": 5}[verb]
    for row in ep.articulated:
        if len(row) > key and row[0] and obj_name and obj_name in str(row[0]):
            rng = row[key]
            if isinstance(rng, (list, tuple)) and len(rng) == 2:
                return [float(rng[0]), float(rng[1])]
    return None


def articulated_margin(ep: Episode, joint: Optional[JointInfo], verb: str,
                       cfg: PredConfig) -> np.ndarray:
    """EXACT replica of LIBERO is_open/is_close/turn_on/turn_off as a continuous margin.
    Source (articulated_objects.py), range = default_<verb>_ranges [r0,r1]:
      is_open  : max(r)<=0 -> q<max(r) ; else q>min(r)
      is_close : min(r)>=0 -> q>min(r) ; else q<max(r)
      turn_on  : q >= min(turnon_range)
      turn_off : q <  max(turnoff_range)
    margin = signed distance to the threshold (>0 iff the predicate holds). No relax.
    """
    if joint is None or ep.joint_qpos is None or ep.joint_qpos.shape[1] == 0:
        return np.zeros(ep.T, dtype=np.float64)
    q = ep.joint_qpos[:, joint.col]
    rng = _articulated_range(ep, joint.obj, verb)
    if rng is None:
        return np.abs(q - float(q[0])) - cfg.articulated_abs_gate
    r0, r1 = float(rng[0]), float(rng[1])
    if verb == "open":
        return (max(r0, r1) - q) if max(r0, r1) <= 0 else (q - min(r0, r1))
    if verb == "close":
        # close direction is OPPOSITE to open: determine from the OPEN range sign.
        open_rng = _articulated_range(ep, joint.obj, "open")
        opens_negative = open_rng is not None and max(float(open_rng[0]), float(open_rng[1])) <= 0
        if opens_negative:
            return q - min(r0, r1)      # is_close: q > min(close_range)
        else:
            return max(r0, r1) - q      # is_close: q < max(close_range)
    if verb == "turn_on":
        return q - min(r0, r1)
    if verb == "turn_off":
        return max(r0, r1) - q
    return np.zeros(ep.T, dtype=np.float64)


# ---------------------------------------------------------------------------------------------------------------------
# the 8 predicate builders — each (ep, step, cfg) -> (T,) margin
# ---------------------------------------------------------------------------------------------------------------------

def pred_pick(ep: Episode, obj_key: str, cfg: PredConfig) -> np.ndarray:
    """pick(o) = min( contact_force(o,gripper) - f_eps ,
                     q_g_max - gripper_width ,
                     r_grasp - dist(eef, o) )   (3-term: force + closure + proximity)
    The proximity term supplies a continuous gradient during approach: contact_force is
    exactly 0 until the gripper touches the object, so without it the predicate is a flat
    -f_eps floor for the whole pre-contact phase.
    """
    f = contact_force_on_object(ep, obj_key)
    w = gripper_width(ep)
    prox = cfg.r_grasp - eef_object_dist(ep, obj_key)
    return np.minimum(np.minimum(f - cfg.f_eps, cfg.q_g_max - w), prox)


def _body_set(ep: Episode, name: str) -> Optional[set]:
    """The set of mujoco body ids belonging to an object/body-name (root + composite
    children), from the recorded obj_body_tree."""
    if not ep.obj_body_tree:
        return None
    if name in ep.obj_body_tree:
        return set(ep.obj_body_tree[name])
    for k, v in ep.obj_body_tree.items():              # substring fallback (canonical -> key)
        if name in k:
            return set(v)
    return None


def _tree_contact_dist(ep: Episode, obj_key: str, target_name: str) -> np.ndarray:
    """Min MuJoCo contact_dist between obj and target per step, using obj_body_tree so
    composite-object contacts (geoms on child bodies) are matched EXACTLY as robosuite's
    check_contact does. inf where no contact. sign(-min_dist) == has_contact."""
    out = np.full(ep.T, np.inf, dtype=np.float64)
    a = _body_set(ep, obj_key) or ({int(_key_to_body_id(ep, obj_key))} if _key_to_body_id(ep, obj_key) is not None else set())
    b = _body_set(ep, target_name)
    if b is None:
        bi = ep.body_index(target_name)
        b = {int(ep.body_ids[bi])} if bi is not None else set()
    if not a or not b:
        return out
    for t in range(ep.T):
        n = int(ep.ncon[t])
        if n <= 0:
            continue
        b1 = ep.contact_body1[t, :n]; b2 = ep.contact_body2[t, :n]; d = ep.contact_dist[t, :n]
        # one body from a, the other from b (tree-matched -> exact check_contact)
        pair = ((np.isin(b1, list(a)) & np.isin(b2, list(b))) |
                (np.isin(b1, list(b)) & np.isin(b2, list(a))))
        if pair.any():
            out[t] = float(d[pair].min())
    return out


def pred_on(ep: Episode, obj_key: str, target_name: str, cfg: PredConfig) -> np.ndarray:
    """LIBERO check_ontop(target, obj): target_z <= obj_z  AND  contact  AND  ||xy||<0.03.
    Continuous margin = min(obj_z - target_z, 0.03 - dxy, contact_term). The contact term
    is >=0 ONLY on a real (tree-matched) contact — bit-exact to robosuite's check_contact;
    off-contact it is <=0 (clamped proximity) so it never falsely satisfies 'on'. The 0.03
    and dz>=0 are robosuite's exact constants."""
    o = ep.object_pos[obj_key]
    tp = target_pos(ep, target_name)
    if tp is None:
        return np.full(ep.T, -1.0, dtype=np.float64)
    dz = o[:, 2] - tp[:, 2]                          # obj above target (LIBERO: target_z <= obj_z)
    dxy = np.linalg.norm(o[:, :2] - tp[:, :2], axis=1)
    cd = _tree_contact_dist(ep, obj_key, target_name)
    # robosuite check_contact returns True if the object pair is in MuJoCo's contact
    # list AT ALL (MuJoCo lists near-touching pairs with dist up to a small margin, so
    # dist can be slightly +). So contact_term is >=0 whenever a tree-matched contact
    # exists (max(0,-dist)), else the clamped proximity gradient (<=0).
    real = np.where(np.isinf(cd), -np.inf, np.maximum(0.0, -cd))
    prox = np.minimum(0.0, cfg.contact_radius - np.linalg.norm(o - tp, axis=1))  # <=0 gradient
    contact_term = np.maximum(real, prox)
    return np.minimum(np.minimum(dz - cfg.eps_z, cfg.dxy_tol - dxy), contact_term)


def pred_stack(ep: Episode, obj_key: str, target_name: str, cfg: PredConfig) -> np.ndarray:
    o = ep.object_pos[obj_key]
    tp = target_pos(ep, target_name)
    dz = (o[:, 2] - tp[:, 2]) if tp is not None else np.full(ep.T, -1.0)
    cm = contact_margin(ep, obj_key, target_name, cfg)
    cont = contain_margin(ep, obj_key, target_name)
    return np.minimum(np.minimum(cm, cont), dz)


def pred_in(ep: Episode, obj_key: str, container_name: str, cfg: PredConfig) -> np.ndarray:
    """LIBERO In = check_contact AND check_contain. For site containers check_contact is
    always True, so In == contain_margin; we also AND contact_margin (the over-opening
    xy proxy) as a belt-and-suspenders gate that helps borderline placements."""
    return np.minimum(contact_margin(ep, obj_key, container_name, cfg),
                     contain_margin(ep, obj_key, container_name))


def pred_open(ep: Episode, level: Optional[str], cfg: PredConfig) -> np.ndarray:
    j = ep.joint_by_level(level)
    return articulated_margin(ep, j, "open", cfg)


def pred_close(ep: Episode, level: Optional[str], cfg: PredConfig) -> np.ndarray:
    j = ep.joint_by_level(level)
    return articulated_margin(ep, j, "close", cfg)


def pred_turn_on(ep: Episode, cfg: PredConfig) -> np.ndarray:
    j = _rotary_joint(ep)
    return articulated_margin(ep, j, "turn_on", cfg)


def pred_turn_off(ep: Episode, cfg: PredConfig) -> np.ndarray:
    j = _rotary_joint(ep)
    return articulated_margin(ep, j, "turn_off", cfg)


def _rotary_joint(ep: Episode) -> Optional[JointInfo]:
    """The stove knob/button joint (obj name contains 'stove')."""
    js = [j for j in ep.joints if "stove" in j.obj]
    if js:
        return js[0]
    return ep.joints[0] if ep.joints else None
