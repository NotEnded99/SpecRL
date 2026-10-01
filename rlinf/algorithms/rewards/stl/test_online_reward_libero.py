#!/usr/bin/env python
# Copyright 2025 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License").
"""End-to-end correctness test for the ONLINE STL reward on REAL libero state.

Single process, DIRECT ``env.sim`` access (no subprocess workers). Rolls out N
episodes with either the real OpenVLA-OFT policy or a random policy, and at every
step computes the STL reward with the *same* machinery as ``LiberoSTLRewardEnv``:

    _extract_privileged_state(env) -> _build_step_episode -> predicates ->
        OnlineRobustnessTracker -> r_t = sparse_bonus + scale * (gamma*rho_t - rho_{t-1})

After each episode it INDEPENDENTLY recomputes the prefix robustness with
``fast_online.online_prefix(...)["true"]`` over the recorded per-atom margin
arrays and asserts the incremental tracker matched it bit-for-bit. This is the
real-data analogue of ``tests/test_stl_tracker_vs_fastonline.py`` (which uses
random streams) — it validates the full extraction->Episode->predicate->tracker
pipeline, not just the tracker maths.

Reward correctness is INDEPENDENT of policy quality, so ``--policy random`` is a
perfectly valid way to run this test without loading the model.

Run (from the RLinf repo, in an env with libero + the openvla repo deps):
    OPENVLA_REPO=/path/to/openvla \
    OPENVLA_MODEL_PATH=/path/to/openvla-checkpoint \
    python rlinf/algorithms/rewards/stl/test_online_reward_libero.py \
        --policy model \
        --task_suite libero_spatial --task_id 0 --num_episodes 10

Defaults target the OpenVLA-OFT SFT model on libero_spatial:
  --model_path "$OPENVLA_MODEL_PATH"
  --task_suite  libero_spatial   (--max_steps 220, the libero_spatial cap)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import types
from types import SimpleNamespace

import numpy as np

# matplotlib imported lazily inside plot_episode() so the script runs headless
# without it (and so --no_plot never pays the import cost).

# --- make the RLinf repo root importable (so `import rlinf` works when this ---
# --- script is run by path from anywhere) -----------------------------------
_RLINF_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _RLINF_ROOT not in sys.path:
    sys.path.insert(0, _RLINF_ROOT)

# --- make the openvla repo importable for libero_utils (get_libero_env, ...).
# IMPORTANT: append, do NOT insert at front — the openvla repo ships a PARTIAL
# `prismatic` package (no vla/constants.py) that would shadow the .venv's complete
# one and break RLinf's OpenVLA-OFT loader (`from prismatic.vla.constants import ...`).
OPENVLA_REPO = os.environ.get("OPENVLA_REPO")
if not OPENVLA_REPO:
    raise RuntimeError("OPENVLA_REPO must point to the OpenVLA checkout")
if OPENVLA_REPO not in sys.path:
    sys.path.append(OPENVLA_REPO)

from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_dummy_action,
    get_libero_env,
    quat2axisangle,
)
from libero.libero import benchmark  # noqa: E402

# NOTE: the openvla-repo model helpers (get_model/get_action/get_processor/...) are
# imported LAZILY inside load_policy()'s model branch, because importing them pulls
# in prismatic -> tensorflow_datasets, whose eager import of hundreds of dataset
# modules is extremely slow when site-packages is on FUSE. --policy random skips it.

# --- the STL reward machinery under test (heavy rlinf import; fine in the run env) ---
from rlinf.algorithms.rewards.stl.stlcg_task_monitor import fast_online  # noqa: E402
from rlinf.algorithms.rewards.stl.stlcg_task_monitor.loader import resolve_object_key  # noqa: E402
from rlinf.algorithms.rewards.stl.stlcg_task_monitor.predicates import (  # noqa: E402
    PredConfig,
    goal_atom_margin,
    pred_pick,
)
from rlinf.envs.libero.libero_env_stl_reward import (  # noqa: E402
    _GEOMETRIC_PREDS,
    OnlineRobustnessTracker,
    _build_step_episode,
    _normalize_goal_atoms,
)
from rlinf.envs.libero.venv import _extract_privileged_state, _get_goal_atoms  # noqa: E402


# -------------------------------------------------------------------------------------------------


def compute_step_margins(ep, goal_atoms, grasp_objs, pred_cfg):
    """One state -> (grasp_margins, goal_margins) lists, len == #subformulas."""
    grasp: list = []
    goal: list = []
    for (pred, obj, target), gobj in zip(goal_atoms, grasp_objs):
        gm, _name = goal_atom_margin(ep, pred, obj, target, pred_cfg)
        goal.append(float(np.asarray(gm, dtype=np.float64).reshape(-1)[-1]))
        if gobj is not None:
            gk = resolve_object_key(gobj, ep)
            if gk is not None:
                pm = pred_pick(ep, gk, pred_cfg)
                grasp.append(float(np.asarray(pm, dtype=np.float64).reshape(-1)[-1]))
            else:
                grasp.append(None)
        else:
            grasp.append(None)
    return grasp, goal


def _resolve_unnorm_key(norm_stats: dict, task_suite: str) -> str:
    """Pick the norm_stats key matching this suite. OFT checkpoints store stats
    under per-dataset keys like 'libero_90_no_noops_trajall' / 'libero_spatial_no_noops',
    while config.json often only carries a stale placeholder (e.g. 'libero_10')."""
    if not norm_stats:
        return task_suite
    matches = [k for k in norm_stats if task_suite in k]
    if matches:
        return matches[0]
    libero_keys = [k for k in norm_stats if "libero" in k]
    if libero_keys:
        return libero_keys[0]
    return next(iter(norm_stats))


def _inject_dataset_stats(model, model_path: str) -> None:
    """OFT checkpoints put the REAL action norm stats in dataset_statistics.json
    (top-level key = dataset name, e.g. {'libero_90_no_noops_trajall': {action,proprio}}).
    The openvla-repo loader only reads config.json's norm_stats (often a stale
    placeholder), so merge the real stats into model.norm_stats here."""
    stats_path = os.path.join(model_path, "dataset_statistics.json")
    if not os.path.isfile(stats_path):
        return
    with open(stats_path) as f:
        ds = json.load(f)
    ns = ds.get("norm_stats", ds) if isinstance(ds, dict) else None
    if isinstance(ns, dict) and ns:
        model.norm_stats.update(ns)


def _patch_openvla_config_for_oft() -> None:
    """openvla-repo's get_vla loads the OFT checkpoint's bundled modeling_prismatic.py
    (trust_remote_code), which reads config.use_proprio / config.proprio_dim. The
    installed (older) prismatic OpenVLAConfig + the checkpoint's config.json don't
    define them -> AttributeError. Monkeypatch OpenVLAConfig.__init__ to add safe
    OFT defaults on every instance (idempotent)."""
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    if getattr(OpenVLAConfig, "_oft_patched", False):
        return
    orig_init = OpenVLAConfig.__init__

    def patched_init(self, *a, **kw):
        orig_init(self, *a, **kw)
        # OFT turned proprioception OFF for these SFT checkpoints; proprio_dim only
        # matters when use_proprio is True.
        if not hasattr(self, "use_proprio"):
            self.use_proprio = False
        if not hasattr(self, "proprio_dim"):
            self.proprio_dim = 0

    OpenVLAConfig.__init__ = patched_init
    OpenVLAConfig._oft_patched = True


def load_policy(args):
    """Return a callable obs->action_chunk, shape (num_action_chunks, 7) np.float64.
    Random policy returns (1, 7) per call. The model branch loads via RLinf's OWN
    OpenVLA-OFT loader + predict_action_batch (the same path the official eval uses),
    NOT the openvla repo's get_vla — the latter mishandles OFT action chunking
    (ALOHA (25,14) shape bug) and yields 0% success."""
    if args.policy == "random":
        rng = np.random.default_rng(args.seed)

        def random_fn(_obs, _task_desc):
            return rng.uniform(-1.0, 1.0, size=(1, 7))

        return random_fn

    if not args.model_path:
        raise RuntimeError(
            "--model_path or OPENVLA_MODEL_PATH is required for --policy model"
        )

    import torch
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.openvla_oft import get_model as get_oft_model

    oft_cfg = OmegaConf.create({
        "model_path": args.model_path,
        "model_type": "openvla_oft",
        "implement_version": args.implement_version,  # "rlinf" for RLinf LIBERO OFT ckpts
        "precision": "bf16",
        "action_dim": 7,
        "num_action_chunks": args.num_action_chunks,
        "add_value_head": False,
        "max_prompt_length": 128,
        "center_crop": True,
        "trust_remote_code": True,
        "num_images_in_input": 1,
        "use_proprio": False,
        "use_film": False,
        "unnorm_key": args.task_suite,
    })
    model = get_oft_model(oft_cfg)
    # RLinf's loader does not move to GPU / eval; do it here.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    # ensure the correct action un-norm key (loader merges dataset_statistics.json)
    model.unnorm_key = _resolve_unnorm_key(getattr(model, "norm_stats", {}), args.task_suite)
    assert getattr(model, "unnorm_key", None) in getattr(model, "norm_stats", {}), (
        f"unnorm key {model.unnorm_key!r} not in norm_stats "
        f"(keys={list(getattr(model, 'norm_stats', {}))})"
    )
    print(f"[policy] RLinf OFFT loader: unnorm_key={model.unnorm_key!r}  "
          f"num_action_chunks={model.num_action_chunks}  action_dim={model.action_dim}")

    top_k = args.top_k  # -1 means disabled (matches official eval config)

    def policy_fn(obs, task_description):
        # The rlinf OFT loader's predict_action_batch expects env_obs with torch
        # tensors: main_images (B,H,W,C) uint8, states (B,proprio_dim). It builds
        # the prompt, runs the processor, does action chunking + un-normalization,
        # and returns actions of shape (B, num_action_chunks, action_dim).
        # libero's agentview_image is upside-down; get_libero_image / RLinf's env
        # worker rotate it 180° (img[::-1, ::-1]) before the model sees it, and the
        # processor does NOT flip — so we must flip here or the policy acts on an
        # inverted view (-> 0% success).
        raw = np.asarray(obs["agentview_image"], dtype=np.uint8)[::-1, ::-1]
        img = torch.from_numpy(np.ascontiguousarray(raw)).unsqueeze(0)
        proprio = np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        )
        env_obs = {
            "task_descriptions": [task_description],
            "main_images": img,
            "states": torch.from_numpy(proprio).unsqueeze(0).float(),
        }
        with torch.no_grad():
            actions, _result = model.predict_action_batch(
                env_obs=env_obs,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_k=top_k,
                calculate_logprobs=False,
                calculate_values=False,
            )
        actions = actions[0] if not isinstance(actions, np.ndarray) else actions[0]
        return np.asarray(actions, dtype=np.float64).reshape(-1, 7)

    return policy_fn


def run_episode(env, policy_fn, task_description, initial_state, goal_atoms, grasp_objs,
                pred_cfg, args):
    """Roll out one episode; return a dict of traces + correctness flag."""
    env.reset()
    obs = env.set_init_state(initial_state)

    tracker = OnlineRobustnessTracker(args.stl_tau, len(goal_atoms))
    prev_rho = 0.0
    grasp_bufs = [[] for _ in goal_atoms]   # for offline cross-check
    goal_bufs = [[] for _ in goal_atoms]
    # per-atom named margin traces (mirror plot_traj_robustness res.atoms):
    #   key = atom predicate string, value = per-step robustness margin list.
    goal_names = [_atom_name(a) for a in goal_atoms]
    grasp_names = [f"pick({g})" if g is not None else None for g in grasp_objs]
    atom_traces: dict[str, list] = {}
    for nm in goal_names + [n for n in grasp_names if n]:
        atom_traces.setdefault(nm, [])
    rho_trace: list = []
    shaping_trace: list = []
    reward_trace: list = []
    sparse_trace: list = []
    frames: list = []  # upright agentview per step (for rollout video)

    success = False
    # settle steps: let objects drop; do NOT feed the tracker (mirrors env reset settling)
    dummy = get_libero_dummy_action("openvla")
    for _ in range(args.num_steps_wait):
        obs, _r, _d, _i = env.step(dummy)

    t = 0
    while t < args.max_steps:
        # policy_fn returns an action CHUNK of shape (num_action_chunks, action_dim);
        # for a random policy it is a single action (1, 7). OpenVLA-OFT is trained to
        # predict a whole chunk, so we execute it open-loop (the official eval does the
        # same via the env's chunk_step) — this is what gives correct success rates.
        chunk = policy_fn(obs, task_description)
        chunk = np.asarray(chunk, dtype=np.float64).reshape(-1, 7)
        for action in chunk:
            if t >= args.max_steps:
                break
            obs, reward, done, info = env.step(action.tolist())
            frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))

            # --- online STL reward on the live post-step privileged state ---
            priv = _extract_privileged_state(env)
            ep = _build_step_episode(priv)
            grasp_margins, goal_margins = compute_step_margins(ep, goal_atoms, grasp_objs, pred_cfg)
            rho_t = tracker.step(grasp_margins, goal_margins)
            shaping = args.stl_gamma * rho_t - prev_rho
            prev_rho = rho_t

            sparse = args.reward_coef * float(done)
            step_reward = sparse + args.stl_reward_scale * shaping

            # record per-atom robustness margins (named) + offline cross-check buffers
            for k, g in enumerate(goal_margins):
                grasp_bufs[k].append(grasp_margins[k])
                goal_bufs[k].append(g)
                atom_traces[goal_names[k]].append(g)               # goal atom margin
                if grasp_names[k] is not None:
                    atom_traces[grasp_names[k]].append(grasp_margins[k])  # grasp atom margin
            rho_trace.append(rho_t)
            shaping_trace.append(shaping)
            reward_trace.append(step_reward)
            sparse_trace.append(sparse)

            success = success or bool(done)
            t += 1
            if done:
                break
        if done:
            break

    # --- INDEPENDENT offline recomputation of prefix robustness ---
    rho_trace = np.asarray(rho_trace, dtype=np.float64)
    correct = True
    ref_rho = None
    if len(goal_bufs) > 0 and len(goal_bufs[0]) > 0:
        ref_rho = fast_online.online_prefix(
            [np.asarray(b, dtype=np.float64) for b in grasp_bufs],
            [np.asarray(b, dtype=np.float64) for b in goal_bufs],
            args.stl_tau,
        )["true"]
        correct = bool(np.allclose(rho_trace, ref_rho, atol=1e-9, rtol=1e-8))

    # per-atom margin arrays + first-step-above-zero (onset), like plot_traj res.atom_steps
    atoms = {nm: np.asarray(tr, dtype=np.float64) for nm, tr in atom_traces.items() if len(tr) == t}
    atom_onsets = {nm: int(np.where(tr >= 0.0)[0][0]) for nm, tr in atoms.items()
                   if (tr >= 0.0).any()}

    return {
        "success": success,
        "steps": t,
        "rho_trace": rho_trace,
        "ref_rho": ref_rho,
        "correct": correct,
        "reward_trace": np.asarray(reward_trace, dtype=np.float64),
        "shaping_trace": np.asarray(shaping_trace, dtype=np.float64),
        "sparse_trace": np.asarray(sparse_trace, dtype=np.float64),
        "goal_bufs": goal_bufs,
        "grasp_bufs": grasp_bufs,
        "atoms": atoms,                 # name -> (T,) per-step robustness margin
        "atom_onsets": atom_onsets,     # name -> first step with margin >= 0
        "frames": frames,               # list of upright agentview uint8 frames
    }


def summarize(ep_name, res, goal_atoms):
    rho = res["rho_trace"]
    shaping = res["shaping_trace"]
    goal_all = np.concatenate([np.asarray(b) for b in res["goal_bufs"] if len(b)]) if res["goal_bufs"] else np.array([])
    monotonic = bool(np.all(np.diff(rho) >= -1e-9)) if len(rho) > 1 else True
    dense_frac = float(np.mean(np.abs(shaping) > 1e-6)) if len(shaping) else 0.0
    print(f"\n=== {ep_name} ===")
    print(f"  success={res['success']}  steps={res['steps']}")
    print(f"  online==offline_rho: {res['correct']}  "
          f"(max|diff|={np.max(np.abs(rho-res['ref_rho'])) if res['ref_rho'] is not None else float('nan'):.2e})")
    print(f"  rho monotonic_nondec: {monotonic}   rho[0]={rho[0]:.4f}  rho[-1]={rho[-1]:.4f}")
    print(f"  dense shaping frac (|shaping|>1e-6): {dense_frac:.3f}")
    print(f"  reward: sum={res['reward_trace'].sum():.3f}  sparse_sum={res['sparse_trace'].sum():.2f}  "
          f"shaping_sum={shaping.sum():.3f}")
    if goal_all.size:
        print(f"  goal margins: min={goal_all.min():.3f} max={goal_all.max():.3f} "
              f"mean={goal_all.mean():.3f} (finite={np.isfinite(goal_all).all()})")
    print(f"  #subformulas(k)={len(goal_atoms)}")
    # per-atom robustness (online margin trace), mirroring plot_traj_robustness res.atoms
    print(f"  per-atom robustness (margin over the episode):")
    for nm, tr in res.get("atoms", {}).items():
        on = res.get("atom_onsets", {}).get(nm)
        on_s = f"onset>0@t={on}" if on is not None else "never>=0"
        sat = "SAT" if tr[-1] >= 0 else "unsat"
        print(f"    {nm:32s} start={tr[0]:+.3f} end={tr[-1]:+.3f} "
              f"max={tr.max():+.3f} [{sat}, {on_s}]")
    return res["correct"] and monotonic


# -------------------------------------------------------------------------------------------------
# plotting (style adapted from experiments/robot/libero/plot_traj_robustness.py)
# -------------------------------------------------------------------------------------------------


def _atom_name(atom) -> str:
    pred, obj, target = atom
    if obj is None:
        return f"{pred}({target})"
    return f"{pred}({obj},{target})"


def _first_above(sig: np.ndarray, thr: float = 0.0):
    idx = np.where(np.asarray(sig) >= thr)[0]
    return int(idx[0]) if len(idx) else None


def plot_episode(out_path, res, goal_atoms, task_description, ep_idx, tau):
    """3-panel figure for one episode:
      (1) per-subformula atomic predicate margins (pick / on / in / ...) vs step
      (2) online prefix rho vs offline fast_online rho (must overlap) + success mark
      (3) reward decomposition: sparse bonus, STL shaping, total reward
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = len(res["rho_trace"])
    t = np.arange(T)
    if T == 0:
        return

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(11, 9), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.15, 0.85]},
    )

    # ---- panel 1: per-atom predicate robustness margins -----------------------
    # each atomic predicate (pick / on / in / open / ...) gets its own trace,
    # annotated with its first-above-zero onset (mirrors plot_traj_robustness).
    ax1.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    cmap = plt.get_cmap("tab10")
    atoms = res.get("atoms", {})
    # order: grasp (pick) atoms first, then goal atoms — derived only from what's
    # present in res["atoms"] (no dependence on main()'s locals).
    order = [nm for nm in atoms if nm.startswith("pick(")]
    order += [nm for nm in (_atom_name(a) for a in goal_atoms) if nm in atoms]
    seen = set()
    li = 0
    for nm in order:
        if nm in seen or nm not in atoms:
            continue
        seen.add(nm)
        tr = atoms[nm]
        is_pick = nm.startswith("pick(")
        ax1.plot(t, tr, color=cmap(li % 10), lw=1.8,
                 ls=":" if is_pick else "-", label=rf"$\mu_{{{nm}}}(t)$")
        li += 1
        on = res.get("atom_onsets", {}).get(nm)
        if on is not None:
            ax1.axvline(on, color=cmap((li - 1) % 10), ls=":", lw=0.9, alpha=0.5)
            ax1.annotate(f"{nm}\n>=0 @t={on}", (on, 0.0),
                         textcoords="offset points", xytext=(4, 6),
                         fontsize=7, color=cmap((li - 1) % 10))
    ax1.set_ylabel("per-atom margin\n(>0 satisfied)")
    ax1.set_title("Per-atom predicate robustness  μ(t)  (live privileged state)")
    ax1.legend(loc="best", fontsize=7.5, ncol=2)
    ax1.grid(alpha=0.25)

    # ---- panel 2: online vs offline robustness --------------------------------
    ax2.axhline(0.0, color="k", lw=0.9, alpha=0.6, label="satisfaction boundary")
    ax2.plot(t, res["rho_trace"], color="crimson", lw=2.4,
             label=r"online $\rho(\phi,\,s[0:t+1])$  (incremental tracker)")
    if res["ref_rho"] is not None:
        ax2.plot(t, res["ref_rho"], color="navy", lw=1.2, ls="--",
                 label=r"offline $\rho$  (fast_online recomputed)")
    max_diff = (float(np.max(np.abs(res["rho_trace"] - res["ref_rho"])))
                if res["ref_rho"] is not None else float("nan"))
    # success / first-sat annotation
    success_step = (T - 1) if res["success"] else None
    if success_step is not None:
        ax2.axvline(success_step, color="green", ls="-.", lw=1.0, alpha=0.7)
        ax2.annotate(f"success\n@ t={success_step}", (success_step, 0.0),
                     textcoords="offset points", xytext=(6, 10), fontsize=8, color="green")
    else:
        # show where the whole-spec robustness first crossed 0 (if ever)
        fs = _first_above(res["rho_trace"])
        if fs is not None:
            ax2.axvline(fs, color="gray", ls="-.", lw=1.0, alpha=0.5)
            ax2.annotate(r"$\rho\geq0$" + f"\n@ t={fs}", (fs, 0.0),
                         textcoords="offset points", xytext=(6, 10), fontsize=8, color="gray")
    sat = "SATISFIED" if (res["rho_trace"][-1] >= 0) else "VIOLATED"
    ax2.set_title(rf"$\phi=\bigwedge_k\diamond(\,grasp_k\wedge\diamond_{{[0,\tau]}}goal_k\,)$"
                  rf"   $\rho_{{final}}$={res['rho_trace'][-1]:+.4f} ({sat})"
                  rf"   online$\equiv$offline (max|diff|={max_diff:.2e})   $\tau$={tau}")
    ax2.set_ylabel("task robustness\nmargin")
    ax2.legend(loc="best", fontsize=8.5)
    ax2.grid(alpha=0.25)

    # ---- panel 3: reward decomposition ----------------------------------------
    ax3.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    ax3.plot(t, res["sparse_trace"], color="tab:olive", lw=1.4, ls=":",
             label="sparse bonus (reward_coef·success)")
    ax3.plot(t, res["shaping_trace"], color="tab:purple", lw=1.4, ls="--",
             label="STL shaping (γ·ρ_t − ρ_{t-1})")
    ax3.plot(t, res["reward_trace"], color="black", lw=2.0,
             label="total reward r_t")
    ax3.set_title(f"Reward decomposition  —  sum={res['reward_trace'].sum():.3f}  "
                  f"(sparse={res['sparse_trace'].sum():.2f}, shaping={res['shaping_trace'].sum():.3f})")
    ax3.set_ylabel("reward")
    ax3.set_xlabel("step t")
    ax3.legend(loc="best", fontsize=8.5)
    ax3.grid(alpha=0.25)
    ax3.set_xlim(0, T - 1)

    fig.suptitle(f"{task_description}    episode {ep_idx}    "
                 f"success={res['success']}  steps={T}  k={len(goal_atoms)}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  plot -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["model", "random"], default="model")
    ap.add_argument("--model_path", default=os.environ.get("OPENVLA_MODEL_PATH"))
    ap.add_argument("--task_suite", default="libero_spatial",
                    help="libero_spatial / libero_object / libero_goal / libero_10 / libero_90")
    ap.add_argument("--task_id", type=int, default=0)
    ap.add_argument("--num_episodes", type=int, default=10)
    ap.add_argument("--stl_tau", type=int, default=60)
    ap.add_argument("--stl_gamma", type=float, default=0.99)
    ap.add_argument("--stl_reward_scale", type=float, default=1.0)
    ap.add_argument("--reward_coef", type=float, default=1.0)
    ap.add_argument("--max_steps", type=int, default=220,
                    help="per-suite rollout cap (libero_spatial=220, libero_90=400, ...)")
    ap.add_argument("--num_steps_wait", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    # model-inference knobs (RLinf OFT loader path; match the official eval defaults)
    ap.add_argument("--implement_version", default="rlinf",
                    help="RLinf OFT loader version: 'rlinf' (LIBERO/ManiSkill RLinf ckpts) or 'official'")
    ap.add_argument("--num_action_chunks", type=int, default=8, help="OFT action chunk size")
    ap.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=True,
                    help="sample actions (eval default True); --no-do_sample for greedy")
    ap.add_argument("--temperature", type=float, default=1.6, help="sampling temperature (eval=1.6)")
    ap.add_argument("--top_k", type=int, default=-1, help="top-k sampling (-1 = off)")
    ap.add_argument("--plot_dir", default="results/stl_reward_figures",
                    help="per-episode PNG output dir (set empty to disable)")
    ap.add_argument("--no_plot", action="store_true", help="disable plotting")
    ap.add_argument("--save_video", action=argparse.BooleanOptionalAction, default=True,
                    help="save per-episode rollout mp4 (upright agentview frames)")
    args = ap.parse_args()

    def _set_seed(s):
        random.seed(s)
        np.random.seed(s)
        try:
            import torch
            torch.manual_seed(s)
        except Exception:
            pass

    _set_seed(args.seed)

    print(f"policy={args.policy}  suite={args.task_suite}  task_id={args.task_id}  "
          f"episodes={args.num_episodes}  tau={args.stl_tau}")
    policy_fn = load_policy(args)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    task = task_suite.get_task(args.task_id)
    initial_states = task_suite.get_task_init_states(args.task_id)
    env, task_description = get_libero_env(task, "openvla", resolution=256)
    print(f"task: {task_description}")

    # goal atoms + grasp mapping (constant per task) — read straight from the live env
    raw_atoms = _get_goal_atoms(env)
    goal_atoms = _normalize_goal_atoms(raw_atoms)
    grasp_objs = [obj if pred in _GEOMETRIC_PREDS else None for (pred, obj, _t) in goal_atoms]
    pred_cfg = PredConfig()
    if not goal_atoms:
        print(f"WARNING: no goal atoms parsed (raw head={raw_atoms[:3]}); "
              "STL reward cannot be computed. Aborting.")
        return 1
    print(f"parsed {len(goal_atoms)} goal atoms: {goal_atoms}")

    n = min(args.num_episodes, len(initial_states))
    all_ok = True
    n_success = 0
    plot_dir = None if (args.no_plot or not args.plot_dir) else args.plot_dir
    for ep in range(n):
        res = run_episode(env, policy_fn, task_description, initial_states[ep],
                          goal_atoms, grasp_objs, pred_cfg, args)
        ok = summarize(f"episode {ep}", res, goal_atoms)
        all_ok = all_ok and ok
        n_success += int(res["success"])
        if plot_dir is not None:
            out_path = os.path.join(plot_dir, args.task_suite,
                                    f"reward_{args.task_suite}_task{args.task_id:02d}_ep{ep:03d}.png")
            try:
                plot_episode(out_path, res, goal_atoms, task_description, ep, args.stl_tau)
            except Exception as e:  # plotting must never fail the test
                print(f"  plot skipped: {e!r}")
        if args.save_video and res.get("frames"):
            try:
                import imageio.v2 as imageio

                vdir = os.path.join(plot_dir or "results/stl_reward_figures",
                                    args.task_suite, "videos")
                os.makedirs(vdir, exist_ok=True)
                vpath = os.path.join(
                    vdir,
                    f"rollout_{args.task_suite}_task{args.task_id:02d}_ep{ep:03d}"
                    f"_{'SUCCESS' if res['success'] else 'fail'}.mp4",
                )
                imageio.mimsave(vpath, list(res["frames"]), fps=20, codec="libx264", quality=6)
                print(f"  video -> {vpath}")
            except Exception as e:  # video must never fail the test
                print(f"  video skipped: {e!r}")

    print("\n================ SUMMARY ================")
    print(f"episodes={n}  success={n_success}/{n}  all_correct={all_ok}")
    print("RESULT:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
