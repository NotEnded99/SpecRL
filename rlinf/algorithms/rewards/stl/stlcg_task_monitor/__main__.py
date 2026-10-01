"""CLI: monitor one episode (with plot) or batch a whole suite (success-rate
cross-check vs ground truth).

Examples:
    python -m experiments.robot.libero.stlcg_task_monitor \
        --traj trajectories_gf/libero_goal/task00_ep000.h5 --plot_dir results/figures
    python -m experiments.robot.libero.stlcg_task_monitor \
        --traj_dir trajectories_gf/libero_goal --out_json results/stl_task_goal.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

from .monitor import monitor_episode
from .ontology import ObjectOntology
from .plot import episode_ids, plot_result
from .predicates import PredConfig


def _gt_success(path: str) -> bool | None:
    """Ground-truth success from the sibling .mp4 filename or h5 attr."""
    import h5py
    try:
        with h5py.File(path, "r") as f:
            return bool(f.attrs.get("success", False))
    except Exception:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj", help="single episode HDF5")
    ap.add_argument("--traj_dir", help="directory of episode HDF5s (batch)")
    ap.add_argument("--tau", type=int, default=60)
    ap.add_argument("--approx", choices=["true", "logsumexp", "softmax"], default="true")
    ap.add_argument("--temperature", type=float, default=10.0)
    ap.add_argument("--f_eps", type=float, default=PredConfig.f_eps)
    ap.add_argument("--q_g_max", type=float, default=PredConfig.q_g_max)
    ap.add_argument("--r_grasp", type=float, default=PredConfig.r_grasp)
    ap.add_argument("--theta_art", type=float, default=PredConfig.theta_art)
    ap.add_argument("--plot_dir", help="if set with --traj, save a PNG here")
    ap.add_argument("--out_json", help="write batch per-episode summary JSON here")
    ap.add_argument("--calib", action="store_true",
                    help="enable goal_flags zero-point calibration (batch). Helps articulated "
                         "suites (goal); pick-place suites have a ~1-2cm gray zone where it "
                         "trades FP<->FN with no net gain, so it is OFF by default.")
    ap.add_argument("--calib_sample", type=int, default=200,
                    help="max episodes used to fit the goal_flags calibration")
    args = ap.parse_args(argv)

    cfg = PredConfig(f_eps=args.f_eps, q_g_max=args.q_g_max, r_grasp=args.r_grasp,
                     theta_art=args.theta_art)
    onto = ObjectOntology()

    if args.traj:
        res = monitor_episode(args.traj, cfg, tau=args.tau,
                              approx_method=args.approx, temperature=args.temperature,
                              onto=onto, compute_online=True)
        print(f"{os.path.basename(args.traj)}  T={res.T}  rho={res.rho_final:+.4f}  "
              f"{'SATISFIED' if res.satisfied else 'VIOLATED'}  first_sat={res.first_sat_step}")
        print(f"  task: {res.task_description}")
        for nm, m in res.atoms.items():
            print(f"    {nm}: max={m.max():+.3f}  first>=0={res.atom_steps[nm]}")
        if args.plot_dir:
            tid, eid = episode_ids(args.traj)
            tag = args.approx
            out = os.path.join(args.plot_dir, f"rob_task{tid:02d}_ep{eid:03d}_approx-{tag}.png")
            plot_result(res, out, title=f"{os.path.basename(args.traj)}  |  {res.task_description}")
        return

    if args.traj_dir:
        files = sorted(glob.glob(os.path.join(args.traj_dir, "task*_ep*.h5")))
        if not files:
            ap.error(f"no task*_ep*.h5 under {args.traj_dir}")

        # fit goal_flags calibration (per goal_flag key) on a sample, then evaluate
        from .calibration import fit_calibration, calibration_report
        calib = fit_calibration(files, cfg, onto, args.calib_sample) if args.calib else {}
        if calib:
            rep = calibration_report(files, calib, cfg, onto, args.calib_sample)
            print("=== goal_flags calibration (offset pins atom zero-point to goal onset) ===")
            for k, v in rep.items():
                print(f"  {k:6s}: offset={v['offset']:+.4f}  step-F1 {v['f1_before']:.3f} -> {v['f1_after']:.3f}  (n={v['n_steps']}, pos_rate={v['pos_rate']:.3f})")

        summaries = []
        tp = tn = fp = fn = 0
        per_task = {}
        for i, path in enumerate(files):
            try:
                res = monitor_episode(path, cfg, tau=args.tau,
                                      approx_method=args.approx, temperature=args.temperature,
                                      onto=onto, compute_online=False, calib=calib)
            except Exception as e:
                print(f"[skip] {os.path.basename(path)}: {e}")
                continue
            gt = _gt_success(path)
            mon = res.satisfied
            if gt is not None:
                if gt and mon: tp += 1
                elif (not gt) and (not mon): tn += 1
                elif gt and not mon: fn += 1
                else: fp += 1
            tid = res.task_description  # group by description
            d = per_task.setdefault(tid, {"n": 0, "sat": 0, "gt": 0})
            d["n"] += 1; d["sat"] += int(mon); d["gt"] += int(gt) if gt else 0
            summaries.append({"episode": os.path.basename(path), "T": res.T,
                              "rho": round(res.rho_final, 4), "sat": mon, "gt": gt,
                              "task": res.task_description})
            if (i + 1) % 50 == 0:
                print(f"  ...{i+1}/{len(files)} processed")
        # per-task table
        print(f"\n=== per-task ({os.path.basename(args.traj_dir)}) ===")
        print(f"{'task':55s} n   sat  gt   SR(mon) SR(gt)")
        for tname, d in sorted(per_task.items()):
            srm = d["sat"] / d["n"] if d["n"] else 0
            srg = d["gt"] / d["n"] if d["n"] else 0
            print(f"{tname[:55]:55s} {d['n']:3d} {d['sat']:3d}  {d['gt']:3d}  {srm:.3f}  {srg:.3f}")
        tot = tp + tn + fp + fn
        if tot:
            print(f"\nmonitor vs ground-truth: TP={tp} TN={tn} FP={fp} FN={fn}  "
                  f"acc={(tp+tn)/tot:.3f}  prec={tp/(tp+fp) if tp+fp else 0:.3f}  "
                  f"rec={tp/(tp+fn) if tp+fn else 0:.3f}")
        if args.out_json:
            os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
            with open(args.out_json, "w") as f:
                json.dump({"summaries": summaries,
                           "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}}, f, indent=2)
            print(f"wrote {len(summaries)} summaries -> {args.out_json}")
        return

    ap.error("provide --traj or --traj_dir")


if __name__ == "__main__":
    main()
