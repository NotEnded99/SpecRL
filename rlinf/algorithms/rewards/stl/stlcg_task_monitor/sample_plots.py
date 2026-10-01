"""Sample 20 random episodes per suite, plot online-prefix robustness (true only).

Uses fast numpy O(T²) exact online prefix (prefix-truncated inner windows).
"""
from __future__ import annotations
import argparse, glob, os, random, re
import h5py
import numpy as np

from .bddl_goals import goal_atoms
from .fast_online import online_prefix
from .formula_builder import build_bddl_bundle
from .loader import load_episode
from .plot import plot_result, plot_three_methods
from .predicates import PredConfig


def _slug(text: str, maxlen: int = 48) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return (s[:maxlen]).rstrip("_") or "task"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="trajectories_gf")
    ap.add_argument("--suites", nargs="+",
                    default=["libero_spatial", "libero_object", "libero_10", "libero_goal"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out_dir",
        default=os.environ.get("STL_PLOT_DIR", "results/stl_reward_figures"),
    )
    ap.add_argument("--tau", type=int, default=60)
    args = ap.parse_args()

    cfg = PredConfig()
    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    for suite in args.suites:
        files = sorted(glob.glob(os.path.join(args.root, suite, "task*_ep*.h5")))
        if not files:
            print(f"[{suite}] no episodes"); continue
        sample = rng.sample(files, min(args.n, len(files)))
        print(f"[{suite}] sampling {len(sample)} / {len(files)}")

        for i, path in enumerate(sample):
            try:
                with h5py.File(path, "r") as f:
                    tid = int(f.attrs.get("task_id", -1))
                    eid = int(f.attrs.get("episode_idx", -1))
                    desc = str(f.attrs.get("task_description", ""))
                    success = bool(f.attrs.get("success", False))
                ep = load_episode(path)
                atoms = goal_atoms(str(ep.attrs.get("suite", suite)), tid)
                bundle, cols = build_bddl_bundle(ep, cfg, atoms)

                # extract grasp+goal pairs
                grasp_margins = []
                goal_margins = []
                for idx, (nm, m, ok, fl) in enumerate(cols):
                    if nm.startswith("grasp("):
                        continue
                    # find matching grasp (same obj_key)
                    g = None
                    for (nm2, m2, ok2, _) in cols:
                        if nm2.startswith("grasp(") and ok2 == ok:
                            g = m2; break
                    grasp_margins.append(g)
                    goal_margins.append(m)

                traces = online_prefix(grasp_margins, goal_margins, args.tau,
                                       ("true", "logsumexp", "softmax"), 10.0)

                class _R: pass
                r = _R()
                r.T = ep.T
                r.task_description = desc
                r.template = "bddl"
                r.atoms = {a.name: a.margin for a in bundle.atoms}
                r.atom_steps = {nm: (int(np.argmax(m >= 0)) if (m >= 0).any() else None)
                                for nm, m in r.atoms.items()}
                r.online_trace = traces["true"]
                r.suffix_trace = traces["true"]
                r.rho_final = float(traces["true"][-1])
                r.satisfied = bool(r.rho_final >= -1e-4)
                r.first_sat_step = next((t for t, v in enumerate(traces["true"]) if v >= -1e-4), None)
                r.approx_method = "true+logsumexp+softmax"

            except Exception as e:
                print(f"  [skip] {os.path.basename(path)}: {e}")
                continue

            tag = "succ" if success else "fail"
            name = f"rob_{suite}__task{tid:02d}_ep{eid:03d}__{tag}__{_slug(desc)}.png"
            out = os.path.join(args.out_dir, name)
            title = f"{suite} task{tid:02d} ep{eid:03d} [{tag}]  |  {desc}"
            plot_three_methods(r, traces, out, title=title)
            print(f"  [{i+1}/{len(sample)}] {name}  rho={r.rho_final:+.4f}")


if __name__ == "__main__":
    main()
