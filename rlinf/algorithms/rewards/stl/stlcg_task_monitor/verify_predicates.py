"""Per-task predicate vs aligned goal_flags verification on libero_goal.
For each task, compute step-F1 of each goal atom's margin sign vs goal_flags,
and whether my margin >=0 at the achievement step (goal_flags first True)."""
import sys, glob
import numpy as np
from .loader import load_episode
from .formula_builder import resolve_steps, build_bundle
from .nl_parser import parse_task
from .ontology import ObjectOntology
from .predicates import PredConfig

onto = ObjectOntology(); cfg = PredConfig()

def f1(ins, gf):
    tp=(ins&gf).sum(); fp=(ins&~gf).sum(); fn=(~ins&gf).sum()
    p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0
    return round(2*p*r/(p+r),3) if p+r else 0

# group by task description
tasks = {}
for f in sorted(glob.glob("trajectories_gf/libero_goal/*.h5")):
    ep = load_episode(f)
    d = str(ep.attrs["task_description"])
    tasks.setdefault(d, []).append(f)

for d, files in sorted(tasks.items()):
    # parse once
    g = parse_task(d, onto)
    by_atom = {}  # atom_name -> [ (ins_bits, gf_bits, ach_margin_ok) per ep ]
    for f in files[:30]:
        ep = load_episode(f)
        st = resolve_steps(g, ep); b = build_bundle(st, ep, cfg)
        for a in b.atoms:
            if not a.flag_key or a.flag_key not in ep.goal_flags:
                continue
            gf = ep.goal_flags[a.flag_key]; m = a.margin
            n = min(len(m), len(gf))
            ins = m[:n] >= 0; gf2 = gf[:n].astype(bool)
            idx = np.where(gf2)[0]
            ach_ok = (m[idx[0]] >= 0) if len(idx) else (m[:n].max() < 0)  # if never True, ok if never pos
            by_atom.setdefault(a.name + f"[{a.flag_key}]", []).append((ins, gf2, ach_ok))
    print(f"\n{d[:55]}")
    for atom, lst in by_atom.items():
        ins = np.concatenate([x[0] for x in lst]); gf = np.concatenate([x[1] for x in lst])
        ach = sum(x[2] for x in lst)
        print(f"  {atom:34s} step-F1={f1(ins,gf)}  ach_step_margin_ok={ach}/{len(lst)}")
