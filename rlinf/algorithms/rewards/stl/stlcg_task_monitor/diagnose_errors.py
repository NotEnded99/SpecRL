"""Per-error attribution (parse vs threshold vs transient vs grasp), only on the
ERROR episodes identified by the batch JSON summaries."""
import sys, glob, h5py, os, json
import numpy as np
from .loader import load_episode
from .formula_builder import resolve_steps, build_bundle
from .nl_parser import parse_task
from .ontology import ObjectOntology
from .predicates import PredConfig

onto = ObjectOntology(); cfg = PredConfig()
CATS = {"PARSE": 0, "THRESHOLD": 0, "TRANSIENT": 0, "GRASP": 0, "OTHER": 0}
DETAIL = {k: [] for k in CATS}

def flag_final(ep, key):
    return bool(ep.goal_flags[key][-1]) if key and key in ep.goal_flags else None

for suite in ["libero_spatial", "libero_object", "libero_10", "libero_goal"]:
    js = f"results/stl_task_{suite}.json"
    if not os.path.exists(js):
        print(f"[{suite}] no summary json, skip"); continue
    errs = [s for s in json.load(open(js))["summaries"] if bool(s["sat"]) != bool(s["gt"])]
    print(f"[{suite}] {len(errs)} error episodes")
    for s in errs:
        f = os.path.join("trajectories_gf", suite, s["episode"])
        ep = load_episode(f); d = str(ep.attrs["task_description"])
        gt = bool(s["gt"]); kind = "FP" if (s["sat"] and not gt) else "FN"
        try:
            g = parse_task(d, onto); steps = resolve_steps(g, ep); bundle = build_bundle(steps, ep, cfg)
        except Exception as e:
            CATS["PARSE"] += 1; DETAIL["PARSE"].append(f"{suite}/{s['episode']}: parse exc"); continue
        atoms = bundle.atoms
        goal_atoms = [a for a in atoms if a.flag_key]
        grasp_atoms = [a for a in atoms if a.name.startswith("grasp")]
        cat = None; why = []
        for a in goal_atoms:
            gf = flag_final(ep, a.flag_key); mx = a.margin.max(); fin = a.margin[-1]
            if gf is True and mx < 0:
                cat = "THRESHOLD"; why.append(f"{a.name}: mymax={mx:+.3f} goalflag(True)")
            elif gf is False and mx >= 0 and fin < 0:
                cat = cat or "TRANSIENT"; why.append(f"{a.name}: max={mx:+.3f} final={fin:+.3f} goalflag(False)")
            elif gf is False and mx >= 0 and fin >= 0:
                cat = cat or "THRESHOLD"; why.append(f"{a.name}: final>=0 goalflag(False)")
            elif gf is None:
                why.append(f"{a.name}: no goal_flag key {a.flag_key}")
        if cat is None and grasp_atoms and any(a.margin.max() < 0 for a in grasp_atoms):
            cat = "GRASP"; why = [f"{a.name}: max={a.margin.max():+.3f}" for a in grasp_atoms if a.margin.max() < 0]
        if cat is None:
            cat = "PARSE"; why.append(f"goal_atoms={[a.name for a in goal_atoms]} grasp={[a.name for a in grasp_atoms]}")
        # if goal atom has no flag_key mapping, that's a PARSE issue
        if any(a.flag_key is None for a in goal_atoms):
            cat = "PARSE"
        CATS[cat] += 1
        DETAIL[cat].append(f"[{suite} {kind}] {s['episode']} ({d[:28]}): " + " | ".join(why)[:160])

print("\n=== ERROR ATTRIBUTION ===")
for k in ["PARSE", "THRESHOLD", "TRANSIENT", "GRASP", "OTHER"]:
    print(f"  {k:10s}: {CATS[k]}")
print(f"  TOTAL: {sum(CATS.values())}")
for k in ["PARSE", "THRESHOLD", "TRANSIENT", "GRASP"]:
    if DETAIL[k]:
        print(f"\n--- {k} samples ---")
        for d in DETAIL[k][:5]:
            print(" ", d)
