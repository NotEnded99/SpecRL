#!/usr/bin/env python
# Copyright 2025 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Parse-only diagnostic: LIBERO NL instruction -> STL formula + atomic predicates.

No env, no model, no rollout. For every task in a LIBERO suite it runs the
existing NL parser (``stlcg_task_monitor.nl_parser.parse_task``) and writes a CSV
with the natural-language instruction, the parsed template, the ordered verb
steps, the derived atomic predicates, and a readable STL formula. When the BDDL
goal atoms can be read they are added as a ground-truth column for comparison.

This lets you eyeball whether the NL->STL parsing is accurate across the suite.

Run (from the RLinf repo):
    python rlinf/algorithms/rewards/stl/parse_libero_to_stl_csv.py \
        --task_suite libero_spatial --out stl_parse_libero_spatial.csv

    # all tasks, default suite:
    python rlinf/algorithms/rewards/stl/parse_libero_to_stl_csv.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

# make the RLinf repo root importable when run by path
_RLINF_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _RLINF_ROOT not in sys.path:
    sys.path.insert(0, _RLINF_ROOT)

from libero.libero import benchmark  # noqa: E402

from rlinf.algorithms.rewards.stl.stlcg_task_monitor.nl_parser import parse_task  # noqa: E402
from rlinf.algorithms.rewards.stl.stlcg_task_monitor.ontology import ObjectOntology  # noqa: E402

# verb -> goal predicate name used by predicates.goal_atom_margin / bddl
_VERB_TO_GOAL_PRED = {
    "place": "on",
    "stack": "on",
    "insert": "in",
    "open": "open",
    "close": "close",
    "turn_on": "turnon",
    "turn_off": "turnoff",
}
_GEOMETRIC_VERBS = {"place", "stack", "insert"}


def step_to_atoms(step) -> list[str]:
    """One TaskStep -> the atomic predicate strings it contributes."""
    obj, tgt, lvl = step.object, step.target, step.level
    v = step.verb
    atoms: list[str] = []
    if v in _GEOMETRIC_VERBS:
        if obj:
            atoms.append(f"pick({obj})")          # grasp atom
        pred = _VERB_TO_GOAL_PRED[v]
        atoms.append(f"{pred}({obj},{tgt})")      # goal atom
    elif v == "pick":
        if obj:
            atoms.append(f"pick({obj})")
    elif v in _VERB_TO_GOAL_PRED:                  # open/close/turn_on/turn_off
        pred = _VERB_TO_GOAL_PRED[v]
        arg = tgt or lvl or obj
        atoms.append(f"{pred}({arg})")
    else:
        # push / twist / unknown -> record verbatim so nothing is silently dropped
        atoms.append(f"{v}({obj},{tgt})")
    return atoms


def steps_to_formula(steps, tau: int) -> str:
    """Readable STL:  phi = AND_k ◇[0,T] ( grasp_k  AND  ◇[0,tau] goal_{k} ).

    Mirrors the structure stlcg_task_monitor/formula_builder emits (grasp-then-goal
    for geometric verbs, bare eventually for articulated verbs)."""
    subs: list[str] = []
    for s in steps:
        v = s.verb
        obj, tgt, lvl = s.object, s.target, s.level
        if v in _GEOMETRIC_VERBS:
            grasp = f"pick({obj})" if obj else None
            pred = _VERB_TO_GOAL_PRED[v]
            goal = f"{pred}({obj},{tgt})"
            inner = f"({grasp} ∧ ◇[0,τ] {goal})" if grasp else f"◇[0,τ] {goal}"
            subs.append(f"◇[0,T] {inner}")
        elif v == "pick":
            subs.append(f"◇[0,T] pick({obj})")
        elif v in _VERB_TO_GOAL_PRED:
            pred = _VERB_TO_GOAL_PRED[v]
            arg = tgt or lvl or obj
            subs.append(f"◇[0,T] {pred}({arg})")
        else:
            subs.append(f"◇[0,T] {v}({obj},{tgt})")
    return " ∧ ".join(subs) if subs else "(unparsed)"


def objects_str(graph) -> str:
    parts = []
    for o in graph.objects:
        parts.append(f"{o.name}[{o.role.value}]")
    return "; ".join(parts)


def steps_str(steps) -> str:
    return " | ".join(
        f"{s.verb}({s.object},{s.target},rel={s.relation},lvl={s.level})" for s in steps
    )


def bddl_ground_truth(suite: str, task_id: int) -> list[tuple] | None:
    """Best-effort: read the exact goal atoms from the task's BDDL file."""
    try:
        from rlinf.algorithms.rewards.stl.stlcg_task_monitor.bddl_goals import goal_atoms
        return goal_atoms(suite, task_id)
    except Exception as e:
        print(f"  [bddl] suite={suite} task={task_id}: ground-truth unavailable ({e!r})",
              file=sys.stderr)
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task_suite", default="libero_spatial",
                    help="libero_spatial / libero_object / libero_goal / libero_10 / libero_90")
    ap.add_argument("--task_ids", default=None,
                    help="comma-separated task ids (default: all tasks in suite)")
    ap.add_argument("--tau", type=int, default=60, help="inner horizon for the formula string")
    ap.add_argument("--out", default=None, help="output CSV path")
    args = ap.parse_args()

    out_path = args.out or f"stl_parse_{args.task_suite}.csv"
    onto = ObjectOntology()
    task_suite = benchmark.get_benchmark_dict()[args.task_suite]()
    n_tasks = task_suite.n_tasks
    task_ids = [int(x) for x in args.task_ids.split(",")] if args.task_ids else list(range(n_tasks))

    rows: list[dict] = []
    n_ok = n_err = 0
    for tid in task_ids:
        task = task_suite.get_task(tid)
        nl = task.language
        row = {"suite": args.task_suite, "task_id": tid, "nl": nl}
        try:
            graph = parse_task(nl, onto=onto)
            atoms: list[str] = []
            for s in graph.steps:
                atoms.extend(step_to_atoms(s))
            row["template"] = graph.template.value
            row["objects"] = objects_str(graph)
            row["steps"] = steps_str(graph.steps)
            row["atoms"] = "; ".join(atoms)
            row["formula"] = steps_to_formula(graph.steps, args.tau)
            row["n_steps"] = len(graph.steps)
            n_ok += 1
        except Exception as e:
            row.update(template="PARSE_ERROR", objects="", steps="", atoms="",
                       formula=f"<error: {e!r}>", n_steps=0)
            n_err += 1
        # ground truth for comparison
        gt = bddl_ground_truth(args.task_suite, tid)
        if gt is not None:
            row["bddl_atoms"] = "; ".join(f"{p}({o},{t})" if t else f"{p}({o})" for p, o, t in gt)
            row["n_bddl_atoms"] = len(gt)
        else:
            row["bddl_atoms"] = ""
            row["n_bddl_atoms"] = ""
        rows.append(row)
        flag = "OK " if row["template"] != "PARSE_ERROR" else "ERR"
        print(f"[{flag}] task {tid:02d}: {nl}")
        print(f"        template={row['template']}  atoms=[{row['atoms']}]")
        if row["bddl_atoms"]:
            print(f"        bddl   =[{row['bddl_atoms']}]")

    fieldnames = ["suite", "task_id", "nl", "template", "objects", "steps",
                  "atoms", "formula", "n_steps", "bddl_atoms", "n_bddl_atoms"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"\nparsed {n_ok}/{len(task_ids)} tasks OK ({n_err} errors) -> {out_path}")
    print(f"columns: {fieldnames}")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
