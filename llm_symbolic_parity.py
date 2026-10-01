#!/usr/bin/env python
"""Compare the LLM-generated symbolic STL formula against the audited one.

For every libero_40 task this builds both formulas and compares them at the
STRUCTURAL level (predicate multiset + per-path stage shape), which is tolerant
to role renaming and path reordering -- i.e. it asks "did the LLM produce the
same STL formula shape", not "did it use the exact same role strings".

By default it uses :class:`OfflineLLMClient`, which returns the audited formula
verbatim, so this script is fully runnable WITHOUT an API key and exercises the
prompt-building / JSON parsing / validation / conversion pipeline end to end
(result: 40/40).  Pass ``--online`` to hit the real OpenAI-compatible endpoint
(set ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` / ``RLINF_LLM_MODEL`` first) and
measure the model's actual structural alignment.

Each record stores the validated LLM output VERBATIM in ``llm_formula``
(structured paths/stages/atoms, lossless).  The flattened ``llm_paths`` /
``llm_atoms`` / ``llm_atom_spatial`` fields are debug-readable views only:
human revision and the revised-JSON build must consume ``llm_formula``.

    .venv_embodied_openpi/bin/python llm_symbolic_parity.py            # offline
    .venv_embodied_openpi/bin/python llm_symbolic_parity.py --online   # real API
    .venv_embodied_openpi/bin/python llm_symbolic_parity.py --prompt "put the bowl on the stove"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Tuple

from rlinf.envs.libero import stl_stage_plan as ssp
from rlinf.envs.libero.nlstl import (
    OfflineLLMClient,
    OpenAIChatClient,
    build_symbolic_plan,
    build_symbolic_plan_llm,
)
from rlinf.envs.libero.nlstl.llm_symbolic import build_messages

LIBERO_40_SUITES: Tuple[str, ...] = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
)


def _fmt_atom(atom) -> str:
    pred, obj, target = ssp.canonical_atom(atom)
    if pred == "pick":
        return f"pick({obj})"
    if target is None:
        return f"{pred}({obj})"
    if obj is None:
        return f"{pred}({target})"
    return f"{pred}({obj},{target})"


def _structural_signature(plan) -> Tuple[tuple, frozenset]:
    """Predicate-multiset + per-path stage shape (role- and order-tolerant)."""
    preds = tuple(sorted(
        a[0] for path in plan.paths for stage in path.stages for a in stage.atoms
    ))
    path_shapes = frozenset(
        tuple(frozenset(a[0] for a in stage.atoms) for stage in path.stages)
        for path in plan.paths
    )
    return preds, path_shapes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--online", action="store_true",
                        help="use the real OpenAI-compatible API (needs OPENAI_API_KEY)")
    parser.add_argument("--prompt", metavar="INSTRUCTION",
                        help="print the chat messages that would be sent for one instruction and exit")
    args = parser.parse_args()

    if args.prompt:
        messages = build_messages(args.prompt)
        print("=== Messages sent to the LLM ===")
        for msg in messages:
            print(f"\n--- {msg['role']} ---\n{msg['content']}")
        return 0

    client = OpenAIChatClient() if args.online else OfflineLLMClient()
    mode = "ONLINE" if args.online else "OFFLINE (audited echo; set --online + OPENAI_API_KEY for the real model)"
    print(f"Client: {type(client).__name__}  ({mode})")

    results = []
    global_id = 0
    for suite in LIBERO_40_SUITES:
        for description in ssp._STANDARD_SUITE_TASKS[suite]:
            audited = build_symbolic_plan(description, "libero_40", global_id)
            try:
                llm, llm_formula = build_symbolic_plan_llm(
                    description, client=client, return_raw=True
                )
                same = _structural_signature(audited) == _structural_signature(llm)
                err = None
            except Exception as e:  # noqa: BLE001
                llm = None
                llm_formula = None
                same = False
                err = f"{type(e).__name__}: {e}"
            results.append({
                "global_id": global_id, "suite": suite, "description": description,
                "same_structure": same, "error": err,
                "audited_atoms": [list(a) for a in audited.atoms],
                # VERBATIM validated LLM output: structured pred/obj/target
                # atoms, path names, stage boundaries, spatial annotations.
                # This -- not the flattened fields below -- is the artifact a
                # human revises and the revised JSON is built from.
                "llm_formula": llm_formula,
                # Flattened convenience views (debug readability only).
                "llm_atoms": [list(a) for a in llm.atoms] if llm else None,
                "llm_paths": [[_fmt_atom(a) for st in p.stages for a in st.atoms] for p in llm.paths] if llm else None,
                "llm_atom_spatial": (
                    [{"atom": _fmt_atom(atom), "spatial": {"relation": sp.relation, "refs": list(sp.refs)}}
                     for atom, sp in llm.atom_spatial]
                    if llm else None
                ),
            })
            global_id += 1

    # worked examples
    print("\n" + "=" * 92)
    for gid in (0, 20, 38):
        r = next(x for x in results if x["global_id"] == gid)
        print(f"[task #{gid:02d}] {r['description']}")
        print(f"  audited : {r['audited_atoms']}")
        print(f"  llm     : {r['llm_atoms']}")
        print(f"  same_shape: {r['same_structure']}")
    print("=" * 92)

    n_same = sum(1 for r in results if r["same_structure"])
    n_err = sum(1 for r in results if r["error"])
    print(f"\nRESULT: {n_same}/{len(results)} tasks have the same STL structure as audited "
          f"({n_err} errors).")

    failed = [r for r in results if not r["same_structure"]]
    if failed:
        print("\nMismatches / errors:")
        for r in failed[:20]:
            tag = r["error"] or "structure differs"
            print(f"  task #{r['global_id']:02d} ({r['suite']}): {tag}")
            print(f"     audited={r['audited_atoms']} llm={r['llm_atoms']}")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "llm_symbolic_parity_results.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"\nJSON results written to: {out_path}")
    return 0 if n_same == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
