#!/usr/bin/env python3
"""Dump the audited AGM stage-plan table to a static JSON file.

The JSON is consumed by ``stl_stage_plan._load_plans_from_json`` when the
environment variable ``RLINF_AGM_STAGE_PLAN_JSON`` points at it (see
``run_SpecRL_P.sh``).

Usage:
    python3 dump_stage_plans.py <output.json>
"""

from __future__ import annotations

import argparse
import json

from rlinf.envs.libero.stl_stage_plan import (
    AUDITED_PLANS_BY_DESCRIPTION,
    StagePlan,
)


def plan_to_dict(plan: StagePlan) -> dict:
    """Serialize one StagePlan into a JSON-compatible dict."""
    return {
        "paths": [
            {
                "name": path.name,
                "stages": [
                    {
                        "name": stage.name,
                        "atoms": [list(atom) for atom in stage.atoms],
                        "virtual": stage.virtual,
                    }
                    for stage in path.stages
                ],
            }
            for path in plan.paths
        ],
        "gated": plan.gated,
        "reason": plan.reason,
        "audited": plan.audited,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="Destination JSON path.")
    args = parser.parse_args()

    data = {
        description: plan_to_dict(plan)
        for description, plan in AUDITED_PLANS_BY_DESCRIPTION.items()
    }

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"wrote {len(data)} stage plans -> {args.output}")


if __name__ == "__main__":
    main()
