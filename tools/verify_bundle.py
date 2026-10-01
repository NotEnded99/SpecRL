#!/usr/bin/env python3
"""Validate the source-only SpecRL bundle without importing ML dependencies."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = (
    "llm_symbolic_parity.py",
    "build_llm_symbolic_revised.py",
    "scripts/run_ppo_8gpu_SpecRL_P.sh",
    "scripts/run_ppo_8gpu_SpecRL_V.sh",
    "scripts/run_yoloe_per_gpu.sh",
    "scripts/yoloe_server.py",
    "examples/embodiment/run_embodiment.sh",
    "examples/embodiment/run_SpecRL_P.sh",
    "examples/embodiment/run_SpecRL_V.sh",
    "examples/embodiment/train_embodied_agent.py",
    "examples/embodiment/agm_stage_runtime/sitecustomize.py",
    "examples/embodiment/visual_agm_stage_runtime/sitecustomize.py",
    "examples/embodiment/config/libero_40_ppo_openpi_pi05_SpecRL_P.yaml",
    "examples/embodiment/config/libero_40_ppo_openpi_pi05_SpecRL_V_goal.yaml",
    "examples/embodiment/config/libero_40_ppo_openpi_pi05_SpecRL_V.yaml",
    "examples/embodiment/config/env/libero_40.yaml",
    "examples/embodiment/config/env/libero_goal.yaml",
    "examples/embodiment/config/model/pi0_5.yaml",
    "examples/embodiment/config/hybrid_engines/fsdp.yaml",
    "examples/embodiment/config/weight_syncer/patch_syncer.yaml",
    "examples/embodiment/config/visual_stl/v51.yaml",
    "rlinf/envs/libero/libero_env.py",
    "rlinf/envs/libero/libero_env_agm_stage.py",
    "rlinf/envs/libero/libero_env_visual_agm_stage.py",
    "rlinf/envs/libero/agm_stage_runtime.py",
    "rlinf/envs/libero/stl_stage_plan.py",
    "rlinf/envs/libero/nlstl/__init__.py",
    "rlinf/envs/libero/nlstl/evidence.py",
    "rlinf/envs/libero/nlstl/grounding.py",
    "rlinf/envs/libero/nlstl/llm_client.py",
    "rlinf/envs/libero/nlstl/llm_symbolic.py",
    "rlinf/envs/libero/nlstl/pipeline.py",
    "rlinf/envs/libero/nlstl/symbolic_plan.py",
    "rlinf/envs/libero/venv.py",
    "rlinf/envs/libero/visual_venv.py",
    "rlinf/envs/libero/yoloe_client.py",
    "rlinf/envs/action_utils.py",
    "rlinf/envs/utils.py",
    "rlinf/envs/venv/__init__.py",
    "rlinf/envs/wrappers/__init__.py",
    "rlinf/algorithms/rewards/stl/stlcg_task_monitor/predicates.py",
    "rlinf/algorithms/rewards/stl/stlcg_task_monitor/stl_aggregation.py",
    "third_party/stlcg-plus-plus/setup.py",
    "third_party/stlcg-plus-plus/stlcgpp/formula.py",
    "llm_symbolic_parity_results_revised.json",
    "requirements/install.sh",
    "requirements/embodied/models/openpi.txt",
    "pyproject.toml",
    "uv.lock",
)

EXTERNAL_ASSETS = (
    ".venv_embodied_openpi",
    ".yoloe_venv",
)

EXCLUDED_ENV_DIRS = (
    "behavior",
    "calvin",
    "d4rl",
    "embodichain",
    "frankasim",
    "genesis",
    "habitat",
    "isaaclab",
    "maniskill",
    "metaworld",
    "polaris",
    "realworld",
    "robocasa",
    "robotwin",
    "roboverse",
    "world_model",
)

LIBERO_RUNTIME_FILES = {
    "__init__.py",
    "agm_pick_predicate.py",
    "agm_stage_runtime.py",
    "libero_env.py",
    "libero_env_agm_stage.py",
    "libero_env_visual_agm_stage.py",
    "stl_stage_plan.py",
    "utils.py",
    "venv.py",
    "visual_agm_shadow_runtime.py",
    "visual_close_margin.py",
    "visual_drawer_close_margin.py",
    "visual_drawer_panel_diagnostic.py",
    "visual_in_margin.py",
    "visual_microwave_close_margin.py",
    "visual_microwave_diagnostic.py",
    "visual_open_margin.py",
    "visual_spatial_relation_margin.py",
    "visual_stl_comparison_plot.py",
    "visual_stl_monitor.py",
    "visual_stl_shadow.py",
    "visual_target_selector.py",
    "visual_turnon_observer.py",
    "visual_venv.py",
    "visual_yoloe_rollout_exporter.py",
    "yoloe_client.py",
}

LIBERO_RUNTIME_DIRS = {"nlstl"}

NLSTL_RUNTIME_FILES = {
    "__init__.py",
    "evidence.py",
    "grounding.py",
    "llm_client.py",
    "llm_symbolic.py",
    "pipeline.py",
    "symbolic_plan.py",
}


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)


def main() -> int:
    errors = 0

    missing = [relative for relative in REQUIRED_PATHS if not (ROOT / relative).exists()]
    if missing:
        errors += len(missing)
        for relative in missing:
            fail(f"required path is missing: {relative}")

    unexpected_envs = [
        name for name in EXCLUDED_ENV_DIRS if (ROOT / "rlinf" / "envs" / name).exists()
    ]
    if unexpected_envs:
        errors += len(unexpected_envs)
        for name in unexpected_envs:
            fail(f"unrelated environment directory is present: rlinf/envs/{name}")
    else:
        print("OK: unrelated environment directories remain excluded")

    libero_dir = ROOT / "rlinf" / "envs" / "libero"
    actual_libero_files = {
        path.name for path in libero_dir.iterdir() if path.is_file()
    }
    actual_libero_dirs = {
        path.name for path in libero_dir.iterdir() if path.is_dir()
    }
    missing_libero = sorted(LIBERO_RUNTIME_FILES - actual_libero_files)
    unexpected_libero = sorted(actual_libero_files - LIBERO_RUNTIME_FILES)
    unexpected_libero_dirs = sorted(actual_libero_dirs - LIBERO_RUNTIME_DIRS)
    missing_libero_dirs = sorted(LIBERO_RUNTIME_DIRS - actual_libero_dirs)
    if missing_libero or unexpected_libero or unexpected_libero_dirs or missing_libero_dirs:
        errors += (
            len(missing_libero)
            + len(unexpected_libero)
            + len(unexpected_libero_dirs)
            + len(missing_libero_dirs)
        )
        for name in missing_libero:
            fail(f"required LIBERO runtime file is missing: {name}")
        for name in unexpected_libero:
            fail(f"unexpected LIBERO file is present: {name}")
        for name in missing_libero_dirs:
            fail(f"required LIBERO subdirectory is missing: {name}")
        for name in unexpected_libero_dirs:
            fail(f"unexpected LIBERO subdirectory is present: {name}")
    else:
        print(f"OK: curated LIBERO runtime files={len(actual_libero_files)}")

    nlstl_dir = libero_dir / "nlstl"
    if nlstl_dir.is_dir():
        actual_nlstl_files = {
            path.name for path in nlstl_dir.iterdir() if path.is_file()
        }
        actual_nlstl_dirs = {
            path.name for path in nlstl_dir.iterdir() if path.is_dir()
        }
        missing_nlstl = sorted(NLSTL_RUNTIME_FILES - actual_nlstl_files)
        unexpected_nlstl = sorted(actual_nlstl_files - NLSTL_RUNTIME_FILES)
        if missing_nlstl or unexpected_nlstl or actual_nlstl_dirs:
            errors += len(missing_nlstl) + len(unexpected_nlstl) + len(actual_nlstl_dirs)
            for name in missing_nlstl:
                fail(f"required NLSTL file is missing: {name}")
            for name in unexpected_nlstl:
                fail(f"unexpected NLSTL file is present: {name}")
            for name in sorted(actual_nlstl_dirs):
                fail(f"unexpected NLSTL subdirectory is present: {name}")
        else:
            print(f"OK: curated NLSTL runtime files={len(actual_nlstl_files)}")

    plan_path = ROOT / "llm_symbolic_parity_results_revised.json"
    if plan_path.is_file():
        try:
            payload = json.loads(plan_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            errors += 1
            fail(f"stage-plan JSON is invalid: {exc}")
        else:
            count = len(payload) if hasattr(payload, "__len__") else 0
            if count == 0:
                errors += 1
                fail("stage-plan JSON is empty")
            else:
                print(f"OK: stage-plan JSON entries={count}")

    python_files = sorted(ROOT.rglob("*.py"))
    syntax_errors = 0
    for path in python_files:
        if "__pycache__" in path.parts:
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeError, SyntaxError) as exc:
            syntax_errors += 1
            fail(f"Python syntax check failed: {path.relative_to(ROOT)}: {exc}")
    errors += syntax_errors
    print(f"OK: parsed Python files={len(python_files) - syntax_errors}")

    present_external = [relative for relative in EXTERNAL_ASSETS if (ROOT / relative).exists()]
    if present_external:
        print("NOTE: external environment directories are present locally:")
        for relative in present_external:
            print(f"  - {relative}")
    else:
        print("OK: external virtual environments remain excluded")

    file_count = sum(1 for path in ROOT.rglob("*") if path.is_file())
    print(f"OK: bundle files={file_count}")
    if errors:
        fail(f"bundle validation failed with {errors} error(s)")
        return 1
    print("BUNDLE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
