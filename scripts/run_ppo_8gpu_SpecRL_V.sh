#!/usr/bin/env bash
# SpecRL visual-grounding launcher (V): proposition satisfaction is evaluated
# from RGB-D observations, robot proprioception, and YOLOE detections.
#
# This launcher enables visual grounding and batched detector inference.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

MODE="${1:-goal}"
case "${MODE}" in
    full) CONFIG_NAME="libero_40_ppo_openpi_pi05_SpecRL_V" ;;
    goal) CONFIG_NAME="libero_40_ppo_openpi_pi05_SpecRL_V_goal" ;;
    *) echo "USAGE: $0 [full|goal]" >&2; exit 2 ;;
esac

TRAIN_VENV="${RLINF_TRAIN_VENV:-${ROOT_DIR}/.venv_embodied_openpi}"
TRAIN_VENV_ACTIVATE="${TRAIN_VENV}/bin/activate"
TRAIN_PYTHON="${TRAIN_VENV}/bin/python"
SFT_CHECKPOINT="${_SFT_CHECKPOINT:-${ROOT_DIR}/checkpoints/openpi_sft/SFT_checkpoint}"

export _SFT_CHECKPOINT="${SFT_CHECKPOINT}"
export VISUAL_GPU_IDS="${VISUAL_GPU_IDS:-0 1 2 3 4 5 6 7}"
export YOLOE_BASE_PORT="${YOLOE_BASE_PORT:-8010}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${ROOT_DIR}/openpi_data}"
export YOLOE_PYTHON="${YOLOE_PYTHON:-${ROOT_DIR}/.yoloe_venv/bin/python}"
export YOLOE_WEIGHTS="${YOLOE_WEIGHTS:-${ROOT_DIR}/weights/yoloe/best.pt}"

# --- Stage-plan table -------------------------------------------------------
PLAN_JSON="${RLINF_AGM_STAGE_PLAN_JSON:-${ROOT_DIR}/llm_symbolic_parity_results_revised.json}"
if [[ ! -f "${PLAN_JSON}" ]]; then
    echo "ERROR: LLM role-level plan table missing: ${PLAN_JSON}" >&2
    echo "  It is produced by llm_symbolic_parity.py --online plus the" >&2
    echo "  role-level revision step.  Or point RLINF_AGM_STAGE_PLAN_JSON at a" >&2
    echo "  grounded export (agm_stage_plans_v1.json / agm_stage_plans_v2.json)." >&2
    exit 5
fi
export RLINF_AGM_STAGE_PLAN_JSON="${PLAN_JSON}"

YOLOE_BATCH_V2_LAUNCHER="${ROOT_DIR}/scripts/run_yoloe_per_gpu.sh"
export YOLOE_GPU_IDS="${VISUAL_GPU_IDS}"
TRAIN_LAUNCHER="${ROOT_DIR}/examples/embodiment/run_SpecRL_V.sh"

die() { echo "ERROR: $*" >&2; exit 1; }

# --- Preflight: files, checkpoint, GPUs, config ----------------------------
[[ -f "${YOLOE_BATCH_V2_LAUNCHER}" ]] || die "YOLOE Batch V2 launcher missing: ${YOLOE_BATCH_V2_LAUNCHER}"
[[ -f "${TRAIN_LAUNCHER}" ]]   || die "training launcher missing: ${TRAIN_LAUNCHER}"
[[ -f "${TRAIN_VENV_ACTIVATE}" ]] || die "training venv activation script missing: ${TRAIN_VENV_ACTIVATE}"
[[ -x "${TRAIN_PYTHON}" ]]     || die "training python missing or not executable: ${TRAIN_PYTHON}"
[[ -x "${YOLOE_PYTHON}" ]]     || die "YOLOE python missing or not executable: ${YOLOE_PYTHON}"
[[ -f "${YOLOE_WEIGHTS}" ]]    || die "YOLOE weights missing: ${YOLOE_WEIGHTS}"
[[ -d "${SFT_CHECKPOINT}" ]]   || die "SFT checkpoint missing: ${SFT_CHECKPOINT}"
[[ -f "examples/embodiment/config/${CONFIG_NAME}.yaml" ]] || die "config missing: ${CONFIG_NAME}.yaml"
export RLINF_EXPECTED_PYTHON="$(readlink -f "${TRAIN_PYTHON}")"

test "$(nvidia-smi -L | wc -l)" -eq 8 || die "expected 8 GPUs, got $(nvidia-smi -L | wc -l)"

echo "MODE=${MODE} CONFIG=${CONFIG_NAME}"
echo "PLAN_JSON=${RLINF_AGM_STAGE_PLAN_JSON} (LLM role-level, grounded at load)"

# --- Preflight: training python env (no packages installed/modified) ------
# The venv's activate script dereferences $PYTHONPATH unguarded, which trips
# set -u on a clean DLC shell; relax it just for the source.
set +u
# shellcheck disable=SC1091
source "${TRAIN_VENV_ACTIVATE}"
set -u
python - <<'PY' || exit 1
import os
import sys
from pathlib import Path

import torch

# Guard against the venv python symlink being dangling (e.g. interpreter not
# present on this instance): a wrong interpreter silently falls through PATH
# to an image env without ray.
actual = Path(sys.executable).resolve()
expected = Path(os.environ["RLINF_EXPECTED_PYTHON"]).resolve()
assert actual == expected, f"wrong interpreter: actual={actual}, expected={expected}"
assert torch.cuda.device_count() == 8, f"GPU_COUNT={torch.cuda.device_count()}"

# Dry-run the plan-table load in the exact training interpreter: this fails
# fast if the JSON cannot be parsed/grounded, instead of inside a Ray worker.
from rlinf.envs.libero import stl_stage_plan as ssp
plans = ssp.AUDITED_PLANS_BY_DESCRIPTION
print(f"TRAIN_ENV_OK torch={torch.__version__} plans={len(plans)}")
PY

# --- 1. Start the 8-way YOLOE service (background supervisor, self-verifying)
echo "Starting YOLOE per-GPU services..."
bash "${YOLOE_BATCH_V2_LAUNCHER}" &
YOLOE_SUPERVISOR_PID=$!

# --- 2. Confirm all 8 ports report "ok batch raw" (with retry) --------------
# The per-GPU supervisor loads the YOLOE weights after being started above,
# so readiness must be polled until the deadline instead of checked once.
YOLOE_READY_TIMEOUT="${YOLOE_READY_TIMEOUT:-600}"
YOLOE_LOG_DIR="${ROOT_DIR}/logs/yoloe_per_gpu"
deadline=$((SECONDS + YOLOE_READY_TIMEOUT))
while true; do
    all_ready=1
    for port in $(seq "${YOLOE_BASE_PORT}" $((YOLOE_BASE_PORT + 7))); do
        health="$(curl -fsS --max-time 3 "http://127.0.0.1:${port}/health" 2>/dev/null || true)"
        if [[ "${health}" != *"ok batch raw"* ]]; then
            all_ready=0
            break
        fi
    done
    (( all_ready == 1 )) && break
    if ! kill -0 "${YOLOE_SUPERVISOR_PID}" 2>/dev/null; then
        echo "ERROR: YOLOE supervisor exited before services became healthy; recent logs:" >&2
        tail -n 30 "${YOLOE_LOG_DIR}"/*.log >&2 2>/dev/null || true
        die "YOLOE supervisor exited during startup"
    fi
    (( SECONDS < deadline )) || {
        echo "ERROR: YOLOE services not healthy after ${YOLOE_READY_TIMEOUT}s; recent logs:" >&2
        tail -n 30 "${YOLOE_LOG_DIR}"/*.log >&2 2>/dev/null || true
        die "YOLOE startup timed out; inspect ${YOLOE_LOG_DIR}"
    }
    sleep 5
done
for port in $(seq "${YOLOE_BASE_PORT}" $((YOLOE_BASE_PORT + 7))); do
    health="$(curl -fsS --max-time 3 "http://127.0.0.1:${port}/health" 2>/dev/null || true)"
    echo "HEALTHY PORT=${port} BODY=${health}"
done

# --- Optional teardown of screen-based YOLOE services on exit --------------
cleanup() {
    if [[ "${V2_TEARDOWN_YOLOE:-0}" == "1" ]]; then
        echo "Stopping YOLOE services..."
        kill "${YOLOE_SUPERVISOR_PID}" 2>/dev/null || true
        pkill -f '[y]oloe_server.py' 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# --- 3. Visual PPO training (foreground; its exit code becomes the job's) --
bash "${TRAIN_LAUNCHER}" "${CONFIG_NAME}"
