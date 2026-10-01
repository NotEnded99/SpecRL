#!/usr/bin/env bash
# SpecRL_V launcher: evaluate proposition satisfaction with lightweight visual
# grounding from RGB-D observations, robot proprioception, and YOLOE detections.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_RUNNER="${SCRIPT_DIR}/run_embodiment.sh"
VISUAL_RUNTIME_DIR="${SCRIPT_DIR}/visual_agm_stage_runtime"
CONFIG_NAME="${1:-libero_40_ppo_openpi_pi05_SpecRL_V}"
GPU_IDS="${VISUAL_GPU_IDS:-0 1 2 3 4 5 6 7}"
BASE_PORT="${YOLOE_BASE_PORT:-8010}"
CONFIG_FILE="${SCRIPT_DIR}/config/${CONFIG_NAME}.yaml"

if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "ERROR: visual PPO config is missing: ${CONFIG_NAME}.yaml" >&2
    exit 2
fi
if [[ ! -f "${BASE_RUNNER}" ]]; then
    echo "ERROR: base launcher is missing: ${BASE_RUNNER}" >&2
    exit 2
fi
if [[ ! -f "${VISUAL_RUNTIME_DIR}/sitecustomize.py" ]]; then
    echo "ERROR: visual runtime hook is missing: ${VISUAL_RUNTIME_DIR}" >&2
    exit 2
fi

# Refuse to run unless the selected config explicitly pins Batch V2.
for setting in \
    'visual_stl_yoloe_per_gpu: true' \
    'visual_stl_yoloe_batch_enabled: true' \
    'visual_stl_yoloe_batch_max_size: 32' \
    'visual_stl_yoloe_raw_transport: true'; do
    count="$(grep -F -c "${setting}" "${CONFIG_FILE}" || true)"
    if [[ "${count}" -lt 2 ]]; then
        echo "ERROR: train/eval config does not pin YOLOE Batch V2: ${setting}" >&2
        exit 4
    fi
done

for gpu in ${GPU_IDS}; do
    port=$((BASE_PORT + gpu))
    health="$(curl -fsS --max-time 3 "http://127.0.0.1:${port}/health" 2>/dev/null || true)"
    if [[ "${health}" != *"batch raw"* ]]; then
        echo "ERROR: YOLOE service is not healthy: http://127.0.0.1:${port}" >&2
        echo "Expected a Batch V2 health body containing: batch raw" >&2
        echo "Start the SpecRL_V Batch V2 per-GPU launcher first." >&2
        exit 3
    fi
    echo "YOLOE_BATCH_V2_OK GPU=${gpu} PORT=${port} HEALTH=${health}"
done

echo "Visual PPO preflight passed for GPUs: ${GPU_IDS}"
echo "Reward source: pure RGB-D + robot proprioception"
echo "YOLOE acceleration: Batch V2 + raw transport + per-GPU routing"
: "${OPENPI_DATA_HOME:?Set OPENPI_DATA_HOME to the OpenPI data directory}"

export RLINF_USE_VISUAL_AGM_STAGE_ENV=1
export PYTHONPATH="${VISUAL_RUNTIME_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OPENPI_DATA_HOME

exec bash "${BASE_RUNNER}" "${CONFIG_NAME}" "${2:-LIBERO}"
