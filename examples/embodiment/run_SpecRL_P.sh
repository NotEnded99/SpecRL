#! /bin/bash
# Privileged-state SpecRL launcher.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_RUNNER="${SCRIPT_DIR}/run_embodiment.sh"
AGM_RUNTIME_DIR="${SCRIPT_DIR}/agm_stage_runtime"
SITE_CUSTOMIZE="${AGM_RUNTIME_DIR}/sitecustomize.py"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PLAN_JSON="${REPO_ROOT}/llm_symbolic_parity_results_revised.json"

if [ ! -f "${BASE_RUNNER}" ]; then
    echo "ERROR: 找不到基础启动脚本：" >&2
    echo "  ${BASE_RUNNER}" >&2
    exit 1
fi

if [ ! -f "${SITE_CUSTOMIZE}" ]; then
    echo "ERROR: 找不到 AGM 环境覆盖文件：" >&2
    echo "  ${SITE_CUSTOMIZE}" >&2
    exit 1
fi

if [ ! -f "${PLAN_JSON}" ]; then
    echo "ERROR: 找不到 stage-plan JSON：" >&2
    echo "  ${PLAN_JSON}" >&2
    exit 1
fi

if [ "$#" -lt 1 ]; then
    echo "用法：" >&2
    echo "  bash $0 <config_name> [ROBOT_PLATFORM]" >&2
    echo >&2
    echo "例如：" >&2
    echo "  bash $0 libero_40_ppo_openpi_pi05_SpecRL_P" >&2
    exit 2
fi

export RLINF_USE_AGM_STAGE_ENV=1

export PYTHONPATH="${AGM_RUNTIME_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

export RLINF_AGM_STAGE_PLAN_JSON="${RLINF_AGM_STAGE_PLAN_JSON:-${PLAN_JSON}}"

: "${OPENPI_DATA_HOME:?Set OPENPI_DATA_HOME to the OpenPI data directory}"
export OPENPI_DATA_HOME

echo "============================================================"
echo "SpecRL_P (privileged-state grounding) training launcher"
echo "Base runner          : ${BASE_RUNNER}"
echo "AGM runtime directory: ${AGM_RUNTIME_DIR}"
echo "AGM environment      : ${RLINF_USE_AGM_STAGE_ENV}"
echo "Stage plan JSON      : ${RLINF_AGM_STAGE_PLAN_JSON}"
echo "OPENPI_DATA_HOME     : ${OPENPI_DATA_HOME}"
echo "Config name          : $1"
echo "Arguments             : $*"
echo "============================================================"

# Delegate to the base training launcher.
exec bash "${BASE_RUNNER}" "$@"
