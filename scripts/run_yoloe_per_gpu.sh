#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER="${ROOT_DIR}/scripts/yoloe_server.py"
PYTHON_BIN="${YOLOE_PYTHON:-python}"
BASE_PORT="${YOLOE_BASE_PORT:-8010}"
GPU_IDS="${YOLOE_GPU_IDS:-0 1 2 3 4 5 6 7}"
LOG_DIR="${YOLOE_LOG_DIR:-${ROOT_DIR}/logs/yoloe_per_gpu}"
STARTUP_TIMEOUT="${YOLOE_STARTUP_TIMEOUT:-300}"

if [[ -z "${YOLOE_WEIGHTS:-}" ]]; then
    echo "ERROR: set YOLOE_WEIGHTS to the YOLOE checkpoint path" >&2
    exit 2
fi
if [[ ! -f "${YOLOE_WEIGHTS}" ]]; then
    echo "ERROR: YOLOE_WEIGHTS does not exist: ${YOLOE_WEIGHTS}" >&2
    exit 2
fi
if [[ ! -f "${SERVER}" ]]; then
    echo "ERROR: YOLOE server is missing: ${SERVER}" >&2
    exit 2
fi

mkdir -p "${LOG_DIR}"
declare -a PIDS=()

cleanup() {
    local pid
    for pid in "${PIDS[@]:-}"; do
        kill -TERM "${pid}" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for gpu in ${GPU_IDS}; do
    port=$((BASE_PORT + gpu))
    if curl -fsS --max-time 2 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
        echo "ERROR: port ${port} already has a healthy service; refusing to replace it" >&2
        exit 3
    fi
    log_file="${LOG_DIR}/gpu${gpu}_port${port}.log"
    echo "START gpu=${gpu} port=${port} log=${log_file}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${SERVER}" \
        --weights "${YOLOE_WEIGHTS}" \
        --host 127.0.0.1 \
        --port "${port}" \
        --device 0 \
        >"${log_file}" 2>&1 &
    pid=$!
    PIDS+=("${pid}")
    echo "${pid}" >"${LOG_DIR}/gpu${gpu}.pid"
done

deadline=$((SECONDS + STARTUP_TIMEOUT))
for gpu in ${GPU_IDS}; do
    port=$((BASE_PORT + gpu))
    until curl -fsS --max-time 2 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; do
        if (( SECONDS >= deadline )); then
            echo "ERROR: YOLOE startup timed out; inspect ${LOG_DIR}" >&2
            exit 4
        fi
        sleep 2
    done
    echo "READY gpu=${gpu} url=http://127.0.0.1:${port}"
done

echo "ALL_READY gpu_ids=${GPU_IDS}"
wait -n
echo "ERROR: a YOLOE service exited; inspect ${LOG_DIR}" >&2
exit 5
