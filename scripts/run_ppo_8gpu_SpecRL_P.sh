#!/usr/bin/env bash
# SpecRL privileged-state launcher.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG_NAME="libero_40_ppo_openpi_pi05_SpecRL_P"
TRAIN_VENV="${RLINF_TRAIN_VENV:-${ROOT_DIR}/.venv_embodied_openpi}"
TRAIN_VENV_ACTIVATE="${TRAIN_VENV}/bin/activate"
TRAIN_PYTHON="${TRAIN_VENV}/bin/python"
SFT_CHECKPOINT="${_SFT_CHECKPOINT:-${ROOT_DIR}/checkpoints/openpi_sft/SFT_checkpoint}"

export _SFT_CHECKPOINT="${SFT_CHECKPOINT}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${ROOT_DIR}/openpi_data}"

TRAIN_LAUNCHER="${ROOT_DIR}/examples/embodiment/run_SpecRL_P.sh"
PLAN_JSON="${ROOT_DIR}/llm_symbolic_parity_results_revised.json"

die() { echo "ERROR: $*" >&2; exit 1; }

# --- Preflight: files and GPUs ---------------------------------------------
[[ -f "${TRAIN_LAUNCHER}" ]] || die "training launcher missing: ${TRAIN_LAUNCHER}"
[[ -f "${TRAIN_VENV_ACTIVATE}" ]] || die "training venv activation script missing: ${TRAIN_VENV_ACTIVATE}"
[[ -x "${TRAIN_PYTHON}" ]]    || die "training python missing or not executable: ${TRAIN_PYTHON}"
[[ -d "${SFT_CHECKPOINT}" ]] || die "SFT checkpoint missing: ${SFT_CHECKPOINT}"
[[ -f "${PLAN_JSON}" ]]      || die "LLM stage plan JSON missing: ${PLAN_JSON}"
[[ -f "examples/embodiment/config/${CONFIG_NAME}.yaml" ]] || die "config missing: ${CONFIG_NAME}.yaml"
export RLINF_EXPECTED_PYTHON="$(readlink -f "${TRAIN_PYTHON}")"

test "$(nvidia-smi -L | wc -l)" -eq 8 || die "expected 8 GPUs, got $(nvidia-smi -L | wc -l)"

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
assert torch.__version__.startswith("2.6.0"), f"unexpected torch: {torch.__version__}"
assert torch.cuda.device_count() == 8, f"GPU_COUNT={torch.cuda.device_count()}"
print("TRAIN_ENV_OK torch=", torch.__version__)
PY

# --- PPO training (foreground; its exit code becomes the job's) -----------
bash "${TRAIN_LAUNCHER}" "${CONFIG_NAME}"
