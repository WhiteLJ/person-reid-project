#!/usr/bin/env bash
# Start the Atlas application with USB camera index 0.
# This script intentionally does not accept or open a video file.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ASCEND_ENV="${ASCEND_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
CAMERA_INDEX="${CAMERA_INDEX:-0}"

if [[ -f "${ASCEND_ENV}" ]]; then
    # shellcheck disable=SC1090
    if ! source "${ASCEND_ENV}"; then
        echo "Failed to load Ascend environment: ${ASCEND_ENV}" >&2
        exit 1
    fi
else
    echo "Ascend environment script not found: ${ASCEND_ENV}" >&2
    exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
    for candidate in \
        "${HOME}/anaconda3/envs/person-reid/bin/python" \
        "${HOME}/miniconda3/envs/person-reid/bin/python" \
        "${HOME}/miniforge3/envs/person-reid/bin/python" \
        "/usr/local/miniconda3/envs/person-reid/bin/python" \
        "/opt/conda/envs/person-reid/bin/python"; do
        if [[ -x "${candidate}" ]]; then
            PYTHON_BIN="${candidate}"
            break
        fi
    done
fi

# CANN's set_env.sh may already expose the board's conda environment in PATH.
if [[ -z "${PYTHON_BIN}" ]] && command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
fi

if [[ -z "${PYTHON_BIN}" ]]; then
    echo "Python environment 'person-reid' was not found." >&2
    echo "Set PYTHON_BIN to that environment's Python executable." >&2
    exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable is not usable: ${PYTHON_BIN}" >&2
    exit 1
fi

cd "${REPO_DIR}"
echo "Starting Person + Vehicle Atlas application on camera ${CAMERA_INDEX}"
exec "${PYTHON_BIN}" "${REPO_DIR}/app.py" \
    --config "${REPO_DIR}/config/config_atlas.yaml" \
    --source "${CAMERA_INDEX}"
