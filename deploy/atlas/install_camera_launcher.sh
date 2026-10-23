#!/usr/bin/env bash
# Install a desktop button for the Atlas USB-camera application.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TEMPLATE="${SCRIPT_DIR}/person-reid-camera.desktop.in"
DESKTOP_DIR="${XDG_DESKTOP_DIR:-${HOME}/Desktop}"
LAUNCHER="${DESKTOP_DIR}/Person-ReID-Camera.desktop"

if [[ ! -f "${TEMPLATE}" ]]; then
    echo "Launcher template not found: ${TEMPLATE}" >&2
    exit 1
fi

mkdir -p "${DESKTOP_DIR}"

# Escape the replacement for the sed delimiter and replacement characters.
escaped_repo="${REPO_DIR//\\/\\\\}"
escaped_repo="${escaped_repo//&/\\&}"
escaped_repo="${escaped_repo//|/\\|}"
sed "s|@REPO_DIR@|${escaped_repo}|g" "${TEMPLATE}" > "${LAUNCHER}"
chmod +x "${LAUNCHER}"

echo "Desktop launcher installed: ${LAUNCHER}"
echo "Click 'Person ReID Camera' to start camera ${CAMERA_INDEX:-0}."
