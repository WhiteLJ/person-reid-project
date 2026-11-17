#!/usr/bin/env bash
# Install the Atlas camera application for the current graphical user session.
# This installs both a desktop launcher and a user-level autostart entry.
# It does not configure login autologin; use configure_camera_autologin.sh for
# that one-time system setting.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LAUNCHER_TEMPLATE="${SCRIPT_DIR}/person-reid-camera.desktop.in"
AUTOSTART_TEMPLATE="${SCRIPT_DIR}/person-reid-camera-autostart.desktop.in"

TARGET_USER="${SUDO_USER:-${USER:-}}"
if [[ -z "${TARGET_USER}" || "${TARGET_USER}" == "root" ]]; then
    echo "Run this script as the graphical desktop user, not as root." >&2
    echo "If you used sudo, keep the original user in SUDO_USER." >&2
    exit 1
fi

TARGET_HOME="$(getent passwd "${TARGET_USER}" | cut -d: -f6)"
if [[ -z "${TARGET_HOME}" || ! -d "${TARGET_HOME}" ]]; then
    echo "Cannot find home directory for user: ${TARGET_USER}" >&2
    exit 1
fi

if [[ ! -f "${LAUNCHER_TEMPLATE}" || ! -f "${AUTOSTART_TEMPLATE}" ]]; then
    echo "Camera launcher template is missing." >&2
    exit 1
fi

if [[ -n "${XDG_DESKTOP_DIR:-}" ]]; then
    DESKTOP_DIR="${XDG_DESKTOP_DIR}"
else
    DESKTOP_DIR="${TARGET_HOME}/Desktop"
fi
AUTOSTART_DIR="${TARGET_HOME}/.config/autostart"
LAUNCHER="${DESKTOP_DIR}/Person-ReID-Camera.desktop"
AUTOSTART_ENTRY="${AUTOSTART_DIR}/Person-ReID-Camera.desktop"

mkdir -p "${DESKTOP_DIR}" "${AUTOSTART_DIR}"

# Escape the replacement for the sed delimiter and replacement characters.
escaped_repo="${REPO_DIR//\\/\\\\}"
escaped_repo="${escaped_repo//&/\\&}"
escaped_repo="${escaped_repo//|/\\|}"

sed "s|@REPO_DIR@|${escaped_repo}|g" "${LAUNCHER_TEMPLATE}" > "${LAUNCHER}"
sed "s|@REPO_DIR@|${escaped_repo}|g" "${AUTOSTART_TEMPLATE}" > "${AUTOSTART_ENTRY}"

chmod +x "${LAUNCHER}"
if [[ "$(id -u)" -eq 0 ]]; then
    chown "${TARGET_USER}:$(id -gn "${TARGET_USER}")" "${LAUNCHER}" "${AUTOSTART_ENTRY}"
fi

echo "Desktop launcher installed: ${LAUNCHER}"
echo "User autostart installed: ${AUTOSTART_ENTRY}"
echo "The application will start after ${TARGET_USER} logs into the graphical desktop."
echo "To remove automatic application start, delete: ${AUTOSTART_ENTRY}"
