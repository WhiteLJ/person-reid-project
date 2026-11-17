#!/usr/bin/env bash
# Configure graphical autologin for the Atlas desktop user.
# The password is requested by sudo during setup only and is never stored.

set -euo pipefail

TARGET_USER="${1:-${TARGET_USER:-${SUDO_USER:-${USER:-}}}}"
if [[ -z "${TARGET_USER}" || ( "${TARGET_USER}" == "root" && "$(id -u)" -ne 0 ) ]]; then
    echo "Run this script as the desktop user, or pass the desktop user after su -:" >&2
    echo "  bash deploy/atlas/configure_camera_autologin.sh" >&2
    echo "  bash deploy/atlas/configure_camera_autologin.sh HwHiAiUser" >&2
    exit 1
fi

if [[ "$(id -u)" -eq 0 && "${TARGET_USER}" == "root" ]]; then
    echo "Do not configure autologin for root; pass the graphical desktop user." >&2
    exit 1
fi

if ! id "${TARGET_USER}" >/dev/null 2>&1; then
    echo "Unknown desktop user: ${TARGET_USER}" >&2
    exit 1
fi

DISPLAY_MANAGER_TARGET="$(readlink -f /etc/systemd/system/display-manager.service 2>/dev/null || true)"
if [[ -z "${DISPLAY_MANAGER_TARGET}" && -e /etc/systemd/system/display-manager.service ]]; then
    DISPLAY_MANAGER_TARGET="$(systemctl show -p FragmentPath --value display-manager.service 2>/dev/null || true)"
fi

case "${DISPLAY_MANAGER_TARGET}" in
    */lightdm.service|*lightdm*)
        if [[ "$(id -u)" -eq 0 ]]; then
            SUDO_CMD=()
        else
            sudo -v
            SUDO_CMD=(sudo)
        fi
        # Keep the privileged operations small so boards with restricted
        # sudoers rules do not need /usr/bin/install permission. When this
        # script is run from `su -`, use the already-root shell directly.
        "${SUDO_CMD[@]}" mkdir -p /etc/lightdm/lightdm.conf.d
        printf '[Seat:*]\nautologin-user=%s\nautologin-user-timeout=0\n' "${TARGET_USER}" \
            | "${SUDO_CMD[@]}" tee /etc/lightdm/lightdm.conf.d/50-person-reid-autologin.conf >/dev/null
        echo "LightDM autologin configured for ${TARGET_USER}."
        ;;
    */sddm.service|*sddm*)
        if [[ "$(id -u)" -eq 0 ]]; then
            SUDO_CMD=()
        else
            sudo -v
            SUDO_CMD=(sudo)
        fi
        # SDDM reads this root-owned drop-in during the next graphical login.
        "${SUDO_CMD[@]}" mkdir -p /etc/sddm.conf.d
        printf '[Autologin]\nUser=%s\nRelogin=true\n' "${TARGET_USER}" \
            | "${SUDO_CMD[@]}" tee /etc/sddm.conf.d/person-reid-autologin.conf >/dev/null
        echo "SDDM autologin configured for ${TARGET_USER}."
        ;;
    */gdm.service|*/gdm3.service|*gdm.service|*gdm3*)
        echo "GDM was detected, but this installer does not modify its existing custom.conf automatically." >&2
        echo "Add these settings under [daemon] in /etc/gdm3/custom.conf:" >&2
        echo "  AutomaticLoginEnable = true" >&2
        echo "  AutomaticLogin = ${TARGET_USER}" >&2
        exit 2
        ;;
    *)
        echo "Unsupported or undetected display manager: ${DISPLAY_MANAGER_TARGET:-unknown}" >&2
        echo "Inspect: systemctl status display-manager.service" >&2
        echo "Configure graphical autologin for ${TARGET_USER} in that display manager, then keep the user autostart entry." >&2
        exit 2
        ;;
esac

echo "Reboot once to verify password-free graphical login and camera autostart."
