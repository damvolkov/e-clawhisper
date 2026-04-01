#!/usr/bin/env bash
# Pre-remove: stop and disable the user service for all active users.
set -euo pipefail

# Stop system service (legacy — in case upgrading from system service)
systemctl stop e-clawhisper.service 2>/dev/null || true
systemctl disable e-clawhisper.service 2>/dev/null || true

# Stop user service for the invoking user
REAL_USER="${SUDO_USER:-$USER}"
REAL_UID="$(id -u "$REAL_USER" 2>/dev/null || true)"
if [ -n "$REAL_UID" ] && [ -S "/run/user/$REAL_UID/bus" ]; then
    sudo -u "$REAL_USER" XDG_RUNTIME_DIR="/run/user/$REAL_UID" \
        systemctl --user stop e-clawhisper.service 2>/dev/null || true
    sudo -u "$REAL_USER" XDG_RUNTIME_DIR="/run/user/$REAL_UID" \
        systemctl --user disable e-clawhisper.service 2>/dev/null || true
fi
