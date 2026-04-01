#!/usr/bin/env bash
# Post-remove: reload systemd, clean runtime state.
set -euo pipefail

# Reload system systemd (legacy cleanup)
systemctl daemon-reload 2>/dev/null || true

# Reload user systemd daemon
REAL_USER="${SUDO_USER:-$USER}"
REAL_UID="$(id -u "$REAL_USER" 2>/dev/null || true)"
if [ -n "$REAL_UID" ] && [ -S "/run/user/$REAL_UID/bus" ]; then
    sudo -u "$REAL_USER" XDG_RUNTIME_DIR="/run/user/$REAL_UID" \
        systemctl --user daemon-reload 2>/dev/null || true
fi

# Note: ~/.config/e-heed/ is preserved (user config).
# /opt/e-heed/ is removed by the package manager.
