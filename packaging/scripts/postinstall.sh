#!/usr/bin/env bash
# Post-install: set up XDG config, notify systemd user daemon.
set -euo pipefail

# Ensure config directory permissions
chmod 755 /etc/e-heed
chmod 644 /etc/e-heed/config.yaml
chmod 640 /etc/e-heed/.env

# Create user-level config (XDG) for the invoking user
REAL_USER="${SUDO_USER:-$USER}"
REAL_HOME="$(getent passwd "$REAL_USER" | cut -d: -f6)"
XDG_DIR="$REAL_HOME/.config/e-heed"

if [ -n "$REAL_HOME" ]; then
    # Config
    if [ ! -f "$XDG_DIR/config.yaml" ]; then
        mkdir -p "$XDG_DIR"
        cp /etc/e-heed/config.yaml "$XDG_DIR/config.yaml"
        chmod 644 "$XDG_DIR/config.yaml"
    fi
    # Env (secrets) — create empty if not exists
    if [ ! -f "$XDG_DIR/.env" ]; then
        cp /etc/e-heed/.env "$XDG_DIR/.env"
        chmod 600 "$XDG_DIR/.env"
    fi
    chown -R "$REAL_USER":"$REAL_USER" "$XDG_DIR"
fi

# Reload user systemd daemon (if running as the user's session)
REAL_UID="$(id -u "$REAL_USER" 2>/dev/null || true)"
if [ -n "$REAL_UID" ] && [ -S "/run/user/$REAL_UID/bus" ]; then
    sudo -u "$REAL_USER" XDG_RUNTIME_DIR="/run/user/$REAL_UID" systemctl --user daemon-reload 2>/dev/null || true
fi

echo ""
echo "e-heed installed successfully."
echo ""
echo "  Config:   ~/.config/e-heed/config.yaml"
echo "  Secrets:  ~/.config/e-heed/.env"
echo "  Start:    systemctl --user start e-heed"
echo "  Stop:     systemctl --user stop e-heed"
echo "  Status:   eheed session status"
echo "  Logs:     eheed session logs"
echo "  Re-init:  eheed config init"
echo ""
echo "  1. Set your API key in ~/.config/e-heed/.env:"
echo "     echo 'GOOGLE_API_KEY=your-key' >> ~/.config/e-heed/.env"
echo "  2. Edit config:  nano ~/.config/e-heed/config.yaml"
echo "  3. Start:        systemctl --user start e-heed"
echo "  4. Auto-start:   systemctl --user enable e-heed"
echo ""
