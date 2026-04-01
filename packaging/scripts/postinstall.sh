#!/usr/bin/env bash
# Post-install: set up XDG config, notify systemd user daemon.
set -euo pipefail

# Ensure config directory permissions
chmod 755 /etc/e-clawhisper
chmod 644 /etc/e-clawhisper/config.yaml
chmod 640 /etc/e-clawhisper/.env

# Create user-level config (XDG) for the invoking user
REAL_USER="${SUDO_USER:-$USER}"
REAL_HOME="$(getent passwd "$REAL_USER" | cut -d: -f6)"
XDG_DIR="$REAL_HOME/.config/e-clawhisper"

if [ -n "$REAL_HOME" ]; then
    # Config
    if [ ! -f "$XDG_DIR/config.yaml" ]; then
        mkdir -p "$XDG_DIR"
        cp /etc/e-clawhisper/config.yaml "$XDG_DIR/config.yaml"
        chmod 644 "$XDG_DIR/config.yaml"
    fi
    # Env (secrets) — create empty if not exists
    if [ ! -f "$XDG_DIR/.env" ]; then
        cp /etc/e-clawhisper/.env "$XDG_DIR/.env"
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
echo "e-clawhisper installed successfully."
echo ""
echo "  Config:   ~/.config/e-clawhisper/config.yaml"
echo "  Secrets:  ~/.config/e-clawhisper/.env"
echo "  Start:    systemctl --user start e-clawhisper"
echo "  Stop:     systemctl --user stop e-clawhisper"
echo "  Status:   eclaw session status"
echo "  Logs:     eclaw session logs"
echo "  Re-init:  eclaw config init"
echo ""
echo "  1. Set your API key in ~/.config/e-clawhisper/.env:"
echo "     echo 'GOOGLE_API_KEY=your-key' >> ~/.config/e-clawhisper/.env"
echo "  2. Edit config:  nano ~/.config/e-clawhisper/config.yaml"
echo "  3. Start:        systemctl --user start e-clawhisper"
echo "  4. Auto-start:   systemctl --user enable e-clawhisper"
echo ""
