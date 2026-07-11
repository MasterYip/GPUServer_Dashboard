#!/usr/bin/env bash
# Uninstall the GPU Server Dashboard systemd user service.
#
# Usage:
#   ./uninstall_service.sh

set -euo pipefail

SERVICE_NAME="gpu-dashboard"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
UNIT_FILE="$UNIT_DIR/$SERVICE_NAME.service"

echo "Stopping $SERVICE_NAME..."
systemctl --user stop "$SERVICE_NAME" 2>/dev/null || true

echo "Disabling $SERVICE_NAME..."
systemctl --user disable "$SERVICE_NAME" 2>/dev/null || true

if [[ -f "$UNIT_FILE" ]]; then
    echo "Removing $UNIT_FILE..."
    rm -f "$UNIT_FILE"
    systemctl --user daemon-reload
fi

echo ""
echo "Service '$SERVICE_NAME' removed."
