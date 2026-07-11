#!/usr/bin/env bash
# Restart the GPU Server Dashboard systemd user service.
#
# Usage:
#   ./restart_service.sh

set -euo pipefail

SERVICE_NAME="gpu-dashboard"

echo "Restarting $SERVICE_NAME..."
systemctl --user restart "$SERVICE_NAME"

echo ""
echo "Service status:"
systemctl --user status "$SERVICE_NAME" --no-pager || true
