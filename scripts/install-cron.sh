#!/usr/bin/env bash
# Copyright 2026 David McCarty. All rights reserved.
# Install hermes-memory-dream systemd units (timer + service).
# Idempotent — re-running is safe.

set -euo pipefail

SYSTEMD_DIR="/etc/systemd/system"
AGENT_DIR="${HOME}/.hermes/hermes-agent"
SOURCE_DIR="${AGENT_DIR}/systemd"

echo "==> Installing hermes-memory-dream systemd units..."

# Verify source files exist
for unit in "${SOURCE_DIR}/hermes-memory-dream.service" "${SOURCE_DIR}/hermes-memory-dream.timer"; do
    if [[ ! -f "${unit}" ]]; then
        echo "ERROR: Unit file not found: ${unit}" >&2
        exit 1
    fi
done

# Copy units to systemd directory
cp "${SOURCE_DIR}/hermes-memory-dream.service" "${SYSTEMD_DIR}/"
cp "${SOURCE_DIR}/hermes-memory-dream.timer" "${SYSTEMD_DIR}/"

# Ensure the log directory exists
mkdir -p "${HOME}/.hermes/logs"

# Reload systemd, enable and start the timer
systemctl daemon-reload
systemctl enable --now hermes-memory-dream.timer

# Verify timer is registered
if systemctl list-timers --no-legend | grep -q "hermes-memory-dream.timer"; then
    echo "==> Timer installed and active:"
    systemctl list-timers hermes-memory-dream.timer
else
    echo "WARNING: Timer not showing in list-timers yet. Check 'systemctl status hermes-memory-dream.timer'"
fi

echo "==> Installation complete."