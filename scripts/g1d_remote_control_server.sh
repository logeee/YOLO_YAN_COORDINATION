#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

args=(
  scripts/g1d_remote_control_server.py
  --bind "${REMOTE_CONTROL_BIND:-0.0.0.0}"
  --port "${REMOTE_CONTROL_PORT:-18086}"
  --sdk-build-dir "${UNITREE_SDK_BUILD_DIR:-/home/unitree/unitree_sdk2/build}"
  --interface "${DDS_INTERFACE:-eth0}"
)

if [ -n "${REMOTE_CONTROL_EXECUTE_ENABLED:-}" ]; then
  args+=(--execute-enabled)
fi

exec "${PYTHON:-python3}" "${args[@]}" "$@"
