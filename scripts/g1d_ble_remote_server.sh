#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

args=(
  scripts/g1d_ble_remote_server.py
  --adapter "${G1D_BLE_ADAPTER:-hci0}"
  --local-name "${G1D_BLE_LOCAL_NAME:-G1D}"
  --adapter-alias "${G1D_BLE_ADAPTER_ALIAS:-${G1D_BLE_LOCAL_NAME:-G1D}}"
  --sdk-build-dir "${UNITREE_SDK_BUILD_DIR:-/home/unitree/unitree_sdk2/build}"
  --interface "${DDS_INTERFACE:-eth0}"
  --watchdog-sec "${G1D_BLE_WATCHDOG_SEC:-0.6}"
  --hold-duration-sec "${G1D_BLE_HOLD_DURATION_SEC:-600}"
)

if [ -n "${G1D_BLE_ADVERTISE_SERVICE_UUID:-}" ]; then
  args+=(--advertise-service-uuid)
fi

if [ -n "${G1D_BLE_EXECUTE_ENABLED:-}" ]; then
  args+=(--execute-enabled)
fi

exec "${PYTHON:-python3}" "${args[@]}" "$@"
