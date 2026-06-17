#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

services=(
  cigarette-pose-yolo.service
  g1d-cigarette-visualizer.service
  g1d-pose-adjust.service
  g1d-remote-control.service
)

for service in "${services[@]}"; do
  sudo cp "systemd/${service}" /etc/systemd/system/
done

sudo systemctl daemon-reload

for service in "${services[@]}"; do
  sudo systemctl enable --now "${service}"
done

for service in "${services[@]}"; do
  printf '%s: ' "${service}"
  systemctl is-enabled "${service}"
done

for service in "${services[@]}"; do
  printf '%s: ' "${service}"
  systemctl is-active "${service}"
done
