#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

legacy_services=(
  cigarette-pose-yolo.service
  g1d-cigarette-visualizer.service
  g1d-pose-adjust.service
  g1d-remote-control.service
  g1d-ble-remote.service
)

# No arguments retain the existing five-service installation behavior.
services=()
if [[ "$#" == 0 ]]; then
  set -- legacy
fi
for selection in "$@"; do
  case "$selection" in
    legacy) services+=("${legacy_services[@]}") ;;
    portal) services+=(g1d-tool-portal.service) ;;
    suction-proxy) services+=(suction-api-proxy.service) ;;
    body-control) services+=(g1d-body-control.service) ;;
    all) services+=("${legacy_services[@]}" g1d-tool-portal.service suction-api-proxy.service g1d-body-control.service) ;;
    *) echo "Usage: $0 [legacy|portal|suction-proxy|body-control|all] [...]" >&2; exit 1 ;;
  esac
done

# Complete all preflight checks before changing system services.
missing_config=0
for service in "${services[@]}"; do
  case "$service" in
    g1d-tool-portal.service) config=config/g1d_tool_portal.env ;;
    suction-api-proxy.service) config=config/suction_api_proxy.env ;;
    g1d-body-control.service) config=config/g1d_body_control.env ;;
    *) continue ;;
  esac
  if [[ ! -f "$config" ]]; then
    cp "${config}.example" "$config"
    echo "Created $config. Configure device addresses, then rerun installation." >&2
    missing_config=1
  fi
done
if [[ "$missing_config" == 1 ]]; then
  exit 1
fi

# Unit templates intentionally match the standard robot account and path.
if [[ "$PWD" != /home/unitree/YOLO_YAN_COORDINATION ]]; then
  echo "Install from /home/unitree/YOLO_YAN_COORDINATION (systemd template path)." >&2
  exit 1
fi

backup_dir="/var/backups/yolo-services/$(date +%Y%m%d-%H%M%S)-$$"
for service in "${services[@]}"; do
  if [[ -f "/etc/systemd/system/$service" ]]; then
    sudo mkdir -p "$backup_dir"
    sudo cp -a "/etc/systemd/system/$service" "$backup_dir/"
    if [[ -d "/etc/systemd/system/${service}.d" ]]; then
      sudo cp -a "/etc/systemd/system/${service}.d" "$backup_dir/"
    fi
    echo "Backed up $service to $backup_dir"
  fi
  sudo cp "systemd/${service}" /etc/systemd/system/
done

sudo systemctl daemon-reload

for service in "${services[@]}"; do
  case "$service" in
    g1d-tool-portal.service|suction-api-proxy.service|g1d-body-control.service)
      sudo systemctl enable "$service"
      # Existing standalone processes must restart to pick up the new code path.
      sudo systemctl restart "$service"
      ;;
    *) sudo systemctl enable --now "$service" ;;
  esac
done

for service in "${services[@]}"; do
  printf '%s: ' "${service}"
  systemctl is-enabled "${service}"
done

for service in "${services[@]}"; do
  printf '%s: ' "${service}"
  systemctl is-active "${service}"
done
