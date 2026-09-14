#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
config_file="$PWD/config/suction_api_proxy.env"
if [[ ! -f "$config_file" ]]; then
  echo "Missing $config_file. Copy config/suction_api_proxy.env.example and configure device addresses first." >&2
  exit 1
fi
# Use simple KEY=value entries compatible with both Bash and systemd.
set -a
source "$config_file"
set +a
exec /usr/bin/python3 -m services.es80z_api_proxy.server "$@"
