#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
config_file="$PWD/config/g1d_tool_portal.env"
if [[ ! -f "$config_file" ]]; then
  echo "Missing $config_file. Copy config/g1d_tool_portal.env.example and configure device addresses first." >&2
  exit 1
fi
# Use simple KEY=value entries compatible with both Bash and systemd.
set -a
source "$config_file"
set +a
exec /usr/bin/python3 -m services.g1d_tool_portal.server "$@"
