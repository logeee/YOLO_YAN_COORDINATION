#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
exec python scripts/g1d_cigarette_visualizer_server.py --bind 0.0.0.0 --port 18085 "$@"
