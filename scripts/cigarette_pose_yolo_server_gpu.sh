#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source /home/unitree/venvs/tv_gpu/bin/activate
export PYTHONPATH="$PWD:$PWD/scripts:${PYTHONPATH:-}"

exec python scripts/cigarette_pose_yolo_server.py --bind 0.0.0.0 --port 18081 --yolo-device cuda:0 "$@"
