#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source /home/unitree/venvs/tv_gpu/bin/activate
export PYTHONPATH="$PWD:$PWD/scripts:${PYTHONPATH:-}"

python scripts/cigarette_pose_optical_api.py --capture --pretty --yolo-device cuda:0 "$@"
