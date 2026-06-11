#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ -f /home/unitree/venvs/tv_gpu/bin/activate ]; then
  source /home/unitree/venvs/tv_gpu/bin/activate
elif [ -f /home/unitree/miniconda3/etc/profile.d/conda.sh ]; then
  source /home/unitree/miniconda3/etc/profile.d/conda.sh
  conda activate tv
elif [ -x /home/unitree/miniconda3/envs/tv/bin/python ]; then
  export PATH="/home/unitree/miniconda3/envs/tv/bin:$PATH"
fi

export PYTHONPATH="$PWD:$PWD/scripts:${PYTHONPATH:-}"

exec python scripts/g1d_pose_adjust_service.py --bind 0.0.0.0 --port 18084 "$@"
