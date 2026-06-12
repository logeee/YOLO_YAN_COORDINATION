#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

for setup in /opt/ros/noetic/setup.bash /opt/ros/melodic/setup.bash; do
  if [ -f "$setup" ]; then
    # Needed when /api/robot_state reads ROS /joint_states through rostopic.
    # shellcheck disable=SC1090
    source "$setup"
    break
  fi
done

exec python scripts/g1d_cigarette_visualizer_server.py \
  --bind "${VISUALIZER_BIND:-0.0.0.0}" \
  --port "${VISUALIZER_PORT:-18085}" \
  --xyz-url "${VISUALIZER_XYZ_URL:-http://127.0.0.1:18081/xyz}" \
  --joint-states-topic "${JOINT_STATES_TOPIC:-/joint_states}" \
  --timeout-sec "${VISUALIZER_TIMEOUT_SEC:-2.0}" \
  "$@"
