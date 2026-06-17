#!/usr/bin/env bash
set -eo pipefail

cd "$(dirname "$0")/.."

for setup in /opt/ros/noetic/setup.bash /opt/ros/melodic/setup.bash; do
  if [ -f "$setup" ]; then
    # Needed when /api/robot_state reads ROS /joint_states through rostopic.
    # shellcheck disable=SC1090
    set +u
    source "$setup"
    set -u
    break
  fi
done

set -u

if [ -d "${UNITREE_SDK2PY_PATH:-/home/unitree/unitree_sdk2_python}" ]; then
  export PYTHONPATH="${UNITREE_SDK2PY_PATH:-/home/unitree/unitree_sdk2_python}${PYTHONPATH:+:$PYTHONPATH}"
fi

exec "${PYTHON:-python3}" scripts/g1d_cigarette_visualizer_server.py \
  --bind "${VISUALIZER_BIND:-0.0.0.0}" \
  --port "${VISUALIZER_PORT:-18085}" \
  --xyz-url "${VISUALIZER_XYZ_URL:-http://127.0.0.1:18081/xyz}" \
  --dds-interface "${DDS_INTERFACE:-eth0}" \
  --dds-lowstate-topic "${DDS_LOWSTATE_TOPIC:-rt/lowstate}" \
  --dds-hispeed-topic "${DDS_HISPEED_TOPIC:-rt/hispeed_state}" \
  --unitree-sdk2py-path "${UNITREE_SDK2PY_PATH:-/home/unitree/unitree_sdk2_python}" \
  --joint-states-topic "${JOINT_STATES_TOPIC:-/joint_states}" \
  --column-raw-min-mm "${COLUMN_RAW_MIN_MM:-0.0}" \
  --column-raw-max-mm "${COLUMN_RAW_MAX_MM:-246.9}" \
  --column-visual-max-mm "${COLUMN_VISUAL_MAX_MM:-420.0}" \
  --timeout-sec "${VISUALIZER_TIMEOUT_SEC:-2.0}" \
  "$@"
