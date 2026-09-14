#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
config_file="$PWD/config/g1d_body_control.env"
if [[ ! -f "$config_file" ]]; then
  echo "Missing $config_file. Copy config/g1d_body_control.env.example and configure this robot first." >&2
  exit 1
fi
set -a
source "$config_file"
set +a

source "${G1D_BODY_ROS_SETUP:-/opt/ros/foxy/setup.bash}"
slam_setup="${G1D_BODY_SLAM_SETUP:-/unitree/module/slamware_service_pc4/install/setup.bash}"
if [[ -f "$slam_setup" ]]; then
  source "$slam_setup"
fi
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export PYTHONUNBUFFERED=1
if [[ -f "${HOME}/cyclonedds.xml" ]]; then
  export CYCLONEDDS_URI="${CYCLONEDDS_URI:-file://${HOME}/cyclonedds.xml}"
fi
# ROS Foxy rclpy belongs to the system interpreter, not the YOLO Conda env.
exec /usr/bin/python3 -m services.g1d_body_control.node "$@"
