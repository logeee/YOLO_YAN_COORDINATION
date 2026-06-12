#!/usr/bin/env python3
"""Serve the G1-D cigarette relative-position visualizer."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
import shutil
import subprocess
import urllib.parse
import urllib.request
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


DEFAULT_BIND = "0.0.0.0"
DEFAULT_PORT = 18085
DEFAULT_XYZ_URL = "http://127.0.0.1:18081/xyz"
DEFAULT_JOINT_STATES_TOPIC = "/joint_states"
DEFAULT_DDS_INTERFACE = "eth0"
DEFAULT_DDS_LOWSTATE_TOPIC = "rt/lowstate"
DEFAULT_DDS_HISPEED_TOPIC = "rt/hispeed_state"
DEFAULT_UNITREE_SDK2PY_PATH = "/home/unitree/unitree_sdk2_python"
VIEWER_DIR = Path(__file__).resolve().parents[1] / "visualization" / "g1d_cigarette_viewer"
DEFAULT_ROBOT_STATE: dict[str, Any] = {
    "ok": True,
    "source": "visualizer_default",
    "column_extension_mm": 420.0,
    "joints": {
        "LZ_mt_Joint": 0.21,
        "LZ_it_Joint": 0.21,
    },
}
DDS_READER: "_DdsStateReader | None" = None

G1D_LOWSTATE_JOINT_MAP: dict[int, str] = {
    12: "torso_Joint",
    14: "Yaw_Joint",
    15: "left_shoulder_pitch_joint",
    16: "left_shoulder_roll_joint",
    17: "left_shoulder_yaw_joint",
    18: "left_elbow_joint",
    19: "left_wrist_roll_joint",
    20: "left_wrist_pitch_joint",
    21: "left_wrist_yaw_joint",
    22: "right_shoulder_pitch_joint",
    23: "right_shoulder_roll_joint",
    24: "right_shoulder_yaw_joint",
    25: "right_elbow_joint",
    26: "right_wrist_roll_joint",
    27: "right_wrist_pitch_joint",
    28: "right_wrist_yaw_joint",
}


def _json_response(handler: SimpleHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(data)


def _fetch_json(url: str, timeout_sec: float) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=float(timeout_sec)) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def _read_member(obj: Any, name: str, default: Any = None) -> Any:
    value = getattr(obj, name, default)
    if callable(value):
        return value()
    return value


def _round_float(value: Any, ndigits: int = 6) -> float | None:
    try:
        return round(float(value), ndigits)
    except Exception:
        return None


class _DdsStateReader:
    def __init__(
        self,
        *,
        network_interface: str,
        lowstate_topic: str,
        hispeed_topic: str,
        sdk2py_path: str,
    ) -> None:
        if sdk2py_path and sdk2py_path not in sys.path and Path(sdk2py_path).exists():
            sys.path.insert(0, sdk2py_path)
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Point32_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

        self.network_interface = network_interface
        self.lowstate_topic = lowstate_topic
        self.hispeed_topic = hispeed_topic
        self.sdk2py_path = sdk2py_path
        self.lock = threading.Lock()
        self.lowstate_msg: Any = None
        self.hispeed_msg: Any = None
        self.lowstate_updated_at = 0.0
        self.hispeed_updated_at = 0.0

        ChannelFactoryInitialize(0, network_interface)
        self.lowstate_subscriber = ChannelSubscriber(lowstate_topic, LowState_)
        self.lowstate_subscriber.Init(self._on_lowstate, 10)
        self.hispeed_subscriber = ChannelSubscriber(hispeed_topic, Point32_)
        self.hispeed_subscriber.Init(self._on_hispeed, 10)

    def _on_lowstate(self, msg: Any) -> None:
        with self.lock:
            self.lowstate_msg = msg
            self.lowstate_updated_at = time.time()

    def _on_hispeed(self, msg: Any) -> None:
        with self.lock:
            self.hispeed_msg = msg
            self.hispeed_updated_at = time.time()

    def state(self, timeout_sec: float) -> dict[str, Any]:
        deadline = time.time() + max(0.05, float(timeout_sec))
        while time.time() < deadline:
            with self.lock:
                lowstate = self.lowstate_msg
                hispeed = self.hispeed_msg
                lowstate_time = self.lowstate_updated_at
                hispeed_time = self.hispeed_updated_at
            if lowstate is not None:
                return self._build_state(lowstate, hispeed, lowstate_time, hispeed_time)
            time.sleep(0.01)
        raise RuntimeError(f"DDS lowstate timeout on {self.lowstate_topic}")

    def _build_state(self, lowstate: Any, hispeed: Any, lowstate_time: float, hispeed_time: float) -> dict[str, Any]:
        motor_state = list(_read_member(lowstate, "motor_state", []) or [])
        joints: dict[str, float] = {}
        joint_states_name: list[str] = []
        joint_states_position: list[float] = []
        joint_states_velocity: list[float] = []
        joint_states_effort: list[float] = []
        raw_motor_states: list[dict[str, Any]] = []

        for index, state in enumerate(motor_state):
            q = _round_float(_read_member(state, "q"))
            dq = _round_float(_read_member(state, "dq"))
            tau_est = _round_float(_read_member(state, "tau_est"))
            raw_motor_states.append({"index": index, "q": q, "dq": dq, "tau_est": tau_est})
            joint_name = G1D_LOWSTATE_JOINT_MAP.get(index)
            if not joint_name or q is None:
                continue
            joints[joint_name] = q
            joint_states_name.append(joint_name)
            joint_states_position.append(q)
            joint_states_velocity.append(dq if dq is not None else 0.0)
            joint_states_effort.append(tau_est if tau_est is not None else 0.0)

        column_height_m = _round_float(_read_member(hispeed, "y")) if hispeed is not None else None
        if column_height_m is not None:
            column_height_m = max(0.0, min(0.42, column_height_m))
            per_joint = column_height_m / 2.0
            joints["LZ_mt_Joint"] = per_joint
            joints["LZ_it_Joint"] = per_joint
            for name in ("LZ_mt_Joint", "LZ_it_Joint"):
                joint_states_name.append(name)
                joint_states_position.append(per_joint)
                joint_states_velocity.append(0.0)
                joint_states_effort.append(0.0)

        imu_state = _read_member(lowstate, "imu_state")
        imu_payload: dict[str, Any] = {}
        if imu_state is not None:
            for key in ("rpy", "gyroscope", "accelerometer", "quaternion"):
                values = _read_member(imu_state, key)
                if values is not None:
                    imu_payload[key] = [_round_float(value) for value in list(values)]

        return {
            "ok": True,
            "source": "unitree_dds_lowstate",
            "dds": {
                "network_interface": self.network_interface,
                "lowstate_topic": self.lowstate_topic,
                "hispeed_topic": self.hispeed_topic,
                "lowstate_age_ms": round((time.time() - lowstate_time) * 1000.0, 1) if lowstate_time else None,
                "hispeed_age_ms": round((time.time() - hispeed_time) * 1000.0, 1) if hispeed_time else None,
                "motor_count": len(motor_state),
            },
            "column_extension_mm": round(column_height_m * 1000.0, 1) if column_height_m is not None else None,
            "joints": joints,
            "joint_states": {
                "name": joint_states_name,
                "position": joint_states_position,
                "velocity": joint_states_velocity,
                "effort": joint_states_effort,
            },
            "imu": imu_payload,
            "raw_motor_states": raw_motor_states,
            "updated_at": datetime.now().isoformat(timespec="milliseconds"),
        }


def _read_dds_robot_state(
    *,
    network_interface: str,
    lowstate_topic: str,
    hispeed_topic: str,
    sdk2py_path: str,
    timeout_sec: float,
) -> dict[str, Any]:
    global DDS_READER
    if DDS_READER is None:
        DDS_READER = _DdsStateReader(
            network_interface=network_interface,
            lowstate_topic=lowstate_topic,
            hispeed_topic=hispeed_topic,
            sdk2py_path=sdk2py_path,
        )
    return DDS_READER.state(timeout_sec)


def _parse_rostopic_joint_state_csv(text: str, topic: str) -> dict[str, Any]:
    rows = [row for row in csv.reader(text.splitlines()) if row]
    if len(rows) < 2:
        raise RuntimeError(f"no joint state sample from {topic}")
    headers = rows[0]
    values = rows[1]
    field_values = {header: values[index] for index, header in enumerate(headers) if index < len(values)}
    names: list[str] = []
    positions: list[float] = []
    velocities: list[float] = []
    efforts: list[float] = []

    for header, name in field_values.items():
        if not header.startswith("field.name"):
            continue
        suffix = header.removeprefix("field.name")
        if not name:
            continue
        position_text = field_values.get(f"field.position{suffix}")
        if position_text in (None, ""):
            continue
        names.append(name)
        positions.append(float(position_text))
        velocity_text = field_values.get(f"field.velocity{suffix}")
        effort_text = field_values.get(f"field.effort{suffix}")
        if velocity_text not in (None, ""):
            velocities.append(float(velocity_text))
        if effort_text not in (None, ""):
            efforts.append(float(effort_text))

    if not names:
        raise RuntimeError(f"joint state sample from {topic} did not contain name/position arrays")

    joint_states: dict[str, Any] = {"name": names, "position": positions}
    if len(velocities) == len(names):
        joint_states["velocity"] = velocities
    if len(efforts) == len(names):
        joint_states["effort"] = efforts
    return {
        "ok": True,
        "source": f"ros1:{topic}",
        "joint_states": joint_states,
        "joints": dict(zip(names, positions)),
        "updated_at": datetime.now().isoformat(timespec="milliseconds"),
    }


def _read_ros_joint_states(topic: str, timeout_sec: float) -> dict[str, Any]:
    rostopic = shutil.which("rostopic")
    if not rostopic:
        raise RuntimeError("rostopic not found; source ROS setup.bash before starting the visualizer")
    completed = subprocess.run(
        [rostopic, "echo", "-n", "1", "-p", topic],
        check=False,
        capture_output=True,
        text=True,
        timeout=float(timeout_sec),
    )
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(message or f"rostopic echo failed for {topic}")
    return _parse_rostopic_joint_state_csv(completed.stdout, topic)


def _merge_query(url: str, params: dict[str, str]) -> str:
    parsed = urllib.parse.urlparse(url)
    query = dict(urllib.parse.parse_qsl(parsed.query))
    query.update({key: value for key, value in params.items() if value})
    return urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(query)))


def _default_robot_state(warning: str | None = None) -> dict[str, Any]:
    payload = json.loads(json.dumps(DEFAULT_ROBOT_STATE))
    payload["updated_at"] = datetime.now().isoformat(timespec="milliseconds")
    if warning:
        payload["warning"] = warning
    return payload


def _read_robot_state(
    robot_state_url: str | None,
    robot_state_file: Path | None,
    dds_interface: str | None,
    dds_lowstate_topic: str | None,
    dds_hispeed_topic: str | None,
    unitree_sdk2py_path: str,
    joint_states_topic: str | None,
    timeout_sec: float,
) -> dict[str, Any]:
    if robot_state_url:
        payload = _fetch_json(robot_state_url, timeout_sec)
        payload.setdefault("source", robot_state_url)
        return payload
    if robot_state_file and robot_state_file.exists():
        payload = json.loads(robot_state_file.read_text(encoding="utf-8"))
        payload.setdefault("source", str(robot_state_file))
        return payload
    if dds_interface and dds_lowstate_topic:
        try:
            return _read_dds_robot_state(
                network_interface=dds_interface,
                lowstate_topic=dds_lowstate_topic,
                hispeed_topic=dds_hispeed_topic or DEFAULT_DDS_HISPEED_TOPIC,
                sdk2py_path=unitree_sdk2py_path,
                timeout_sec=timeout_sec,
            )
        except Exception as exc:
            return _default_robot_state(f"DDS state unavailable: {exc}")
    if joint_states_topic:
        try:
            return _read_ros_joint_states(joint_states_topic, timeout_sec)
        except Exception as exc:
            return _default_robot_state(f"joint states unavailable: {exc}")
    return _default_robot_state()


def make_handler(
    default_xyz_url: str,
    timeout_sec: float,
    robot_state_url: str | None,
    robot_state_file: Path | None,
    dds_interface: str | None,
    dds_lowstate_topic: str | None,
    dds_hispeed_topic: str | None,
    unitree_sdk2py_path: str,
    joint_states_topic: str | None,
) -> type[SimpleHTTPRequestHandler]:
    class VisualizerHandler(SimpleHTTPRequestHandler):
        server_version = "G1DCigaretteVisualizer/1.0"

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(VIEWER_DIR), **kwargs)

        def end_headers(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            super().end_headers()

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self.end_headers()

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/health":
                _json_response(
                    self,
                    200,
                    {
                        "ok": True,
                        "service": "g1d_cigarette_visualizer",
                        "viewer_dir": str(VIEWER_DIR),
                        "default_xyz_url": default_xyz_url,
                        "robot_state_url": robot_state_url,
                        "robot_state_file": str(robot_state_file) if robot_state_file else None,
                        "dds_interface": dds_interface,
                        "dds_lowstate_topic": dds_lowstate_topic,
                        "dds_hispeed_topic": dds_hispeed_topic,
                        "unitree_sdk2py_path": unitree_sdk2py_path,
                        "joint_states_topic": joint_states_topic,
                    },
                )
                return
            if parsed.path == "/api/xyz":
                query = urllib.parse.parse_qs(parsed.query)
                source_url = query.get("url", [default_xyz_url])[-1]
                label = query.get("label", [""])[-1]
                request_url = _merge_query(source_url, {"label": label})
                try:
                    payload = _fetch_json(request_url, timeout_sec)
                    _json_response(self, 200 if payload.get("ok", True) else 502, {"ok": True, "url": request_url, "pose": payload})
                except Exception as exc:
                    _json_response(self, 502, {"ok": False, "url": request_url, "error": str(exc)})
                return
            if parsed.path == "/api/robot_state":
                query = urllib.parse.parse_qs(parsed.query)
                source_url = query.get("url", [robot_state_url or ""])[-1] or None
                query_dds_interface = query.get("dds_interface", [dds_interface or ""])[-1] or None
                query_dds_lowstate_topic = query.get("dds_lowstate_topic", [dds_lowstate_topic or ""])[-1] or None
                query_dds_hispeed_topic = query.get("dds_hispeed_topic", [dds_hispeed_topic or ""])[-1] or None
                topic = query.get("joint_states_topic", [joint_states_topic or ""])[-1] or None
                try:
                    payload = _read_robot_state(
                        source_url,
                        robot_state_file,
                        query_dds_interface,
                        query_dds_lowstate_topic,
                        query_dds_hispeed_topic,
                        unitree_sdk2py_path,
                        topic,
                        timeout_sec,
                    )
                    _json_response(self, 200 if payload.get("ok", True) else 502, {"ok": True, "state": payload})
                except Exception as exc:
                    _json_response(self, 502, {"ok": False, "error": str(exc)})
                return
            super().do_GET()

        def log_message(self, fmt: str, *args: Any) -> None:
            print("%s - - [%s] %s" % (self.client_address[0], self.log_date_time_string(), fmt % args))

    return VisualizerHandler


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the G1-D cigarette visualization page.")
    parser.add_argument("--bind", default=DEFAULT_BIND)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--xyz-url", default=DEFAULT_XYZ_URL)
    parser.add_argument("--robot-state-url", default="")
    parser.add_argument("--robot-state-file", type=Path)
    parser.add_argument("--dds-interface", default=DEFAULT_DDS_INTERFACE)
    parser.add_argument("--dds-lowstate-topic", default=DEFAULT_DDS_LOWSTATE_TOPIC)
    parser.add_argument("--dds-hispeed-topic", default=DEFAULT_DDS_HISPEED_TOPIC)
    parser.add_argument("--unitree-sdk2py-path", default=DEFAULT_UNITREE_SDK2PY_PATH)
    parser.add_argument("--joint-states-topic", default=DEFAULT_JOINT_STATES_TOPIC)
    parser.add_argument("--timeout-sec", type=float, default=8.0)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if not VIEWER_DIR.exists():
        raise FileNotFoundError(f"viewer directory not found: {VIEWER_DIR}")
    server = ThreadingHTTPServer(
        (args.bind, int(args.port)),
        make_handler(
            args.xyz_url,
            args.timeout_sec,
            args.robot_state_url or None,
            args.robot_state_file,
            args.dds_interface or None,
            args.dds_lowstate_topic or None,
            args.dds_hispeed_topic or None,
            args.unitree_sdk2py_path,
            args.joint_states_topic or None,
        ),
    )
    print(f"serving G1-D cigarette visualizer on http://{args.bind}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
