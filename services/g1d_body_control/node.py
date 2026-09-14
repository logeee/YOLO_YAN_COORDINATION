#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS2 topic API for G1-D body atomic actions.

Command topic:
  /g1d_body_control/task_command  std_msgs/msg/String JSON

Status topic:
  /g1d_body_control/task_status   std_msgs/msg/String JSON
"""

from __future__ import annotations

import argparse
import os
import json
import math
import subprocess
import threading
import time
import uuid
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any, Dict, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


TERMINAL_STATES = {"DONE", "FAILED", "TIMEOUT", "STOPPED", "REJECTED"}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


def finite_or_none(value: Any, digits: Optional[int] = None) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits) if digits is not None else number


def clamp(value: Any, default: float, lo: float, hi: float) -> float:
    number = finite_or_none(value)
    if number is None:
        number = default
    return max(lo, min(hi, float(number)))


def normalize_phase(raw_phase: Any) -> str:
    phase = str(raw_phase or "").strip().upper().replace("-", "_")
    aliases = {
        "FORWARD": "BASE_FORWARD",
        "BACK": "BASE_BACK",
        "BACKWARD": "BASE_BACK",
        "TURN_LEFT": "BASE_TURN_LEFT",
        "LEFT": "BASE_TURN_LEFT",
        "TURN_RIGHT": "BASE_TURN_RIGHT",
        "RIGHT": "BASE_TURN_RIGHT",
        "STOP": "BASE_STOP",
        "BASE_FORWARD": "BASE_FORWARD",
        "BASE_BACK": "BASE_BACK",
        "BASE_BACKWARD": "BASE_BACK",
        "BASE_TURN_LEFT": "BASE_TURN_LEFT",
        "BASE_TURN_RIGHT": "BASE_TURN_RIGHT",
        "BASE_STOP": "BASE_STOP",
        "UP": "COLUMN_UP",
        "DOWN": "COLUMN_DOWN",
        "COLUMN_UP": "COLUMN_UP",
        "COLUMN_DOWN": "COLUMN_DOWN",
        "COLUMN_STOP": "COLUMN_STOP",
        "LIFT_UP": "COLUMN_UP",
        "LIFT_DOWN": "COLUMN_DOWN",
        "LIFT_STOP": "COLUMN_STOP",
        "MOVE_TO": "COLUMN_MOVE_TO",
        "COLUMN_MOVE_TO": "COLUMN_MOVE_TO",
        "LIFT_MOVE_TO": "COLUMN_MOVE_TO",
        "HEIGHT_MOVE_TO": "COLUMN_MOVE_TO",
    }
    return aliases.get(phase, phase)


class G1DBodyControlNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("g1d_body_control_api")
        self.args = args
        self.command_topic = args.command_topic
        self.status_topic = args.status_topic
        self.worker_lock = threading.Lock()
        self.current_stop_event: Optional[threading.Event] = None
        self.current_task_id: Optional[str] = None
        self.current_process: Optional[subprocess.Popen[str]] = None

        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.command_sub = self.create_subscription(String, self.command_topic, self.on_command, 10)
        self.get_logger().info(
            f"G1-D body control API ready command={self.command_topic} status={self.status_topic}"
        )

    def publish_status(self, payload: Dict[str, Any]) -> None:
        payload = dict(payload)
        payload.setdefault("timestamp", now_iso())
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.status_pub.publish(msg)

    def on_command(self, msg: String) -> None:
        self.get_logger().info(f"received command: {msg.data[:300]}")
        try:
            command = json.loads(msg.data)
            if not isinstance(command, dict):
                raise ValueError("JSON command must be an object")
        except Exception as exc:
            self.publish_status(
                {
                    "ok": False,
                    "task_id": "",
                    "phase": "",
                    "state": "REJECTED",
                    "error": f"invalid JSON command: {exc}",
                    "raw": msg.data,
                }
            )
            return

        phase = normalize_phase(command.get("phase") or command.get("action"))
        task_id = str(command.get("task_id") or command.get("taskId") or "").strip()
        if not task_id:
            task_id = f"body_{phase.lower() or 'unknown'}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
        command["task_id"] = task_id
        command["phase"] = phase

        if phase in {"BASE_STOP", "COLUMN_STOP"}:
            self.request_stop(task_id, phase, command)
            return

        with self.worker_lock:
            if self.current_task_id:
                if self.current_task_id == task_id:
                    self.publish_status(
                        {
                            "ok": None,
                            "task_id": task_id,
                            "phase": phase,
                            "state": "RUNNING",
                            "message": "duplicate command ignored; task already running",
                            "duplicate": True,
                            "command": command,
                        }
                    )
                    return
                self.publish_status(
                    {
                        "ok": False,
                        "task_id": task_id,
                        "phase": phase,
                        "state": "REJECTED",
                        "error": f"busy running task {self.current_task_id}",
                        "command": command,
                    }
                )
                return
            stop_event = threading.Event()
            self.current_stop_event = stop_event
            self.current_task_id = task_id

        self.publish_status(
            {
                "ok": None,
                "task_id": task_id,
                "phase": phase,
                "state": "ACCEPTED",
                "message": "task accepted",
                "command": command,
            }
        )
        thread = threading.Thread(target=self.run_task, args=(command, stop_event), daemon=True)
        thread.start()

    def request_stop(self, task_id: str, phase: str, command: Dict[str, Any]) -> None:
        with self.worker_lock:
            running_task = self.current_task_id
            stop_event = self.current_stop_event
            proc = self.current_process
            if stop_event:
                stop_event.set()
            if proc and proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass
        result = self.run_simple_control(["stop"], timeout=5.0)
        self.publish_status(
            {
                "ok": bool(result.get("ok")),
                "task_id": task_id,
                "phase": phase,
                "state": "STOPPED" if result.get("ok") else "FAILED",
                "message": f"stop requested; previous task={running_task or 'none'}",
                "command": command,
                "execution": result,
            }
        )

    def run_task(self, command: Dict[str, Any], stop_event: threading.Event) -> None:
        task_id = str(command["task_id"])
        phase = str(command["phase"])
        started = time.time()
        started_at = now_iso()
        final_state = "FAILED"
        try:
            self.publish_status(
                {
                    "ok": None,
                    "task_id": task_id,
                    "phase": phase,
                    "state": "RUNNING",
                    "started_at": started_at,
                    "command": command,
                }
            )
            if phase in {"BASE_FORWARD", "BASE_BACK", "BASE_TURN_LEFT", "BASE_TURN_RIGHT"}:
                result = self.execute_simple_motion(command, stop_event)
            elif phase in {"COLUMN_UP", "COLUMN_DOWN"}:
                result = self.execute_column_jog(command, stop_event)
            elif phase == "COLUMN_MOVE_TO":
                result = self.execute_column_move_to(command, stop_event)
            else:
                result = {"ok": False, "state": "REJECTED", "error": f"unsupported phase {phase}"}

            final_state = str(result.get("state") or ("DONE" if result.get("ok") else "FAILED"))
            status = {
                "ok": bool(result.get("ok")),
                "task_id": task_id,
                "phase": phase,
                "state": final_state,
                "started_at": started_at,
                "finished_at": now_iso(),
                "elapsed_sec": round(time.time() - started, 3),
                "command": command,
            }
            status.update(result)
            self.publish_status(status)
        except Exception as exc:
            final_state = "FAILED"
            self.publish_status(
                {
                    "ok": False,
                    "task_id": task_id,
                    "phase": phase,
                    "state": "FAILED",
                    "error": str(exc),
                    "started_at": started_at,
                    "finished_at": now_iso(),
                    "elapsed_sec": round(time.time() - started, 3),
                    "command": command,
                }
            )
        finally:
            with self.worker_lock:
                if self.current_task_id == task_id:
                    self.current_task_id = None
                    self.current_stop_event = None
                    self.current_process = None
            self.get_logger().info(f"task {task_id} {phase} ended as {final_state}")

    def execute_simple_motion(self, command: Dict[str, Any], stop_event: threading.Event) -> Dict[str, Any]:
        action_map = {
            "BASE_FORWARD": "forward",
            "BASE_BACK": "back",
            "BASE_TURN_LEFT": "turn_left",
            "BASE_TURN_RIGHT": "turn_right",
        }
        phase = str(command["phase"])
        speed = clamp(command.get("speed"), self.args.default_base_speed, 0.0, 1.0)
        duration = clamp(command.get("duration_sec", command.get("durationSec")), 0.5, 0.05, self.args.max_duration_sec)
        if stop_event.is_set():
            return {"ok": False, "state": "STOPPED", "message": "stopped before execution"}
        result = self.run_simple_control([action_map[phase], f"{speed:.3f}", f"{duration:.3f}"], timeout=duration + 8.0)
        if stop_event.is_set():
            return {"ok": False, "state": "STOPPED", "execution": result}
        return {
            "ok": bool(result.get("ok")),
            "state": "DONE" if result.get("ok") else "FAILED",
            "speed": speed,
            "duration_sec": duration,
            "execution": result,
        }

    def execute_column_jog(self, command: Dict[str, Any], stop_event: threading.Event) -> Dict[str, Any]:
        action = "up" if command["phase"] == "COLUMN_UP" else "down"
        speed = clamp(command.get("speed"), self.args.default_column_speed, 0.0, 1.0)
        duration = clamp(command.get("duration_sec", command.get("durationSec")), 0.5, 0.05, self.args.max_duration_sec)
        if stop_event.is_set():
            return {"ok": False, "state": "STOPPED", "message": "stopped before execution"}
        before = self.read_lift_height_status()
        result = self.run_simple_control([action, f"{speed:.3f}", f"{duration:.3f}"], timeout=duration + 8.0)
        after = self.read_lift_height_status()
        if stop_event.is_set():
            return {"ok": False, "state": "STOPPED", "execution": result, "lift_height_before": before, "lift_height_after": after}
        return {
            "ok": bool(result.get("ok")),
            "state": "DONE" if result.get("ok") else "FAILED",
            "speed": speed,
            "duration_sec": duration,
            "execution": result,
            "lift_height_before": before,
            "lift_height_after": after,
        }

    def execute_column_move_to(self, command: Dict[str, Any], stop_event: threading.Event) -> Dict[str, Any]:
        target_physical = finite_or_none(
            command.get("target_physical_height_m", command.get("targetPhysicalHeightM", command.get("physical_height_m")))
        )
        if target_physical is None:
            return {"ok": False, "state": "REJECTED", "error": "target_physical_height_m is required"}
        timeout = clamp(command.get("timeout_sec", command.get("timeoutSec")), self.args.default_column_timeout_sec, 1.0, 300.0)
        tolerance = clamp(command.get("tolerance_m", command.get("toleranceM")), self.args.column_tolerance_m, 0.002, 0.05)
        stable_required = int(clamp(command.get("stable_count", command.get("stableCount")), self.args.column_stable_count, 1, 20))
        before = self.read_lift_height_status()
        if not before.get("ok"):
            return {"ok": False, "state": "FAILED", "error": "cannot read current lift height", "lift_height_before": before}

        physical_min = finite_or_none(before.get("physical_min_m")) or 0.0
        physical_max = finite_or_none(before.get("physical_max_m"))
        if physical_max is None:
            physical_max = finite_or_none(before.get("full_travel_m")) or self.args.default_full_travel_m
        if target_physical < physical_min or target_physical > physical_max:
            return {
                "ok": False,
                "state": "REJECTED",
                "error": f"target_physical_height_m out of range [{physical_min:.3f}, {physical_max:.3f}]",
                "target_physical_height_m": round(target_physical, 6),
                "lift_height_before": before,
            }

        lift_offset = finite_or_none(before.get("lift_offset_m"))
        if lift_offset is None:
            hispeed_y = finite_or_none(before.get("hispeed_y_m"))
            physical_now = finite_or_none(before.get("physical_height_m"))
            if hispeed_y is not None and physical_now is not None:
                lift_offset = hispeed_y - physical_now
        if lift_offset is None:
            return {"ok": False, "state": "FAILED", "error": "lift_offset_m unavailable", "lift_height_before": before}
        raw_target = float(target_physical) + float(lift_offset)

        argv = [self.args.height_control_bin, self.args.interface, f"{raw_target:.6f}"]
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        with self.worker_lock:
            self.current_process = proc

        deadline = time.time() + timeout
        stable_count = 0
        last_height: Dict[str, Any] = before
        status_interval = max(0.1, self.args.status_interval_sec)
        next_status = 0.0
        while time.time() < deadline:
            if stop_event.is_set():
                self.terminate_process(proc)
                self.run_simple_control(["stop"], timeout=5.0)
                return {
                    "ok": False,
                    "state": "STOPPED",
                    "target_physical_height_m": round(target_physical, 6),
                    "target_height_m": round(raw_target, 6),
                    "lift_height_before": before,
                    "lift_height_after": last_height,
                }
            last_height = self.read_lift_height_status()
            current = finite_or_none(last_height.get("physical_height_m"))
            error_m = None if current is None else target_physical - current
            if error_m is not None and abs(error_m) <= tolerance:
                stable_count += 1
            else:
                stable_count = 0
            now = time.time()
            if now >= next_status:
                self.publish_status(
                    {
                        "ok": None,
                        "task_id": command["task_id"],
                        "phase": command["phase"],
                        "state": "RUNNING",
                        "message": "moving column to target physical height",
                        "target_physical_height_m": round(target_physical, 6),
                        "target_height_m": round(raw_target, 6),
                        "current_physical_height_m": finite_or_none(current, 6),
                        "error_m": finite_or_none(error_m, 6),
                        "tolerance_m": tolerance,
                        "stable_count": stable_count,
                        "stable_required": stable_required,
                    }
                )
                next_status = now + status_interval
            if stable_count >= stable_required:
                self.terminate_process(proc)
                stop_result = self.run_simple_control(["stop"], timeout=5.0)
                after = self.read_lift_height_status()
                return {
                    "ok": True,
                    "state": "DONE",
                    "target_physical_height_m": round(target_physical, 6),
                    "target_height_m": round(raw_target, 6),
                    "current_physical_height_m": finite_or_none(after.get("physical_height_m"), 6),
                    "error_m": finite_or_none(target_physical - float(after.get("physical_height_m", target_physical)), 6),
                    "tolerance_m": tolerance,
                    "stop_execution": stop_result,
                    "lift_height_before": before,
                    "lift_height_after": after,
                }
            if proc.poll() is not None:
                break
            time.sleep(status_interval)

        stdout, stderr = self.collect_process(proc)
        after = self.read_lift_height_status()
        current_after = finite_or_none(after.get("physical_height_m"))
        error_after = None if current_after is None else target_physical - current_after
        if error_after is not None and abs(error_after) <= tolerance:
            return {
                "ok": True,
                "state": "DONE",
                "target_physical_height_m": round(target_physical, 6),
                "target_height_m": round(raw_target, 6),
                "current_physical_height_m": finite_or_none(current_after, 6),
                "error_m": finite_or_none(error_after, 6),
                "tolerance_m": tolerance,
                "lift_height_before": before,
                "lift_height_after": after,
                "execution": {"argv": argv, "returncode": proc.returncode, "stdout": stdout, "stderr": stderr},
            }
        timed_out = time.time() >= deadline
        if timed_out:
            self.run_simple_control(["stop"], timeout=5.0)
        return {
            "ok": False,
            "state": "TIMEOUT" if timed_out else "FAILED",
            "error": "column move timed out" if timed_out else "height control exited before reaching target",
            "target_physical_height_m": round(target_physical, 6),
            "target_height_m": round(raw_target, 6),
            "current_physical_height_m": finite_or_none(current_after, 6),
            "error_m": finite_or_none(error_after, 6),
            "tolerance_m": tolerance,
            "lift_height_before": before,
            "lift_height_after": after,
            "execution": {"argv": argv, "returncode": proc.returncode, "stdout": stdout, "stderr": stderr},
        }

    def run_simple_control(self, args: list[str], timeout: float) -> Dict[str, Any]:
        argv = [self.args.simple_control_bin, self.args.interface] + list(args)
        started = time.time()
        try:
            completed = subprocess.run(
                argv,
                text=True,
                capture_output=True,
                timeout=max(1.0, timeout),
            )
            return {
                "ok": completed.returncode == 0,
                "argv": argv,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "elapsed_sec": round(time.time() - started, 3),
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "ok": False,
                "argv": argv,
                "returncode": None,
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "",
                "elapsed_sec": round(time.time() - started, 3),
                "error": "command timeout",
            }

    @staticmethod
    def terminate_process(proc: subprocess.Popen[str]) -> None:
        if proc.poll() is not None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=1.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    @staticmethod
    def collect_process(proc: subprocess.Popen[str]) -> tuple[str, str]:
        if proc.poll() is None:
            G1DBodyControlNode.terminate_process(proc)
        try:
            stdout, stderr = proc.communicate(timeout=1.0)
        except Exception:
            stdout, stderr = "", ""
        return stdout or "", stderr or ""

    def read_lift_height_status(self) -> Dict[str, Any]:
        started = time.time()
        try:
            req = urllib.request.Request(self.args.lift_height_url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=self.args.lift_height_timeout_sec) as resp:
                raw = resp.read().decode("utf-8", "replace")
                source_payload = json.loads(raw) if raw else {}
                status_code = getattr(resp, "status", 200)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return {"ok": False, "error": str(exc), "url": self.args.lift_height_url}

        def pick_m(*names: str) -> Optional[float]:
            for name in names:
                value = finite_or_none(source_payload.get(name))
                if value is not None:
                    return value / 1000.0 if name.endswith("_mm") else value
            return None

        physical_height_m = pick_m("physical_height_m", "physicalHeightM", "height_m", "heightM")
        hispeed_y_m = pick_m("hispeed_y_m", "hispeedYM")
        lift_offset_m = pick_m("lift_offset_m", "liftOffsetM")
        full_travel_m = pick_m("full_travel_m", "fullTravelM") or self.args.default_full_travel_m
        physical_min_m = pick_m("physical_min_m", "physicalMinM")
        physical_max_m = pick_m("physical_max_m", "physicalMaxM")
        sdk_min_m = pick_m("sdk_min_m", "sdkMinM")
        sdk_max_m = pick_m("sdk_max_m", "sdkMaxM")

        if physical_height_m is None and hispeed_y_m is not None and lift_offset_m is not None:
            physical_height_m = hispeed_y_m - lift_offset_m
        if lift_offset_m is None and hispeed_y_m is not None and physical_height_m is not None:
            lift_offset_m = hispeed_y_m - physical_height_m
        if physical_min_m is None:
            physical_min_m = 0.0
        if physical_max_m is None:
            physical_max_m = full_travel_m
        if sdk_min_m is None and lift_offset_m is not None:
            sdk_min_m = physical_min_m + lift_offset_m
        if sdk_max_m is None and lift_offset_m is not None:
            sdk_max_m = physical_max_m + lift_offset_m

        return {
            "ok": True,
            "url": self.args.lift_height_url,
            "http_status": status_code,
            "source": source_payload.get("source") or source_payload.get("service") or "lift_height_service",
            "physical_height_m": finite_or_none(physical_height_m, 6),
            "hispeed_y_m": finite_or_none(hispeed_y_m, 6),
            "lift_offset_m": finite_or_none(lift_offset_m, 6),
            "full_travel_m": finite_or_none(full_travel_m, 6),
            "physical_min_m": finite_or_none(physical_min_m, 6),
            "physical_max_m": finite_or_none(physical_max_m, 6),
            "sdk_min_m": finite_or_none(sdk_min_m, 6),
            "sdk_max_m": finite_or_none(sdk_max_m, 6),
            "data_age_sec": finite_or_none(source_payload.get("data_age_sec"), 3),
            "timestamp": source_payload.get("timestamp"),
            "elapsed_ms": round((time.time() - started) * 1000.0, 1),
            "raw": source_payload,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="G1-D body atomic action ROS2 topic API.")
    parser.add_argument("--command-topic", default=os.environ.get("G1D_BODY_COMMAND_TOPIC", "/g1d_body_control/task_command"))
    parser.add_argument("--status-topic", default=os.environ.get("G1D_BODY_STATUS_TOPIC", "/g1d_body_control/task_status"))
    parser.add_argument("--interface", default=os.environ.get("G1D_BODY_INTERFACE", "eth0"))
    parser.add_argument("--simple-control-bin", default=os.environ.get("G1D_BODY_SIMPLE_CONTROL_BIN", "/home/unitree/unitree_sdk2/build/bin/g1d_simple_control"))
    parser.add_argument("--height-control-bin", default=os.environ.get("G1D_BODY_HEIGHT_CONTROL_BIN", "/home/unitree/unitree_sdk2/build/bin/g1d_height_control"))
    parser.add_argument("--lift-height-url", default=os.environ.get("G1D_BODY_LIFT_HEIGHT_URL", "http://127.0.0.1:28089/api/basic_status"))
    parser.add_argument("--lift-height-timeout-sec", type=float, default=1.0)
    parser.add_argument("--default-base-speed", type=float, default=0.1)
    parser.add_argument("--default-column-speed", type=float, default=0.2)
    parser.add_argument("--max-duration-sec", type=float, default=30.0)
    parser.add_argument("--default-column-timeout-sec", type=float, default=45.0)
    parser.add_argument("--default-full-travel-m", type=float, default=0.427)
    parser.add_argument("--column-tolerance-m", type=float, default=0.01)
    parser.add_argument("--column-stable-count", type=int, default=3)
    parser.add_argument("--status-interval-sec", type=float, default=0.25)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rclpy.init()
    node = G1DBodyControlNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
