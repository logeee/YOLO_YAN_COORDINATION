#!/usr/bin/env python3
"""Serve a browser control panel for G1-D manual movement.

The server is safe by default: it records and returns command previews unless
started with --execute-enabled. This lets the page be tested locally before the
same code is copied to the robot.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import re
import subprocess
import threading
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


DEFAULT_BIND = "0.0.0.0"
DEFAULT_PORT = 18086
DEFAULT_INTERFACE = "eth0"
DEFAULT_SDK_BUILD_DIR = Path("/home/unitree/unitree_sdk2/build")
DEFAULT_HOLD_DURATION_SEC = 600.0
PAGE_DIR = Path(__file__).resolve().parents[1] / "visualization" / "g1d_remote_control"

BASE_ACTIONS = {"forward", "back", "turn_left", "turn_right"}
COLUMN_ACTIONS = {"column_up", "column_down"}
STOP_ACTIONS = {"stop"}
ALL_ACTIONS = BASE_ACTIONS | COLUMN_ACTIONS | STOP_ACTIONS
SDK_ACTIONS = {
    "column_up": "up",
    "column_down": "down",
}
ACTIVE_HOLD_LOCK = threading.Lock()
ACTIVE_HOLD_PROCESS: subprocess.Popen[str] | None = None
ACTIVE_HOLD_COMMAND: dict[str, Any] | None = None


@dataclass(frozen=True)
class RemoteControlConfig:
    bind: str = DEFAULT_BIND
    port: int = DEFAULT_PORT
    sdk_build_dir: str = str(DEFAULT_SDK_BUILD_DIR).replace("\\", "/")
    interface: str = DEFAULT_INTERFACE
    execute_enabled: bool = False
    default_speed: float = 0.1
    default_turn_speed: float = 0.1
    default_column_speed: float = 0.05
    default_duration_sec: float = 0.25
    min_duration_sec: float = 0.05
    max_duration_sec: float = 2.0
    max_speed: float = 0.3
    max_turn_speed: float = 0.6
    max_column_speed: float = 1.0
    hold_duration_sec: float = DEFAULT_HOLD_DURATION_SEC
    command_timeout_extra_sec: float = 3.0


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _json_response(handler: SimpleHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(data)


def _is_usable_ipv4(value: str) -> bool:
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return False
    benchmark_net = ipaddress.ip_network("198.18.0.0/15")
    return (
        ip.version == 4
        and not ip.is_loopback
        and not ip.is_link_local
        and not ip.is_multicast
        and ip not in benchmark_net
    )


def _net_kind(name: str) -> str:
    lower = name.lower()
    if lower.startswith(("wl", "wlan")) or any(word in lower for word in ("wifi", "wi-fi", "wireless", "无线")):
        return "wireless"
    if lower.startswith(("en", "eth")) or any(word in lower for word in ("lan", "ethernet", "以太网")):
        return "wired"
    return "other"


def _looks_virtual_interface(name: str) -> bool:
    lower = name.lower()
    return any(
        word in lower
        for word in (
            "virtual",
            "vethernet",
            "vmware",
            "virtualbox",
            "hyper-v",
            "loopback",
            "docker",
            "wsl",
            "tailscale",
            "zerotier",
            "bluetooth",
        )
    )


def _choose_network_candidate(candidates: list[dict[str, str]]) -> dict[str, Any] | None:
    physical = [item for item in candidates if not _looks_virtual_interface(item.get("interface", ""))]
    ordered = physical or candidates
    for kind in ("wireless", "wired", "other"):
        for item in ordered:
            if item["kind"] == kind:
                return item
    return None


def _linux_iface_ipv4(name: str) -> str | None:
    try:
        completed = subprocess.run(
            ["ip", "-o", "-4", "addr", "show", "dev", name],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    match = re.search(r"\binet\s+(\d+\.\d+\.\d+\.\d+)/", completed.stdout)
    if not match:
        return None
    ip = match.group(1)
    return ip if _is_usable_ipv4(ip) else None


def _detect_linux_network_ip() -> dict[str, Any] | None:
    sys_net = Path("/sys/class/net")
    if not sys_net.exists():
        return None
    candidates: list[dict[str, str]] = []
    for iface in sorted(path.name for path in sys_net.iterdir() if path.is_dir()):
        if iface == "lo":
            continue
        try:
            operstate = (sys_net / iface / "operstate").read_text(encoding="utf-8").strip()
        except Exception:
            operstate = ""
        if operstate and operstate not in ("up", "unknown"):
            continue
        ip = _linux_iface_ipv4(iface)
        if ip:
            candidates.append({"interface": iface, "ip": ip, "kind": _net_kind(iface)})
    item = _choose_network_candidate(candidates)
    return {**item, "source": "linux_ip_addr"} if item else None


def _detect_windows_network_ip() -> dict[str, Any] | None:
    detected = _detect_windows_network_ip_powershell()
    if detected:
        return detected
    try:
        completed = subprocess.run(
            ["ipconfig"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except Exception:
        return None
    blocks = re.split(r"\r?\n\r?\n", completed.stdout)
    candidates: list[dict[str, str]] = []
    for block in blocks:
        if not block.strip():
            continue
        header = block.splitlines()[0].strip().rstrip(":")
        match = re.search(r"IPv4[^\r\n:]*:\s*(\d+\.\d+\.\d+\.\d+)", block)
        if not match:
            continue
        ip = match.group(1)
        if not _is_usable_ipv4(ip):
            continue
        candidates.append({"interface": header, "ip": ip, "kind": _net_kind(header), "source": "windows_ipconfig"})
    return _choose_network_candidate(candidates)


def _detect_windows_network_ip_powershell() -> dict[str, Any] | None:
    command = (
        "Get-NetIPConfiguration | "
        "Where-Object {$_.IPv4Address -and $_.NetAdapter.Status -eq 'Up'} | "
        "ForEach-Object {[pscustomobject]@{"
        "InterfaceAlias=$_.InterfaceAlias;"
        "InterfaceDescription=$_.InterfaceDescription;"
        "IPv4Address=$_.IPv4Address[0].IPAddress"
        "}} | ConvertTo-Json -Compress"
    )
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except Exception:
        return None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    try:
        data = json.loads(completed.stdout)
    except Exception:
        return None
    rows = [data] if isinstance(data, dict) else data if isinstance(data, list) else []
    candidates: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        ip = str(row.get("IPv4Address") or "")
        if not _is_usable_ipv4(ip):
            continue
        alias = str(row.get("InterfaceAlias") or "")
        description = str(row.get("InterfaceDescription") or "")
        name = alias or description
        candidates.append(
            {
                "interface": name,
                "ip": ip,
                "kind": _net_kind(f"{alias} {description}"),
                "source": "windows_powershell",
            }
        )
    return _choose_network_candidate(candidates)


def detect_network_ip() -> dict[str, Any]:
    detected = _detect_linux_network_ip() or _detect_windows_network_ip()
    if detected:
        return detected
    return {"ip": None, "interface": None, "kind": None, "source": "not_found"}


def _read_request_json(handler: SimpleHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or 0)
    if length <= 0:
        return {}
    data = json.loads(handler.rfile.read(length).decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("request body must be a JSON object")
    return data


def _normalize_action(value: Any) -> str:
    action = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "backward": "back",
        "left": "turn_left",
        "right": "turn_right",
        "up": "column_up",
        "down": "column_down",
        "lift_up": "column_up",
        "lift_down": "column_down",
    }
    action = aliases.get(action, action)
    if action not in ALL_ACTIONS:
        raise ValueError(f"unsupported action: {value!r}")
    return action


def _number(values: dict[str, Any], key: str, fallback: float) -> float:
    raw = values.get(key, fallback)
    if raw in (None, ""):
        return float(fallback)
    return float(raw)


def _command_values(values: dict[str, Any], config: RemoteControlConfig) -> dict[str, Any]:
    action = _normalize_action(values.get("action"))
    duration = _clamp(
        _number(values, "duration_sec", config.default_duration_sec),
        config.min_duration_sec,
        config.max_duration_sec,
    )
    if action in ("turn_left", "turn_right"):
        speed = _clamp(_number(values, "speed", config.default_turn_speed), 0.0, config.max_turn_speed)
    elif action in COLUMN_ACTIONS:
        speed = _clamp(_number(values, "speed", config.default_column_speed), 0.0, config.max_column_speed)
    else:
        speed = _clamp(_number(values, "speed", config.default_speed), 0.0, config.max_speed)
    return {
        "action": action,
        "speed": round(speed, 4),
        "duration_sec": round(duration, 4),
    }


def _base_binary(config: RemoteControlConfig) -> Path:
    return Path(config.sdk_build_dir) / "bin" / "g1d_simple_control"


def _display_base_binary(config: RemoteControlConfig) -> str:
    return config.sdk_build_dir.rstrip("/\\") + "/bin/g1d_simple_control"


def _argv_for_command(config: RemoteControlConfig, command: dict[str, Any]) -> list[str]:
    action = str(command["action"])
    if action == "stop":
        return [_display_base_binary(config), config.interface, "stop"]
    sdk_action = SDK_ACTIONS.get(action, action)
    if action in BASE_ACTIONS | COLUMN_ACTIONS:
        return [
            _display_base_binary(config),
            config.interface,
            sdk_action,
            str(command["speed"]),
            str(command["duration_sec"]),
        ]
    raise ValueError(f"unsupported action: {action}")


def _execute(config: RemoteControlConfig, argv: list[str], duration_sec: float) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        argv,
        cwd=config.sdk_build_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=float(duration_sec) + float(config.command_timeout_extra_sec),
        check=False,
    )
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 1),
    }


def _stop_active_hold() -> dict[str, Any] | None:
    global ACTIVE_HOLD_COMMAND, ACTIVE_HOLD_PROCESS
    with ACTIVE_HOLD_LOCK:
        process = ACTIVE_HOLD_PROCESS
        command = ACTIVE_HOLD_COMMAND
        ACTIVE_HOLD_PROCESS = None
        ACTIVE_HOLD_COMMAND = None
    if process is None:
        return None
    result: dict[str, Any] = {
        "pid": process.pid,
        "command": command,
        "was_running": process.poll() is None,
    }
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=0.5)
            result["terminated"] = True
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1.0)
            result["terminated"] = True
            result["killed"] = True
    else:
        result["terminated"] = False
        result["returncode"] = process.returncode
    return result


def _cleanup_stale_control_processes(config: RemoteControlConfig) -> dict[str, Any]:
    binary_name = Path(_display_base_binary(config)).name
    patterns = [
        f"{binary_name} {config.interface}",
        str(_display_base_binary(config)),
    ]
    attempts: list[dict[str, Any]] = []
    for signal_name in ("-TERM", "-KILL"):
        signal_attempts: list[dict[str, Any]] = []
        for pattern in patterns:
            try:
                completed = subprocess.run(
                    ["pkill", signal_name, "-f", pattern],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=0.5,
                )
                signal_attempts.append(
                    {
                        "pattern": pattern,
                        "returncode": completed.returncode,
                        "stderr": completed.stderr.strip(),
                    }
                )
            except FileNotFoundError:
                return {"ok": False, "skipped": True, "reason": "pkill not found"}
            except Exception as exc:
                signal_attempts.append({"pattern": pattern, "error": str(exc)})
        attempts.append({"signal": signal_name, "attempts": signal_attempts})
        time.sleep(0.05)
        if not _has_stale_control_process(config):
            return {"ok": True, "cleared": True, "attempts": attempts}
    return {"ok": not _has_stale_control_process(config), "cleared": False, "attempts": attempts}


def _has_stale_control_process(config: RemoteControlConfig) -> bool:
    pattern = f"{Path(_display_base_binary(config)).name} {config.interface}"
    try:
        completed = subprocess.run(
            ["pgrep", "-f", pattern],
            check=False,
            capture_output=True,
            text=True,
            timeout=0.5,
        )
    except Exception:
        return False
    return completed.returncode == 0 and bool(completed.stdout.strip())


def _start_hold_command(config: RemoteControlConfig, command: dict[str, Any], argv: list[str]) -> dict[str, Any]:
    stopped = _stop_active_hold()
    cleanup = _cleanup_stale_control_processes(config)
    started = time.perf_counter()
    process = subprocess.Popen(
        argv,
        cwd=config.sdk_build_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    global ACTIVE_HOLD_COMMAND, ACTIVE_HOLD_PROCESS
    with ACTIVE_HOLD_LOCK:
        ACTIVE_HOLD_PROCESS = process
        ACTIVE_HOLD_COMMAND = dict(command)
    return {
        "ok": True,
        "mode": "hold_started",
        "pid": process.pid,
        "replaced_previous_hold": stopped,
        "stale_process_cleanup": cleanup,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 1),
    }


def _handle_command(config: RemoteControlConfig, values: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    command = _command_values(values, config)
    hold = bool(values.get("hold"))
    if hold and command["action"] != "stop":
        command = dict(command)
        command["duration_sec"] = round(float(config.hold_duration_sec), 4)
    try:
        argv = _argv_for_command(config, command)
    except Exception as exc:
        return 400, {
            "ok": False,
            "execute_enabled": config.execute_enabled,
            "command": command,
            "error": str(exc),
            "updated_at": _now_iso(),
        }

    payload: dict[str, Any] = {
        "ok": True,
        "execute_enabled": config.execute_enabled,
        "command": command,
        "argv": argv,
        "updated_at": _now_iso(),
    }
    if not config.execute_enabled:
        payload["executed"] = False
        payload["reason"] = "preview only; restart server with --execute-enabled to call the SDK"
        return 200, payload

    if hold and command["action"] != "stop":
        execution = _start_hold_command(config, command, argv)
        payload["hold"] = True
    elif command["action"] == "stop":
        payload["stopped_hold"] = _stop_active_hold()
        payload["stale_process_cleanup"] = _cleanup_stale_control_processes(config)
        execution = _execute(config, argv, float(command["duration_sec"]))
    else:
        payload["stopped_hold"] = _stop_active_hold()
        payload["stale_process_cleanup"] = _cleanup_stale_control_processes(config)
        execution = _execute(config, argv, float(command["duration_sec"]))
    payload["executed"] = True
    payload["execution"] = execution
    payload["ok"] = bool(execution.get("ok"))
    return (200 if payload["ok"] else 500), payload


def make_handler(config: RemoteControlConfig) -> type[SimpleHTTPRequestHandler]:
    class RemoteControlHandler(SimpleHTTPRequestHandler):
        server_version = "G1DRemoteControl/1.0"

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(PAGE_DIR), **kwargs)

        def end_headers(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            super().end_headers()

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self.end_headers()

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/health":
                network = detect_network_ip()
                _json_response(
                    self,
                    200,
                    {
                        "ok": True,
                        "service": "g1d_remote_control",
                        "execute_enabled": config.execute_enabled,
                        "sdk_build_dir": config.sdk_build_dir,
                        "interface": config.interface,
                        "robot_ip": network.get("ip"),
                        "network_interface": network.get("interface"),
                        "network_interface_kind": network.get("kind"),
                        "network_ip_source": network.get("source"),
                        "page_dir": str(PAGE_DIR),
                        "column_control_configured": True,
                        "column_actions": {"column_up": "up", "column_down": "down"},
                        "endpoints": ["/health", "/api/command", "/api/stop"],
                    },
                )
                return
            super().do_GET()

        def do_POST(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            try:
                values = _read_request_json(self)
                if parsed.path == "/api/command":
                    status, payload = _handle_command(config, values)
                    _json_response(self, status, payload)
                    return
                if parsed.path == "/api/stop":
                    status, payload = _handle_command(config, {"action": "stop"})
                    _json_response(self, status, payload)
                    return
                _json_response(self, 404, {"ok": False, "error": f"unknown endpoint: {parsed.path}"})
            except Exception as exc:
                _json_response(self, 500, {"ok": False, "error": str(exc), "updated_at": _now_iso()})

        def log_message(self, fmt: str, *args: Any) -> None:
            print("%s - - [%s] %s" % (self.client_address[0], self.log_date_time_string(), fmt % args))

    return RemoteControlHandler


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve a safe G1-D browser remote control page.")
    parser.add_argument("--bind", default=DEFAULT_BIND)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--sdk-build-dir", default=str(DEFAULT_SDK_BUILD_DIR).replace("\\", "/"))
    parser.add_argument("--interface", default=DEFAULT_INTERFACE)
    parser.add_argument("--execute-enabled", action="store_true")
    parser.add_argument("--default-speed", type=float, default=0.1)
    parser.add_argument("--default-turn-speed", type=float, default=0.1)
    parser.add_argument("--default-column-speed", type=float, default=0.05)
    parser.add_argument("--default-duration-sec", type=float, default=0.25)
    parser.add_argument("--min-duration-sec", type=float, default=0.05)
    parser.add_argument("--max-duration-sec", type=float, default=2.0)
    parser.add_argument("--max-speed", type=float, default=0.3)
    parser.add_argument("--max-turn-speed", type=float, default=0.6)
    parser.add_argument("--max-column-speed", type=float, default=1.0)
    parser.add_argument("--hold-duration-sec", type=float, default=DEFAULT_HOLD_DURATION_SEC)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if not PAGE_DIR.exists():
        raise FileNotFoundError(f"page directory not found: {PAGE_DIR}")
    config = RemoteControlConfig(
        bind=args.bind,
        port=args.port,
        sdk_build_dir=str(args.sdk_build_dir).replace("\\", "/"),
        interface=args.interface,
        execute_enabled=bool(args.execute_enabled),
        default_speed=args.default_speed,
        default_turn_speed=args.default_turn_speed,
        default_column_speed=args.default_column_speed,
        default_duration_sec=args.default_duration_sec,
        min_duration_sec=args.min_duration_sec,
        max_duration_sec=args.max_duration_sec,
        max_speed=args.max_speed,
        max_turn_speed=args.max_turn_speed,
        max_column_speed=args.max_column_speed,
        hold_duration_sec=args.hold_duration_sec,
    )
    server = ThreadingHTTPServer((config.bind, config.port), make_handler(config))
    mode = "EXECUTE" if config.execute_enabled else "PREVIEW"
    print(f"serving G1-D remote control on http://{config.bind}:{config.port} [{mode}]", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
