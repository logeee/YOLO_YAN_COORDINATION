#!/usr/bin/env python3
"""BLE GATT remote control bridge for G1-D.

The robot advertises a BLE peripheral named G1D by default. A phone writes compact
commands to the control characteristic:

  H forward 0.20
  H back 0.20
  H turn_left 0.20
  H turn_right 0.20
  H column_up 0.05
  H column_down 0.05
  S

Preview mode is the default and only logs commands. Use --execute-enabled on
the robot after BLE pairing and command delivery have been verified.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import dbus
    import dbus.exceptions
    import dbus.mainloop.glib
    import dbus.service
    from gi.repository import GLib
except ImportError as exc:  # pragma: no cover - only happens off robot
    raise SystemExit(
        "This BLE service must run on Ubuntu with BlueZ Python DBus bindings "
        f"installed. Missing dependency: {exc}"
    )


BLUEZ_SERVICE_NAME = "org.bluez"
DBUS_OM_IFACE = "org.freedesktop.DBus.ObjectManager"
DBUS_PROP_IFACE = "org.freedesktop.DBus.Properties"
LE_ADVERTISING_MANAGER_IFACE = "org.bluez.LEAdvertisingManager1"
LE_ADVERTISEMENT_IFACE = "org.bluez.LEAdvertisement1"
GATT_MANAGER_IFACE = "org.bluez.GattManager1"
GATT_SERVICE_IFACE = "org.bluez.GattService1"
GATT_CHRC_IFACE = "org.bluez.GattCharacteristic1"
ADAPTER_IFACE = "org.bluez.Adapter1"

MAIN_LOOP: GLib.MainLoop | None = None

SERVICE_UUID = "6f9d0001-7f70-4f8f-9f25-41f0a7a1b001"
CONTROL_UUID = "6f9d0002-7f70-4f8f-9f25-41f0a7a1b001"
STATUS_UUID = "6f9d0003-7f70-4f8f-9f25-41f0a7a1b001"

DEFAULT_INTERFACE = "eth0"
DEFAULT_SDK_BUILD_DIR = Path("/home/unitree/unitree_sdk2/build")
DEFAULT_HOLD_DURATION_SEC = 600.0
DEFAULT_WATCHDOG_SEC = 0.6
BASE_ACTIONS = {"forward", "back", "turn_left", "turn_right"}
COLUMN_ACTIONS = {"column_up", "column_down"}
ALL_ACTIONS = BASE_ACTIONS | COLUMN_ACTIONS
SDK_ACTIONS = {"column_up": "up", "column_down": "down"}


@dataclass(frozen=True)
class BleRemoteConfig:
    adapter: str = "hci0"
    local_name: str = "G1D"
    adapter_alias: str = ""
    advertise_service_uuid: bool = False
    sdk_build_dir: str = str(DEFAULT_SDK_BUILD_DIR)
    interface: str = DEFAULT_INTERFACE
    execute_enabled: bool = False
    hold_duration_sec: float = DEFAULT_HOLD_DURATION_SEC
    watchdog_sec: float = DEFAULT_WATCHDOG_SEC
    max_speed: float = 0.3
    max_turn_speed: float = 0.6
    max_column_speed: float = 1.0
    command_timeout_extra_sec: float = 3.0


def now_text() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def byte_array_to_text(value: list[Any]) -> str:
    return bytes(int(item) for item in value).decode("utf-8", errors="replace").strip()


def text_to_dbus_bytes(text: str) -> list[dbus.Byte]:
    return [dbus.Byte(item) for item in text.encode("utf-8")]


def normalize_action(value: str) -> str:
    action = value.strip().lower().replace("-", "_")
    aliases = {
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


def parse_control_message(text: str) -> dict[str, Any]:
    if not text:
        raise ValueError("empty BLE command")
    if text.startswith("{"):
        payload = json.loads(text)
        action = str(payload.get("action", "")).strip().lower()
        if action == "stop":
            return {"type": "stop"}
        return {
            "type": "hold",
            "action": normalize_action(action),
            "speed": float(payload.get("speed", 0.1)),
        }

    parts = text.split()
    kind = parts[0].upper()
    if kind in {"S", "STOP"}:
        return {"type": "stop"}
    if kind in {"P", "PING"}:
        return {"type": "ping"}
    if kind not in {"H", "HOLD"}:
        raise ValueError(f"unsupported BLE command: {text!r}")
    if len(parts) < 2:
        raise ValueError("hold command requires action")
    speed = float(parts[2]) if len(parts) >= 3 else 0.1
    return {"type": "hold", "action": normalize_action(parts[1]), "speed": speed}


class G1DBleController:
    def __init__(self, config: BleRemoteConfig) -> None:
        self.config = config
        self.lock = threading.Lock()
        self.active_process: subprocess.Popen[str] | None = None
        self.active_command: dict[str, Any] | None = None
        self.last_heartbeat = 0.0
        self.last_status = "idle"
        self.last_status_at = now_text()

    def status_text(self) -> str:
        with self.lock:
            active = self.active_command.get("action") if self.active_command else "idle"
            return json.dumps(
                {
                    "ok": True,
                    "mode": "execute" if self.config.execute_enabled else "preview",
                    "active": active,
                    "status": self.last_status,
                    "updated_at": self.last_status_at,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )

    def handle_message(self, text: str) -> str:
        try:
            message = parse_control_message(text)
            msg_type = message["type"]
            if msg_type == "ping":
                return self._set_status("pong")
            if msg_type == "stop":
                return self.stop("ble_stop")
            if msg_type == "hold":
                return self.hold(str(message["action"]), float(message["speed"]))
            raise ValueError(f"unsupported message type: {msg_type}")
        except Exception as exc:
            return self._set_status(f"error {exc}")

    def hold(self, action: str, speed: float) -> str:
        speed = self._clamp_speed(action, speed)
        command = {"action": action, "speed": round(speed, 4), "duration_sec": self.config.hold_duration_sec}
        with self.lock:
            same_command = self.active_command == command
            process_alive = self.active_process is not None and self.active_process.poll() is None
            self.last_heartbeat = time.monotonic()
        if same_command and (process_alive or not self.config.execute_enabled):
            return self._set_status(f"hold {action} {speed:.2f}")

        if not self.config.execute_enabled:
            self._stop_active_process()
            with self.lock:
                self.active_command = command
                self.active_process = None
                self.last_heartbeat = time.monotonic()
            return self._set_status(f"preview hold {action} {speed:.2f}")

        self._stop_active_process()
        cleanup = self._cleanup_stale_control_processes()
        argv = self._argv_for_command(command)
        process = subprocess.Popen(
            argv,
            cwd=self.config.sdk_build_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        with self.lock:
            self.active_command = command
            self.active_process = process
            self.last_heartbeat = time.monotonic()
        return self._set_status(f"hold {action} {speed:.2f} pid={process.pid} cleanup={cleanup.get('ok')}")

    def stop(self, reason: str) -> str:
        stopped = self._stop_active_process()
        cleanup = self._cleanup_stale_control_processes() if self.config.execute_enabled else {"ok": True, "preview": True}
        if self.config.execute_enabled:
            command = {"action": "stop", "speed": 0.0, "duration_sec": 0.0}
            result = self._execute(self._argv_for_command(command), 0.0)
            return self._set_status(f"stop {reason} sdk_ok={result.get('ok')} stopped={bool(stopped)} cleanup={cleanup.get('ok')}")
        return self._set_status(f"preview stop {reason}")

    def watchdog_tick(self) -> bool:
        with self.lock:
            active = self.active_command is not None
            expired = active and (time.monotonic() - self.last_heartbeat) > self.config.watchdog_sec
        if expired:
            self.stop("watchdog")
        return True

    def _set_status(self, text: str) -> str:
        status = f"{now_text()} {text}"
        with self.lock:
            self.last_status = text
            self.last_status_at = now_text()
        print(status, flush=True)
        return status

    def _clamp_speed(self, action: str, speed: float) -> float:
        if action in ("turn_left", "turn_right"):
            return clamp(speed, 0.0, self.config.max_turn_speed)
        if action in COLUMN_ACTIONS:
            return clamp(speed, 0.0, self.config.max_column_speed)
        return clamp(speed, 0.0, self.config.max_speed)

    def _display_binary(self) -> str:
        return self.config.sdk_build_dir.rstrip("/\\") + "/bin/g1d_simple_control"

    def _argv_for_command(self, command: dict[str, Any]) -> list[str]:
        if command["action"] == "stop":
            return [self._display_binary(), self.config.interface, "stop"]
        sdk_action = SDK_ACTIONS.get(str(command["action"]), str(command["action"]))
        return [
            self._display_binary(),
            self.config.interface,
            sdk_action,
            str(command["speed"]),
            str(command["duration_sec"]),
        ]

    def _execute(self, argv: list[str], duration_sec: float) -> dict[str, Any]:
        started = time.perf_counter()
        completed = subprocess.run(
            argv,
            cwd=self.config.sdk_build_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=duration_sec + self.config.command_timeout_extra_sec,
            check=False,
        )
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 1),
        }

    def _stop_active_process(self) -> dict[str, Any] | None:
        with self.lock:
            process = self.active_process
            command = self.active_command
            self.active_process = None
            self.active_command = None
        if process is None:
            return None
        result = {"pid": process.pid, "command": command, "was_running": process.poll() is None}
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
        return result

    def _cleanup_stale_control_processes(self) -> dict[str, Any]:
        binary_name = Path(self._display_binary()).name
        pattern = f"{binary_name} {self.config.interface}"
        attempts: list[dict[str, Any]] = []
        for signal_name in ("-TERM", "-KILL"):
            try:
                completed = subprocess.run(
                    ["pkill", signal_name, "-f", pattern],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=0.5,
                )
                attempts.append({"signal": signal_name, "returncode": completed.returncode})
            except FileNotFoundError:
                return {"ok": False, "skipped": True, "reason": "pkill not found"}
            except Exception as exc:
                attempts.append({"signal": signal_name, "error": str(exc)})
            time.sleep(0.05)
            if not self._has_stale_control_process():
                return {"ok": True, "attempts": attempts}
        return {"ok": not self._has_stale_control_process(), "attempts": attempts}

    def _has_stale_control_process(self) -> bool:
        pattern = f"{Path(self._display_binary()).name} {self.config.interface}"
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


class Application(dbus.service.Object):
    def __init__(self, bus: dbus.SystemBus) -> None:
        self.path = "/com/bwton/g1d_ble_remote"
        self.services: list[Service] = []
        super().__init__(bus, self.path)

    def get_path(self) -> dbus.ObjectPath:
        return dbus.ObjectPath(self.path)

    def add_service(self, service: "Service") -> None:
        self.services.append(service)

    @dbus.service.method(DBUS_OM_IFACE, out_signature="a{oa{sa{sv}}}")
    def GetManagedObjects(self) -> dict[dbus.ObjectPath, Any]:
        response: dict[dbus.ObjectPath, Any] = {}
        for service in self.services:
            response[service.get_path()] = service.get_properties()
            for characteristic in service.characteristics:
                response[characteristic.get_path()] = characteristic.get_properties()
        return response


class Service(dbus.service.Object):
    PATH_BASE = "/com/bwton/g1d_ble_remote/service"

    def __init__(self, bus: dbus.SystemBus, index: int, uuid: str, primary: bool) -> None:
        self.path = self.PATH_BASE + str(index)
        self.bus = bus
        self.uuid = uuid
        self.primary = primary
        self.characteristics: list[Characteristic] = []
        super().__init__(bus, self.path)

    def get_properties(self) -> dict[str, Any]:
        return {
            GATT_SERVICE_IFACE: {
                "UUID": self.uuid,
                "Primary": self.primary,
                "Characteristics": dbus.Array(
                    [characteristic.get_path() for characteristic in self.characteristics],
                    signature="o",
                ),
            }
        }

    def get_path(self) -> dbus.ObjectPath:
        return dbus.ObjectPath(self.path)

    def add_characteristic(self, characteristic: "Characteristic") -> None:
        self.characteristics.append(characteristic)

    @dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
    def GetAll(self, interface: str) -> dict[str, Any]:
        if interface != GATT_SERVICE_IFACE:
            raise InvalidArgsException()
        return self.get_properties()[GATT_SERVICE_IFACE]


class Characteristic(dbus.service.Object):
    def __init__(self, bus: dbus.SystemBus, index: int, uuid: str, flags: list[str], service: Service) -> None:
        self.path = service.path + "/char" + str(index)
        self.bus = bus
        self.uuid = uuid
        self.service = service
        self.flags = flags
        super().__init__(bus, self.path)

    def get_properties(self) -> dict[str, Any]:
        return {
            GATT_CHRC_IFACE: {
                "Service": self.service.get_path(),
                "UUID": self.uuid,
                "Flags": self.flags,
            }
        }

    def get_path(self) -> dbus.ObjectPath:
        return dbus.ObjectPath(self.path)

    @dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
    def GetAll(self, interface: str) -> dict[str, Any]:
        if interface != GATT_CHRC_IFACE:
            raise InvalidArgsException()
        return self.get_properties()[GATT_CHRC_IFACE]

    @dbus.service.method(GATT_CHRC_IFACE, in_signature="a{sv}", out_signature="ay")
    def ReadValue(self, options: dict[str, Any]) -> list[dbus.Byte]:
        raise NotSupportedException()

    @dbus.service.method(GATT_CHRC_IFACE, in_signature="aya{sv}")
    def WriteValue(self, value: list[Any], options: dict[str, Any]) -> None:
        raise NotSupportedException()


class ControlCharacteristic(Characteristic):
    def __init__(self, bus: dbus.SystemBus, index: int, service: Service, controller: G1DBleController) -> None:
        super().__init__(bus, index, CONTROL_UUID, ["write", "write-without-response"], service)
        self.controller = controller

    @dbus.service.method(GATT_CHRC_IFACE, in_signature="aya{sv}")
    def WriteValue(self, value: list[Any], options: dict[str, Any]) -> None:
        text = byte_array_to_text(value)
        self.controller.handle_message(text)


class StatusCharacteristic(Characteristic):
    def __init__(self, bus: dbus.SystemBus, index: int, service: Service, controller: G1DBleController) -> None:
        super().__init__(bus, index, STATUS_UUID, ["read"], service)
        self.controller = controller

    @dbus.service.method(GATT_CHRC_IFACE, in_signature="a{sv}", out_signature="ay")
    def ReadValue(self, options: dict[str, Any]) -> list[dbus.Byte]:
        return text_to_dbus_bytes(self.controller.status_text())


class Advertisement(dbus.service.Object):
    PATH_BASE = "/com/bwton/g1d_ble_remote/advertisement"

    def __init__(self, bus: dbus.SystemBus, index: int, local_name: str, include_service_uuid: bool) -> None:
        self.path = self.PATH_BASE + str(index)
        self.bus = bus
        self.ad_type = "peripheral"
        self.local_name = local_name
        self.service_uuids = [SERVICE_UUID] if include_service_uuid else []
        super().__init__(bus, self.path)

    def get_path(self) -> dbus.ObjectPath:
        return dbus.ObjectPath(self.path)

    def get_properties(self) -> dict[str, Any]:
        includes = ["local-name", "tx-power"]
        props: dict[str, Any] = {
            LE_ADVERTISEMENT_IFACE: {
                "Type": self.ad_type,
                "Includes": dbus.Array(includes, signature="s"),
            }
        }
        if self.service_uuids:
            props[LE_ADVERTISEMENT_IFACE]["ServiceUUIDs"] = dbus.Array(self.service_uuids, signature="s")
        return props

    @dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
    def GetAll(self, interface: str) -> dict[str, Any]:
        if interface != LE_ADVERTISEMENT_IFACE:
            raise InvalidArgsException()
        return self.get_properties()[LE_ADVERTISEMENT_IFACE]

    @dbus.service.method(LE_ADVERTISEMENT_IFACE, in_signature="", out_signature="")
    def Release(self) -> None:
        print("BLE advertisement released", flush=True)


class InvalidArgsException(dbus.exceptions.DBusException):
    _dbus_error_name = "org.freedesktop.DBus.Error.InvalidArgs"


class NotSupportedException(dbus.exceptions.DBusException):
    _dbus_error_name = "org.bluez.Error.NotSupported"


def find_adapter(bus: dbus.SystemBus, adapter_name: str) -> str:
    remote_om = dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, "/"), DBUS_OM_IFACE)
    objects = remote_om.GetManagedObjects()
    wanted = f"/org/bluez/{adapter_name}"
    for path, interfaces in objects.items():
        if ADAPTER_IFACE in interfaces and str(path) == wanted:
            return str(path)
    for path, interfaces in objects.items():
        if ADAPTER_IFACE in interfaces:
            return str(path)
    raise RuntimeError("Bluetooth adapter not found")


def configure_adapter(bus: dbus.SystemBus, adapter_path: str, alias: str) -> None:
    props = dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, adapter_path), DBUS_PROP_IFACE)
    props.Set(ADAPTER_IFACE, "Powered", dbus.Boolean(True))
    if alias:
        props.Set(ADAPTER_IFACE, "Alias", dbus.String(alias))


def register_app_cb() -> None:
    print("GATT application registered", flush=True)


def register_ad_cb() -> None:
    print("BLE advertisement registered", flush=True)


def register_error_cb(error: Exception) -> None:
    print(f"BLE registration failed: {error}", file=sys.stderr, flush=True)
    if MAIN_LOOP is not None:
        MAIN_LOOP.quit()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the G1-D BLE remote control GATT service.")
    parser.add_argument("--adapter", default=os.environ.get("G1D_BLE_ADAPTER", "hci0"))
    parser.add_argument("--local-name", default=os.environ.get("G1D_BLE_LOCAL_NAME", "G1D"))
    parser.add_argument("--adapter-alias", default=os.environ.get("G1D_BLE_ADAPTER_ALIAS", ""))
    parser.add_argument(
        "--advertise-service-uuid",
        action="store_true",
        default=bool(os.environ.get("G1D_BLE_ADVERTISE_SERVICE_UUID")),
        help="Include the 128-bit GATT service UUID in advertisements. Disabled by default to keep the packet small.",
    )
    parser.add_argument("--sdk-build-dir", default=os.environ.get("UNITREE_SDK_BUILD_DIR", str(DEFAULT_SDK_BUILD_DIR)))
    parser.add_argument("--interface", default=os.environ.get("DDS_INTERFACE", DEFAULT_INTERFACE))
    parser.add_argument("--execute-enabled", action="store_true", default=bool(os.environ.get("G1D_BLE_EXECUTE_ENABLED")))
    parser.add_argument("--hold-duration-sec", type=float, default=float(os.environ.get("G1D_BLE_HOLD_DURATION_SEC", DEFAULT_HOLD_DURATION_SEC)))
    parser.add_argument("--watchdog-sec", type=float, default=float(os.environ.get("G1D_BLE_WATCHDOG_SEC", DEFAULT_WATCHDOG_SEC)))
    parser.add_argument("--max-speed", type=float, default=float(os.environ.get("G1D_BLE_MAX_SPEED", 0.3)))
    parser.add_argument("--max-turn-speed", type=float, default=float(os.environ.get("G1D_BLE_MAX_TURN_SPEED", 0.6)))
    parser.add_argument("--max-column-speed", type=float, default=float(os.environ.get("G1D_BLE_MAX_COLUMN_SPEED", 1.0)))
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    config = BleRemoteConfig(
        adapter=args.adapter,
        local_name=args.local_name,
        adapter_alias=args.adapter_alias or args.local_name,
        advertise_service_uuid=bool(args.advertise_service_uuid),
        sdk_build_dir=str(args.sdk_build_dir).replace("\\", "/"),
        interface=args.interface,
        execute_enabled=bool(args.execute_enabled),
        hold_duration_sec=args.hold_duration_sec,
        watchdog_sec=args.watchdog_sec,
        max_speed=args.max_speed,
        max_turn_speed=args.max_turn_speed,
        max_column_speed=args.max_column_speed,
    )
    mode = "EXECUTE" if config.execute_enabled else "PREVIEW"
    print(f"starting G1-D BLE remote {config.local_name} on {config.adapter} [{mode}]", flush=True)

    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
    bus = dbus.SystemBus()
    adapter_path = find_adapter(bus, config.adapter)
    configure_adapter(bus, adapter_path, config.adapter_alias)
    adapter = bus.get_object(BLUEZ_SERVICE_NAME, adapter_path)

    controller = G1DBleController(config)
    app = Application(bus)
    service = Service(bus, 0, SERVICE_UUID, True)
    service.add_characteristic(ControlCharacteristic(bus, 0, service, controller))
    service.add_characteristic(StatusCharacteristic(bus, 1, service, controller))
    app.add_service(service)
    advertisement = Advertisement(bus, 0, config.local_name, config.advertise_service_uuid)

    service_manager = dbus.Interface(adapter, GATT_MANAGER_IFACE)
    ad_manager = dbus.Interface(adapter, LE_ADVERTISING_MANAGER_IFACE)
    service_manager.RegisterApplication(app.get_path(), {}, reply_handler=register_app_cb, error_handler=register_error_cb)
    ad_manager.RegisterAdvertisement(advertisement.get_path(), {}, reply_handler=register_ad_cb, error_handler=register_error_cb)

    global MAIN_LOOP
    MAIN_LOOP = GLib.MainLoop()
    GLib.timeout_add(int(config.watchdog_sec * 500), controller.watchdog_tick)

    def stop_loop(signum: int, frame: Any) -> None:
        controller.stop(f"signal_{signum}")
        try:
            ad_manager.UnregisterAdvertisement(advertisement.get_path())
        except Exception:
            pass
        if MAIN_LOOP is not None:
            MAIN_LOOP.quit()

    signal.signal(signal.SIGTERM, stop_loop)
    signal.signal(signal.SIGINT, stop_loop)
    MAIN_LOOP.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
