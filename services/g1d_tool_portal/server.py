#!/usr/bin/env python3
"""Read-only directory and capability inventory for G1-D debugging tools."""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .catalog import TOOLS


PORTAL_VERSION = "2.0.0"
STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML = (STATIC_DIR / "index.html").read_text(encoding="utf-8")


def _fact(label: str, value: Any) -> Dict[str, str]:
    return {"label": label, "value": str(value)}


def _seconds(value: Any) -> Optional[str]:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}min"
    return f"{seconds / 3600:.1f}h"


class Portal:
    def __init__(
        self,
        box_public_host: str,
        box_internal_host: str,
        timeout: float,
        es80z_location: str = "box",
        suction_type: str = "es80z",
        control_api_enabled: bool = True,
        arm_ik_version: str = "",
    ) -> None:
        if suction_type not in ("evs01", "es80z"):
            raise ValueError("suction_type must be evs01 or es80z")
        if es80z_location not in ("body", "box"):
            raise ValueError("es80z_location must be body or box")

        self.tools = [dict(tool) for tool in TOOLS]
        if not control_api_enabled:
            self.tools = [tool for tool in self.tools if tool["id"] != "control_api"]
        if suction_type == "evs01":
            self.tools = [tool for tool in self.tools if tool["id"] != "es80z_backend"]
            suction = next(tool for tool in self.tools if tool["id"] == "suction")
            suction.update(
                path="/",
                name="EVS01 吸盘控制台",
                description="原吸盘状态、初始化、吸取、停止、释放与参数设置",
            )
        for tool in self.tools:
            if tool["id"] == "es80z_backend":
                tool["location"] = es80z_location
                tool["description"] = (
                    ("本体" if es80z_location == "body" else "盒子")
                    + "侧串口、Modbus RTU 与吸盘状态服务"
                )
            if tool["id"] == "arm_ik" and arm_ik_version.strip():
                tool["configured_version"] = arm_ik_version.strip()

        self.box_public_host = box_public_host
        self.box_internal_host = box_internal_host
        self.timeout = max(0.1, float(timeout))

    @staticmethod
    def _body_public_host(host_header: str) -> str:
        value = (host_header or "").strip()
        if value.startswith("["):
            return value.split("]", 1)[0] + "]"
        return value.rsplit(":", 1)[0] if ":" in value else value or "127.0.0.1"

    @staticmethod
    def _base_details(tool: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        details: Dict[str, Any] = {"features": [], "facts": []}
        version = tool.get("configured_version") or payload.get("version")
        if version not in (None, ""):
            details["version"] = str(version)
        service = payload.get("service") or payload.get("adapter")
        if service:
            details["service"] = str(service)
        return details

    @classmethod
    def _details_from_payload(
        cls, tool: Dict[str, Any], payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        details = cls._base_details(tool, payload)
        features: List[str] = details["features"]
        facts: List[Dict[str, str]] = details["facts"]
        adapter = tool.get("details_adapter", "")

        if adapter == "arm_ik":
            latest = payload.get("active_status") or payload.get("latest_status") or {}
            bridge = payload.get("bridge") or {}
            mode = latest.get("waist_ik_mode")
            joint_names = latest.get("controlled_joint_names") or []
            if mode == "coupled":
                features.append("腰部联合 IK")
            elif mode:
                features.append(f"IK {mode}")
            if joint_names:
                features.append(f"{len(joint_names)} DoF")
            if latest.get("continuous_pick_route_enabled"):
                features.append("连续抓取轨迹")
            interpolation = latest.get("continuous_interpolation")
            if interpolation:
                features.append(str(interpolation))
            if latest.get("slsqp_stop_on_first_safe"):
                features.append("快速 SLSQP")
            if latest.get("waist_axis_line_lock_enabled"):
                features.append("轴线锁腰")

            if mode:
                facts.append(_fact("模式", mode))
            if joint_names:
                facts.append(_fact("控制关节", len(joint_names)))
            stage = latest.get("stage")
            status_text = latest.get("status_text")
            if stage or status_text:
                facts.append(_fact("最近任务", " / ".join(x for x in (status_text, stage) if x)))
            actual = latest.get("waist_yaw_actual_deg")
            if actual is not None:
                facts.append(_fact("腰部角度", f"{float(actual):.1f}°"))
            subscribers = bridge.get("command_subscriber_count")
            publishers = bridge.get("status_publisher_count")
            if subscribers is not None or publishers is not None:
                facts.append(_fact("ROS2", f"命令订阅 {subscribers or 0} / 状态发布 {publishers or 0}"))
            details["runtime_ok"] = bool(bridge.get("running") and bridge.get("arm_node_seen"))
            details["subscriber_count"] = subscribers
            bridge_error = bridge.get("error")
            if bridge_error:
                details["reason"] = str(bridge_error)

        elif adapter in ("suction", "arm_console"):
            device = payload.get("device") or {}
            bridge = payload.get("bridge") or {}
            model = device.get("model")
            if model:
                features.append(str(model).split("-")[0])
                facts.append(_fact("设备", model))
            protocol = device.get("protocol")
            if protocol:
                features.append("Modbus RTU" if "modbus" in str(protocol).lower() else str(protocol))
            if payload.get("serial_error") in (None, "") and model:
                features.append("串口在线")
            if bridge:
                if bridge.get("mode"):
                    features.append(str(bridge["mode"]).upper())
                facts.append(_fact("机械臂节点", "在线" if bridge.get("arm_node_seen") else "未发现"))
                facts.append(_fact("命令订阅", bridge.get("command_subscriber_count", 0)))

        elif adapter == "yolo":
            model = os.path.basename(str(payload.get("yolo_model") or ""))
            device = payload.get("resolved_device")
            classes = payload.get("yolo_class_names") or []
            if device:
                features.append(str(device).upper())
            if payload.get("stereo_available"):
                features.append("双目")
            intrinsics = payload.get("intrinsics_effective_defaults") or {}
            if any(abs(float(x)) > 1e-12 for x in intrinsics.get("dist_coeffs") or []):
                features.append("畸变校正")
            if classes:
                features.append(f"{len(classes)} 类")
            if model:
                facts.append(_fact("模型", model))
            if device:
                facts.append(_fact("推理设备", device))
            if payload.get("model_cache_size") is not None:
                facts.append(_fact("模型缓存", payload["model_cache_size"]))

        elif adapter == "slam":
            for key, label in (
                ("has_map", "地图"),
                ("has_odom", "里程计"),
                ("has_scan", "激光"),
                ("has_point_cloud", "点云"),
                ("has_arm_task_status", "动作链"),
            ):
                if payload.get(key):
                    features.append(label)
            uptime = _seconds(payload.get("uptime_s"))
            if uptime:
                facts.append(_fact("运行时间", uptime))
            if payload.get("fault_snapshot_count") is not None:
                facts.append(_fact("故障快照", payload["fault_snapshot_count"]))

        elif adapter == "pose_adjust":
            endpoints = payload.get("endpoints") or []
            if "/adjust" in endpoints:
                features.append("基础微调")
            if "/adjust_target_angle" in endpoints:
                features.append("目标角策略")
            if "/adjust_right_entry" in endpoints:
                features.append("右侧安全入位")
            features.append("Dry Run")
            if payload.get("interface"):
                facts.append(_fact("控制网卡", payload["interface"]))

        elif adapter == "lift_height":
            if payload.get("offset_valid"):
                features.append("偏移有效")
            calibration = payload.get("auto_calibration") or {}
            if calibration.get("enabled"):
                features.append("开机自动标定")
            physical = payload.get("physical_height_m")
            if physical is not None:
                facts.append(_fact("当前高度", f"{float(physical):.3f} m"))
            source = payload.get("calibration_source")
            if source:
                facts.append(_fact("标定来源", source))
            age = payload.get("data_age_sec")
            if age is not None:
                facts.append(_fact("数据年龄", f"{float(age):.2f}s"))

        elif adapter == "x_nav":
            features.extend(("x_nav 兼容", "18083 转发"))
            upstream = payload.get("upstream")
            if upstream:
                facts.append(_fact("上游", upstream))
            uptime = _seconds(payload.get("uptime_s"))
            if uptime:
                facts.append(_fact("运行时间", uptime))

        elif adapter == "arm_relay":
            if payload.get("service"):
                facts.append(_fact("服务", payload["service"]))
            features.extend(("HTTP", "ROS2 Topic"))

        details["features"] = list(dict.fromkeys(str(value) for value in features if value))[:8]
        details["facts"] = facts[:6]
        return {key: value for key, value in details.items() if value not in (None, "", [], {})}

    @staticmethod
    def _state_from_payload(
        tool: Dict[str, Any], payload: Dict[str, Any], details: Dict[str, Any]
    ) -> str:
        if payload.get("ok") is False or payload.get("success") is False:
            return "degraded"
        if tool.get("details_adapter") == "x_nav" and payload.get("status") != "success":
            return "degraded"
        if tool.get("details_adapter") == "arm_ik":
            if not details.get("runtime_ok"):
                return "degraded"
            subscribers = details.get("subscriber_count")
            if subscribers is not None and int(subscribers) != 1:
                details["reason"] = f"机械臂命令订阅者数量异常：{subscribers}"
                return "degraded"
        return "online"

    def _probe(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        internal_host = "127.0.0.1" if tool["location"] == "body" else self.box_internal_host
        url = f"http://{internal_host}:{tool['port']}{tool['health_path']}"
        started = time.perf_counter()
        try:
            request = Request(url, headers={"User-Agent": f"g1d-tool-portal/{PORTAL_VERSION}"})
            with urlopen(
                request, timeout=max(self.timeout, float(tool.get("probe_timeout", 0)))
            ) as response:
                raw = response.read(524288)
                status_code = int(response.status)
            payload: Dict[str, Any] = {}
            if raw:
                try:
                    decoded = json.loads(raw.decode("utf-8", errors="replace"))
                    if isinstance(decoded, dict):
                        payload = decoded
                except (UnicodeDecodeError, json.JSONDecodeError):
                    pass
            details = self._details_from_payload(tool, payload)
            state = self._state_from_payload(tool, payload, details)
            error = None
            if state == "degraded":
                error = str(payload.get("error") or payload.get("message") or details.get("reason") or "服务返回异常状态")
            return {
                "state": state,
                "http_status": status_code,
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 1),
                "error": error,
                "details": details,
            }
        except HTTPError as exc:
            return {
                "state": "degraded",
                "http_status": int(exc.code),
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 1),
                "error": str(exc),
                "details": self._base_details(tool, {}),
            }
        except Exception as exc:
            return {
                "state": "offline",
                "http_status": None,
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 1),
                "error": str(exc),
                "details": self._base_details(tool, {}),
            }

    def catalog(self, host_header: str) -> Dict[str, Any]:
        body_public_host = self._body_public_host(host_header)
        with ThreadPoolExecutor(max_workers=8) as executor:
            probes = list(executor.map(self._probe, self.tools))
        result = []
        for source, probe in zip(self.tools, probes):
            item = dict(source)
            public_host = body_public_host if item["location"] == "body" else self.box_public_host
            item["public_host"] = public_host
            item["url"] = f"http://{public_host}:{item['port']}{item['path']}"
            item.update(probe)
            result.append(item)
        return {
            "ok": True,
            "service": "g1d_tool_portal",
            "version": PORTAL_VERSION,
            "body_public_host": body_public_host,
            "box_public_host": self.box_public_host,
            "box_internal_host": self.box_internal_host,
            "tools": result,
            "timestamp": time.time(),
        }


def make_handler(portal: Portal):
    class Handler(BaseHTTPRequestHandler):
        server_version = f"G1DToolPortal/{PORTAL_VERSION}"

        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            assets = {
                "/static/portal.css": ("portal.css", "text/css; charset=utf-8"),
                "/static/portal.js": ("portal.js", "text/javascript; charset=utf-8"),
            }
            if path in assets:
                filename, content_type = assets[path]
                payload = (STATIC_DIR / filename).read_bytes()
                self._send(200, content_type, payload)
                return
            if path == "/":
                self._send(200, "text/html; charset=utf-8", INDEX_HTML.encode("utf-8"))
                return
            if path == "/health":
                payload = {
                    "ok": True,
                    "service": "g1d_tool_portal",
                    "version": PORTAL_VERSION,
                    "port": self.server.server_port,
                }
                self._send(
                    200,
                    "application/json; charset=utf-8",
                    json.dumps(payload, separators=(",", ":")).encode("utf-8"),
                )
                return
            if path == "/api/catalog":
                payload = portal.catalog(self.headers.get("Host", ""))
                body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                self._send(200, "application/json; charset=utf-8", body)
                return
            self._send(404, "application/json; charset=utf-8", b'{"ok":false,"error":"not found"}')

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[{self.log_date_time_string()}] {self.client_address[0]} {fmt % args}", flush=True)

    return Handler


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def main() -> None:
    parser = argparse.ArgumentParser(description="G1-D debugging tool directory")
    parser.add_argument("--host", default=os.environ.get("G1D_PORTAL_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("G1D_PORTAL_PORT", "18079")))
    parser.add_argument("--box-public-host", default=os.environ.get("G1D_BOX_PUBLIC_HOST", "192.168.61.132"))
    parser.add_argument("--box-internal-host", default=os.environ.get("G1D_BOX_INTERNAL_HOST", "192.168.123.5"))
    parser.add_argument("--probe-timeout", type=float, default=float(os.environ.get("G1D_PORTAL_PROBE_TIMEOUT", "1.2")))
    args = parser.parse_args()
    portal = Portal(
        args.box_public_host,
        args.box_internal_host,
        args.probe_timeout,
        os.environ.get("G1D_ES80Z_LOCATION", "box"),
        os.environ.get("G1D_SUCTION_TYPE", "es80z"),
        _env_bool("G1D_CONTROL_API_ENABLED", True),
        os.environ.get("G1D_ARM_IK_VERSION", ""),
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(portal))
    print(f"G1-D tool portal {PORTAL_VERSION} listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
