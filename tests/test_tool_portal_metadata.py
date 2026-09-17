"""Metadata extraction tests; no robot or ROS access is used."""
import unittest

from services.g1d_tool_portal.server import PORTAL_VERSION, Portal


class ToolPortalMetadataTests(unittest.TestCase):
    def setUp(self):
        self.portal = Portal(
            "box.example",
            "127.0.0.1",
            0.1,
            es80z_location="body",
            arm_ik_version="6.2",
        )

    def tool(self, tool_id):
        return next(tool for tool in self.portal.tools if tool["id"] == tool_id)

    def test_arm_ik_features_come_from_runtime_status(self):
        payload = {
            "latest_status": {
                "status_text": "DONE",
                "stage": "IDLE",
                "waist_ik_mode": "coupled",
                "controlled_joint_names": ["torso_Joint"] + [f"arm_{index}" for index in range(7)],
                "continuous_pick_route_enabled": True,
                "continuous_interpolation": "monotone_cubic",
                "slsqp_stop_on_first_safe": True,
                "waist_axis_line_lock_enabled": True,
                "waist_yaw_actual_deg": 4.25,
            },
            "bridge": {
                "running": True,
                "arm_node_seen": True,
                "command_subscriber_count": 1,
                "status_publisher_count": 1,
            },
        }
        tool = self.tool("arm_ik")
        details = self.portal._details_from_payload(tool, payload)
        self.assertEqual(details["version"], "6.2")
        self.assertIn("腰部联合 IK", details["features"])
        self.assertIn("8 DoF", details["features"])
        self.assertIn("连续抓取轨迹", details["features"])
        self.assertEqual(self.portal._state_from_payload(tool, payload, details), "online")

    def test_arm_ik_requires_exactly_one_command_subscriber(self):
        payload = {
            "latest_status": {"waist_ik_mode": "coupled"},
            "bridge": {
                "running": True,
                "arm_node_seen": True,
                "command_subscriber_count": 2,
            },
        }
        tool = self.tool("arm_ik")
        details = self.portal._details_from_payload(tool, payload)
        self.assertEqual(self.portal._state_from_payload(tool, payload, details), "degraded")
        self.assertIn("订阅者数量异常", details["reason"])

    def test_existing_services_expose_selected_metadata(self):
        suction = self.portal._details_from_payload(
            self.tool("suction"),
            {
                "ok": True,
                "service": "es80z-api",
                "version": "1.1.0",
                "device": {"model": "ES80Z-S4-RS-N-M1.5", "protocol": "Modbus RTU"},
                "serial_error": None,
            },
        )
        self.assertEqual(suction["version"], "1.1.0")
        self.assertIn("Modbus RTU", suction["features"])

        yolo = self.portal._details_from_payload(
            self.tool("yolo"),
            {
                "yolo_model": "models/YanHe20class_2.pt",
                "resolved_device": "cuda:0",
                "stereo_available": True,
                "yolo_class_names": [str(index) for index in range(20)],
                "intrinsics_effective_defaults": {"dist_coeffs": [0.05, 0, 0, 0, 0]},
            },
        )
        self.assertIn("双目", yolo["features"])
        self.assertIn("畸变校正", yolo["features"])
        self.assertIn("20 类", yolo["features"])

    def test_portal_version_is_declared(self):
        self.assertEqual(PORTAL_VERSION, "2.0.0")


if __name__ == "__main__":
    unittest.main()
