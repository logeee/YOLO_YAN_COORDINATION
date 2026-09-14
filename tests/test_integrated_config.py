"""Configuration checks without ROS initialization or hardware commands."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from services.g1d_tool_portal import server as portal
from services.es80z_api_proxy import server as proxy

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which('bash')
if BASH is None and Path('D:/APP/Git/bin/bash.exe').exists():
    BASH = 'D:/APP/Git/bin/bash.exe'


class ConfigTests(unittest.TestCase):
    def test_http_environment_and_cli_precedence(self):
        with patch.dict(os.environ, {'SUCTION_PROXY_HOST': '127.0.0.1', 'SUCTION_PROXY_PORT': '19080', 'SUCTION_PROXY_UPSTREAM': 'http://stub:123', 'SUCTION_PROXY_TIMEOUT': '4'}):
            with patch.object(sys, 'argv', ['server', '--port', '19081']), patch.object(proxy, 'ThreadingHTTPServer') as server, patch.object(proxy, 'make_handler') as handler:
                proxy.main()
                self.assertEqual(server.call_args.args[0], ('127.0.0.1', 19081))
                handler.assert_called_once_with('http://stub:123', 4.0)
        with patch.dict(os.environ, {'G1D_BOX_PUBLIC_HOST': 'public.example', 'G1D_BOX_INTERNAL_HOST': 'internal.example', 'G1D_PORTAL_PORT': '19079'}):
            with patch.object(sys, 'argv', ['server']), patch.object(portal, 'ThreadingHTTPServer') as server, patch.object(portal, 'Portal') as factory:
                portal.main()
                self.assertEqual(factory.call_args.args[:2], ('public.example', 'internal.example'))
                self.assertEqual(server.call_args.args[0][1], 19079)

    def test_body_parser_without_ros_runtime(self):
        stubs = {name: types.ModuleType(name) for name in ['rclpy', 'rclpy.node', 'std_msgs', 'std_msgs.msg']}
        stubs['rclpy.node'].Node = object
        stubs['std_msgs.msg'].String = object
        spec = importlib.util.spec_from_file_location('body_config_test', ROOT / 'services/g1d_body_control/node.py')
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, stubs):
            spec.loader.exec_module(module)
        with patch.dict(os.environ, {'G1D_BODY_INTERFACE': 'eth_test', 'G1D_BODY_LIFT_HEIGHT_URL': 'http://stub:28089/status'}):
            parser = module.build_parser()
            self.assertEqual(parser.parse_args([]).interface, 'eth_test')
            self.assertEqual(parser.parse_args([]).lift_height_url, 'http://stub:28089/status')
            self.assertEqual(parser.parse_args(['--interface', 'cli_eth']).interface, 'cli_eth')

    @unittest.skipUnless(BASH, 'Bash not installed')
    def test_installer_creates_templates_without_overwriting_device_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'scripts').mkdir()
            (root / 'config').mkdir()
            target = root / 'scripts/install_autostart_services.sh'
            shutil.copyfile(ROOT / 'scripts/install_autostart_services.sh', target)
            for name in ['g1d_tool_portal', 'suction_api_proxy', 'g1d_body_control']:
                shutil.copyfile(ROOT / 'config' / (name + '.env.example'), root / 'config' / (name + '.env.example'))
            custom = root / 'config/g1d_tool_portal.env'
            custom.write_text('G1D_BOX_PUBLIC_HOST=keep-this-host\n', encoding='utf-8')
            result = subprocess.run([BASH, str(target), 'portal', 'suction-proxy', 'body-control'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn('Created config/suction_api_proxy.env', result.stderr)
            self.assertIn('Created config/g1d_body_control.env', result.stderr)
            self.assertEqual(custom.read_text(), 'G1D_BOX_PUBLIC_HOST=keep-this-host\n')

    @unittest.skipUnless(BASH, 'Bash not installed')
    def test_shell_syntax_and_config_loading(self):
        for name in ['install_autostart_services', 'g1d_tool_portal_server', 'suction_api_proxy_server', 'g1d_body_control']:
            subprocess.run([BASH, '-n', str(ROOT / 'scripts' / (name + '.sh'))], check=True, capture_output=True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'scripts').mkdir()
            (root / 'config').mkdir()
            # Replace the Bash exec builtin only in this test process. No Python service runs.
            harness = 'exec() { printf "PORT=%s BOX=%s UPSTREAM=%s IFACE=%s\\n" "${G1D_PORTAL_PORT-}" "${G1D_BOX_PUBLIC_HOST-}" "${SUCTION_PROXY_UPSTREAM-}" "${G1D_BODY_INTERFACE-}"; }; export -f exec; bash "$1"'
            for script, config, text, expected in [
                ('g1d_tool_portal_server', 'g1d_tool_portal', 'G1D_PORTAL_PORT=19079\nG1D_BOX_PUBLIC_HOST=example.box\n', 'PORT=19079 BOX=example.box'),
                ('suction_api_proxy_server', 'suction_api_proxy', 'SUCTION_PROXY_UPSTREAM=http://example.box:19089\n', 'UPSTREAM=http://example.box:19089'),
                ('g1d_body_control', 'g1d_body_control', 'G1D_BODY_ROS_SETUP=/dev/null\nG1D_BODY_SLAM_SETUP=/dev/null\nG1D_BODY_INTERFACE=test0\n', 'IFACE=test0'),
            ]:
                target = root / 'scripts' / (script + '.sh')
                shutil.copyfile(ROOT / 'scripts' / target.name, target)
                missing = subprocess.run([BASH, str(target)], capture_output=True, text=True)
                self.assertNotEqual(missing.returncode, 0)
                self.assertIn('Missing', missing.stderr)
                (root / 'config' / (config + '.env')).write_text(text, encoding='utf-8')
                result = subprocess.run([BASH, '-c', harness, '_', str(target)], capture_output=True, text=True, check=True)
                self.assertIn(expected, result.stdout)


if __name__ == '__main__':
    unittest.main()
