"""Local-only HTTP checks; never contact robot hardware."""
import json
import threading
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

from services.g1d_tool_portal import server as portal
from services.es80z_api_proxy import server as proxy


@contextmanager
def running(handler):
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield 'http://127.0.0.1:%d' % server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class IntegrationTests(unittest.TestCase):
    def test_portal_catalog_and_assets(self):
        instance = portal.Portal('box.example', '127.0.0.1', 0.1)
        with patch.object(instance, '_probe', return_value={'state': 'online'}):
            with running(portal.make_handler(instance)) as base:
                for route in ['/', '/static/portal.css', '/static/portal.js', '/health']:
                    with urlopen(base + route) as response:
                        self.assertEqual(response.status, 200)
                        self.assertTrue(response.read())
                req = Request(base + '/api/catalog', headers={'Host': 'body.example:18079'})
                with urlopen(req) as response:
                    data = json.load(response)
                self.assertEqual(data['body_public_host'], 'body.example')
                for tool in data['tools']:
                    host = 'body.example' if tool['location'] == 'body' else 'box.example'
                    self.assertTrue(tool['url'].startswith('http://' + host + ':'))
                with self.assertRaises(HTTPError) as error:
                    urlopen(base + '/static/../server.py')
                self.assertEqual(error.exception.code, 404)

    def test_proxy_forwards_method_path_body_and_error(self):
        received = []

        class Backend(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(409 if self.path == '/error' else 200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"ok":true}')

            def do_POST(self):
                received.append((self.path, self.rfile.read(int(self.headers['Content-Length']))))
                self.do_GET()

        with running(Backend) as upstream:
            with running(proxy.make_handler(upstream, 1)) as base:
                for route in ['/', '/ui', '/index.html', '/static/suction.css', '/static/suction.js', '/api/v1/health']:
                    with urlopen(base + route) as response:
                        self.assertEqual(response.status, 200)
                        self.assertTrue(response.read())
                req = Request(base + '/api/v1/command?test=1', data=b'{"command":"test"}', headers={'Content-Type': 'application/json'})
                with urlopen(req) as response:
                    self.assertEqual(response.status, 200)
                self.assertEqual(received, [('/api/v1/command?test=1', b'{"command":"test"}')])
                with self.assertRaises(HTTPError) as error:
                    urlopen(base + '/error')
                self.assertEqual(error.exception.code, 409)

    def test_proxy_unavailable_returns_502(self):
        with patch.object(proxy, 'urlopen', side_effect=OSError('unavailable')):
            with running(proxy.make_handler('http://unused.invalid', 1)) as base:
                with self.assertRaises(HTTPError) as error:
                    urlopen(base + '/api/v1/health')
                self.assertEqual(error.exception.code, 502)
                self.assertFalse(json.load(error.exception)['ok'])


if __name__ == '__main__':
    unittest.main()
