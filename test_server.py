"""Loopback HTTP checks against the locally trained transformer checkpoint."""
import http.client
import json
import threading
import unittest

from server import LessonServer, ROOT


class LessonServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checkpoint = ROOT / "runs/transformer-v1/last-valid.pt"
        if not checkpoint.is_file():
            raise unittest.SkipTest("Train runs/transformer-v1 before testing the real local API.")
        cls.server = LessonServer(("127.0.0.1", 0), checkpoint)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.origin = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=15)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def generate(self, payload, origin=None):
        headers = {"Content-Type": "application/json", "Origin": origin or self.origin}
        status, headers, body = self.request("POST", "/api/generate", json.dumps(payload), headers)
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(headers["Cache-Control"], "no-store")
        return status, json.loads(body)

    def test_health_reports_loaded_checkpoint(self):
        status, headers, body = self.request("GET", "/api/health")
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["mode"], "local Python backend")
        self.assertEqual(payload["parameters"], sum(p.numel() for p in self.server.model.parameters()))
        self.assertEqual(payload["trained_steps"], self.server.state["step"])
        self.assertGreater(payload["trained_steps"], 0)
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(headers["Referrer-Policy"], "no-referrer")

    def test_generation_matches_real_model_and_repeats(self):
        payload = {"prompt": "the quiet fox", "maxTokens": 8, "temperature": 0.8, "seed": 31}
        expected = self.server.model.generate(self.server.tokenizer, payload["prompt"], 8, 0.8, 31)
        status, generated = self.generate(payload)
        self.assertEqual(status, 200)
        self.assertEqual(generated, {"text": expected, "mode": "local Python generation"})
        self.assertEqual(len(generated["text"]), len(payload["prompt"]) + 8)
        self.assertEqual(self.generate(payload), (status, generated))

    def test_prompt_and_generation_types_are_rejected(self):
        invalid = [
            {"prompt": None}, {"prompt": ""}, {"prompt": "THE"}, {"prompt": "🦊"},
            {"prompt": "a" * 513}, {"maxTokens": 0}, {"maxTokens": 97},
            {"maxTokens": 1.5}, {"maxTokens": True}, {"maxTokens": "8"},
            {"temperature": None}, {"temperature": "0.8"}, {"temperature": True},
            {"temperature": 0.1}, {"temperature": 1.6}, {"temperature": float("nan")},
            {"seed": -1}, {"seed": 2**32}, {"seed": 1.5}, {"seed": True},
        ]
        for payload in invalid:
            with self.subTest(payload=payload):
                status, result = self.generate(payload)
                self.assertEqual(status, 400)
                self.assertIsInstance(result["error"], str)
        # Invalid requests must release the model lock for the next valid call.
        self.assertEqual(self.generate({"prompt": "the ", "maxTokens": 1})[0], 200)

    def test_bad_json_nonobject_and_content_type(self):
        for body in ("{", "null", "[]", '"text"', "true", "42"):
            with self.subTest(body=body):
                status, _, response = self.request("POST", "/api/generate", body,
                                                   {"Content-Type": "application/json"})
                self.assertEqual(status, 400)
                self.assertIn("error", json.loads(response))
        status, _, _ = self.request("POST", "/api/generate", "{}", {"Content-Type": "text/plain"})
        self.assertEqual(status, 415)

    def test_request_body_size_and_length_are_bounded(self):
        for length, body, expected in (("0", b"", 413), ("-1", b"", 413),
                                       ("4097", b"x" * 4097, 413), ("invalid", b"", 400)):
            with self.subTest(length=length):
                status, _, _ = self.request("POST", "/api/generate", body,
                                           {"Content-Type": "application/json", "Content-Length": length})
                self.assertEqual(status, expected)

    def test_external_and_null_origins_are_rejected(self):
        for origin in ("https://example.com", "null", "http://127.0.0.1:1", self.origin + ".example.com"):
            with self.subTest(origin=origin):
                status, payload = self.generate({"prompt": "the ", "maxTokens": 1}, origin)
                self.assertEqual(status, 403)
                self.assertIn("same-origin", payload["error"])
        localhost = f"http://localhost:{self.server.server_port}"
        self.assertEqual(self.generate({"prompt": "the ", "maxTokens": 1}, localhost)[0], 200)

    def test_busy_generation_is_rejected_without_blocking(self):
        self.server.generation_lock.acquire()
        try:
            status, payload = self.generate({"prompt": "the ", "maxTokens": 1})
            self.assertEqual(status, 503)
            self.assertIn("busy", payload["error"])
        finally:
            self.server.generation_lock.release()

    def test_static_root_is_public_and_private_paths_are_inaccessible(self):
        for path, file in (("/", ROOT / "public/index.html"), ("/inference.js", ROOT / "public/inference.js")):
            with self.subTest(path=path):
                status, _, body = self.request("GET", path)
                self.assertEqual(status, 200)
                self.assertEqual(body, file.read_bytes())
        for path in ("/server.py", "/PREP.md", "/STATUS.md", "/.git/config", "/.venv/pyvenv.cfg",
                     "/runs/transformer-v1/last-valid.pt", "/../server.py", "/%2e%2e/server.py",
                     "/%2e%2e%2fserver.py", "/api/generate"):
            with self.subTest(path=path):
                self.assertEqual(self.request("GET", path)[0], 404)
        self.assertEqual(self.request("POST", "/api/unknown", "{}", {"Content-Type": "application/json"})[0], 404)


if __name__ == "__main__":
    unittest.main()
