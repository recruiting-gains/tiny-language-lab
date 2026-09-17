"""Local teaching frontend and optional Python generation API; loopback only.

The hosted static frontend runs the exported model in a browser worker instead.
"""
import argparse
import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import torch

from train_transformer import load_checkpoint

ROOT = Path(__file__).resolve().parent


class LessonServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, checkpoint):
        torch.set_num_threads(2)
        self.model, self.tokenizer, self.state = load_checkpoint(checkpoint)
        self.model.eval()
        self.generation_lock = threading.Lock()
        super().__init__(address, partial(Handler, directory=str(ROOT / "public")))


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        # Do not log user prompts or query strings.
        pass

    def end_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        super().end_headers()

    def list_directory(self, path):
        self.send_error(404)
        return None

    def send_json(self, status, payload):
        body = json.dumps(payload, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if urlparse(self.path).path == "/api/health":
            return self.send_json(200, {"status": "ready", "mode": "local Python backend",
                                        "parameters": sum(p.numel() for p in self.server.model.parameters()),
                                        "trained_steps": self.server.state["step"]})
        return super().do_GET()

    def do_POST(self):
        if urlparse(self.path).path != "/api/generate":
            return self.send_json(404, {"error": "Unknown endpoint."})
        allowed_origins = {f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"}
        if self.headers.get("Origin") and self.headers["Origin"] not in allowed_origins:
            return self.send_json(403, {"error": "Only same-origin local requests are accepted."})
        if self.headers.get_content_type() != "application/json":
            return self.send_json(415, {"error": "Use application/json."})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 1 <= length <= 4096:
                return self.send_json(413, {"error": "JSON request is empty or too large."})
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object.")
            prompt, count = payload.get("prompt", "the "), payload.get("maxTokens", 48)
            temperature, seed = payload.get("temperature", 0.8), payload.get("seed", 7)
            if not isinstance(prompt, str) or type(count) is not int or not 1 <= count <= 96:
                raise ValueError("Use a text prompt and 1–96 new characters.")
            if type(temperature) not in (int, float) or type(seed) is not int or not 0 <= seed <= 2**32 - 1:
                raise ValueError("Use numeric temperature and a nonnegative 32-bit seed.")
            if not self.server.generation_lock.acquire(blocking=False):
                return self.send_json(503, {"error": "Generation is busy; try again shortly."})
            try:
                output = self.server.model.generate(self.server.tokenizer, prompt, count, temperature, seed)
            finally:
                self.server.generation_lock.release()
            return self.send_json(200, {"text": output, "mode": "local Python generation"})
        except (ValueError, TypeError, OverflowError) as exc:
            return self.send_json(400, {"error": str(exc)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8847)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/transformer-v1/last-valid.pt")
    args = parser.parse_args()
    service = LessonServer(("127.0.0.1", args.port), args.checkpoint)
    print(f"Tiny Language Lab: http://127.0.0.1:{service.server_port}", flush=True)
    service.serve_forever()
