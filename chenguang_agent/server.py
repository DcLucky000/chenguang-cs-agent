"""Minimal demo HTTP: health check and one-turn ask. No auth, no tenants."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from chenguang_agent.agent import CustomerServiceAgent
from chenguang_agent.kb import load_kb
from chenguang_agent.safety import INPUT_TOO_LONG, RATE_LIMITED, RateLimiter


def make_server(
    host: str,
    port: int,
    *,
    use_llm: bool = False,
    planner: str | None = None,
    limiter: RateLimiter | None = None,
) -> ThreadingHTTPServer:
    kb = load_kb()
    agent = CustomerServiceAgent(kb, use_llm=use_llm, planner=planner, limiter=limiter)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path != "/health":
                self._send(404, {"ok": False, "error": "not_found"})
                return
            self._send(200, {"ok": True, "service": "chenguang-agent", "kb_version": kb.kb_version})

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path != "/ask":
                self._send(404, {"ok": False, "error": "not_found"})
                return
            length = int(self.headers.get("Content-Length") or "0")
            if length > 100_000:
                self._send(413, {"ok": False, "error": "payload_too_large"})
                return
            raw = self.rfile.read(length) if length else b""
            try:
                body = json.loads(raw.decode("utf-8")) if raw else {}
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._send(400, {"ok": False, "error": "bad_json"})
                return
            if not isinstance(body, dict):
                self._send(400, {"ok": False, "error": "bad_json"})
                return
            question = body.get("question")
            if not isinstance(question, str) or not question.strip():
                self._send(400, {"ok": False, "error": "question_required"})
                return
            session_id = body.get("session_id")
            if session_id is not None and not isinstance(session_id, str):
                self._send(400, {"ok": False, "error": "bad_session_id"})
                return
            client = self.client_address[0] if self.client_address else "unknown"
            rate_key = session_id or f"ip:{client}"
            turn = agent.run(question, session_id=session_id or None, observe=True, rate_key=rate_key)
            status = 200
            if turn.flags.get(RATE_LIMITED):
                status = 429
            elif turn.flags.get(INPUT_TOO_LONG):
                status = 413
            self._send(status, turn.to_dict())

        def log_message(self, fmt: str, *args) -> None:
            return

        def _send(self, status: int, payload: dict) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)

    return ThreadingHTTPServer((host, port), Handler)


def serve(host: str, port: int, *, use_llm: bool = False, planner: str | None = None) -> int:
    server = make_server(host, port, use_llm=use_llm, planner=planner)
    bound_host, bound_port = server.server_address[:2]
    print(f"listening http://{bound_host}:{bound_port}")
    print("GET /health    POST /ask {\"question\": \"...\", \"session_id\": \"可选\"}")
    print("没有鉴权。这是本机演示，不是线上接入。Ctrl+C 结束。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
    return 0
