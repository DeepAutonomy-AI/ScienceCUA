from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .gateway import CapabilityGateway, CapabilityGatewayError, CapabilityInvocation


def _handler(
    gateway: CapabilityGateway,
    *,
    max_request_bytes: int,
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:  # noqa: N802 - stdlib hook
            if self.path != "/invoke":
                self._respond(404, {"ok": False, "error": {"code": "not_found"}})
                return
            try:
                length = int(self.headers.get("content-length", "0"))
            except ValueError:
                length = 0
            if length <= 0 or length > max_request_bytes:
                self._respond(
                    413,
                    {
                        "ok": False,
                        "error": {
                            "code": "request_too_large",
                            "message": "Capability request size is invalid",
                        },
                    },
                )
                return
            raw = self.rfile.read(length)
            authorization = self.headers.get("authorization", "")
            token = authorization[7:] if authorization.startswith("Bearer ") else ""
            try:
                invocation = CapabilityInvocation.model_validate_json(raw)
                result = gateway.invoke(token, invocation)
            except CapabilityGatewayError as exc:
                self._respond(
                    exc.status_code,
                    {"ok": False, "error": {"code": exc.code, "message": str(exc)}},
                )
                return
            except Exception:
                self._respond(
                    400,
                    {
                        "ok": False,
                        "error": {
                            "code": "invalid_request",
                            "message": "Capability request is invalid",
                        },
                    },
                )
                return
            self._respond(200, {"ok": True, "result": result})

        def _respond(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.send_header("cache-control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    return Handler


class CapabilityGatewayServer:
    """Private, token-protected HTTP bridge for the DSH worker."""

    def __init__(
        self,
        gateway: CapabilityGateway,
        *,
        max_request_bytes: int = 512 * 1024,
    ) -> None:
        self.gateway = gateway
        self.max_request_bytes = max_request_bytes
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._endpoint_url: str | None = None
        self._lock = threading.Lock()

    @property
    def endpoint_url(self) -> str:
        with self._lock:
            if self._endpoint_url is None:
                self._start_locked()
            return self._endpoint_url  # type: ignore[return-value]

    def close(self) -> None:
        with self._lock:
            server = self._server
            thread = self._thread
            self._server = None
            self._thread = None
            self._endpoint_url = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=2)

    def _start_locked(self) -> None:
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            _handler(self.gateway, max_request_bytes=self.max_request_bytes),
        )
        server.daemon_threads = True
        port = int(server.server_address[1])
        thread = threading.Thread(
            target=server.serve_forever,
            name="capability-gateway",
            daemon=True,
        )
        thread.start()
        self._server = server
        self._thread = thread
        self._endpoint_url = f"http://127.0.0.1:{port}/invoke"


def find_free_loopback_port() -> int:
    """Test helper retained here so bridge tests can avoid fixed ports."""

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
