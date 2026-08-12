"""Zero-download local product tour for the bundled Guided Sample.

The tour intentionally uses only the Python standard library.  It serves a
small, explicit allowlist of package resources on the IPv4 loopback address;
it does not initialize a model runtime, download assets, or emit telemetry.
"""

from __future__ import annotations

import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from typing import Any
from urllib.parse import unquote, urlsplit

TOUR_HOST = "127.0.0.1"

_ROUTES: dict[str, tuple[tuple[str, ...], str]] = {
    "/": (("tour", "index.html"), "text/html; charset=utf-8"),
    "/index.html": (("tour", "index.html"), "text/html; charset=utf-8"),
    "/sample/manifest.json": (
        ("studio_app", "assets", "pocket-kingdom", "manifest.json"),
        "application/json; charset=utf-8",
    ),
    "/sample/prepared/source.png": (
        ("studio_app", "assets", "pocket-kingdom", "prepared", "source.png"),
        "image/png",
    ),
    "/sample/prepared/control.png": (
        ("studio_app", "assets", "pocket-kingdom", "prepared", "control.png"),
        "image/png",
    ),
    "/sample/scout/scout-01.png": (
        ("studio_app", "assets", "pocket-kingdom", "scout", "scout-01.png"),
        "image/png",
    ),
    "/sample/scout/scout-02.png": (
        ("studio_app", "assets", "pocket-kingdom", "scout", "scout-02.png"),
        "image/png",
    ),
    "/sample/scout/scout-03.png": (
        ("studio_app", "assets", "pocket-kingdom", "scout", "scout-03.png"),
        "image/png",
    ),
    "/sample/scout/scout-04.png": (
        ("studio_app", "assets", "pocket-kingdom", "scout", "scout-04.png"),
        "image/png",
    ),
}


def _resource_bytes(parts: tuple[str, ...]) -> bytes:
    resource = files("aisketcher")
    for part in parts:
        resource = resource.joinpath(part)
    return resource.read_bytes()


class TourRequestHandler(BaseHTTPRequestHandler):
    """Serve only the static tour and its bundled sample resources."""

    server_version = "AIsketcherTour/1"

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._serve(include_body=True)

    def do_HEAD(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._serve(include_body=False)

    def _serve(self, *, include_body: bool) -> None:
        path = unquote(urlsplit(self.path).path)
        if path == "/favicon.ico":
            self.send_response(204)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return

        route = _ROUTES.get(path)
        if route is None:
            self.send_error(404, "Not found")
            return

        resource_parts, content_type = route
        try:
            payload = _resource_bytes(resource_parts)
        except (FileNotFoundError, OSError):
            self.send_error(500, "Packaged tour resource is unavailable")
            return

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self'; connect-src 'self'; "
            "style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
            "object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
        )
        self.end_headers()
        if include_body:
            self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:
        """Keep the product tour quiet unless the CLI reports an error."""


class TourHTTPServer(ThreadingHTTPServer):
    """Thread-per-request server that shuts down cleanly with the CLI."""

    daemon_threads = True
    allow_reuse_address = True


def build_tour_server(*, port: int | None = None) -> TourHTTPServer:
    """Create a loopback-only tour server without starting its event loop.

    ``None`` asks the operating system for an available ephemeral port.  A
    caller-provided port is validated before any socket is opened.
    """

    if port is not None and not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    return TourHTTPServer((TOUR_HOST, port or 0), TourRequestHandler)


def tour_url(server: TourHTTPServer) -> str:
    """Return the browser URL for a bound tour server."""

    host_value, port_value = server.server_address[:2]
    host = host_value.decode("ascii") if isinstance(host_value, bytes) else str(host_value)
    port = int(port_value)
    return f"http://{host}:{port}/"


def launch_tour(*, port: int | None = None, open_browser: bool = True) -> int:
    """Run the local tour until interrupted with Ctrl+C."""

    with build_tour_server(port=port) as server:
        url = tour_url(server)
        print("AIsketcher Guided Tour")
        print(f"Local only · no model download · no telemetry\n{url}", flush=True)
        if open_browser:
            opened = webbrowser.open_new_tab(url)
            if not opened:
                print("The browser did not open automatically. Open the URL above.")
        print("Press Ctrl+C to stop.", flush=True)
        try:
            server.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            print("\nAIsketcher tour stopped.")
    return 0


__all__ = [
    "TOUR_HOST",
    "TourHTTPServer",
    "TourRequestHandler",
    "build_tour_server",
    "launch_tour",
    "tour_url",
]
