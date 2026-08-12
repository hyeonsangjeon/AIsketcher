from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from aisketcher.tour import TOUR_HOST, build_tour_server, tour_url


def _read(url: str, *, method: str = "GET") -> tuple[bytes, object]:
    request = Request(url, method=method)
    with urlopen(request, timeout=5) as response:
        return response.read(), response.headers


def test_tour_server_renders_bilingual_guided_sample_from_package() -> None:
    with build_tour_server() as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = tour_url(server)
        try:
            html, headers = _read(base_url)
            manifest, _ = _read(base_url + "sample/manifest.json")
            image, image_headers = _read(base_url + "sample/scout/scout-01.png")
            head_body, head_headers = _read(base_url, method="HEAD")
        finally:
            server.shutdown()
            thread.join(timeout=5)

    assert server.server_address[0] == TOUR_HOST == "127.0.0.1"
    assert b"One sketch." in html
    assert "하나의 스케치.".encode() in html
    assert b"No Gradio. No Torch. No model download." in html
    assert b'fetch("/sample/manifest.json"' in html
    assert b"response.text()" in html
    assert b"rawManifest.matchAll" in html
    assert b'"schema": "aisketcher.manifest/v1"' in manifest
    assert image.startswith(b"\x89PNG\r\n\x1a\n")
    assert image_headers.get_content_type() == "image/png"
    assert "default-src 'self'" in headers["Content-Security-Policy"]
    assert "connect-src 'self'" in headers["Content-Security-Policy"]
    assert head_body == b""
    assert int(head_headers["Content-Length"]) == len(html)


def test_tour_server_rejects_unlisted_and_traversal_paths() -> None:
    with build_tour_server() as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = tour_url(server)
        try:
            for path in ("pyproject.toml", "../pyproject.toml", "%2e%2e/pyproject.toml"):
                with pytest.raises(HTTPError) as error:
                    _read(base_url + path)
                assert error.value.code == 404
        finally:
            server.shutdown()
            thread.join(timeout=5)


@pytest.mark.parametrize("port", (0, -1, 65536))
def test_tour_server_rejects_invalid_explicit_ports(port: int) -> None:
    with pytest.raises(ValueError, match="port must be between 1 and 65535"):
        build_tour_server(port=port)


def test_tour_import_does_not_load_optional_model_or_ui_runtimes() -> None:
    repository = Path(__file__).resolve().parents[1]
    script = r'''
import builtins
import sys

real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name.split(".", 1)[0] in {"diffusers", "gradio", "huggingface_hub", "torch"}:
        raise AssertionError(f"optional runtime imported: {name}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from aisketcher.tour import build_tour_server

with build_tour_server() as server:
    assert server.server_address[0] == "127.0.0.1"

assert not ({"diffusers", "gradio", "huggingface_hub", "torch"} & set(sys.modules))
'''
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
