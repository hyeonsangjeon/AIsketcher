"""Smoke-test the installed wheel without importing the repository examples."""

from __future__ import annotations

import tempfile
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path

import aisketcher
from aisketcher.cli import main
from aisketcher.studio_app import AppController, GuidedSampleCatalog, build_app


def run() -> None:
    assert version("AIsketcher") == aisketcher.__version__ == "0.3.0"
    assert find_spec("AIsketcher") is None
    assert find_spec("aisketcher.modelPipe") is None
    sample = GuidedSampleCatalog().load()
    assert len(sample.candidates) == 4
    assert sample.manifest_path.is_file()

    with tempfile.TemporaryDirectory(prefix="aisketcher-wheel-smoke-") as temporary:
        root = Path(temporary)
        config = root / "aisketcher.yaml"
        assert (
            main(
                [
                    "init",
                    "--path",
                    str(config),
                    "--outputs",
                    "1",
                    "--seed-mode",
                    "locked",
                    "--seed",
                    "6764547109648557242",
                    "--offline",
                ]
            )
            == 0
        )
        assert config.is_file()
        controller = AppController(workspace_root=root / "workspace")
        app = build_app(controller)
        assert app._studio_launch_kwargs["server_name"] == "127.0.0.1"
        assert controller.guided.available is True
        close = getattr(app, "close", None)
        if callable(close):
            close()

    print("installed wheel, CLI, Studio, and Guided Sample smoke test passed")


if __name__ == "__main__":
    run()
