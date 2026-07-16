"""Packaged AIsketcher Studio application.

The module deliberately avoids importing Gradio at import time.  Base SDK
users can therefore import the controller and inspect the bundled Guided
Sample without installing the optional ``demo`` dependencies.
"""

from __future__ import annotations

from typing import Any

from .runtime import AppController, AppState, GuidedSampleCatalog

__all__ = ["AppController", "AppState", "GuidedSampleCatalog", "build_app"]


def build_app(*args: Any, **kwargs: Any) -> Any:
    """Build the Gradio application, importing Gradio only when requested."""

    from .app import build_app as _build_app

    return _build_app(*args, **kwargs)
