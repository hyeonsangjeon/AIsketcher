"""Compatibility launcher for :mod:`aisketcher.studio_app.app`.

The repository command ``python -m examples.studio_app.app`` remains valid;
installed-package users should use ``python -m aisketcher.studio_app.app``.
"""

from __future__ import annotations

from typing import Any

from aisketcher.studio_app import app as _canonical

build_app = _canonical.build_app
main = _canonical.main

# These helpers were imported by the original example tests and may also be
# useful to downstream example customizations.
_lock_choices = _canonical._lock_choices
_response_values = _canonical._response_values
_variation_choices = _canonical._variation_choices

__all__ = ["build_app", "main"]


def __getattr__(name: str) -> Any:
    """Delegate legacy module attributes to the canonical packaged app."""

    return getattr(_canonical, name)


if __name__ == "__main__":
    main()
