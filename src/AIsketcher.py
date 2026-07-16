"""Deprecated uppercase import facade for AIsketcher 0.0.x users."""

from __future__ import annotations

import warnings

from aisketcher import *  # noqa: F401,F403
from aisketcher import __all__ as _modern_all
from aisketcher import modelPipe as modelPipe
from aisketcher.modelPipe import correct_image_orientation, img2img, resize_image

warnings.warn(
    "Importing 'AIsketcher' is deprecated; use lowercase 'aisketcher'. "
    "The uppercase facade will be removed in 0.3.0.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    *_modern_all,
    "correct_image_orientation",
    "img2img",
    "modelPipe",
    "resize_image",
]
