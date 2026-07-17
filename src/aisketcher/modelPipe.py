"""Deprecated v0.x compatibility helpers.

New code should use :class:`aisketcher.Studio`. This module is intentionally
dependency-lazy and contains no AWS behavior.
"""

from __future__ import annotations

import importlib
import warnings
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from .controls import prepare
from .errors import OptionalDependencyError, RemovedFeatureError, ValidationError
from .models import CannyConfig


def _warn(name: str) -> None:
    warnings.warn(
        f"aisketcher.modelPipe.{name} is deprecated and will be removed in 0.3.0; "
        "use Studio instead",
        DeprecationWarning,
        stacklevel=2,
    )


def correct_image_orientation(image: Image.Image) -> Image.Image:
    _warn("correct_image_orientation")
    return ImageOps.exif_transpose(image).copy()


def resize_image(image_path: str | Path, pixels: int) -> Image.Image:
    _warn("resize_image")
    if pixels < 1:
        raise ValidationError("pixels must be positive")
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB").copy()
    scale = pixels / max(image.size)
    size = (
        max(1, int(image.width * scale)),
        max(1, int(image.height * scale)),
    )
    return image.resize(size, Image.Resampling.LANCZOS)


def img2img(
    img_path: str | Path,
    prompt: str,
    num_steps: int = 20,
    guidance_scale: float = 7,
    seed: int = 0,
    low: int = 100,
    high: int = 200,
    pipe: Any | None = None,
    trans_info: Any | None = None,
    **kwargs: Any,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    """Run the legacy supplied-pipeline call without AWS translation."""

    _warn("img2img")
    if trans_info is not None or "aws" in kwargs or "translate" in kwargs:
        raise RemovedFeatureError(
            "AWS translation and credential arguments were removed in AIsketcher 0.2.0"
        )
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise ValidationError(f"Unknown legacy img2img argument(s): {unknown}")
    if pipe is None:
        raise ValidationError("pipe must be a configured Diffusers ControlNet pipeline")
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise OptionalDependencyError(
            "Legacy img2img requires: pip install 'aisketcher[local]'"
        ) from exc

    with Image.open(img_path) as opened:
        original = ImageOps.exif_transpose(opened).convert("RGB").copy()
    prepared = prepare(
        original,
        max_side=800,
        canny=CannyConfig(low=low, high=high),
    )
    output = pipe(
        prompt,
        negative_prompt=None,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=torch.manual_seed(seed),
        image=prepared.control,
    ).images[0]
    return original, prepared.control, output.convert("RGB").resize(original.size)


__all__ = ["correct_image_orientation", "img2img", "resize_image"]
