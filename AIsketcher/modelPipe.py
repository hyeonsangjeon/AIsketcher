"""Image preparation, ControlNet inference, and prompt translation helpers."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Union
from urllib.parse import urlparse

from PIL import Image, ImageOps

ImageSource = Union[str, os.PathLike[str], Image.Image]

DEFAULT_PROMPT = (
    "(8k, best quality, masterpiece:1.2), "
    "(realistic, photo-realistic:1.37), ultra-detailed"
)
DEFAULT_NEGATIVE_PROMPT = (
    "NSFW, lowres, ((bad anatomy)), ((bad hands)), text, missing finger, extra digits, "
    "fewer digits, blurry, ((mutated hands and fingers)), (poorly drawn face), "
    "((mutation)), ((deformed face)), (ugly), ((bad proportions)), ((extra limbs)), "
    "extra face, (double head), (extra head), ((extra feet)), monster, logo, cropped, "
    "worst quality, low quality, normal quality, jpeg, humpbacked, long body, long neck, "
    "((jpeg artifacts))"
)


def img2img(
    img_path: ImageSource,
    prompt: str,
    num_steps: int = 20,
    guidance_scale: float = 7,
    seed: int = 0,
    low: int = 100,
    high: int = 200,
    pipe: Any = None,
    trans_info: Mapping[str, Any] | None = None,
    *,
    image_size: int = 800,
    default_prompt: str = DEFAULT_PROMPT,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    """Generate a ControlNet-guided image from an image and text prompt.

    The original positional arguments remain compatible with AIsketcher 0.0.x.
    ``pipe`` must be a loaded Diffusers ControlNet pipeline. When ``trans_info`` is
    provided, the prompt is translated with Amazon Translate before inference.
    """
    _validate_generation_inputs(
        prompt=prompt,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        low=low,
        high=high,
        pipe=pipe,
        image_size=image_size,
    )

    image = _load_image(img_path)
    control_source = _align_to_multiple(_resize_image(image, image_size), multiple=8)
    canny_image = _create_canny_image(control_source, low=low, high=high)

    translated_prompt = (
        translate_language(prompt, trans_info) if trans_info is not None else prompt.strip()
    )
    input_prompt = _join_prompts(default_prompt, translated_prompt)

    result = pipe(
        input_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=_seeded_generator(seed),
        image=canny_image,
    )
    images = getattr(result, "images", None)
    if not images:
        raise RuntimeError("The diffusion pipeline returned no images.")

    output_image = images[0]
    if not isinstance(output_image, Image.Image):
        raise TypeError("The diffusion pipeline must return PIL images.")

    return image, canny_image, output_image.resize(image.size, Image.Resampling.LANCZOS)


def correct_image_orientation(image: Image.Image) -> Image.Image:
    """Apply an image's EXIF orientation without changing its pixel content otherwise."""
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image instance")
    return ImageOps.exif_transpose(image)


def resize_image(image_path: ImageSource, pixels: int) -> Image.Image:
    """Resize an image so its longest side is ``pixels`` while preserving its ratio."""
    if not isinstance(pixels, int) or isinstance(pixels, bool) or pixels <= 0:
        raise ValueError("pixels must be a positive integer")
    return _resize_image(_load_image(image_path), pixels)


def translate_language(
    text: str,
    trans_info: Mapping[str, Any],
    *,
    client: Any = None,
) -> str:
    """Translate ``text`` with Amazon Translate.

    ``trans_info`` accepts the original AIsketcher keys. Explicit AWS keys remain
    supported, but omitting them uses boto3's standard credential provider chain.
    A preconfigured ``client`` can be injected for tests or custom AWS sessions.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    if not isinstance(trans_info, Mapping):
        raise TypeError("trans_info must be a mapping")

    source_language = _clean_string(trans_info.get("SourceLanguageCode"), default="auto")
    target_language = _clean_string(trans_info.get("TargetLanguageCode"))
    if not target_language:
        raise ValueError("trans_info must include TargetLanguageCode")

    translate_client = client or _create_translate_client(trans_info)
    response = translate_client.translate_text(
        Text=text.strip(),
        SourceLanguageCode=source_language or "auto",
        TargetLanguageCode=target_language,
    )
    translated_text = response.get("TranslatedText")
    if not isinstance(translated_text, str) or not translated_text.strip():
        raise RuntimeError("Amazon Translate returned an empty translation.")
    return translated_text.strip()


def _create_translate_client(trans_info: Mapping[str, Any]) -> Any:
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - covered by package dependencies
        raise ImportError("Amazon Translate support requires boto3.") from exc

    client_kwargs: dict[str, Any] = {
        "service_name": "translate",
        "use_ssl": True,
    }
    region_name = _clean_string(trans_info.get("region_name"))
    if region_name:
        client_kwargs["region_name"] = region_name

    use_iam_credentials = bool(trans_info.get("iam_access"))
    access_key = _clean_string(trans_info.get("aws_access_key_id"))
    secret_key = _clean_string(trans_info.get("aws_secret_access_key"))
    session_token = _clean_string(trans_info.get("aws_session_token"))
    if not use_iam_credentials and bool(access_key) != bool(secret_key):
        raise ValueError(
            "aws_access_key_id and aws_secret_access_key must be provided together"
        )
    if not use_iam_credentials and access_key and secret_key:
        client_kwargs["aws_access_key_id"] = access_key
        client_kwargs["aws_secret_access_key"] = secret_key
        if session_token:
            client_kwargs["aws_session_token"] = session_token

    return boto3.client(**client_kwargs)


def _load_image(source: ImageSource) -> Image.Image:
    if isinstance(source, Image.Image):
        image = source.copy()
    elif isinstance(source, (str, os.PathLike)):
        location = os.fspath(source)
        scheme = urlparse(location).scheme.lower()
        if scheme in {"http", "https"}:
            try:
                from diffusers.utils import load_image
            except ImportError as exc:  # pragma: no cover - covered by package dependencies
                raise ImportError("URL image loading requires diffusers.") from exc
            image = load_image(location)
        else:
            with Image.open(location) as opened_image:
                image = opened_image.copy()
    else:
        raise TypeError("img_path must be a path, URL, or PIL.Image.Image")

    return correct_image_orientation(image).convert("RGB")


def _resize_image(image: Image.Image, pixels: int) -> Image.Image:
    scale = pixels / max(image.size)
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _align_to_multiple(image: Image.Image, multiple: int) -> Image.Image:
    width = max(multiple, image.width - (image.width % multiple))
    height = max(multiple, image.height - (image.height % multiple))
    if image.size == (width, height):
        return image
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _create_canny_image(image: Image.Image, low: int, high: int) -> Image.Image:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover - covered by package dependencies
        message = "Canny edge detection requires numpy and opencv-python-headless."
        raise ImportError(message) from exc

    rgb_image = np.asarray(image)
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_image, low, high)
    return Image.fromarray(np.repeat(edges[:, :, None], 3, axis=2))


def _seeded_generator(seed: int) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - covered by package dependencies
        raise ImportError("Image generation requires torch.") from exc
    return torch.Generator(device="cpu").manual_seed(seed)


def _join_prompts(prefix: str, prompt: str) -> str:
    clean_prefix = prefix.strip().rstrip(",")
    clean_prompt = prompt.strip().lstrip(",")
    return ", ".join(part for part in (clean_prefix, clean_prompt) if part)


def _clean_string(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def _validate_generation_inputs(
    *,
    prompt: str,
    num_steps: int,
    guidance_scale: float,
    seed: int,
    low: int,
    high: int,
    pipe: Any,
    image_size: int,
) -> None:
    if pipe is None or not callable(pipe):
        raise ValueError("pipe must be a callable Diffusers ControlNet pipeline")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if not isinstance(num_steps, int) or isinstance(num_steps, bool) or num_steps <= 0:
        raise ValueError("num_steps must be a positive integer")
    if (
        not isinstance(guidance_scale, (int, float))
        or isinstance(guidance_scale, bool)
        or guidance_scale < 0
    ):
        raise ValueError("guidance_scale must be zero or greater")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer between 0 and 2**63 - 1")
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in (low, high)):
        raise ValueError("low and high must be integers")
    if not 0 <= low < high <= 255:
        raise ValueError("Canny thresholds must satisfy 0 <= low < high <= 255")
    if not isinstance(image_size, int) or isinstance(image_size, bool) or image_size < 8:
        raise ValueError("image_size must be an integer of at least 8 pixels")
