"""Sketch normalization, Canny controls and transparent technical scoring."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import replace
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .errors import ValidationError
from .models import (
    MAX_GENERATION_DIMENSION,
    MAX_SOURCE_PIXELS,
    Candidate,
    CannyConfig,
    PreparationDiagnostics,
    PreparedSketch,
    TechnicalScores,
)

ImageSource = str | Path | Image.Image


def image_sha256(image: Image.Image) -> str:
    """Hash normalized PNG pixels, not source metadata or a local path."""

    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG", optimize=False)
    return sha256(buffer.getvalue()).hexdigest()


def normalize_image(
    source: ImageSource, *, max_pixels: int = 50_000_000
) -> tuple[Image.Image, tuple[int, int]]:
    """Load an image, apply EXIF orientation and return a metadata-free RGB copy."""

    if not 1 <= max_pixels <= MAX_SOURCE_PIXELS:
        raise ValidationError(f"max_pixels must be 1..{MAX_SOURCE_PIXELS:,}")
    if isinstance(source, Image.Image):
        opened = source
        close_after = False
    else:
        try:
            opened = Image.open(Path(source))
            close_after = True
        except (OSError, ValueError) as exc:
            raise ValidationError("source must be a readable image path or PIL image") from exc

    try:
        oriented = ImageOps.exif_transpose(opened)
        original_size = oriented.size
        if oriented.width * oriented.height > max_pixels:
            raise ValidationError(
                f"source contains {oriented.width * oriented.height:,} pixels; "
                f"the limit is {max_pixels:,}"
            )
        if oriented.mode in ("RGBA", "LA") or "transparency" in oriented.info:
            rgba = oriented.convert("RGBA")
            flattened = Image.alpha_composite(
                Image.new("RGBA", rgba.size, "white"), rgba
            ).convert("RGB")
        else:
            flattened = oriented.convert("RGB")
        # Round-trip through a new pixel buffer to avoid carrying EXIF/ICC/info.
        rgb = Image.fromarray(np.asarray(flattened, dtype=np.uint8).copy(), "RGB")
    finally:
        if close_after:
            opened.close()
    return rgb, original_size


def resize_to_multiple(
    image: Image.Image,
    *,
    max_side: int = 1024,
    multiple: int = 8,
    upscale: bool = True,
) -> Image.Image:
    """Resize proportionally and make both dimensions safe for latent diffusion."""

    if multiple < 1:
        raise ValidationError("multiple must be positive")
    if not 64 <= max_side <= MAX_GENERATION_DIMENSION or max_side % multiple:
        raise ValidationError(
            f"max_side must be 64..{MAX_GENERATION_DIMENSION} and divisible by multiple"
        )

    scale = max_side / max(image.size)
    if not upscale:
        scale = min(1.0, scale)

    def quantize(value: float) -> int:
        units = value / multiple
        rounded = math.floor(units) if not upscale else round(units)
        return max(multiple, rounded * multiple)

    width = quantize(image.width * scale)
    height = quantize(image.height * scale)
    resized = (
        image.copy()
        if (width, height) == image.size
        else image.resize((width, height), Image.Resampling.LANCZOS)
    )
    safe_width = max(64, width)
    safe_height = max(64, height)
    if (safe_width, safe_height) == resized.size:
        return resized
    canvas = Image.new("RGB", (safe_width, safe_height), "white")
    canvas.paste(
        resized,
        ((safe_width - resized.width) // 2, (safe_height - resized.height) // 2),
    )
    return canvas


def make_canny(image: Image.Image, config: CannyConfig) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(
        gray,
        config.low,
        config.high,
        apertureSize=config.aperture_size,
        L2gradient=config.l2_gradient,
    )
    return Image.fromarray(np.repeat(edges[:, :, None], 3, axis=2), "RGB")


def _recommended_canny(gray: np.ndarray) -> CannyConfig:
    # Canny thresholds operate on gradients, so sampling paper brightness would
    # recommend values that are far too high for faint pencil strokes.
    horizontal = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    vertical = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradients = cv2.magnitude(horizontal, vertical)
    active = gradients[gradients > 1.0]
    if active.size == 0:
        return CannyConfig(low=50, high=150)
    # Pencil scans often have a dense low-gradient mode from paper grain and a
    # sharp upper mode from intentional strokes. The 90th percentile lands on
    # that upper mode for the canonical scan, while the clamps keep truly faint
    # drawings usable and prevent saturated ink from forcing 255/255 behavior.
    high = int(np.clip(round(float(np.percentile(active, 90))), 32, 224))
    low = int(np.clip(round(high * 0.4), 12, high - 1))
    return CannyConfig(low=low, high=high)


def diagnose(image: Image.Image, control: Image.Image) -> PreparationDiagnostics:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    edge = np.asarray(control.convert("L"), dtype=np.uint8) > 0
    pixels = edge.size
    edge_pixels = int(edge.sum())
    edge_density = edge_pixels / pixels

    components, _ = cv2.connectedComponents(edge.astype(np.uint8), connectivity=8)
    component_count = max(0, int(components) - 1)
    fragmentation = component_count / max(edge_pixels / 1_000, 1.0)

    # A five-percent band catches strokes that touch/cross the crop boundary;
    # Canny places the detected edge just inside a thick border stroke.
    band = max(1, min(edge.shape) // 20)
    border = np.zeros_like(edge)
    border[:band, :] = True
    border[-band:, :] = True
    border[:, :band] = True
    border[:, -band:] = True
    border_edge_ratio = float((edge & border).sum()) / max(edge_pixels, 1)

    contrast = float(np.std(gray)) / 127.5
    return PreparationDiagnostics(
        contrast=contrast,
        edge_density=edge_density,
        component_count=component_count,
        fragmentation=fragmentation,
        border_edge_ratio=border_edge_ratio,
        low_contrast=contrast < 0.12,
        edge_sparse=edge_density < 0.005,
        edge_dense=edge_density > 0.25,
        crop_risk=border_edge_ratio > 0.15,
        recommended_canny=_recommended_canny(gray),
    )


def prepare(
    source: ImageSource,
    *,
    max_side: int = 1024,
    canny: CannyConfig | None = None,
    upscale: bool = True,
    max_pixels: int = 50_000_000,
) -> PreparedSketch:
    """Prepare an input image and produce model-independent diagnostics."""

    normalized, original_size = normalize_image(source, max_pixels=max_pixels)
    resized = resize_to_multiple(
        normalized, max_side=max_side, multiple=8, upscale=upscale
    )
    if canny is None:
        gray = cv2.cvtColor(np.asarray(resized, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
        config = _recommended_canny(gray)
    else:
        config = canny
    control = make_canny(resized, config)
    return PreparedSketch(
        image=resized,
        control=control,
        original_size=original_size,
        prepared_size=resized.size,
        source_sha256=image_sha256(resized),
        control_sha256=image_sha256(control),
        canny=config,
        diagnostics=diagnose(resized, control),
    )


def _edge_mask(image: Image.Image, size: tuple[int, int], config: CannyConfig) -> np.ndarray:
    fitted = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    return np.asarray(make_canny(fitted, config).convert("L"), dtype=np.uint8) > 0


def score_images(
    control: Image.Image,
    images: Iterable[Image.Image],
    *,
    config: CannyConfig,
) -> list[TechnicalScores]:
    """Score structural adherence, edge cleanliness and visual distinctiveness.

    These measurements are descriptive only. They intentionally avoid claiming
    that one output is aesthetically superior to another.
    """

    image_list = list(images)
    if not image_list:
        return []
    size = control.size
    reference = np.asarray(control.convert("L"), dtype=np.uint8) > 0
    reference_density = float(reference.mean())
    masks = [_edge_mask(image, size, config) for image in image_list]
    rgb = [
        np.asarray(image.convert("RGB").resize(size, Image.Resampling.BILINEAR), dtype=np.float32)
        for image in image_list
    ]

    structures: list[float] = []
    cleanliness: list[float] = []
    distinctiveness: list[float] = []
    for index, mask in enumerate(masks):
        union = float((reference | mask).sum())
        structures.append(float((reference & mask).sum()) / union if union else 1.0)
        density_delta = abs(float(mask.mean()) - reference_density)
        cleanliness.append(max(0.0, 1.0 - density_delta / max(reference_density, 0.01)))
        if len(rgb) == 1:
            distinctiveness.append(0.0)
        else:
            distances = [
                float(np.mean(np.abs(rgb[index] - other))) / 255.0
                for other_index, other in enumerate(rgb)
                if other_index != index
            ]
            distinctiveness.append(float(np.mean(distances)))

    badge_map: list[list[str]] = [[] for _ in image_list]
    badge_map[int(np.argmax(structures))].append("closest structure")
    badge_map[int(np.argmax(cleanliness))].append("cleanest edges")
    if len(image_list) > 1:
        badge_map[int(np.argmax(distinctiveness))].append("most distinct")

    return [
        TechnicalScores(
            structure_similarity=structures[index],
            edge_cleanliness=cleanliness[index],
            distinctiveness=distinctiveness[index],
            badges=tuple(badge_map[index]),
        )
        for index in range(len(image_list))
    ]


def rescore_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    candidate_list = list(candidates)
    if not candidate_list:
        return []
    prepared = candidate_list[0].prepared
    scores = score_images(
        prepared.control,
        [candidate.image for candidate in candidate_list],
        config=prepared.canny,
    )
    return [
        replace(candidate, scores=score)
        for candidate, score in zip(candidate_list, scores, strict=True)
    ]
