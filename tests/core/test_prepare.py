from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from aisketcher import CannyConfig, ValidationError, prepare


def test_prepare_normalizes_transparency_and_latent_size() -> None:
    image = Image.new("RGBA", (321, 167), (0, 0, 0, 0))
    ImageDraw.Draw(image).line((10, 10, 300, 150), fill=(0, 0, 0, 255), width=4)
    prepared = prepare(image, max_side=256)
    assert prepared.image.mode == "RGB"
    assert prepared.control.mode == "RGB"
    assert max(prepared.prepared_size) == 256
    assert prepared.prepared_size[0] % 8 == 0
    assert prepared.prepared_size[1] % 8 == 0
    assert len(prepared.source_sha256) == 64
    assert len(prepared.control_sha256) == 64
    assert prepared.image.getpixel((250, 80)) == (255, 255, 255)


def test_prepare_applies_exif_orientation(tmp_path: Path) -> None:
    path = tmp_path / "private-original-name.jpg"
    image = Image.new("RGB", (40, 80), "white")
    exif = Image.Exif()
    exif[274] = 6
    image.save(path, exif=exif)
    prepared = prepare(path, max_side=80, upscale=False)
    assert prepared.original_size == (80, 40)
    assert prepared.prepared_size == (80, 64)
    assert not hasattr(prepared, "source_name")


def test_diagnostics_flag_blank_low_contrast_sketch() -> None:
    prepared = prepare(Image.new("L", (128, 128), 245), max_side=128)
    assert prepared.diagnostics.low_contrast
    assert prepared.diagnostics.edge_sparse
    assert prepared.diagnostics.edge_density == 0
    assert prepared.diagnostics.recommended_canny.low < prepared.diagnostics.recommended_canny.high


def test_diagnostics_detect_border_crop_risk() -> None:
    image = Image.new("RGB", (128, 128), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 127, 127), outline="black", width=5)
    prepared = prepare(image, max_side=128, canny=CannyConfig(50, 100))
    assert prepared.diagnostics.border_edge_ratio > 0
    assert prepared.diagnostics.crop_risk


def test_control_is_deterministic() -> None:
    array = np.full((96, 144, 3), 255, dtype=np.uint8)
    array[20:75, 20:125] = 10
    image = Image.fromarray(array)
    first = prepare(image, max_side=144, upscale=False)
    second = prepare(image, max_side=144, upscale=False)
    assert first.control_sha256 == second.control_sha256
    assert first.diagnostics.to_dict() == second.diagnostics.to_dict()


def test_recommended_thresholds_follow_faint_pencil_gradients() -> None:
    image = Image.new("RGB", (128, 128), "white")
    ImageDraw.Draw(image).line((8, 110, 64, 15, 120, 110), fill=(225, 225, 225), width=2)
    recommendation = prepare(image, max_side=128).diagnostics.recommended_canny
    assert recommendation.low < 100
    assert recommendation.high < 200


def test_default_prepare_applies_its_adaptive_recommendation() -> None:
    image = Image.new("RGB", (128, 128), "white")
    ImageDraw.Draw(image).line((8, 110, 64, 15, 120, 110), fill=(225, 225, 225), width=2)
    prepared = prepare(image, max_side=128)
    assert prepared.canny == prepared.diagnostics.recommended_canny


def test_extreme_aspect_ratio_is_padded_to_model_safe_minimum() -> None:
    image = Image.new("RGB", (512, 8), "black")
    prepared = prepare(image, max_side=512, upscale=False)
    assert prepared.prepared_size == (512, 64)
    assert prepared.image.getpixel((10, 0)) == (255, 255, 255)
    assert prepared.image.getpixel((10, 32)) == (0, 0, 0)


def test_prepare_rejects_images_over_the_configured_pixel_limit() -> None:
    with pytest.raises(ValidationError, match="the limit is 100"):
        prepare(Image.new("RGB", (11, 10), "white"), max_side=128, max_pixels=100)


def test_canonical_pencil_scan_suppresses_paper_texture() -> None:
    source = Path(__file__).parents[2] / "docs/assets/pocket-kingdom/source.png"
    assert source.is_file(), "canonical source fixture is required"
    prepared = prepare(source, max_side=1024, upscale=False)
    assert 60 <= prepared.canny.low <= 100
    assert 150 <= prepared.canny.high <= 220
    assert 0.035 <= prepared.diagnostics.edge_density <= 0.06
    assert prepared.diagnostics.component_count < 500
