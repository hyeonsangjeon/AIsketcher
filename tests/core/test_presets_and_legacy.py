from __future__ import annotations

import importlib
import json
import sys
import warnings
from pathlib import Path
from types import ModuleType

import pytest
from PIL import Image

from aisketcher import (
    DiffusersBackend,
    ModelUnavailableError,
    PresetManager,
    RemovedFeatureError,
    ValidationError,
)
from aisketcher.modelPipe import img2img, resize_image
from aisketcher.presets import SDXL_BASE, get_preset


def materialize_allowed_files(destination: Path, patterns: tuple[str, ...]) -> None:
    for pattern in patterns:
        relative = pattern.replace("*.json", "fixture.json").replace("*", "fixture.json")
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")


def test_preset_revisions_are_immutable_commit_hashes() -> None:
    for name in ("sdxl-canny-lite@1", "sdxl-canny@1"):
        preset = get_preset(name)
        assert preset.name == name
        assert all(len(model.revision) == 40 for model in preset.models)
        assert all(set(model.revision) <= set("0123456789abcdef") for model in preset.models)
    assert get_preset("quality").name == "sdxl-canny@1"


def test_install_requires_displayed_confirmation_and_respects_disable(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    manager = PresetManager(cache, allow_downloads=False)
    plan = manager.plan_install("sdxl-canny-lite@1")
    assert not plan.installed
    assert plan.estimated_bytes == 7_261_425_974
    assert plan.download_bytes == plan.estimated_bytes
    assert plan.cached_bytes == 0
    assert plan.items[0].allow_patterns == (
        "model_index.json",
        "scheduler/*.json",
        "text_encoder/config.json",
        "text_encoder/model.fp16.safetensors",
        "text_encoder_2/config.json",
        "text_encoder_2/model.fp16.safetensors",
        "tokenizer/*",
        "tokenizer_2/*",
        "unet/config.json",
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.fp16.safetensors",
    )
    assert plan.items[1].allow_patterns == (
        "config.json",
        "diffusion_pytorch_model.fp16.safetensors",
    )
    assert not cache.exists()
    with pytest.raises(ValidationError, match="confirm=True"):
        manager.install(plan.preset)
    with pytest.raises(ModelUnavailableError, match="disabled"):
        manager.install(plan.preset, confirm=True)


def test_preset_manager_recognizes_only_matching_marker(tmp_path: Path) -> None:
    manager = PresetManager(tmp_path)
    destination = manager._destination(SDXL_BASE)
    destination.mkdir(parents=True)
    marker = destination / ".aisketcher-model.json"
    marker.write_text(
        json.dumps({"repo_id": SDXL_BASE.repo_id, "revision": "wrong"}), encoding="utf-8"
    )
    assert manager.model_path(SDXL_BASE) is None
    marker.write_text(
        json.dumps(
            {
                "repo_id": SDXL_BASE.repo_id,
                "revision": SDXL_BASE.revision,
                "download_policy": "fp16-components-v1",
                "allow_patterns": list(manager.plan_install("sdxl-canny-lite@1").items[0].allow_patterns),
                "safe_tensors_only": True,
            }
        ),
        encoding="utf-8",
    )
    patterns = manager.plan_install("sdxl-canny-lite@1").items[0].allow_patterns
    materialize_allowed_files(destination, patterns)
    assert manager.model_path(SDXL_BASE) == destination
    (destination / "unet/diffusion_pytorch_model.fp16.safetensors").unlink()
    assert manager.model_path(SDXL_BASE) is None
    materialize_allowed_files(destination, patterns)
    (destination / "legacy.bin").write_bytes(b"unsafe")
    assert manager.model_path(SDXL_BASE) is None


def test_install_uses_only_curated_fp16_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []

    def fake_snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        materialize_allowed_files(
            Path(str(kwargs["local_dir"])), tuple(kwargs["allow_patterns"])
        )
        return str(kwargs["local_dir"])

    fake_hub = ModuleType("huggingface_hub")
    fake_hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    manager = PresetManager(tmp_path)
    result = manager.install("sdxl-canny-lite@1", confirm=True)
    assert result.downloaded == (
        "stabilityai/stable-diffusion-xl-base-1.0",
        "diffusers/controlnet-canny-sdxl-1.0-small",
    )
    assert len(calls) == 2
    assert "*.safetensors" not in calls[0]["allow_patterns"]
    assert "unet/diffusion_pytorch_model.fp16.safetensors" in calls[0][
        "allow_patterns"
    ]
    assert calls[1]["allow_patterns"] == [
        "config.json",
        "diffusion_pytorch_model.fp16.safetensors",
    ]
    assert "*.py" in calls[0]["ignore_patterns"]
    installed_plan = manager.plan_install("sdxl-canny-lite@1")
    assert installed_plan.installed
    assert installed_plan.download_bytes == 0
    assert installed_plan.cached_bytes == installed_plan.estimated_bytes


def test_constructing_diffusers_backend_is_dependency_lazy() -> None:
    before = set(sys.modules)
    backend = DiffusersBackend(device="auto")
    after = set(sys.modules)
    assert backend.name == "diffusers"
    assert not ({"torch", "diffusers"} & (after - before))


def test_download_policy_cannot_override_disabled_manager(tmp_path: Path) -> None:
    manager = PresetManager(tmp_path, allow_downloads=False)
    with pytest.raises(ValidationError, match="conflicts"):
        DiffusersBackend(preset_manager=manager, local_files_only=False)


def test_legacy_resize_warns_and_preserves_aspect_ratio(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    Image.new("RGB", (160, 80), "white").save(path)
    with pytest.warns(DeprecationWarning, match="removed in 0.3.0"):
        result = resize_image(path, 80)
    assert result.size == (80, 40)


def test_legacy_aws_translation_argument_is_explicitly_removed(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    Image.new("RGB", (80, 80), "white").save(path)
    with (
        pytest.warns(DeprecationWarning),
        pytest.raises(RemovedFeatureError, match="AWS translation"),
    ):
        img2img(path, "castle", pipe=object(), trans_info={"region": "legacy"})


def test_uppercase_facade_warns() -> None:
    sys.modules.pop("AIsketcher", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.import_module("AIsketcher")
    assert module.img2img
    assert module.modelPipe.resize_image
    assert any("uppercase facade" in str(item.message) for item in caught)
