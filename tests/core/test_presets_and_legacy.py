from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from types import ModuleType

import pytest

from aisketcher import (
    DiffusersBackend,
    ModelUnavailableError,
    PresetManager,
    ValidationError,
)
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
    assert get_preset("auto").name == "flux2-klein-edit@1"
    assert get_preset("flux").name == "flux2-klein-edit@1"


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
    plan = manager.plan_install("sdxl-canny-lite@1")
    expected_groups = [
        [pattern]
        for item in plan.items
        for pattern in item.allow_patterns
    ]
    assert [call["allow_patterns"] for call in calls] == expected_groups
    assert all("*.safetensors" not in call["allow_patterns"] for call in calls)
    assert ["unet/diffusion_pytorch_model.fp16.safetensors"] in expected_groups
    assert expected_groups[-2:] == [
        ["config.json"],
        ["diffusion_pytorch_model.fp16.safetensors"],
    ]
    assert all("*.py" in call["ignore_patterns"] for call in calls)
    installed_plan = manager.plan_install("sdxl-canny-lite@1")
    assert installed_plan.installed
    assert installed_plan.download_bytes == 0
    assert installed_plan.cached_bytes == installed_plan.estimated_bytes


def test_install_honours_pre_cancel_without_creating_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []

    def fake_snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(kwargs["local_dir"])

    fake_hub = ModuleType("huggingface_hub")
    fake_hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    token = threading.Event()
    token.set()
    manager = PresetManager(tmp_path)

    with pytest.raises(ModelUnavailableError, match="cancelled safely"):
        manager.install(
            "sdxl-canny-lite@1",
            confirm=True,
            cancellation_token=token,
        )

    assert calls == []
    assert not (tmp_path / "models").exists()


def test_install_cancel_after_first_file_group_skips_later_calls_and_cleans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []
    cancelled = threading.Event()

    def fake_snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        materialize_allowed_files(
            Path(str(kwargs["local_dir"])), tuple(kwargs["allow_patterns"])
        )
        cancelled.set()
        return str(kwargs["local_dir"])

    fake_hub = ModuleType("huggingface_hub")
    fake_hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    manager = PresetManager(tmp_path)
    plan = manager.plan_install("sdxl-canny-lite@1")

    with pytest.raises(ModelUnavailableError, match="cancelled safely"):
        manager.install(
            plan.preset,
            confirm=True,
            cancellation_token=cancelled,
        )

    assert len(calls) == 1
    assert calls[0]["repo_id"] == plan.items[0].repo_id
    assert calls[0]["allow_patterns"] == ["model_index.json"]
    assert not plan.items[0].destination.exists()
    assert not plan.items[1].destination.exists()


def test_install_cancel_between_artifacts_keeps_completed_item_and_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    checks = 0

    def fake_snapshot_download(**kwargs: object) -> str:
        repo_id = str(kwargs["repo_id"])
        calls.append(repo_id)
        materialize_allowed_files(
            Path(str(kwargs["local_dir"])), tuple(kwargs["allow_patterns"])
        )
        return str(kwargs["local_dir"])

    def should_cancel() -> bool:
        nonlocal checks
        checks += 1
        first_item_boundaries = 2 + (2 * len(plan.items[0].allow_patterns))
        return checks >= first_item_boundaries

    fake_hub = ModuleType("huggingface_hub")
    fake_hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    manager = PresetManager(tmp_path)
    plan = manager.plan_install("sdxl-canny-lite@1")

    with pytest.raises(ModelUnavailableError, match="cancelled safely"):
        manager.install(
            plan.preset,
            confirm=True,
            should_cancel=should_cancel,
        )

    assert manager.model_path(get_preset(plan.preset).models[0]) is not None
    assert not plan.items[1].destination.exists()
    result = manager.install(plan.preset, confirm=True)
    assert result.downloaded == ("diffusers/controlnet-canny-sdxl-1.0-small",)
    assert manager.plan_install(plan.preset).installed


def test_failed_snapshot_removes_only_incomplete_current_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_snapshot_download(**kwargs: object) -> str:
        destination = Path(str(kwargs["local_dir"]))
        materialize_allowed_files(destination, tuple(kwargs["allow_patterns"]))
        if kwargs["repo_id"] == "diffusers/controlnet-canny-sdxl-1.0-small":
            raise OSError("network interrupted")
        return str(destination)

    fake_hub = ModuleType("huggingface_hub")
    fake_hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    manager = PresetManager(tmp_path)
    preset = get_preset("sdxl-canny-lite@1")

    with pytest.raises(OSError, match="network interrupted"):
        manager.install(preset.name, confirm=True)

    assert manager.model_path(preset.models[0]) is not None
    assert manager.model_path(preset.models[1]) is None
    assert not manager.model_destination(preset.models[1]).exists()


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
