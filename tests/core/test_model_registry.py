from __future__ import annotations

import re
from dataclasses import replace

import pytest

from aisketcher import ValidationError
from aisketcher.model_registry import (
    CURATED_MODEL_PROFILES,
    MODEL_ARTIFACTS,
    ControlType,
    HashPolicy,
    InputMode,
    ModelStatus,
    RuntimeFamily,
    get_model_profile,
    zero_click_profiles,
)


def test_registry_contains_the_2026_refresh_profiles() -> None:
    assert tuple(CURATED_MODEL_PROFILES) == (
        "auto",
        "flux2-klein-4b",
        "z-image-turbo-union-lite",
        "qwen-image-edit-quality",
        "sdxl-canny-legacy",
        "mage-flow-edit-turbo-experimental",
    )

    auto = get_model_profile("auto")
    assert auto.runtime_family is RuntimeFamily.AUTO_ROUTER
    assert auto.status is ModelStatus.READY
    assert not auto.zero_click_enabled
    assert tuple((route.input_kind, route.profile_id) for route in auto.auto_routes) == (
        ("sparse-sketch-or-line-art", "flux2-klein-4b"),
        ("photo-or-semantic-edit", "flux2-klein-4b"),
    )
    assert auto.tested_devices == (
        "NVIDIA Tesla T4 16 GB / Azure Standard_NC4as_T4_v3",
    )


def test_every_artifact_has_immutable_revision_and_verified_payloads() -> None:
    for key, artifact in MODEL_ARTIFACTS.items():
        assert key == artifact.artifact_id
        assert re.fullmatch(r"[0-9a-f]{40}", artifact.revision)
        assert artifact.hash_policy is HashPolicy.PINNED_COMMIT_AND_LFS_SHA256
        assert artifact.download_bytes == sum(item.size_bytes for item in artifact.files)
        assert all(re.fullmatch(r"[0-9a-f]{64}", item.sha256) for item in artifact.files)
        assert all(item.path.endswith((".safetensors", "tokenizer.json")) for item in artifact.files)
        assert artifact.model_card_url.endswith(artifact.revision)


def test_zero_click_policy_is_stricter_than_general_registry_inclusion() -> None:
    profiles = zero_click_profiles()
    assert tuple(profile.profile_id for profile in profiles) == ("flux2-klein-4b",)

    for profile in profiles:
        assert profile.license_ids == ("apache-2.0",)
        assert not profile.gated
        assert not profile.territory_exclusions
        assert all(artifact.public for artifact in profile.artifacts)
        assert all(artifact.commercial_use for artifact in profile.artifacts)
        assert all(not artifact.gated for artifact in profile.artifacts)
        assert all(artifact.files for artifact in profile.artifacts)

    assert not get_model_profile("sdxl-canny-legacy").zero_click_enabled
    assert not get_model_profile("z-image-turbo-union-lite").zero_click_enabled
    assert not get_model_profile("qwen-image-edit-quality").zero_click_enabled
    assert not get_model_profile(
        "mage-flow-edit-turbo-experimental"
    ).zero_click_enabled

    with pytest.raises(ValidationError, match="Zero-click profiles"):
        replace(get_model_profile("sdxl-canny-legacy"), zero_click_enabled=True)


def test_structural_profile_pins_the_reviewed_2602_lite_adapter() -> None:
    profile = get_model_profile("z-image-turbo-union-lite")
    assert profile.model_ids == (
        "Tongyi-MAI/Z-Image-Turbo",
        "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1",
    )
    assert profile.download_bytes == 34_860_389_932
    assert {
        ControlType.CANNY,
        ControlType.HED,
        ControlType.SCRIBBLE,
        ControlType.DEPTH,
        ControlType.POSE,
        ControlType.TILE,
        ControlType.INPAINT,
    } == set(profile.control_types)

    adapter = profile.artifacts[1]
    assert adapter.revision == "5155fc56d17821007d6f62ac192c09e0f0e72016"
    assert len(adapter.files) == 1
    assert (
        adapter.files[0].path
        == "Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2602-8steps.safetensors"
    )
    assert (
        adapter.files[0].sha256
        == "3ea098db9bd145be525c7e2366920b6d76c5ffd46b3d7aa8169bbc943fdaee35"
    )


def test_profiles_expose_hardware_and_download_expectations() -> None:
    flux = get_model_profile("flux2-klein-4b")
    assert flux.download_bytes == 16_225_156_608
    assert (flux.minimum_vram_gb, flux.recommended_vram_gb) == (13, 16)
    assert flux.status is ModelStatus.READY
    assert InputMode.SKETCH in flux.input_modes
    assert InputMode.MULTI_REFERENCE not in flux.input_modes
    assert flux.control_types == (ControlType.IMAGE_EDIT,)
    assert flux.model_ids == (
        "black-forest-labs/FLUX.2-klein-4B",
        "black-forest-labs/FLUX.2-small-decoder",
    )
    assert flux.tested_devices == (
        "NVIDIA Tesla T4 16 GB / Azure Standard_NC4as_T4_v3",
    )
    decoder = MODEL_ARTIFACTS["flux2-small-decoder"]
    assert decoder.revision == "a3efc24f613ef42d9428af62fdbd6f5fd8856c4a"
    assert decoder.download_bytes == 249_521_340
    assert (
        decoder.files[0].sha256
        == "d8d52ba036475f5fb07c8b435e176d3d97ebfa82f0d1a1c317f9cc1e25bd013b"
    )

    qwen = get_model_profile("qwen-image-edit-quality")
    assert qwen.download_bytes == 57_710_671_694
    assert (qwen.minimum_vram_gb, qwen.recommended_vram_gb) == (40, 80)
    assert qwen.status is ModelStatus.PRO_QUALITY

    legacy = get_model_profile("sdxl-canny-legacy")
    assert legacy.download_bytes == 7_258_248_609
    assert legacy.status is ModelStatus.LEGACY


def test_registry_mappings_are_read_only_and_lookup_errors_are_clear() -> None:
    with pytest.raises(TypeError):
        CURATED_MODEL_PROFILES["other"] = get_model_profile("auto")  # type: ignore[index]
    with pytest.raises(TypeError):
        MODEL_ARTIFACTS["other"] = MODEL_ARTIFACTS["flux2-klein-4b"]  # type: ignore[index]
    with pytest.raises(ValidationError, match="Available profiles"):
        get_model_profile("not-real")
