from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest
from PIL import Image

import aisketcher.presets as presets_module
from aisketcher import (
    BackendCapabilities,
    FakeBackend,
    Intent,
    PresetManager,
    Recipe,
    Studio,
)
from aisketcher.errors import ModelUnavailableError
from aisketcher.flux2_backend import Flux2KleinBackend
from aisketcher.model_registry import MODEL_ARTIFACTS, VerifiedFile
from aisketcher.presets import (
    FLUX2_KLEIN_BASE,
    FLUX2_SMALL_DECODER,
    get_preset,
    resolve_recipe,
)

_FIXTURE_CONTENT = b"fixture"
_FIXTURE_SHA256 = hashlib.sha256(_FIXTURE_CONTENT).hexdigest()


@pytest.fixture(autouse=True)
def _use_tiny_verified_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    tiny_artifacts = {}
    for key, artifact in MODEL_ARTIFACTS.items():
        tiny_artifacts[key] = replace(
            artifact,
            files=tuple(
                VerifiedFile(
                    path=required.path,
                    size_bytes=len(_FIXTURE_CONTENT),
                    sha256=_FIXTURE_SHA256,
                )
                for required in artifact.files
            ),
        )
    monkeypatch.setattr(presets_module, "MODEL_ARTIFACTS", tiny_artifacts)


def test_flux2_edit_preset_is_pinned_and_uses_validated_t4_defaults() -> None:
    preset = get_preset("flux2-klein-edit@1")

    assert preset.models == (FLUX2_KLEIN_BASE, FLUX2_SMALL_DECODER)
    assert tuple(model.role for model in preset.models) == ("base-edit", "decoder")
    assert all(len(model.revision) == 40 for model in preset.models)
    assert preset.steps == 4
    assert preset.guidance_scale == 1.0
    assert preset.scheduler == "flow-match-euler"
    assert preset.negative_prompt == ""
    assert preset.required_control == "reference-image"
    assert preset.max_dimension == 1024
    assert preset.estimated_bytes == 16_229_653_713


def test_flux2_recipe_requires_reference_image_not_canny() -> None:
    capabilities = BackendCapabilities(
        controls=("reference-image",),
        supports_negative_prompt=False,
        schedulers=("flow-match-euler",),
    )

    resolved = resolve_recipe(
        "flux2-klein-edit@1",
        Intent(
            "원본의 구도를 유지한 종이 공예",
            model_prompt="A paper craft that preserves the source composition",
            prompt_metadata={"translation": "local"},
        ),
        None,
        backend_name="flux2-klein",
        capabilities=capabilities,
    )

    assert resolved.capability_report.supported
    assert resolved.prompt == "원본의 구도를 유지한 종이 공예"
    assert resolved.model_prompt == "A paper craft that preserves the source composition"
    assert resolved.prompt_metadata == {"translation": "local"}
    assert resolved.steps == 4
    assert resolved.guidance_scale == 1.0
    assert resolved.scheduler == "flow-match-euler"

    canny_only = resolve_recipe(
        "flux2-klein-edit@1",
        Intent("paper craft"),
        None,
        backend_name="legacy",
        capabilities=BackendCapabilities(),
    )
    assert not canny_only.capability_report.supported
    assert canny_only.capability_report.errors[0].requested == "reference-image"


def test_flux2_recipe_rejects_dimensions_above_profile_maximum() -> None:
    resolved = resolve_recipe(
        "flux2-klein-edit@1",
        Intent("paper craft"),
        Recipe(width=1536, height=1024),
        backend_name="flux2-klein",
        capabilities=BackendCapabilities(
            controls=("reference-image",),
            supports_negative_prompt=False,
            schedulers=("flow-match-euler",),
        ),
    )

    assert not resolved.capability_report.supported
    issue = next(
        issue
        for issue in resolved.capability_report.errors
        if issue.setting == "dimensions"
    )
    assert issue.requested == "1536x1024"
    assert issue.applied is None


def test_flux2_install_plan_uses_role_scoped_safe_files(tmp_path: Path) -> None:
    manager = PresetManager(tmp_path, allow_downloads=False)

    plan = manager.plan_install("flux2-klein-edit@1")

    assert plan.estimated_bytes == 16_229_653_713
    assert plan.download_bytes == plan.estimated_bytes
    assert tuple(item.role for item in plan.items) == ("base-edit", "decoder")
    assert tuple(item.estimated_bytes for item in plan.items) == (
        15_980_131_531,
        249_522_182,
    )
    base_patterns, decoder_patterns = (
        plan.items[0].allow_patterns,
        plan.items[1].allow_patterns,
    )
    assert "transformer/diffusion_pytorch_model.safetensors" in base_patterns
    assert "text_encoder/model-00001-of-00002.safetensors" in base_patterns
    assert "text_encoder/model-00002-of-00002.safetensors" in base_patterns
    assert "*.safetensors" not in base_patterns
    assert decoder_patterns == (
        "config.json",
        "diffusion_pytorch_model.safetensors",
    )
    assert not (tmp_path / "models").exists()


def test_flux2_install_uses_safe_doubles_and_role_specific_markers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        destination = Path(str(kwargs["local_dir"]))
        for pattern in kwargs["allow_patterns"]:
            assert isinstance(pattern, str)
            relative = pattern.replace("*", "fixture")
            path = destination / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
        return str(destination)

    fake_hub = ModuleType("huggingface_hub")
    fake_hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    manager = PresetManager(tmp_path)

    result = manager.install("flux2-klein-edit@1", confirm=True)

    assert result.downloaded == (
        FLUX2_KLEIN_BASE.repo_id,
        FLUX2_SMALL_DECODER.repo_id,
    )
    expected_groups = [
        [pattern]
        for item in manager.plan_install("flux2-klein-edit@1").items
        for pattern in item.allow_patterns
    ]
    assert [call["allow_patterns"] for call in calls] == expected_groups
    assert all("*.py" in call["ignore_patterns"] for call in calls)
    plan = manager.plan_install("flux2-klein-edit@1")
    assert plan.installed
    for item in plan.items:
        marker = json.loads(
            (item.destination / ".aisketcher-model.json").read_text(encoding="utf-8")
        )
        assert marker["download_policy"] == "safetensors-components-v1"
        assert marker["safe_tensors_only"] is True
        assert marker["trust_remote_code"] is False
        assert marker["schema"] == "aisketcher-model-cache"
        assert marker["schema_version"] == 2
        assert marker["policy_version"] == 2
        assert marker["hash_policy"] == "pinned-commit-and-runtime-sha256-v2"
        assert marker["artifact_fingerprint"][0]["sha256"] == _FIXTURE_SHA256


def test_studio_from_flux2_preset_constructs_exact_lazy_backend(tmp_path: Path) -> None:
    manager = PresetManager(tmp_path, allow_downloads=False)

    studio = Studio.from_preset(
        "flux2-klein-edit@1",
        device="cuda",
        preset_manager=manager,
        local_files_only=True,
    )

    assert isinstance(studio.backend, Flux2KleinBackend)
    assert studio.preset == "flux2-klein-edit@1"
    assert studio.backend.model.repo_id == FLUX2_KLEIN_BASE.repo_id
    assert studio.backend.model.revision == FLUX2_KLEIN_BASE.revision
    assert studio.backend.model.decoder_model_id == FLUX2_SMALL_DECODER.repo_id
    assert studio.backend.model.decoder_revision == FLUX2_SMALL_DECODER.revision
    assert studio.backend.model.cache_dir == tmp_path
    plan = manager.plan_install("flux2-klein-edit@1")
    assert studio.backend.model.base_path == plan.items[0].destination
    assert studio.backend.model.decoder_path == plan.items[1].destination
    assert studio.backend.settings.num_inference_steps == 4
    assert studio.backend.settings.guidance_scale == 1.0
    assert studio.backend.local_files_only is True


def test_flux2_generation_rejects_an_unmarked_managed_cache(tmp_path: Path) -> None:
    manager = PresetManager(tmp_path, allow_downloads=False)
    studio = Studio.from_preset(
        "flux2-klein-edit@1",
        device="cuda",
        preset_manager=manager,
        local_files_only=True,
    )
    prepared = studio.prepare(Image.new("RGB", (64, 64), "white"), max_side=64)

    with pytest.raises(ModelUnavailableError, match="not installed"):
        studio.explore(
            prepared,
            intent=Intent("A small paper kingdom"),
            outputs=1,
        )


def test_flux2_generation_revalidates_a_marked_but_incomplete_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_snapshot_download(**kwargs: object) -> str:
        destination = Path(str(kwargs["local_dir"]))
        for pattern in kwargs["allow_patterns"]:
            assert isinstance(pattern, str)
            path = destination / pattern.replace("*", "fixture")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
        return str(destination)

    fake_hub = ModuleType("huggingface_hub")
    fake_hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    manager = PresetManager(tmp_path)
    manager.install("flux2-klein-edit@1", confirm=True)
    installed = manager.plan_install("flux2-klein-edit@1")
    assert installed.installed
    (
        installed.items[0].destination
        / "transformer"
        / "diffusion_pytorch_model.safetensors"
    ).unlink()

    studio = Studio.from_preset(
        "flux2-klein-edit@1",
        device="cuda",
        preset_manager=manager,
        local_files_only=True,
    )
    prepared = studio.prepare(Image.new("RGB", (64, 64), "white"), max_side=64)

    with pytest.raises(ModelUnavailableError, match="not installed"):
        studio.explore(
            prepared,
            intent=Intent("A small paper kingdom"),
            outputs=1,
        )


def test_studio_from_flux2_preset_preserves_an_injected_backend() -> None:
    backend = FakeBackend()

    studio = Studio.from_preset("flux2-klein-edit@1", backend=backend)

    assert studio.backend is backend
    assert studio.preset == "flux2-klein-edit@1"


def test_studio_from_preset_defaults_to_the_validated_flux_profile() -> None:
    studio = Studio.from_preset(backend=FakeBackend())

    assert studio.preset == "flux2-klein-edit@1"
