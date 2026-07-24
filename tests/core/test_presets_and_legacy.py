from __future__ import annotations

import hashlib
import json
import sys
import threading
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest

import aisketcher.presets as presets_module
from aisketcher import (
    DiffusersBackend,
    ModelUnavailableError,
    PresetManager,
    ValidationError,
)
from aisketcher.model_registry import MODEL_ARTIFACTS, VerifiedFile
from aisketcher.presets import SDXL_BASE, SDXL_CANNY_QUALITY, get_preset

_FIXTURE_CONTENT = b"fixture"
_FIXTURE_SHA256 = hashlib.sha256(_FIXTURE_CONTENT).hexdigest()


@pytest.fixture(autouse=True)
def _use_tiny_verified_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep installer tests tiny while exercising real SHA-256 verification."""

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


def materialize_allowed_files(destination: Path, patterns: tuple[str, ...]) -> None:
    for pattern in patterns:
        relative = pattern.replace("*.json", "fixture.json").replace("*", "fixture.json")
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_FIXTURE_CONTENT)


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
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "text_encoder/model.fp16.safetensors",
        "text_encoder_2/config.json",
        "text_encoder_2/model.fp16.safetensors",
        "tokenizer/merges.txt",
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json",
        "tokenizer_2/merges.txt",
        "tokenizer_2/special_tokens_map.json",
        "tokenizer_2/tokenizer_config.json",
        "tokenizer_2/vocab.json",
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
    upgraded_raw = marker.read_bytes()
    upgraded = json.loads(upgraded_raw)
    assert upgraded["schema"] == "aisketcher-model-cache"
    assert upgraded["schema_version"] == 2
    assert upgraded["policy_version"] == 2
    assert upgraded["hash_policy"] == "pinned-commit-and-runtime-sha256-v2"
    assert upgraded["revision"] == SDXL_BASE.revision
    assert len(upgraded["artifact_fingerprint"]) == 18
    assert {
        item["path"] for item in upgraded["artifact_fingerprint"]
    } == set(patterns)
    assert all(
        item["size_bytes"] == len(_FIXTURE_CONTENT)
        and item["sha256"] == _FIXTURE_SHA256
        for item in upgraded["artifact_fingerprint"]
    )
    assert upgraded_raw == presets_module._canonical_json_bytes(upgraded)
    (destination / "unet/diffusion_pytorch_model.fp16.safetensors").unlink()
    assert manager.model_path(SDXL_BASE) is None
    materialize_allowed_files(destination, patterns)
    (destination / "legacy.bin").write_bytes(b"unsafe")
    assert manager.model_path(SDXL_BASE) is None


def test_matching_legacy_marker_cannot_approve_a_fake_small_payload(
    tmp_path: Path,
) -> None:
    manager = PresetManager(tmp_path)
    destination = manager.model_destination(SDXL_BASE)
    destination.mkdir(parents=True)
    patterns = manager.plan_install("sdxl-canny-lite@1").items[0].allow_patterns
    materialize_allowed_files(destination, patterns)
    (destination / "text_encoder/model.fp16.safetensors").write_bytes(b"x")
    marker = destination / ".aisketcher-model.json"
    marker.write_text(
        json.dumps(
            {
                "repo_id": SDXL_BASE.repo_id,
                "revision": SDXL_BASE.revision,
                "download_policy": "fp16-components-v1",
                "safe_tensors_only": True,
            }
        ),
        encoding="utf-8",
    )

    assert manager.model_path(SDXL_BASE) is None
    assert json.loads(marker.read_text(encoding="utf-8"))["download_policy"] == (
        "fp16-components-v1"
    )


def test_verified_cache_rejects_undeclared_safetensors_runtime_file(
    tmp_path: Path,
) -> None:
    manager = PresetManager(tmp_path)
    destination = manager.model_destination(SDXL_BASE)
    destination.mkdir(parents=True)
    patterns = manager.plan_install("sdxl-canny-lite@1").items[0].allow_patterns
    materialize_allowed_files(destination, patterns)
    (destination / ".aisketcher-model.json").write_text(
        json.dumps(
            {"repo_id": SDXL_BASE.repo_id, "revision": SDXL_BASE.revision}
        ),
        encoding="utf-8",
    )
    assert manager.model_path(SDXL_BASE) == destination

    undeclared_weights = destination / "text_encoder/model.safetensors"
    undeclared_weights.write_bytes(b"unreviewed runtime weights")

    assert manager.model_path(SDXL_BASE) is None


def test_verified_cache_allows_only_expected_hugging_face_bookkeeping(
    tmp_path: Path,
) -> None:
    manager = PresetManager(tmp_path)
    destination = manager.model_destination(SDXL_BASE)
    destination.mkdir(parents=True)
    patterns = manager.plan_install("sdxl-canny-lite@1").items[0].allow_patterns
    materialize_allowed_files(destination, patterns)
    (destination / ".aisketcher-model.json").write_text(
        json.dumps(
            {"repo_id": SDXL_BASE.repo_id, "revision": SDXL_BASE.revision}
        ),
        encoding="utf-8",
    )
    assert manager.model_path(SDXL_BASE) == destination

    hf_cache = destination / ".cache/huggingface"
    hf_cache.mkdir(parents=True)
    (hf_cache / ".gitignore").write_text("*", encoding="utf-8")
    (hf_cache / "CACHEDIR.TAG").write_text(
        "Signature: 8a477f597d28d172789f06886806bc55",
        encoding="utf-8",
    )
    metadata = hf_cache / "download/text_encoder/config.json.metadata"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("commit\netag\n0\n", encoding="utf-8")
    metadata.with_suffix(".lock").touch()
    tree_cache = hf_cache / f"trees/{SDXL_BASE.revision}.json"
    tree_cache.parent.mkdir()
    tree_cache.write_text(
        json.dumps({"revision": SDXL_BASE.revision, "files": list(patterns)}),
        encoding="utf-8",
    )

    assert manager.model_path(SDXL_BASE) == destination

    (tree_cache.parent / f"{'0' * 40}.json").write_text("{}", encoding="utf-8")
    assert manager.model_path(SDXL_BASE) is None


def test_post_verification_tamper_forces_rehash_and_is_rejected(
    tmp_path: Path,
) -> None:
    manager = PresetManager(tmp_path)
    destination = manager.model_destination(SDXL_BASE)
    destination.mkdir(parents=True)
    patterns = manager.plan_install("sdxl-canny-lite@1").items[0].allow_patterns
    materialize_allowed_files(destination, patterns)
    marker = destination / ".aisketcher-model.json"
    marker.write_text(
        json.dumps(
            {"repo_id": SDXL_BASE.repo_id, "revision": SDXL_BASE.revision}
        ),
        encoding="utf-8",
    )
    assert manager.model_path(SDXL_BASE) == destination

    required = destination / "text_encoder/model.fp16.safetensors"
    required.write_bytes(b"tamper!")

    assert manager.model_path(SDXL_BASE) is None


def test_fresh_manager_rejects_tampered_runtime_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_snapshot_download(**kwargs: object) -> str:
        destination = Path(str(kwargs["local_dir"]))
        materialize_allowed_files(destination, tuple(kwargs["allow_patterns"]))
        return str(destination)

    fake_hub = ModuleType("huggingface_hub")
    fake_hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    manager = PresetManager(tmp_path)
    manager.install("sdxl-canny-lite@1", confirm=True)

    destination = manager.model_destination(SDXL_BASE)
    config = destination / "model_index.json"
    original_stat = config.stat()
    config.write_bytes(b"fixturE")
    assert config.stat().st_size == original_stat.st_size
    config.touch()

    assert PresetManager(tmp_path).model_path(SDXL_BASE) is None


def test_unchanged_verified_cache_uses_process_local_stat_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = PresetManager(tmp_path)
    destination = manager.model_destination(SDXL_BASE)
    destination.mkdir(parents=True)
    patterns = manager.plan_install("sdxl-canny-lite@1").items[0].allow_patterns
    materialize_allowed_files(destination, patterns)
    (destination / ".aisketcher-model.json").write_text(
        json.dumps(
            {"repo_id": SDXL_BASE.repo_id, "revision": SDXL_BASE.revision}
        ),
        encoding="utf-8",
    )
    assert manager.model_path(SDXL_BASE) == destination

    def unexpected_rehash(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("unchanged process-local receipt should skip rehash")

    monkeypatch.setattr(manager, "_verify_required_artifacts", unexpected_rehash)
    assert manager.model_path(SDXL_BASE) == destination


def test_quick_plan_never_hashes_a_fresh_process_marker_but_reuses_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_snapshot_download(**kwargs: object) -> str:
        destination = Path(str(kwargs["local_dir"]))
        materialize_allowed_files(destination, tuple(kwargs["allow_patterns"]))
        return str(destination)

    fake_hub = ModuleType("huggingface_hub")
    fake_hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    PresetManager(tmp_path).install("sdxl-canny-lite@1", confirm=True)

    fresh_manager = PresetManager(tmp_path)
    original_verifier = fresh_manager._verify_required_artifacts
    verified: list[str] = []

    def record_verification(
        destination: Path, *args: object, **kwargs: object
    ) -> object:
        verified.append(destination.name)
        return original_verifier(destination, *args, **kwargs)

    monkeypatch.setattr(
        fresh_manager,
        "_verify_required_artifacts",
        record_verification,
    )

    quick_before = fresh_manager.plan_install(
        "sdxl-canny-lite@1",
        verify_cache=False,
    )
    assert not quick_before.installed
    assert verified == []

    full = fresh_manager.plan_install("sdxl-canny-lite@1")
    assert full.installed
    assert len(verified) == 2

    quick_after = fresh_manager.plan_install(
        "sdxl-canny-lite@1",
        verify_cache=False,
    )
    assert quick_after.installed
    assert len(verified) == 2


def test_hash_verification_observes_cancellation_between_streamed_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = PresetManager(tmp_path)
    destination = manager.model_destination(SDXL_BASE)
    destination.mkdir(parents=True)
    patterns = manager.plan_install("sdxl-canny-lite@1").items[0].allow_patterns
    materialize_allowed_files(destination, patterns)
    marker = destination / ".aisketcher-model.json"
    marker.write_text(
        json.dumps(
            {"repo_id": SDXL_BASE.repo_id, "revision": SDXL_BASE.revision}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(presets_module, "_HASH_CHUNK_SIZE", 2)
    checks = 0

    def should_cancel() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 2

    with pytest.raises(ModelUnavailableError, match="cancelled safely"):
        manager.model_path(SDXL_BASE, should_cancel=should_cancel)

    assert "schema_version" not in json.loads(marker.read_text(encoding="utf-8"))


def test_symlinked_cache_entry_is_rejected_and_removal_never_follows_target(
    tmp_path: Path,
) -> None:
    manager = PresetManager(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    destination = manager.model_destination(SDXL_BASE)
    destination.parent.mkdir(parents=True)
    destination.symlink_to(outside, target_is_directory=True)

    assert manager.model_path(SDXL_BASE) is None
    manager._remove_incomplete_destination(destination)
    assert not destination.exists()
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_symlinked_models_root_is_never_followed_for_removal(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    outside = tmp_path / "outside-models"
    destination = (
        outside
        / f"stabilityai--stable-diffusion-xl-base-1.0@{SDXL_BASE.revision}"
    )
    destination.mkdir(parents=True)
    sentinel = destination / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    (cache / "models").symlink_to(outside, target_is_directory=True)
    manager = PresetManager(cache)

    assert manager.model_path(SDXL_BASE) is None
    with pytest.raises(ValidationError, match="symlinked model cache boundary"):
        manager._remove_incomplete_destination(manager.model_destination(SDXL_BASE))
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_symlinked_cache_ancestor_is_never_followed_for_removal(
    tmp_path: Path,
) -> None:
    outside_parent = tmp_path / "outside-parent"
    outside_parent.mkdir()
    symlinked_parent = tmp_path / "cache-parent"
    symlinked_parent.symlink_to(outside_parent, target_is_directory=True)
    manager = PresetManager(symlinked_parent / "cache")
    destination = manager.model_destination(SDXL_BASE)
    destination.mkdir(parents=True)
    sentinel = destination / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    assert manager.model_path(SDXL_BASE) is None
    with pytest.raises(ValidationError, match="symlinked model cache boundary"):
        manager._remove_incomplete_destination(destination)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_incomplete_runtime_file_manifest_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = dict(presets_module.MODEL_ARTIFACTS)
    base = artifacts["sdxl-base-1.0"]
    artifacts["sdxl-base-1.0"] = replace(
        base,
        files=tuple(
            required
            for required in base.files
            if required.path != "model_index.json"
        ),
    )
    monkeypatch.setattr(presets_module, "MODEL_ARTIFACTS", artifacts)

    with pytest.raises(
        ValidationError,
        match="no complete registry-backed runtime SHA-256",
    ):
        PresetManager(tmp_path).plan_install("sdxl-canny-lite@1")


def test_preset_without_complete_registry_digests_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incomplete = {
        key: artifact
        for key, artifact in presets_module.MODEL_ARTIFACTS.items()
        if artifact.model_id != SDXL_CANNY_QUALITY.repo_id
    }
    monkeypatch.setattr(presets_module, "MODEL_ARTIFACTS", incomplete)
    manager = PresetManager(tmp_path)

    with pytest.raises(
        ValidationError,
        match="no complete registry-backed runtime SHA-256",
    ):
        manager.plan_install("sdxl-canny@1")


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

    def fake_snapshot_download(**kwargs: object) -> str:
        repo_id = str(kwargs["repo_id"])
        calls.append(repo_id)
        materialize_allowed_files(
            Path(str(kwargs["local_dir"])), tuple(kwargs["allow_patterns"])
        )
        return str(kwargs["local_dir"])

    def should_cancel() -> bool:
        return (plan.items[0].destination / ".aisketcher-model.json").is_file()

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
