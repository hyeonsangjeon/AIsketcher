from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from aisketcher import (
    FakeBackend,
    IntegrityError,
    Intent,
    ReplayError,
    Studio,
    ValidationError,
)
from aisketcher.manifest import (
    MANIFEST_SCHEMA,
    canonical_sha256,
    load_manifest,
    verify_manifest_files,
)
from aisketcher.models import GenerationRequest, GenerationResult


def build_study(prompt: str = "paper castle"):
    image = Image.new("RGB", (160, 120), "white")
    ImageDraw.Draw(image).polygon(((20, 100), (80, 15), (140, 100)), outline="black")
    studio = Studio(FakeBackend())
    study = studio.explore(studio.prepare(image, max_side=128), intent=Intent(prompt), outputs=2)
    study.pick(0)
    return studio, study


def test_export_is_sanitized_and_replayable(tmp_path: Path) -> None:
    studio, study = build_study()
    manifest_path = study.export(tmp_path / "run")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    rendered = manifest_path.read_text(encoding="utf-8")
    assert value["schema"] == "aisketcher.manifest/v1"
    assert "private-original-name" not in rendered
    assert str(tmp_path) not in rendered
    assert "EXIF" not in rendered
    assert value["selection"] == study[0].id
    assert len(value["files"]) == 5
    assert all(not Path(item["path"]).is_absolute() for item in value["files"].values())
    assert value["candidates"][0]["backend_metadata"] == {
        "algorithm": "guided-preview-v1",
        "backend": "fake",
    }
    report = studio.replay(manifest_path)
    assert report.replayed
    assert report.drift == ()
    assert len(report.verified_files) == 5
    assert report.study.selected is report.study[0]
    assert report.exact_candidate_match


def test_export_keeps_only_safe_built_in_backend_metadata(tmp_path: Path) -> None:
    _, study = build_study()
    secret = "sk_" + "abcdefghijklmnopqrstuv"
    study[0].backend_metadata = {
        "backend": "diffusers",
        "device": "mps",
        "mps_isolated": True,
        "mps_retries": 1,
        "sequential": True,
        "shared_pipeline_components": True,
        "vae_dtype": "float32",
        "algorithm": f"token={secret}",
        "token": secret,
        "local_path": str(tmp_path),
        "nested": {"unsafe": "value"},
    }

    manifest_path = study.export(tmp_path / "run")
    rendered = manifest_path.read_text(encoding="utf-8")
    value = json.loads(rendered)

    assert value["candidates"][0]["backend_metadata"] == {
        "backend": "diffusers",
        "device": "mps",
        "mps_isolated": True,
        "mps_retries": 1,
        "sequential": True,
        "shared_pipeline_components": True,
        "vae_dtype": "float32",
    }
    assert secret not in rendered
    assert str(tmp_path) not in rendered


def test_variation_export_replays_parent_lineage(tmp_path: Path) -> None:
    studio, study = build_study()
    variants = studio.vary(study[0], outputs=2, strength="balanced")
    manifest_path = variants.export(tmp_path / "variation")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert value["files"]["parent"]["path"] == "parent.png"
    assert value["lineage"]["parent_id"] == study[0].id
    report = studio.replay(manifest_path)
    assert report.study.kind == "variation"
    assert all(item.parent_id == study[0].id for item in report.study)


def test_modified_artifact_is_rejected(tmp_path: Path) -> None:
    studio, study = build_study()
    manifest_path = study.export(tmp_path / "run")
    source = manifest_path.parent / "source.png"
    source.write_bytes(source.read_bytes() + b"tampered")
    with pytest.raises(IntegrityError, match="hash mismatch"):
        studio.replay(manifest_path)


def test_unsafe_relative_artifact_path_is_rejected(tmp_path: Path) -> None:
    studio, study = build_study()
    manifest_path = study.export(tmp_path / "run")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    value["files"]["source"]["path"] = "../source.png"
    manifest_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(IntegrityError, match="Unsafe"):
        studio.replay(manifest_path)


def test_secret_looking_prompt_is_not_exported(tmp_path: Path) -> None:
    secret = "sk_" + "abcdefghijklmnopqrst"
    _, study = build_study(f"use token={secret} in the scene")
    with pytest.raises(ValidationError, match="credential"):
        study.export(tmp_path / "run")


def test_manifest_top_level_must_be_an_object(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]", encoding="utf-8")

    with pytest.raises(ReplayError, match="JSON object"):
        load_manifest(manifest)


def test_manifest_artifact_descriptor_must_be_an_object(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"schema": MANIFEST_SCHEMA, "files": {"source": []}}),
        encoding="utf-8",
    )
    path, value = load_manifest(manifest)

    with pytest.raises(IntegrityError, match="invalid artifact descriptor"):
        verify_manifest_files(path, value)


def test_replay_normalizes_malformed_capability_issues(tmp_path: Path) -> None:
    studio, study = build_study()
    manifest = study.export(tmp_path / "run")
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["recipe"]["capability_report"]["issues"] = [1]
    value["recipe_sha256"] = canonical_sha256(value["recipe"])
    manifest.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ReplayError, match="resolved recipe is invalid"):
        studio.replay(manifest)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("canny", []),
        ("diagnostics", {"flags": [], "recommended_canny": {}}),
        ("diagnostics", {"flags": {}, "recommended_canny": []}),
    ],
)
def test_replay_normalizes_malformed_nested_source_metadata(
    tmp_path: Path, field: str, value: object
) -> None:
    studio, study = build_study()
    manifest = study.export(tmp_path / field / str(len(str(value))))
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["source"][field] = value
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ReplayError, match="source metadata is invalid"):
        studio.replay(manifest)


def test_replay_normalizes_malformed_variation_lineage(tmp_path: Path) -> None:
    studio, study = build_study()
    variation = studio.vary(study[0], outputs=1)
    manifest = variation.export(tmp_path / "variation")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["lineage"] = []
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ReplayError, match="lineage must be an object"):
        studio.replay(manifest)


@pytest.mark.parametrize(
    "secret",
    [
        "sk-" + "a" * 32,
        "sk-proj-" + "b" * 32,
        "hf_" + "c" * 32,
        "ASIA" + "D" * 16,
        "ghp_" + "e" * 36,
        "github_pat_" + "f" * 40,
    ],
)
def test_provider_tokens_are_not_exported(tmp_path: Path, secret: str) -> None:
    _, study = build_study(f"draw an icon beside {secret}")
    with pytest.raises(ValidationError, match="credential"):
        study.export(tmp_path / "run")


def test_overwrite_replaces_only_an_intact_owned_export(tmp_path: Path) -> None:
    _, study = build_study()
    output = tmp_path / "run"
    first_manifest = study.export(output)
    first_manifest_bytes = first_manifest.read_bytes()

    second_manifest = study.export(output, overwrite=True)

    assert second_manifest.is_file()
    assert second_manifest.read_bytes() == first_manifest_bytes


def test_overwrite_preserves_unrelated_directory_with_manifest_json(tmp_path: Path) -> None:
    _, study = build_study()
    output = tmp_path / "run"
    output.mkdir()
    marker = output / "do-not-delete.txt"
    marker.write_text("unrelated user data", encoding="utf-8")
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps({"schema": "aisketcher.manifest/v1", "files": {}}),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="canonical AIsketcher export"):
        study.export(output, overwrite=True)

    assert marker.read_text(encoding="utf-8") == "unrelated user data"
    assert manifest_path.is_file()


def test_export_preserves_existing_directory_without_manifest(tmp_path: Path) -> None:
    _, study = build_study()
    output = tmp_path / "run"
    output.mkdir()
    existing = {
        "source.png": b"private source",
        "control.png": b"private control",
        "notes.txt": b"unrelated user data",
    }
    for relative, contents in existing.items():
        (output / relative).write_bytes(contents)

    with pytest.raises(ValidationError, match="already exists"):
        study.export(output)

    assert {path.name: path.read_bytes() for path in output.iterdir()} == existing
    assert not (output / "manifest.json").exists()
    assert not (output / "candidates").exists()


def test_overwrite_rejects_symlinked_artifact_without_touching_target(
    tmp_path: Path,
) -> None:
    _, study = build_study()
    output = tmp_path / "run"
    study.export(output)
    external = tmp_path / "external.png"
    external.write_bytes((output / "source.png").read_bytes())
    (output / "source.png").unlink()
    (output / "source.png").symlink_to(external)

    with pytest.raises(ValidationError, match="symbolic link"):
        study.export(output, overwrite=True)

    assert external.is_file()
    assert (output / "manifest.json").is_file()


class OtherFakeBackend(FakeBackend):
    name = "other-fake"


def test_backend_drift_requires_compatible_mode(tmp_path: Path) -> None:
    _, study = build_study()
    manifest_path = study.export(tmp_path / "run")
    studio = Studio(OtherFakeBackend())
    with pytest.raises(ReplayError, match="backend changed"):
        studio.replay(manifest_path, mode="strict")
    report = studio.replay(manifest_path, mode="compatible")
    assert report.replayed
    assert "backend changed" in report.drift[0]


def test_model_revision_drift_never_loads_unpinned_model(tmp_path: Path) -> None:
    studio, study = build_study()
    manifest_path = study.export(tmp_path / "run")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    value["recipe"]["models"][0] = {
        "repo_id": "untrusted/arbitrary-model",
        "revision": "main",
        "role": "base",
    }
    value["recipe_sha256"] = canonical_sha256(value["recipe"])
    manifest_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ReplayError, match="model revision"):
        studio.replay(manifest_path)
    report = studio.replay(manifest_path, mode="compatible")
    assert "model revision" in report.drift[0]
    assert report.study.recipe.models == study.recipe.models


class CountingFakeBackend(FakeBackend):
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, request: GenerationRequest) -> list[GenerationResult]:
        self.calls += 1
        return super().generate(request)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("width", 65),
        ("width", 4104),
        ("steps", 9999),
        ("guidance_scale", 999.0),
        ("control_strength", 999.0),
    ],
)
def test_strict_replay_rejects_unsafe_recipe_before_generation(
    tmp_path: Path, field: str, value: object
) -> None:
    _, study = build_study()
    manifest_path = study.export(tmp_path / "run")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["recipe"][field] = value
    payload["recipe_sha256"] = canonical_sha256(payload["recipe"])
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    backend = CountingFakeBackend()
    with pytest.raises(ReplayError, match="resolved recipe"):
        Studio(backend).replay(manifest_path)
    assert backend.calls == 0


def test_replay_rejects_source_metadata_mismatch_before_generation(tmp_path: Path) -> None:
    _, study = build_study()
    manifest_path = study.export(tmp_path / "run")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["source"]["prepared_size"] = [4096, 4096]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    backend = CountingFakeBackend()
    with pytest.raises(ReplayError, match="source metadata"):
        Studio(backend).replay(manifest_path)
    assert backend.calls == 0


def test_replay_rejects_oversized_source_artifact_before_generation(tmp_path: Path) -> None:
    _, study = build_study()
    manifest_path = study.export(tmp_path / "run")
    source_path = manifest_path.parent / "source.png"
    Image.new("RGB", (4104, 64), "white").save(source_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["files"]["source"]["sha256"] = sha256(source_path.read_bytes()).hexdigest()
    payload["source"]["prepared_size"] = [4104, 64]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    backend = CountingFakeBackend()
    with pytest.raises(ReplayError, match="unsafe dimensions"):
        Studio(backend).replay(manifest_path)
    assert backend.calls == 0


def test_replay_rejects_candidate_seed_count_mismatch_before_generation(
    tmp_path: Path,
) -> None:
    _, study = build_study()
    manifest_path = study.export(tmp_path / "run")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["seed_plan"]["seeds"].append(7)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    backend = CountingFakeBackend()
    with pytest.raises(ReplayError, match="candidate list"):
        Studio(backend).replay(manifest_path)
    assert backend.calls == 0
