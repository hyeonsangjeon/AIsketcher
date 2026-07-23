from __future__ import annotations

import json
import stat
import threading
import time
import zipfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from aisketcher.manifest import canonical_sha256
from aisketcher.prompt_normalization import (
    MarianKoreanEnglishTranslator,
    TranslatorMetadata,
)
from aisketcher.studio_app.i18n import navigation_choices, structure_choices, text
from aisketcher.studio_app.runtime import (
    MAX_REPLAY_FILES,
    AppController,
    AppState,
    GuidedSampleCatalog,
    RunRegistry,
    StudioAppError,
    StudioJobCancelled,
    _seed_plan,
    prepare_replay_input,
    sanitize_upload,
)


def test_locked_seed_accepts_recorded_63_bit_heritage_seed() -> None:
    seed = 6764547109648557242

    plan = _seed_plan("locked", 1, str(seed))

    assert plan.resolve(1) == (seed,)


def test_locked_seed_requires_exactly_one_value() -> None:
    with pytest.raises(StudioAppError, match="exactly one"):
        _seed_plan("locked", 1, "1, 2")


def test_locked_seed_rejects_duplicate_output_requests() -> None:
    with pytest.raises(StudioAppError, match="requires one output"):
        _seed_plan("locked", 4, "42")


def test_scout_seed_plan_accepts_an_explicit_studio_base_seed() -> None:
    plan = _seed_plan("scout", 4, "", scout_base_seed=19)

    assert plan.base_seed == 19
    assert len(plan.resolve(4)) == 4


@dataclass
class FakeCandidate:
    image: Image.Image
    seed: int
    technical_badge: str
    reason: str
    recipe: Any = None


class FakeStudy:
    def __init__(self, candidates: list[FakeCandidate]) -> None:
        self.candidates = candidates
        self.picked: int | None = None

    def pick(self, index: int) -> FakeCandidate:
        self.picked = index
        return self.candidates[index]

    def export(self, destination: str) -> Path:
        root = Path(destination)
        root.mkdir(parents=True, exist_ok=True)
        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps({"schema": "aisketcher.manifest/v1", "backend": "fake"}),
            encoding="utf-8",
        )
        return manifest


class FakeStudio:
    def __init__(self, preset: str) -> None:
        self.preset = preset
        self.prepare_calls: list[str] = []
        self.explore_calls: list[dict[str, Any]] = []
        self.vary_calls: list[dict[str, Any]] = []

    def prepare(self, source: str) -> str:
        self.prepare_calls.append(source)
        return source

    @staticmethod
    def _candidates(count: int, offset: int = 0) -> list[FakeCandidate]:
        return [
            FakeCandidate(
                Image.new("RGB", (48, 48), (40 + index * 20, 90, 150)),
                seed=1000 + offset + index,
                technical_badge="Closest structure" if index == 0 else f"Direction {index + 1}",
                reason="Recorded edge alignment; no aesthetic score.",
            )
            for index in range(count)
        ]

    def explore(
        self,
        prepared: str,
        *,
        intent: Any,
        outputs: int,
        seed_plan: Any,
        recipe: Any,
        overrides: dict[str, Any],
    ) -> FakeStudy:
        self.explore_calls.append(
            {
                "prepared": prepared,
                "intent": intent,
                "outputs": outputs,
                "seed_plan": seed_plan,
                "recipe": recipe,
                "overrides": overrides,
            }
        )
        return FakeStudy(self._candidates(outputs))

    def vary(
        self,
        selected: FakeCandidate,
        *,
        outputs: int,
        strength: str,
        locks: tuple[str, ...],
        additional_instruction: str = "",
        refinement_prompt: str = "",
    ) -> FakeStudy:
        self.vary_calls.append(
            {
                "selected": selected,
                "outputs": outputs,
                "strength": strength,
                "locks": locks,
                "additional_instruction": additional_instruction,
                "refinement_prompt": refinement_prompt,
            }
        )
        return FakeStudy(self._candidates(outputs, offset=100))


@dataclass(frozen=True)
class PromptRecipeStub:
    prompt: str
    model_prompt: str | None
    prompt_metadata: dict[str, Any]


class RecordingTranslator:
    def __init__(self, translations: dict[str, str]) -> None:
        self.translations = translations
        self.calls: list[str] = []

    @property
    def metadata(self) -> TranslatorMetadata:
        return TranslatorMetadata(
            provider="test-double",
            model_id="example/ko-en",
            revision="test-revision",
            local_files_only=True,
        )

    def translate(self, value: str) -> str:
        self.calls.append(value)
        return self.translations.get(value, "")


class PromptAwareStudio(FakeStudio):
    def explore(
        self,
        prepared: str,
        *,
        intent: Any,
        outputs: int,
        seed_plan: Any,
        recipe: Any,
        overrides: dict[str, Any],
    ) -> FakeStudy:
        self.explore_calls.append(
            {
                "prepared": prepared,
                "intent": intent,
                "outputs": outputs,
                "seed_plan": seed_plan,
                "recipe": recipe,
                "overrides": overrides,
            }
        )
        resolved = PromptRecipeStub(
            prompt=intent.prompt,
            model_prompt=intent.model_prompt,
            prompt_metadata=dict(intent.prompt_metadata),
        )
        return FakeStudy(
            [
                FakeCandidate(
                    image=candidate.image,
                    seed=candidate.seed,
                    technical_badge=candidate.technical_badge,
                    reason=candidate.reason,
                    recipe=resolved,
                )
                for candidate in self._candidates(outputs)
            ]
        )


def _source(path: Path, *, size: tuple[int, int] = (64, 48)) -> Path:
    image = Image.new("RGB", size, "white")
    exif = Image.Exif()
    exif[274] = 1
    exif[315] = "private-author"
    image.save(path, format="JPEG", exif=exif)
    return path


def _explore(controller: AppController, source: Path, state: dict[str, Any] | None = None):
    return controller.explore(
        state or controller.initial_state(),
        source,
        "Layered paper shapes in navy, coral, and gold",
        "graphic_design",
        "balanced",
        "sdxl-canny-lite@1",
        4,
        "scout",
        "",
        True,
        30,
        5.0,
        ("structure",),
    )


def test_app_state_payload_contains_only_lightweight_values() -> None:
    payload = (
        AppState.new("ko")
        .replace(run_id="abc123", selected_index=2, advanced_overrides=True)
        .payload()
    )

    assert payload["language"] == "ko"
    assert set(payload) == {
        "session_id",
        "run_id",
        "selected_index",
        "language",
        "view",
        "advanced_overrides",
        "guided",
    }
    assert all(value is None or isinstance(value, (str, int, bool)) for value in payload.values())


def test_sanitize_upload_removes_metadata_and_writes_rgb_png(tmp_path: Path) -> None:
    source = _source(tmp_path / "private.jpg")

    clean_path = sanitize_upload(source, tmp_path / "session")

    assert clean_path.suffix == ".png"
    with Image.open(clean_path) as clean:
        assert clean.mode == "RGB"
        assert clean.size == (64, 48)
        assert len(clean.getexif()) == 0
        assert "private-author" not in repr(clean.info)


def test_sanitize_upload_rejects_pixel_limit(tmp_path: Path) -> None:
    source = _source(tmp_path / "large.jpg", size=(20, 20))

    with pytest.raises(StudioAppError, match="50 megapixel"):
        sanitize_upload(source, tmp_path / "session", max_pixels=399)


def test_guided_sample_requires_manifest_and_every_asset(tmp_path: Path) -> None:
    catalog = GuidedSampleCatalog(tmp_path)
    assert catalog.available is False

    (tmp_path / "source.png").write_bytes(b"not-decoded-here")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "aisketcher.manifest/v1",
                "source": {"path": "source.png"},
                "candidates": [{"path": "missing.png"}],
            }
        ),
        encoding="utf-8",
    )

    assert catalog.available is False
    with pytest.raises(StudioAppError):
        catalog.load()


def test_bundled_guided_sample_is_hash_verified_and_ready() -> None:
    catalog = GuidedSampleCatalog()

    assert catalog.available is True
    sample = catalog.load()
    assert len(sample.candidates) == 4
    assert sample.selected_index == 0
    assert sample.candidates[0].seed == 6764547109648557242
    assert sample.candidates[0].label == "cleanest edges"
    assert sample.prompt == (
        "An elaborate miniature fantasy country built in the sketch's tall "
        "letter A silhouette, populated by delightful tiny mascots. Painted "
        "metal towers, enamel creatures, glowing glass windows, layered "
        "architectural depth, vivid cyan, cobalt, vermilion, yellow and gold, "
        "crisp handcrafted details, dramatic studio lighting, premium concept "
        "maquette photograph."
    )
    assert sample.profile == "graphic_design"
    assert sample.structure == "balanced"


def test_guided_sample_response_surfaces_manifest_recipe(tmp_path: Path) -> None:
    controller = AppController(workspace_root=tmp_path / "work")

    opened = controller.open_guided_sample(controller.initial_state("en"))

    sample = controller.guided.load()
    assert opened.prompt == sample.prompt
    assert opened.profile == sample.profile
    assert opened.structure == sample.structure


def test_guided_sample_export_round_trips_every_declared_artifact(tmp_path: Path) -> None:
    controller = AppController(workspace_root=tmp_path / "work")
    opened = controller.open_guided_sample(controller.initial_state("en"))

    archive_path, _ = controller.export(opened.state)
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        assert "manifest.json" in names
        assert "prepared/source.png" in names
        assert "prepared/control.png" in names
        assert "scout/contact-sheet.png" in names
        assert {f"scout/scout-{index:02d}.png" for index in range(1, 5)} <= names

    staged = prepare_replay_input(archive_path, tmp_path / "replay")
    payload = json.loads(staged.read_text(encoding="utf-8"))
    for descriptor in payload["files"].values():
        assert (staged.parent / descriptor["path"]).is_file()


def test_expired_run_removes_its_workspace(tmp_path: Path) -> None:
    registry = RunRegistry(ttl_seconds=1)
    controller = AppController(workspace_root=tmp_path / "work", registry=registry)
    opened = controller.open_guided_sample(controller.initial_state("en"))
    record = registry.get(str(opened.state["run_id"]), str(opened.state["session_id"]))
    record.workspace.mkdir(parents=True)
    (record.workspace / "private.png").write_bytes(b"private")
    record.touched_at -= 2

    with pytest.raises(StudioAppError, match="no longer available"):
        registry.get(str(opened.state["run_id"]), str(opened.state["session_id"]))

    assert not record.workspace.exists()


def test_controller_close_cleans_owned_root_but_preserves_custom_root(tmp_path: Path) -> None:
    owned = AppController()
    owned_root = owned.workspace_root
    owned.open_guided_sample(owned.initial_state("en"))
    owned.close()
    assert not owned_root.exists()

    custom_root = tmp_path / "custom-workspace"
    custom = AppController(workspace_root=custom_root)
    opened = custom.open_guided_sample(custom.initial_state("en"))
    archive_path, _ = custom.export(opened.state)
    assert Path(archive_path).is_file()
    custom.close()
    assert custom_root.is_dir()
    assert not Path(archive_path).exists()


def test_active_guided_run_localizes_display_without_mutating_manifest_labels(
    tmp_path: Path,
) -> None:
    controller = AppController(workspace_root=tmp_path / "work")
    opened = controller.open_guided_sample(controller.initial_state("en"))

    localized = controller.localize_active_run(opened.state, "ko")

    assert localized is not None
    assert localized.state["language"] == "ko"
    assert [label for _, label in localized.gallery] == [
        "가장 깔끔한 윤곽",
        "구조 유사도 최고",
        "방향 3",
        "방향 4",
    ]
    assert "가이드 샘플을 열었습니다" in localized.status
    assert "Guided Sample loaded" not in localized.status
    assert "이 방향의 기술적 특징: 가장 깔끔한 윤곽 · 시드 6764547109648557242" in (
        localized.recommendation
    )
    assert "Why this direction" not in localized.recommendation
    record = controller.registry.get(str(opened.state["run_id"]), str(opened.state["session_id"]))
    assert record.candidates[0].label == "cleanest edges"
    assert record.candidates[1].label == "closest structure"


def test_guided_sample_reads_canonical_manifest_without_relabeling_external_art(
    tmp_path: Path,
) -> None:
    for filename in ("source.png", "one.png", "two.png"):
        Image.new("RGB", (8, 8), "white").save(tmp_path / filename)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "aisketcher.manifest/v1",
                "backend": "GPT Image 2 reference",
                "source": {"original_size": [8, 8]},
                "files": {
                    "source": {
                        "path": "source.png",
                        "sha256": sha256((tmp_path / "source.png").read_bytes()).hexdigest(),
                    },
                    "candidate:one": {
                        "path": "one.png",
                        "sha256": sha256((tmp_path / "one.png").read_bytes()).hexdigest(),
                    },
                    "candidate:two": {
                        "path": "two.png",
                        "sha256": sha256((tmp_path / "two.png").read_bytes()).hexdigest(),
                    },
                },
                "candidates": [
                    {
                        "id": "one",
                        "file": "candidate:one",
                        "seed": 41,
                        "scores": {"badges": ["Reference direction"]},
                    },
                    {"id": "two", "file": "candidate:two", "seed": 42},
                ],
                "selection": "two",
            }
        ),
        encoding="utf-8",
    )

    sample = GuidedSampleCatalog(tmp_path).load()

    assert sample.selected_index == 1
    assert sample.provenance == "GPT Image 2 reference"
    assert sample.prompt is None
    assert sample.profile is None
    assert sample.structure is None
    assert sample.candidates[0].label == "Reference direction"
    assert "AIsketcher" not in " ".join(candidate.label for candidate in sample.candidates)


def test_guided_sample_preserves_unsupported_recipe_without_overwriting_controls(
    tmp_path: Path,
) -> None:
    for filename in ("source.png", "candidate.png"):
        Image.new("RGB", (8, 8), "white").save(tmp_path / filename)
    recipe = {
        "prompt": "A valid prompt",
        "profile": "unsupported-profile",
        "structure": "balanced",
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "aisketcher.manifest/v1",
                "files": {
                    filename.removesuffix(".png"): {
                        "path": filename,
                        "sha256": sha256((tmp_path / filename).read_bytes()).hexdigest(),
                    }
                    for filename in ("source.png", "candidate.png")
                },
                "candidates": [{"path": "candidate.png"}],
                "recipe": recipe,
                "recipe_sha256": canonical_sha256(recipe),
            }
        ),
        encoding="utf-8",
    )

    sample = GuidedSampleCatalog(tmp_path).load()

    assert sample.prompt == "A valid prompt"
    assert sample.profile is None
    assert sample.structure == "balanced"


def test_guided_sample_does_not_display_recipe_with_a_mismatched_hash(
    tmp_path: Path,
) -> None:
    for filename in ("source.png", "candidate.png"):
        Image.new("RGB", (8, 8), "white").save(tmp_path / filename)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "aisketcher.manifest/v1",
                "files": {
                    filename.removesuffix(".png"): {
                        "path": filename,
                        "sha256": sha256((tmp_path / filename).read_bytes()).hexdigest(),
                    }
                    for filename in ("source.png", "candidate.png")
                },
                "candidates": [{"path": "candidate.png"}],
                "recipe": {
                    "prompt": "A tampered prompt",
                    "profile": "graphic_design",
                    "structure": "balanced",
                },
                "recipe_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )

    sample = GuidedSampleCatalog(tmp_path).load()

    assert sample.prompt is None
    assert sample.profile is None
    assert sample.structure is None


def test_guided_sample_ignores_a_non_ascii_recipe_hash(tmp_path: Path) -> None:
    for filename in ("source.png", "candidate.png"):
        Image.new("RGB", (8, 8), "white").save(tmp_path / filename)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "aisketcher.manifest/v1",
                "files": {
                    filename.removesuffix(".png"): {
                        "path": filename,
                        "sha256": sha256((tmp_path / filename).read_bytes()).hexdigest(),
                    }
                    for filename in ("source.png", "candidate.png")
                },
                "candidates": [{"path": "candidate.png"}],
                "recipe": {"prompt": "Do not display this"},
                "recipe_sha256": "é" * 64,
            }
        ),
        encoding="utf-8",
    )

    sample = GuidedSampleCatalog(tmp_path).load()

    assert sample.prompt is None


def test_guided_sample_rejects_paths_outside_bundle(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.png"
    outside.write_bytes(b"outside")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "aisketcher.manifest/v1",
                "source": {"original_size": [1, 1]},
                "files": {
                    "source": {"path": "../outside.png", "sha256": sha256(b"outside").hexdigest()},
                    "candidate:one": {
                        "path": "../outside.png",
                        "sha256": sha256(b"outside").hexdigest(),
                    },
                },
                "candidates": [{"id": "one", "file": "candidate:one"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(StudioAppError, match="unsafe path"):
        GuidedSampleCatalog(tmp_path).load()


def test_manifest_upload_without_sibling_artifacts_recommends_export_zip(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "aisketcher.manifest/v1",
                "files": {
                    "source": {
                        "path": "source.png",
                        "sha256": "0" * 64,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(StudioAppError, match="Upload the canonical export ZIP instead"):
        prepare_replay_input(manifest, tmp_path / "staged")


def test_manifest_upload_rejects_oversized_sibling_before_hashing(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    with source.open("wb") as handle:
        handle.truncate(50 * 1024 * 1024 + 1)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "aisketcher.manifest/v1",
                "files": {"source": {"path": "source.png", "sha256": "0" * 64}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(StudioAppError, match="exceeds the 50 MB limit"):
        prepare_replay_input(manifest, tmp_path / "staged")

    assert not (tmp_path / "staged").exists()


def test_replay_zip_is_extracted_into_a_bounded_session_directory(tmp_path: Path) -> None:
    source_bytes = b"canonical-source"
    manifest = {
        "schema": "aisketcher.manifest/v1",
        "files": {
            "source": {
                "path": "source.png",
                "sha256": sha256(source_bytes).hexdigest(),
            }
        },
    }
    bundle = tmp_path / "study.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("study/manifest.json", json.dumps(manifest))
        archive.writestr("study/source.png", source_bytes)

    staged = prepare_replay_input(bundle, tmp_path / "session" / "upload")

    assert staged == tmp_path / "session" / "upload" / "study" / "manifest.json"
    assert (staged.parent / "source.png").read_bytes() == source_bytes


def test_replay_zip_rejects_an_artifact_hash_mismatch(tmp_path: Path) -> None:
    bundle = tmp_path / "mismatch.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "manifest.json",
            json.dumps(
                {
                    "schema": "aisketcher.manifest/v1",
                    "files": {"source": {"path": "source.png", "sha256": "0" * 64}},
                }
            ),
        )
        archive.writestr("source.png", b"not-the-declared-hash")

    with pytest.raises(StudioAppError, match="failed SHA-256 verification"):
        prepare_replay_input(bundle, tmp_path / "stage")

    assert not (tmp_path / "stage").exists()


def test_replay_zip_rejects_path_traversal_without_writing_outside_stage(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "traversal.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("../escape.txt", "blocked")
        archive.writestr("manifest.json", "{}")

    with pytest.raises(StudioAppError, match="unsafe path"):
        prepare_replay_input(bundle, tmp_path / "stage")

    assert not (tmp_path / "escape.txt").exists()
    assert not (tmp_path / "stage").exists()


def test_replay_zip_rejects_symbolic_links(tmp_path: Path) -> None:
    link = zipfile.ZipInfo("source.png")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    bundle = tmp_path / "symlink.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr(link, "../outside.png")
        archive.writestr(
            "manifest.json",
            json.dumps(
                {
                    "schema": "aisketcher.manifest/v1",
                    "files": {"source": {"path": "source.png", "sha256": "0" * 64}},
                }
            ),
        )

    with pytest.raises(StudioAppError, match="symbolic links"):
        prepare_replay_input(bundle, tmp_path / "stage")

    assert not (tmp_path / "stage").exists()


def test_replay_zip_rejects_excessive_entry_count(tmp_path: Path) -> None:
    bundle = tmp_path / "too-many-files.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        for index in range(MAX_REPLAY_FILES + 1):
            archive.writestr(f"entry-{index}.txt", "x")

    with pytest.raises(StudioAppError, match=f"{MAX_REPLAY_FILES} entries"):
        prepare_replay_input(bundle, tmp_path / "stage")

    assert not (tmp_path / "stage").exists()


def test_replay_requires_backend_to_accept_explicit_strict_mode(tmp_path: Path) -> None:
    class ReplayWithoutMode:
        called = False

        def replay(self, manifest: str) -> FakeStudy:
            self.called = True
            return FakeStudy(FakeStudio._candidates(1))

    source_bytes = b"source"
    manifest = {
        "schema": "aisketcher.manifest/v1",
        "files": {
            "source": {
                "path": "source.png",
                "sha256": sha256(source_bytes).hexdigest(),
            }
        },
    }
    bundle = tmp_path / "study.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("source.png", source_bytes)
    backend = ReplayWithoutMode()
    controller = AppController(
        studio_factory=lambda preset: backend,
        workspace_root=tmp_path / "work",
    )

    with pytest.raises(StudioAppError, match="Strict replay failed"):
        controller.replay_manifest(controller.initial_state(), bundle, "sdxl-canny-lite@1")

    assert backend.called is False


def test_successful_replay_bundle_is_owned_by_the_run_workspace(tmp_path: Path) -> None:
    class ReplayBackend:
        def replay(self, manifest: str, *, mode: str) -> FakeStudy:
            assert Path(manifest).is_file()
            assert mode == "strict"
            return FakeStudy(FakeStudio._candidates(1))

    source_bytes = b"source"
    bundle = tmp_path / "study.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr(
            "manifest.json",
            json.dumps(
                {
                    "schema": "aisketcher.manifest/v1",
                    "files": {
                        "source": {
                            "path": "source.png",
                            "sha256": sha256(source_bytes).hexdigest(),
                        }
                    },
                }
            ),
        )
        archive.writestr("source.png", source_bytes)
    controller = AppController(
        studio_factory=lambda preset: ReplayBackend(),
        workspace_root=tmp_path / "work",
    )

    response = controller.replay_manifest(controller.initial_state(), bundle, "sdxl-canny-lite@1")
    record = controller.registry.get(
        str(response.state["run_id"]), str(response.state["session_id"])
    )

    assert (record.workspace / "replay-bundle" / "manifest.json").is_file()
    controller.close()
    assert not record.workspace.exists()


def test_fake_studio_explore_select_refine_and_export(tmp_path: Path) -> None:
    studios: dict[str, FakeStudio] = {}

    def factory(preset: str) -> FakeStudio:
        studio = FakeStudio(preset)
        studios[preset] = studio
        return studio

    controller = AppController(studio_factory=factory, workspace_root=tmp_path / "work")
    response = _explore(controller, _source(tmp_path / "source.jpg"))

    assert len(response.gallery) == 4
    assert Path(response.source or "").is_file()
    assert Path(response.selected or "").is_file()
    assert response.state["guided"] is False
    studio = studios["sdxl-canny-lite@1"]
    assert studio.explore_calls[0]["outputs"] == 4
    assert studio.explore_calls[0]["overrides"]["steps"] == 30
    assert studio.explore_calls[0]["recipe"].guidance_scale == 5.0

    selected_state, selected_path, recommendation, _ = controller.select_candidate(
        response.state, 2
    )
    assert selected_state["selected_index"] == 2
    selected_record = controller.registry.get(
        str(response.state["run_id"]),
        str(response.state["session_id"]),
    )
    assert selected_record.study.picked == 2
    assert Path(selected_path).is_file()
    assert "Direction 3" in recommendation

    refined = controller.refine(
        selected_state,
        "subtle",
        ("structure",),
        "Use warmer paper and fewer decorative details.",
    )
    assert len(refined.gallery) == 4
    assert studio.vary_calls[0]["locks"] == ("structure",)
    assert (
        studio.vary_calls[0]["additional_instruction"]
        == "Use warmer paper and fewer decorative details."
    )
    archive_path, _ = controller.export(refined.state)
    with zipfile.ZipFile(archive_path) as archive:
        assert "manifest.json" in archive.namelist()
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["refinement"] == {
        "additional_instruction": "Use warmer paper and fewer decorative details.",
        "automatic": False,
        "original_brief": "Layered paper shapes in navy, coral, and gold",
        "parent_run_id": response.state["run_id"],
    }


def test_korean_explore_and_refine_preserve_original_and_model_prompts(
    tmp_path: Path,
) -> None:
    original_brief = "귀여운 종이 왕국"
    original_instruction = "따뜻한 색감과 단순한 장식을 사용해 주세요."
    model_brief = "A cute paper kingdom."
    model_instruction = "Use warm colors and simple decorations."
    translator = RecordingTranslator(
        {
            original_brief: model_brief,
            original_instruction: model_instruction,
        }
    )
    studio = PromptAwareStudio("sdxl-canny-lite@1")
    controller = AppController(
        studio_factory=lambda preset: studio,
        prompt_translator=translator,
        workspace_root=tmp_path / "work",
    )
    state = controller.initial_state("ko")

    explored = controller.explore(
        state,
        _source(tmp_path / "source.jpg"),
        original_brief,
        "graphic_design",
        "balanced",
        "sdxl-canny-lite@1",
        4,
        "scout",
        "",
        True,
        30,
        5.0,
        ("structure",),
    )

    intent = studio.explore_calls[0]["intent"]
    assert intent.prompt == original_brief
    assert intent.model_prompt == model_brief
    assert intent.prompt_metadata == {
        "normalization": {
            "detected_language": "ko",
            "status": "translated",
            "translator": {
                "provider": "test-double",
                "model_id": "example/ko-en",
                "revision": "test-revision",
                "local_files_only": True,
            },
            "enhancement_applied": False,
        }
    }
    explored_record = controller.registry.get(
        str(explored.state["run_id"]),
        str(explored.state["session_id"]),
    )
    assert explored_record.request["brief"] == original_brief
    assert explored_record.request["model_prompt"] == model_brief

    refined = controller.refine(
        explored.state,
        "subtle",
        ("structure",),
        original_instruction,
    )

    selected_recipe = studio.vary_calls[0]["selected"].recipe
    assert selected_recipe.prompt == (
        f"{original_brief}\n\n추가 발전 지시: {original_instruction}"
    )
    assert selected_recipe.model_prompt == (
        f"{model_brief}\n\nRefinement instruction: {model_instruction}"
    )
    assert studio.vary_calls[0]["additional_instruction"] == model_instruction
    assert studio.vary_calls[0]["refinement_prompt"] == selected_recipe.model_prompt
    refinement = selected_recipe.prompt_metadata["refinements"][0]
    assert refinement == {
        "original_instruction": original_instruction,
        "model_instruction": model_instruction,
        "automatic": False,
        "normalization": {
            "detected_language": "ko",
            "status": "translated",
            "translator": {
                "provider": "test-double",
                "model_id": "example/ko-en",
                "revision": "test-revision",
                "local_files_only": True,
            },
            "enhancement_applied": False,
        },
    }
    refined_record = controller.registry.get(
        str(refined.state["run_id"]),
        str(refined.state["session_id"]),
    )
    assert refined_record.request["refinement_instruction"] == original_instruction
    assert refined_record.request["refinement_model_instruction"] == model_instruction
    assert translator.calls == [original_brief, original_instruction]


def test_korean_translation_failure_is_friendly_and_starts_no_model_work(
    tmp_path: Path,
) -> None:
    translator = RecordingTranslator({"한국어 브리프": ""})
    created: list[FakeStudio] = []

    def factory(preset: str) -> FakeStudio:
        studio = FakeStudio(preset)
        created.append(studio)
        return studio

    controller = AppController(
        studio_factory=factory,
        prompt_translator=translator,
        workspace_root=tmp_path / "work",
    )

    with pytest.raises(
        StudioAppError,
        match="원문은 보존되었고 이미지 모델로 전송되지 않았습니다",
    ):
        controller.explore(
            controller.initial_state("ko"),
            _source(tmp_path / "source.jpg"),
            "한국어 브리프",
            "graphic_design",
            "balanced",
            "sdxl-canny-lite@1",
            4,
            "scout",
            "",
            True,
            30,
            5.0,
            ("structure",),
        )

    assert created == []
    assert not list((tmp_path / "work").rglob("source-*.png"))


def test_default_prompt_translator_is_lazy_and_cache_only(tmp_path: Path) -> None:
    controller = AppController(
        studio_factory=FakeStudio,
        workspace_root=tmp_path / "work",
    )

    assert isinstance(controller.prompt_translator, MarianKoreanEnglishTranslator)
    assert controller.prompt_translator.metadata.local_files_only is True


def test_failed_generation_removes_the_new_run_workspace(tmp_path: Path) -> None:
    class FailingStudio(FakeStudio):
        def prepare(self, source: str) -> str:
            raise RuntimeError("backend unavailable")

    root = tmp_path / "work"
    controller = AppController(
        studio_factory=lambda preset: FailingStudio(preset),
        workspace_root=root,
    )

    with pytest.raises(StudioAppError, match="Generation failed"):
        _explore(controller, _source(tmp_path / "source.jpg"))

    assert not list(root.rglob("source-*.png"))
    assert not [path for path in root.rglob("*") if path.is_file()]


def test_process_generation_guard_serializes_controllers(tmp_path: Path) -> None:
    active = 0
    maximum_active = 0
    activity_lock = threading.Lock()

    class MeasuringStudio(FakeStudio):
        def explore(self, *args: Any, **kwargs: Any) -> FakeStudy:
            nonlocal active, maximum_active
            with activity_lock:
                active += 1
                maximum_active = max(maximum_active, active)
            try:
                time.sleep(0.05)
                return super().explore(
                    *args,
                    **{
                        key: value
                        for key, value in kwargs.items()
                        if key
                        in {
                            "intent",
                            "outputs",
                            "seed_plan",
                            "recipe",
                            "overrides",
                        }
                    },
                )
            finally:
                with activity_lock:
                    active -= 1

    source = _source(tmp_path / "source.jpg")
    first = AppController(
        studio_factory=MeasuringStudio,
        workspace_root=tmp_path / "first",
    )
    second = AppController(
        studio_factory=MeasuringStudio,
        workspace_root=tmp_path / "second",
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_explore, controller, source) for controller in (first, second)]
        responses = [future.result(timeout=5) for future in futures]

    assert maximum_active == 1
    assert all(len(response.gallery) == 4 for response in responses)


def test_duplicate_generation_in_one_session_is_rejected(tmp_path: Path) -> None:
    entered = threading.Event()
    release = threading.Event()

    class BlockingStudio(FakeStudio):
        def explore(self, *args: Any, **kwargs: Any) -> FakeStudy:
            entered.set()
            assert release.wait(timeout=5)
            return super().explore(
                *args,
                **{
                    key: value
                    for key, value in kwargs.items()
                    if key
                    in {
                        "intent",
                        "outputs",
                        "seed_plan",
                        "recipe",
                        "overrides",
                    }
                },
            )

    controller = AppController(
        studio_factory=BlockingStudio,
        workspace_root=tmp_path / "work",
    )
    source = _source(tmp_path / "source.jpg")
    state = controller.initial_state("ko")

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(_explore, controller, source, state)
        assert entered.wait(timeout=5)
        with pytest.raises(StudioAppError, match="이미 생성 작업"):
            _explore(controller, source, state)
        release.set()
        assert len(first.result(timeout=5).gallery) == 4


def test_stop_keeps_session_claimed_until_backend_unwinds(tmp_path: Path) -> None:
    entered = threading.Event()
    release = threading.Event()

    class SlowCancellationStudio(FakeStudio):
        def explore(
            self,
            prepared: str,
            *,
            cancellation_token: threading.Event,
            **kwargs: Any,
        ) -> FakeStudy:
            entered.set()
            assert release.wait(timeout=5)
            if cancellation_token.is_set():
                raise RuntimeError("provider callback interrupted")
            raise AssertionError("test expected cancellation")

    controller = AppController(
        studio_factory=SlowCancellationStudio,
        workspace_root=tmp_path / "work",
    )
    source = _source(tmp_path / "source.jpg")
    state = controller.initial_state("ko")

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(_explore, controller, source, state)
        assert entered.wait(timeout=5)
        controller.cancel_operation(state)
        controller.clear_operation(state)
        try:
            with pytest.raises(StudioAppError, match="이미 생성 작업"):
                _explore(controller, source, state)
        finally:
            release.set()
        with pytest.raises(StudioJobCancelled):
            first.result(timeout=5)


def test_cuda_oom_evicts_failed_runtime_and_next_request_recovers(
    tmp_path: Path,
) -> None:
    created: list[FakeStudio] = []

    class OOMStudio(FakeStudio):
        def __init__(self, preset: str, *, fail: bool) -> None:
            super().__init__(preset)
            self.fail = fail
            self.closed = False

        def explore(self, *args: Any, **kwargs: Any) -> FakeStudy:
            if self.fail:
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
            return super().explore(
                *args,
                **{
                    key: value
                    for key, value in kwargs.items()
                    if key
                    in {
                        "intent",
                        "outputs",
                        "seed_plan",
                        "recipe",
                        "overrides",
                    }
                },
            )

        def close(self) -> None:
            self.closed = True

    def factory(preset: str) -> OOMStudio:
        studio = OOMStudio(preset, fail=not created)
        created.append(studio)
        return studio

    controller = AppController(
        studio_factory=factory,
        workspace_root=tmp_path / "work",
    )
    source = _source(tmp_path / "source.jpg")

    with pytest.raises(StudioAppError, match="GPU ran out of memory"):
        _explore(controller, source)

    assert created[0].closed is True
    recovered = _explore(controller, source)
    assert len(created) == 2
    assert len(recovered.gallery) == 4


def test_backend_callback_error_after_cancel_is_reported_as_cancelled(
    tmp_path: Path,
) -> None:
    created: list[FakeStudio] = []

    class CallbackCancelledStudio(FakeStudio):
        def __init__(self, preset: str) -> None:
            super().__init__(preset)
            self.closed = False

        def explore(
            self,
            prepared: str,
            *,
            cancellation_token: threading.Event,
            **kwargs: Any,
        ) -> FakeStudy:
            cancellation_token.set()
            raise RuntimeError("provider callback interrupted")

        def close(self) -> None:
            self.closed = True

    def factory(preset: str) -> CallbackCancelledStudio:
        studio = CallbackCancelledStudio(preset)
        created.append(studio)
        return studio

    controller = AppController(
        studio_factory=factory,
        workspace_root=tmp_path / "work",
    )

    with pytest.raises(StudioJobCancelled, match="Stopped by user"):
        _explore(controller, _source(tmp_path / "source.jpg"))

    assert created[0].closed is True


def test_refine_callback_error_after_cancel_is_reported_as_cancelled(
    tmp_path: Path,
) -> None:
    class CallbackCancelledVariationStudio(FakeStudio):
        def __init__(self, preset: str) -> None:
            super().__init__(preset)
            self.closed = False

        def vary(
            self,
            selected: FakeCandidate,
            *,
            cancellation_token: threading.Event,
            **kwargs: Any,
        ) -> FakeStudy:
            cancellation_token.set()
            raise RuntimeError("provider callback interrupted")

        def close(self) -> None:
            self.closed = True

    controller = AppController(
        studio_factory=CallbackCancelledVariationStudio,
        workspace_root=tmp_path / "work",
    )
    explored = _explore(controller, _source(tmp_path / "source.jpg"))
    active = controller.pool.get("sdxl-canny-lite@1")

    with pytest.raises(StudioJobCancelled, match="Stopped by user"):
        controller.refine(explored.state, "subtle", ("structure",), "Sharper edges")

    assert active.closed is True


def test_failed_variation_removes_the_partial_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = AppController(
        studio_factory=FakeStudio,
        workspace_root=tmp_path / "work",
    )
    explored = _explore(controller, _source(tmp_path / "source.jpg"))
    session_dir = next((tmp_path / "work").iterdir())
    original_runs = {path.name for path in session_dir.iterdir() if path.is_dir()}

    def fail_materialization(study: Any, destination: Path) -> tuple[Any, ...]:
        raise RuntimeError("cannot encode candidate")

    monkeypatch.setattr("aisketcher.studio_app.runtime._materialize_study", fail_materialization)
    with pytest.raises(StudioAppError, match="Variation failed"):
        controller.refine(explored.state, "subtle", ("structure",))

    assert {path.name for path in session_dir.iterdir() if path.is_dir()} == original_runs


def test_try_again_uses_a_fresh_scout_seed_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = iter((11, 29))
    monkeypatch.setattr(
        "aisketcher.studio_app.runtime.secrets.randbits",
        lambda bits: next(values),
    )
    studio = FakeStudio("sdxl-canny-lite@1")
    controller = AppController(
        studio_factory=lambda preset: studio,
        workspace_root=tmp_path / "work",
    )

    first = _explore(controller, _source(tmp_path / "source.jpg"))
    second = controller.try_again(first.state)

    first_plan = studio.explore_calls[0]["seed_plan"]
    second_plan = studio.explore_calls[1]["seed_plan"]
    assert first_plan.base_seed == 11
    assert second_plan.base_seed == 29
    assert first_plan.resolve(4) != second_plan.resolve(4)
    assert first.state["run_id"] != second.state["run_id"]


def test_model_install_requires_explicit_confirmation(tmp_path: Path) -> None:
    installed: list[tuple[str, dict[str, Any]]] = []
    translator_prepares: list[dict[str, Any]] = []

    def installer(preset: str, **kwargs: Any) -> None:
        installed.append((preset, kwargs))

    class Translator:
        def translate(self, value: str) -> str:
            return value

        def prepare(self, **kwargs: Any) -> None:
            translator_prepares.append(kwargs)

    controller = AppController(
        studio_factory=FakeStudio,
        model_installer=installer,
        prompt_translator=Translator(),
        workspace_root=tmp_path,
    )

    with pytest.raises(StudioAppError, match="confirm"):
        controller.install_model("sdxl-canny-lite@1", False)

    assert controller.install_model("sdxl-canny-lite@1", True) == "Local model is ready."
    assert len(installed) == 1
    installed_preset, installed_kwargs = installed[0]
    assert installed_preset == "sdxl-canny-lite@1"
    assert installed_kwargs["confirm"] is True
    assert installed_kwargs["trust_remote_code"] is False
    assert installed_kwargs["safe_tensors_only"] is True
    assert installed_kwargs["cancellation_token"] is installed_kwargs["cancel_event"]
    assert callable(installed_kwargs["should_cancel"])
    assert installed_kwargs["should_cancel"]() is False
    assert len(translator_prepares) == 1
    assert translator_prepares[0]["confirm"] is True
    assert translator_prepares[0]["cancellation_token"] is translator_prepares[0][
        "cancel_event"
    ]
    assert callable(translator_prepares[0]["should_cancel"])


def test_translator_prepare_failure_releases_the_session_for_retry(
    tmp_path: Path,
) -> None:
    installs: list[str] = []

    def installer(preset: str, **kwargs: Any) -> None:
        assert kwargs["confirm"] is True
        installs.append(preset)

    class RetryTranslator:
        def __init__(self) -> None:
            self.attempts = 0

        def translate(self, value: str) -> str:
            return value

        def prepare(self, *, confirm: bool) -> None:
            assert confirm is True
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("translation setup unavailable")

    translator = RetryTranslator()
    controller = AppController(
        studio_factory=FakeStudio,
        model_installer=installer,
        prompt_translator=translator,
        workspace_root=tmp_path,
    )
    state = controller.initial_state("ko")

    with pytest.raises(StudioAppError, match="translation setup unavailable"):
        controller.install_model("flux2-klein-edit@1", True, "ko", state)

    assert controller.install_model("flux2-klein-edit@1", True, "ko", state) == (
        "로컬 모델 준비를 마쳤습니다."
    )
    assert translator.attempts == 2
    assert installs == ["flux2-klein-edit@1", "flux2-klein-edit@1"]


def test_model_prepare_stop_targets_the_active_translator(
    tmp_path: Path,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    def installer(preset: str, **kwargs: Any) -> None:
        assert preset == "flux2-klein-edit@1"
        assert kwargs["should_cancel"]() is False

    class CancellableTranslator:
        def __init__(self) -> None:
            self.cancel_called = False

        def translate(self, value: str) -> str:
            return value

        def prepare(
            self,
            *,
            confirm: bool,
            cancellation_token: threading.Event,
            should_cancel: Callable[[], bool],
        ) -> None:
            assert confirm is True
            entered.set()
            assert release.wait(timeout=5)
            assert cancellation_token.is_set()
            assert should_cancel() is True
            raise RuntimeError("translator stopped at a safe boundary")

        def cancel(self) -> None:
            self.cancel_called = True

    translator = CancellableTranslator()
    controller = AppController(
        studio_factory=FakeStudio,
        model_installer=installer,
        prompt_translator=translator,
        workspace_root=tmp_path,
    )
    state = controller.initial_state("ko")

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            controller.install_model,
            "flux2-klein-edit@1",
            True,
            "ko",
            state,
        )
        assert entered.wait(timeout=5)
        controller.cancel_operation(state)
        release.set()
        with pytest.raises(StudioJobCancelled):
            future.result(timeout=5)

    assert translator.cancel_called is True


def test_cancel_operation_sets_the_session_token_and_localizes_status(
    tmp_path: Path,
) -> None:
    controller = AppController(studio_factory=FakeStudio, workspace_root=tmp_path)
    state = controller.initial_state("ko")
    operation_event = controller.begin_operation(state)

    message = controller.cancel_operation(state)

    assert operation_event.is_set()
    assert message.startswith("사용자가 작업을 중지했습니다.")
    controller.clear_operation(state)


def test_language_catalog_keeps_stable_values() -> None:
    assert dict(navigation_choices("ko"))["고급 설정"] == "advanced"
    assert dict(structure_choices("ko"))["균형 있게"] == "balanced"
    assert text("ko", "headline").startswith("한 장의")
    assert text("ko", "unavailable").startswith("가이드 샘플")
    assert text("ko", "empty").startswith("스케치를")
    assert text("ko", "canny_info").startswith("현재 SDXL")
    assert text("ko", "badge_closest_structure") == "구조 유사도 최고"
    assert text("ko", "seed_label") == "시드"
    assert text("ko", "refine_prompt").startswith("이 방향")
    assert text("ko", "guided_refine_title").startswith("이 방향")
    assert text("ko", "stop") == "작업 중지"
    assert text("unknown", "headline").startswith("Turn one")
