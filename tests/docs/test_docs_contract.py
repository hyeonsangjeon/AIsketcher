from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"

REQUIRED_PAGES = {
    "index.md",
    "getting-started.md",
    "ko/quickstart.md",
    "concepts/design-lineage.md",
    "canonical-sample.md",
    "sdk/workflow.md",
    "sdk/prepare.md",
    "sdk/explore-refine.md",
    "sdk/export-replay.md",
    "studio/simple-advanced.md",
    "studio/models-cache.md",
    "guides/seeds.md",
    "guides/migration.md",
    "guides/privacy.md",
    "guides/troubleshooting.md",
    "reference/configuration.md",
    "changelog.md",
    "releases/0.2.0.md",
}

MARKDOWN_LINK = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")
NAV_PAGE = re.compile(r"^\s*-\s+[^:]+:\s+([^\s#]+\.md)\s*$", re.MULTILINE)
PRIVATE_REFERENCE_NAMES = ("gpt" + " image", "gem" + "ini")


def markdown_files() -> list[Path]:
    return [ROOT / "README.md", *sorted(DOCS.rglob("*.md"))]


def local_target(source: Path, raw_target: str) -> Path | None:
    target = raw_target.strip().strip("<>").split("#", 1)[0]
    if not target or target.startswith(("http://", "https://", "mailto:")):
        return None
    return (source.parent / target).resolve()


def test_required_pages_and_site_assets_exist() -> None:
    missing = sorted(page for page in REQUIRED_PAGES if not (DOCS / page).is_file())
    assert not missing, f"missing required documentation pages: {missing}"
    assert (ROOT / "mkdocs.yml").is_file()
    assert (DOCS / "stylesheets/extra.css").is_file()
    assert (DOCS / "javascripts/sample-gallery.js").is_file()
    assert (DOCS / "assets/aisketcher-social-preview.png").is_file()
    assert (DOCS / "overrides/main.html").is_file()


def test_social_preview_has_explicit_non_execution_provenance() -> None:
    image = DOCS / "assets/aisketcher-social-preview.png"
    provenance = json.loads(
        (DOCS / "assets/aisketcher-social-preview.provenance.json").read_text(
            encoding="utf-8"
        )
    )

    assert provenance["schema"] == "aisketcher.marketing-artwork-provenance/v1"
    assert provenance["canonical_sdk_output"] is False
    assert hashlib.sha256(image.read_bytes()).hexdigest() == provenance["asset_sha256"]
    image_module = pytest.importorskip("PIL.Image")
    with image_module.open(image) as opened:
        assert list(opened.size) == provenance["dimensions"]

    repository_preview = provenance["repository_social_preview"]
    preview_path = DOCS / "assets" / repository_preview["asset"]
    assert preview_path.stat().st_size == repository_preview["bytes"]
    assert preview_path.stat().st_size < 1_000_000
    assert hashlib.sha256(preview_path.read_bytes()).hexdigest() == repository_preview[
        "asset_sha256"
    ]
    with image_module.open(preview_path) as opened:
        assert list(opened.size) == repository_preview["dimensions"] == [1280, 640]

    override = (DOCS / "overrides/main.html").read_text(encoding="utf-8")
    assert "og:image" in override
    assert "twitter:card" in override
    assert "aisketcher-social-preview-github.jpg" in override


def test_studio_screenshots_are_real_traceable_and_language_mapped() -> None:
    image_module = pytest.importorskip("PIL.Image")
    contracts = (
        {
            "asset": "aisketcher-studio-heritage-fixed-seed-en.jpg",
            "language": "en",
            "documents": (
                ROOT / "README.md",
                DOCS / "index.md",
                DOCS / "studio/simple-advanced.md",
            ),
            "excluded": (DOCS / "ko/quickstart.md",),
        },
        {
            "asset": "aisketcher-studio-guided-sample-ko.jpg",
            "language": "ko",
            "documents": (DOCS / "ko/quickstart.md",),
            "excluded": (
                ROOT / "README.md",
                DOCS / "index.md",
                DOCS / "studio/simple-advanced.md",
            ),
        },
    )

    for contract in contracts:
        image = DOCS / "assets" / contract["asset"]
        provenance_path = image.with_name(f"{image.stem}.provenance.json")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))

        assert provenance["schema"] == "aisketcher.ui-screenshot-provenance/v1"
        assert provenance["language"] == contract["language"]
        assert provenance["private_user_data_present"] is False
        assert provenance["model_downloaded_for_capture"] is False
        assert image.stat().st_size == provenance["asset_bytes"]
        assert hashlib.sha256(image.read_bytes()).hexdigest() == provenance[
            "asset_sha256"
        ]
        manifest = DOCS / "assets" / provenance["fixture_manifest"]
        assert hashlib.sha256(manifest.read_bytes()).hexdigest() == provenance[
            "fixture_manifest_sha256"
        ]
        assert (image.parent / provenance["license_notice"]).resolve().is_file()

        with image_module.open(image) as opened:
            assert list(opened.size) == provenance["dimensions"] == [1280, 946]
            assert opened.format == "JPEG"
            assert not opened.getexif()
            metadata = {
                key.lower()
                for key, value in opened.info.items()
                if value not in (None, b"", "")
            }
            assert not metadata & {"exif", "xmp", "xml", "photoshop", "icc_profile"}

        for document in contract["documents"]:
            assert image.name in document.read_text(encoding="utf-8")
        for document in contract["excluded"]:
            assert image.name not in document.read_text(encoding="utf-8")

    english_provenance = json.loads(
        (
            DOCS
            / "assets/aisketcher-studio-heritage-fixed-seed-en.provenance.json"
        ).read_text(encoding="utf-8")
    )
    assert english_provenance["selected_seed"] == 6764547109648557242
    fixture_provenance = DOCS / "assets" / english_provenance["fixture_provenance"]
    assert hashlib.sha256(fixture_provenance.read_bytes()).hexdigest() == (
        english_provenance["fixture_provenance_sha256"]
    )
    assert english_provenance["fixture_generated_with_model"] is True
    assert english_provenance["image_uploaded_for_capture"] is False


def test_studio_heritage_fixture_is_fixed_seed_and_hash_verified() -> None:
    root = DOCS / "assets/heritage-studio-fixed-seed-en"
    provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    manifest_path = root / provenance["generation"]["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == provenance[
        "generation"
    ]["manifest_sha256"]
    assert provenance["source"]["camera_or_gps_metadata_present"] is False
    assert provenance["source"]["private_name_present"] is False
    assert provenance["heritage_record"]["legacy_translation_recovered"] is False
    assert provenance["generation"]["offline"] is True
    assert provenance["generation"]["downloaded_bytes"] == 0

    selected = provenance["generation"]["selected_candidate"]
    assert manifest["selection"] == selected
    selected_candidate = next(
        candidate for candidate in manifest["candidates"] if candidate["id"] == selected
    )
    assert selected_candidate["seed"] == provenance["generation"]["selected_seed"]
    assert selected_candidate["seed"] == 6764547109648557242
    assert manifest["seed_plan"] == {
        "mode": "explicit",
        "seeds": [
            6764547109648557242,
            6854547109648557242,
            6634547109688557242,
            6764547109648557243,
        ],
    }
    assert manifest["recipe"]["preset"] == "sdxl-canny-lite@1"
    assert manifest["recipe"]["steps"] == 32
    assert manifest["recipe"]["guidance_scale"] == 6.5
    assert manifest["recipe"]["control_strength"] == 0.55
    assert manifest["source"]["canny"] == {
        "aperture_size": 3,
        "high": 160,
        "l2_gradient": False,
        "low": 140,
    }
    assert manifest["recipe"]["prompt"] == provenance["heritage_record"][
        "effective_prompt_en"
    ]
    for descriptor in manifest["files"].values():
        artifact = root / descriptor["path"]
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == descriptor["sha256"]


def test_studio_hpo_hero_fixture_is_selected_and_hash_verified() -> None:
    root = DOCS / "assets/heritage-studio-hero-hpo-en"
    provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    manifest_path = root / provenance["generation"]["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == provenance[
        "generation"
    ]["manifest_sha256"]
    assert provenance["schema"] == "aisketcher.studio-hero-hpo-fixture/v1"
    assert provenance["source"]["camera_or_gps_metadata_present"] is False
    assert provenance["source"]["private_name_present"] is False
    assert provenance["generation"]["offline"] is True
    assert provenance["generation"]["downloaded_bytes"] == 0
    assert provenance["hpo"]["new_candidates_generated"] == 12
    assert sum(round_["candidates"] for round_ in provenance["hpo"]["rounds"]) == 12
    assert provenance["hpo"]["technical_scores_are_aesthetic_claims"] is False

    selected = provenance["generation"]["selected_candidate"]
    assert manifest["selection"] == selected == "candidate-ffd8b272e6ec"
    selected_candidate = next(
        candidate for candidate in manifest["candidates"] if candidate["id"] == selected
    )
    assert selected_candidate["seed"] == provenance["generation"]["selected_seed"]
    assert selected_candidate["seed"] == 6764547109648557242
    assert manifest["seed_plan"] == {
        "mode": "explicit",
        "seeds": [
            6764547109648557242,
            6854547109648557242,
            6634547109688557242,
            6764547109648557243,
        ],
    }
    assert manifest["recipe"]["preset"] == "sdxl-canny-lite@1"
    assert manifest["recipe"]["steps"] == 36
    assert manifest["recipe"]["guidance_scale"] == 7.0
    assert manifest["recipe"]["control_strength"] == 0.55
    assert manifest["recipe"]["prompt"] == provenance["heritage_record"][
        "effective_prompt_en"
    ]
    assert manifest["source"]["canny"] == {
        "aperture_size": 3,
        "high": 160,
        "l2_gradient": False,
        "low": 140,
    }
    for descriptor in manifest["files"].values():
        artifact = root / descriptor["path"]
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == descriptor["sha256"]


def test_heritage_seed_study_is_new_hash_verified_evidence() -> None:
    root = DOCS / "assets/heritage-seed-study"
    provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    manifest_path = root / provenance["comparison"]["manifest_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    legacy = provenance["legacy_record"]
    assert legacy["active_seed"] == 6764547109648557242
    assert legacy["notebook_cell_index_zero_based"] == 7
    assert legacy["repository_url"].endswith("/aws-korea-2023-coding-school")
    expected_snapshot = f'{legacy["repository_url"]}/tree/{legacy["repository_commit"]}'
    expected_notebook = (
        f'{legacy["repository_url"]}/blob/{legacy["repository_commit"]}/'
        f'{legacy["notebook_path"]}'
    )
    assert legacy["repository_snapshot_url"] == expected_snapshot
    assert legacy["notebook_url"] == expected_notebook
    assert legacy["notebook_sha256"] == (
        "7b70430607850ea0f88ebac8bcf5674a5857b6ed70f27b5de9efd20310ed225a"
    )
    assert legacy["notebook_cell_execution_count"] == 9
    assert legacy["notebook_cell_source_sha256"] == (
        "91dd47656914cd220be042a83736c9eb582f36f1f2884eee621891b1af47f188"
    )
    legacy_input = legacy["legacy_input_reference"]
    assert legacy_input["path"] == legacy["file_name"] == "present_image1.jpeg"
    assert legacy_input["sha256"] == (
        "93c63301bc2b720fe1a265e834c83c64fb41909ea14589f757b25042f8986d6c"
    )
    assert legacy_input["dimensions"] == [800, 600]
    assert legacy_input["copied_into_v2_bundle"] is False
    assert legacy_input["gps_exif_present"] is True
    assert legacy["alternative_seeds"] == [
        6854547109648557242,
        6634547109688557242,
    ]
    assert legacy["num_inference_steps"] == 40
    assert legacy["guidance_scale"] == 7
    assert legacy["canny_low_threshold"] == 140
    assert legacy["canny_high_threshold"] == 160
    assert legacy["documented_brief_ko"] == (
        "알파벳 A, 글자, 추상화, 멋진 펜아트"
    )
    assert legacy["recorded_runtime_prompt_ko"] == (
        "특이한 풍경, 환상적인 그림, 복잡한 나라, 귀여운 케릭터"
    )
    assert legacy["documented_brief_location"] == {
        "cell_index_zero_based": 0,
        "kind": "markdown",
    }
    assert legacy["recorded_runtime_prompt_location"] == {
        "cell_index_zero_based": 7,
        "output_index_zero_based": 0,
        "kind": "stream",
    }
    assert legacy["model_stack"] == {
        "base_repo_id": "Lykon/DreamShaper",
        "base_revision": None,
        "controlnet_repo_id": "lllyasviel/sd-controlnet-canny",
        "controlnet_revision": None,
        "scheduler": "PNDM",
    }
    showcase = legacy["external_showcase"]
    assert showcase["sha256"] == (
        "3b07c2753ed89c0e655b675240ffb2d704a7fa448cbe592d05611e65a7ad6be1"
    )
    assert showcase["dimensions"] == [800, 600]
    assert showcase["linked_from_opening_cell"] is True
    assert showcase["active_seed_attribution_verified"] is False
    assert showcase["matches_embedded_execution_output"] is False
    embedded = legacy["embedded_execution_output"]
    embedded_path = root / embedded["path"]
    assert embedded["verified"] is True
    assert embedded["dimensions"] == [1600, 600]
    assert embedded["output_index_zero_based"] == 2
    assert hashlib.sha256(embedded_path.read_bytes()).hexdigest() == embedded["sha256"]
    assert provenance["source"]["legacy_jpeg_copied"] is False
    assert provenance["comparison"]["offline"] is True
    assert provenance["comparison"]["downloaded_bytes"] == 0
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == provenance[
        "comparison"
    ]["manifest_sha256"]
    assert manifest["seed_plan"]["seeds"] == [
        6764547109648557242,
        6854547109648557242,
        6634547109688557242,
    ]
    assert manifest["selection"] == provenance["comparison"]["selected_candidate"]
    assert provenance["comparison"]["selected_seed"] == 6764547109648557242
    assert provenance["comparison"]["technical_badge_leader"] == (
        "candidate-51a2dec3b190"
    )

    for descriptor in manifest["files"].values():
        artifact = root / descriptor["path"]
        assert artifact.is_file()
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == descriptor["sha256"]

    image_module = pytest.importorskip("PIL.Image")
    for path in root.rglob("*.png"):
        with image_module.open(path) as opened:
            metadata = {
                key.lower()
                for key, value in opened.info.items()
                if value not in (None, b"", "")
            }
            metadata.discard("dpi")
            assert not opened.getexif()
            assert not metadata & {"exif", "xmp", "xml", "photoshop", "icc_profile"}


def test_every_mkdocs_nav_page_exists() -> None:
    config = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    nav_pages = set(NAV_PAGE.findall(config))
    assert nav_pages >= REQUIRED_PAGES
    missing = sorted(page for page in nav_pages if not (DOCS / page).is_file())
    assert not missing, f"mkdocs nav references missing pages: {missing}"


@pytest.mark.parametrize("source", markdown_files(), ids=lambda path: str(path.relative_to(ROOT)))
def test_local_markdown_links_resolve(source: Path) -> None:
    text = source.read_text(encoding="utf-8")
    missing: list[str] = []
    for raw_target in MARKDOWN_LINK.findall(text):
        target = local_target(source, raw_target)
        if target is not None and not target.exists():
            missing.append(raw_target)
    assert not missing, f"{source.relative_to(ROOT)} has missing local links: {missing}"


def test_private_art_direction_providers_are_not_named_in_public_docs() -> None:
    public_text = "\n".join(path.read_text(encoding="utf-8") for path in markdown_files()).lower()
    found = [name for name in PRIVATE_REFERENCE_NAMES if name in public_text]
    assert not found, f"private art-direction provider names leaked into public docs: {found}"


def test_canonical_page_does_not_claim_an_unavailable_result() -> None:
    page = (DOCS / "canonical-sample.md").read_text(encoding="utf-8").lower()
    manifest = DOCS / "assets/pocket-kingdom/manifest.json"
    if manifest.exists():
        assert "verified local scout ready" in page
        assert "sample-status--ready" in page
        assert "fixture pending" not in page
    else:
        assert "fixture pending" in page
        assert "sample-status--pending" in page


def test_pocket_kingdom_images_have_no_embedded_metadata() -> None:
    image_paths = [
        path
        for path in sorted((DOCS / "assets/pocket-kingdom").rglob("*"))
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ]
    if not image_paths:
        return

    image_module = pytest.importorskip("PIL.Image")
    failures: list[str] = []
    for path in image_paths:
        with image_module.open(path) as image:
            metadata_keys = {
                key.lower()
                for key, value in image.info.items()
                if value not in (None, b"", "")
            }
            metadata_keys.discard("dpi")
            if image.getexif() or metadata_keys & {"exif", "xmp", "xml", "photoshop", "icc_profile"}:
                failures.append(path.name)
    assert not failures, f"canonical images contain embedded metadata: {failures}"


def test_pocket_kingdom_preparation_provenance_matches_assets() -> None:
    asset_dir = DOCS / "assets/pocket-kingdom"
    provenance = json.loads((asset_dir / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["schema"] == "aisketcher.sample-provenance/v1"
    assert provenance["status"] == "scout-ready"
    assert provenance["results"]["status"] == "generated"
    assert provenance["artwork"]["model_training_use"] is False

    image_module = pytest.importorskip("PIL.Image")
    for key in ("source", "control"):
        descriptor = provenance[key]
        relative = Path(descriptor["path"])
        assert not relative.is_absolute()
        path = asset_dir / relative
        assert path.is_file()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == descriptor["file_sha256"]
        with image_module.open(path) as image:
            assert list(image.size) == descriptor["dimensions"]

    results = provenance["results"]
    manifest_descriptor = results["manifest"]
    manifest_path = asset_dir / manifest_descriptor["path"]
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == manifest_descriptor[
        "file_sha256"
    ]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == "aisketcher.manifest/v1"
    assert manifest["selection"] == results["selection"]["parent_candidate_id"]
    for descriptor in manifest["files"].values():
        artifact = asset_dir / descriptor["path"]
        assert artifact.is_file()
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == descriptor["sha256"]

    variation_descriptor = results["variation"]
    variation_manifest_path = asset_dir / variation_descriptor["manifest_path"]
    assert hashlib.sha256(variation_manifest_path.read_bytes()).hexdigest() == variation_descriptor[
        "manifest_file_sha256"
    ]
    variation_manifest = json.loads(variation_manifest_path.read_text(encoding="utf-8"))
    assert variation_manifest["kind"] == "variation"
    assert variation_manifest["selection"] == results["selection"]["candidate_id"]
    assert variation_manifest["lineage"]["parent_id"] == manifest["selection"]
    for descriptor in variation_manifest["files"].values():
        artifact = variation_manifest_path.parent / descriptor["path"]
        assert artifact.is_file()
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == descriptor["sha256"]

    result_path = asset_dir / results["selection"]["result_path"]
    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == results["selection"][
        "result_file_sha256"
    ]

    serialized = json.dumps(provenance, sort_keys=True).lower()
    for forbidden in ("authorization", "access_token", "secret_key", "original_path"):
        assert forbidden not in serialized


def test_gallery_enhancement_has_accessibility_contract() -> None:
    script = (DOCS / "javascripts/sample-gallery.js").read_text(encoding="utf-8")
    css = (DOCS / "stylesheets/extra.css").read_text(encoding="utf-8")
    for required in ("showModal", "aria-label", "aria-valuetext", 'event.key === "Enter"', 'event.key === " "'):
        assert required in script
    assert "prefers-reduced-motion" in css
    assert ":focus-visible" in css
