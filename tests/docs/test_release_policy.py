from __future__ import annotations

import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github/workflows"
SCANNER = ROOT / "tests/docs/scan_distribution.py"
STAGED_UPLOAD_VERIFIER = ROOT / "tests/docs/verify_staged_upload.py"

WHEEL = "aisketcher-0.3.0-py3-none-any.whl"
SDIST = "aisketcher-0.3.0.tar.gz"


def _release_tag_is_in_default_history(
    *,
    comparison_status: str,
    merge_base_sha: str,
    tag_sha: str,
) -> bool:
    return (
        merge_base_sha == tag_sha
        and comparison_status in {"ahead", "identical"}
    )


@pytest.mark.parametrize(
    ("comparison_status", "merge_base_sha", "expected"),
    [
        ("ahead", "tag", True),
        ("identical", "tag", True),
        ("behind", "other", False),
        ("diverged", "other", False),
    ],
)
def test_release_tag_ancestry_truth_table(
    comparison_status: str,
    merge_base_sha: str,
    expected: bool,
) -> None:
    assert (
        _release_tag_is_in_default_history(
            comparison_status=comparison_status,
            merge_base_sha=merge_base_sha,
            tag_sha="tag",
        )
        is expected
    )


@pytest.mark.parametrize(
    "name",
    ["ci.yml", "pages.yml", "publish-pypi.yml"],
)
def test_required_workflow_exists(name: str) -> None:
    assert (WORKFLOWS / name).is_file()


def test_root_readme_is_not_shadowed_by_github_metadata() -> None:
    assert (ROOT / "README.md").is_file()
    shadowing_readmes = [
        path
        for path in (ROOT / ".github").iterdir()
        if path.is_file() and (path.name.casefold() == "readme" or path.stem.casefold() == "readme")
    ]
    assert shadowing_readmes == []
    assert (ROOT / ".github/WORKFLOWS.md").is_file()


def test_normal_ci_has_read_only_permissions_and_expected_matrix() -> None:
    workflow = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    assert "permissions:\n  contents: read" in workflow
    assert '["3.10", "3.12", "3.14"]' in workflow
    assert "mkdocs build --strict" in workflow
    assert "scan_distribution.py --repository . dist/*" in workflow
    assert "wheel_smoke.py" in workflow
    assert "python -m playwright install --with-deps chromium" in workflow
    assert 'AISKETCHER_BROWSER_E2E: "1"' in workflow
    assert "python -m pytest tests/e2e/test_studio_browser.py" in workflow
    assert "gh-action-pypi-publish" not in workflow
    assert workflow.count("actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7") == 6
    assert workflow.count("actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97 # v7") == 6
    assert "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a # v7" in workflow
    assert workflow.count("persist-credentials: false") == 6


def test_pages_deployment_is_manual_only() -> None:
    workflow = (WORKFLOWS / "pages.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in workflow
    assert "push:" not in workflow
    assert "pull_request:" not in workflow


def test_pypi_uses_oidc_and_protected_environment() -> None:
    workflow = (WORKFLOWS / "publish-pypi.yml").read_text(encoding="utf-8")
    assert "release:\n    types: [published]" in workflow
    assert "workflow_dispatch:" in workflow
    assert "github.event_name == 'release' || github.event_name == 'workflow_dispatch'" in workflow
    assert "environment: pypi" in workflow
    assert "id-token: write" in workflow
    assert "cache: pip" not in workflow
    assert (
        "pypa/gh-action-pypi-publish@ba38be9e461d3875417946c167d0b5f3d385a247 # v1.14.1"
    ) in workflow
    assert "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7" in workflow
    assert "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97 # v7" in workflow
    assert "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a # v7" in workflow
    assert (
        workflow.count("actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c # v8")
        == 2
    )
    assert "needs: [build, publish]" in workflow
    build_start = workflow.index("  build:")
    publish_start = workflow.index("  publish:")
    release_gate = workflow.index(
        "Verify published release and default-branch ancestry", build_start
    )
    exact_ci = workflow.index("Require successful CI for the exact release commit")
    checkout = workflow.index("Check out selected revision", build_start)
    prepublish_gate = workflow.index("Recheck release gate immediately before PyPI")
    pypi_state = workflow.index(
        "Plan a safe new publication or partial-upload recovery"
    )
    source_date_epoch = workflow.index("Pin the release build timestamp")
    package_build = workflow.index("Build wheel and source archive")
    record_digests = workflow.index("Record the canonical Actions artifact digests")
    artifact_upload = workflow.index("Upload verified release distributions")
    artifact_identity = workflow.index("Verify the approved Actions artifact identity")
    trusted_publish = workflow.index(
        "Publish only missing distributions with PyPI Trusted Publishing"
    )
    published_identity = workflow.index(
        "Verify and preserve the complete canonical PyPI distribution set"
    )
    canonical_upload = workflow.index(
        "Upload the complete canonical PyPI distributions for Release recovery"
    )
    assert release_gate < exact_ci < checkout < publish_start
    assert publish_start < prepublish_gate < pypi_state
    assert source_date_epoch < package_build < record_digests < artifact_upload
    package_build_step = workflow[package_build:record_digests]
    assert "python -m build --sdist" in package_build_step
    assert "python -m build --wheel" in package_build_step
    assert "run: python -m build\n" not in package_build_step
    assert artifact_identity < prepublish_gate < pypi_state < trusted_publish
    assert trusted_publish < published_identity < canonical_upload
    assert 'git show -s --format=%ct "${GITHUB_SHA}"' in workflow
    assert 'echo "SOURCE_DATE_EPOCH=${source_date_epoch}" >> "${GITHUB_ENV}"' in workflow
    assert "wheel_sha256: ${{ steps.distributions.outputs.wheel_sha256 }}" in workflow
    assert "sdist_sha256: ${{ steps.distributions.outputs.sdist_sha256 }}" in workflow
    assert "Downloaded wheel differs from the build job artifact." in workflow
    assert "Downloaded source archive differs from the build job artifact." in workflow
    recovery_plan = workflow[pypi_state:trusted_publish]
    assert "EXPECTED_WHEEL: ${{ needs.build.outputs.wheel_filename }}" in recovery_plan
    assert (
        "EXPECTED_WHEEL_SHA256: ${{ needs.build.outputs.wheel_sha256 }}"
        in recovery_plan
    )
    assert 'expected_wheel = os.environ["EXPECTED_WHEEL"]' in recovery_plan
    assert (
        'expected_wheel_sha256 = os.environ["EXPECTED_WHEEL_SHA256"]'
        in recovery_plan
    )
    assert "name == expected_wheel" in recovery_plan
    assert "downloaded_digest != expected_wheel_sha256" in recovery_plan
    assert (
        "Pre-existing PyPI wheel bytes do not match the approved "
        in recovery_plan
    )
    published_verification = workflow[published_identity:canonical_upload]
    assert "filename == expected_wheel or filename in uploaded_names" in (
        published_verification
    )
    assert "if requires_actions_exact and recorded != expected[filename]" in (
        published_verification
    )
    assert "requires_actions_exact" in published_verification
    assert "PyPI {filename} bytes do not match the approved " in workflow
    assert "EXPECTED_SDIST_SHA256" in workflow[published_identity:canonical_upload]
    assert "EXPECTED_WHEEL_SHA256" in workflow[published_identity:canonical_upload]
    assert "is not in " in workflow
    assert "merge_base_commit.sha" in workflow
    assert '"${comparison_status}" != "ahead"' in workflow
    assert '"${comparison_status}" != "identical"' in workflow
    assert "release_prerelease" in workflow
    assert 'gh api "repos/${GITHUB_REPOSITORY}/commits/${default_branch}"' in workflow
    assert 'gh api "repos/${GITHUB_REPOSITORY}/commits/${expected_tag}"' in workflow
    assert '"repos/${GITHUB_REPOSITORY}/compare/${expected_tag}...${default_branch}"' in workflow
    assert "Verify matching published GitHub Release" in workflow
    assert workflow.count("EVENT_SHA: ${{ github.sha }}") == 4
    assert workflow.count('"${tag_sha}" != "${EVENT_SHA}"') == 3
    assert workflow.count("release_prerelease=") == 3
    assert 'gh api "repos/${GITHUB_REPOSITORY}/releases/tags/${expected_tag}"' in workflow
    assert '"${GITHUB_REF}" != "refs/tags/${expected_tag}"' in workflow
    assert "release_draft=\"$(jq -r '.draft'" in workflow
    assert "gh release upload" in workflow
    assert "gh release download" in workflow
    assert "cmp -s" in workflow
    assert "already exists with different bytes" in workflow
    assert "actions: read # Required only to verify CI" in workflow
    assert 'actions/workflows/ci.yml/runs"' in workflow
    assert '-f "head_sha=${EVENT_SHA}"' in workflow
    assert '.conclusion == "success"' in workflow
    assert '.head_repository.full_name == $repository' in workflow
    assert "Plan a safe new publication or partial-upload recovery" in workflow
    assert "published_names.issubset(local_names)" in workflow
    assert "missing_names = sorted(local_names - published_names)" in workflow
    assert "shutil.copy2(local_by_name[name], upload_dir / name)" in workflow
    assert "pypi_state.outputs.upload_needed == 'true'" in workflow
    assert "packages-dir: upload-dist/" in workflow
    assert "python tests/docs/verify_staged_upload.py" in workflow
    assert "if path.is_file() and path.name in expected" in workflow
    assert "verify_distribution_equivalence.py" in workflow
    assert "published-python-distributions-" in workflow
    assert "Downloaded PyPI distribution {filename} failed its SHA-256 check." in workflow
    assert 'parsed.hostname != "files.pythonhosted.org"' in workflow
    assert "canonical-dist/*" in workflow
    assert "--clobber" not in workflow
    assert "skip-existing" not in workflow
    assert "DISPATCH_CONFIRM" in workflow
    assert "password:" not in workflow
    assert "api-token" not in workflow.lower()
    assert "EXPECTED_VERSION" in workflow
    assert "refs/tags/v*" in workflow
    assert "persist-credentials: false" in workflow
    assert "wheel_smoke.py" in workflow
    assert "source archive must contain exactly one PKG-INFO file" in workflow
    assert "python -m pytest" in workflow
    assert "scan_distribution.py --repository . dist/*" in workflow


def _verify_staged_upload(directory: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(STAGED_UPLOAD_VERIFIER),
            "--directory",
            str(directory),
            "--expected",
            WHEEL,
            "--expected",
            SDIST,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("staged", [(WHEEL, SDIST), (WHEEL,), (SDIST,)])
def test_staged_upload_accepts_only_expected_attestation_sidecars(
    tmp_path: Path,
    staged: tuple[str, ...],
) -> None:
    for name in (*staged, *(f"{name}.publish.attestation" for name in staged)):
        (tmp_path / name).write_bytes(b"reviewed")

    result = _verify_staged_upload(tmp_path)

    assert result.returncode == 0, result.stderr
    for name in staged:
        assert name in result.stdout


@pytest.mark.parametrize(
    "unexpected_name",
    [
        "notes.txt",
        f"{WHEEL}.attestation",
        f"{WHEEL}.publish.attestation.json",
        "another-package.whl.publish.attestation",
    ],
)
def test_staged_upload_rejects_every_other_file(
    tmp_path: Path,
    unexpected_name: str,
) -> None:
    (tmp_path / WHEEL).write_bytes(b"reviewed")
    (tmp_path / unexpected_name).write_bytes(b"unreviewed")

    result = _verify_staged_upload(tmp_path)

    assert result.returncode != 0
    assert unexpected_name in result.stderr


def test_staged_upload_rejects_orphan_expected_attestation(
    tmp_path: Path,
) -> None:
    attestation = f"{SDIST}.publish.attestation"
    (tmp_path / WHEEL).write_bytes(b"reviewed")
    (tmp_path / attestation).write_bytes(b"orphan")

    result = _verify_staged_upload(tmp_path)

    assert result.returncode != 0
    assert attestation in result.stderr


def test_release_docs_define_the_exact_wheel_and_sdist_recovery_boundary() -> None:
    workflow_docs = (ROOT / ".github/WORKFLOWS.md").read_text(encoding="utf-8")
    release_notes = (ROOT / "docs/releases/0.3.0.md").read_text(encoding="utf-8")

    for text in (workflow_docs, release_notes):
        assert "pre-existing source archive" in text
        assert "normalized" in text
        assert "byte-reproducible" in text
        assert "wheel" in text

    assert "must also match the approved Actions wheel SHA-256 and bytes" in (
        workflow_docs
    )
    assert "The wheel must always match the Azure-approved Actions SHA-256" in (
        workflow_docs
    )
    assert "normalized-recovery exception" in release_notes
    assert "canonical PyPI/Release distribution set" in release_notes


def test_pages_uses_reviewed_full_sha_actions() -> None:
    workflow = (WORKFLOWS / "pages.yml").read_text(encoding="utf-8")
    assert "  build:" in workflow
    assert "  deploy:" in workflow
    assert "    needs: build" in workflow
    assert "permissions:\n  contents: read" in workflow
    build = workflow[workflow.index("  build:") : workflow.index("  deploy:")]
    assert "    permissions:\n      contents: read\n      pages: read" in build
    deploy = workflow[workflow.index("  deploy:") :]
    assert "      pages: write" in deploy
    assert "      id-token: write" in deploy
    assert "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7" in workflow
    assert "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97 # v7" in workflow
    assert "actions/configure-pages@45bfe0192ca1faeb007ade9deae92b16b8254a0d # v6" in workflow
    assert "actions/upload-pages-artifact@fc324d3547104276b827a68afc52ff2a11cc49c9 # v5" in workflow
    assert "actions/deploy-pages@cd2ce8fcbc39b97be8ca5fce6e763baed58fa128 # v5" in workflow
    assert "persist-credentials: false" in workflow


def test_every_workflow_action_is_pinned_to_a_full_commit_sha() -> None:
    for path in sorted(WORKFLOWS.glob("*.yml")):
        workflow = path.read_text(encoding="utf-8")
        actions = re.findall(r"^\s*uses:\s+([^\s#]+)", workflow, flags=re.MULTILINE)
        assert actions, f"{path.name} contains no actions to audit"
        for action in actions:
            assert re.fullmatch(r"[^@\s]+@[0-9a-f]{40}", action), (
                f"{path.name} has an unpinned action: {action}"
            )


def test_artwork_is_explicitly_excluded_from_mit() -> None:
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    artwork = (ROOT / "ARTWORK_LICENSE.md").read_text(encoding="utf-8")
    assert "MIT License" in license_text
    assert "Copyright (c) 2026 hyeonsangjeon" in license_text
    assert "does **not** grant rights" in artwork
    assert "docs/assets/" in artwork
    assert "unmodified" in artwork
    assert "official AIsketcher source or Python distribution" in artwork
    assert 'license = "MIT AND LicenseRef-AIsketcher-Artwork"' in (
        ROOT / "pyproject.toml"
    ).read_text(encoding="utf-8")


def test_release_version_is_consistent_across_metadata_and_notes() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    package = (ROOT / "src/aisketcher/__init__.py").read_text(encoding="utf-8")
    manifest = (ROOT / "src/aisketcher/manifest.py").read_text(encoding="utf-8")
    lockfile = (ROOT / "uv.lock").read_text(encoding="utf-8")
    notes = (ROOT / "docs/releases/0.3.0.md").read_text(encoding="utf-8")

    assert 'version = "0.3.0"' in pyproject
    assert '__version__ = "0.3.0"' in package
    assert 'package_version = "0.3.0"' in manifest
    assert 'name = "aisketcher"\nversion = "0.3.0"' in lockfile
    assert notes.startswith("# AIsketcher 0.3.0\n")


def test_readme_exposes_the_packaged_first_run() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "aisketcher init && aisketcher studio" in readme
    assert "\npip install aisketcher\n" in readme
    assert "aisketcher[demo]==0.3.0" in readme
    assert "lowercase `aisketcher`" in readme
    assert "Until PyPI lists" not in readme
    assert "releases/download/v0.2.0" not in readme


def test_v03_removes_uppercase_legacy_facades() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    wheel_smoke = (ROOT / "tests/docs/wheel_smoke.py").read_text(encoding="utf-8")

    assert not (ROOT / "src/AIsketcher.py").exists()
    assert not (ROOT / "src/aisketcher/modelPipe.py").exists()
    assert 'py-modules = ["AIsketcher"]' not in pyproject
    assert "import AIsketcher" not in wheel_smoke
    assert "AIsketcher.img2img" not in wheel_smoke


def test_pypi_description_uses_the_current_product_boundary() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert 'readme = "README.md"' in pyproject
    assert "model-agnostic Python SDK" in readme
    assert 'preset = "flux2-klein-edit@1"' in readme
    assert "Keep exploring the sample" in readme
    assert "Korean→English helper" in readme
    assert "aisketcher-studio-heritage-fixed-seed-en.jpg" in readme
    assert "6764547109648557242" in readme
    assert "AWS Translate" not in readme
    assert "aws_access_key_id" not in readme


def test_distribution_scanner_detects_current_service_token_shapes(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.whl"
    synthetic = "\n".join(
        (
            "ASIA" + "A" * 16,
            "sk-proj-" + "b" * 32,
            "hf_" + "c" * 32,
            "github_pat_" + "d" * 40,
        )
    )
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("unsafe.txt", synthetic)

    result = subprocess.run(
        [sys.executable, str(SCANNER), str(archive)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "AWS access key" in result.stdout
    assert "OpenAI API key" in result.stdout
    assert "Hugging Face token" in result.stdout
    assert "GitHub token" in result.stdout


def test_repository_scanner_detects_a_secret_before_build(tmp_path: Path) -> None:
    (tmp_path / "settings.txt").write_text("token=" + "z" * 32, encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCANNER), "--repository", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "repository/settings.txt" in result.stdout


@pytest.mark.parametrize("suffix", [".bin", ".onnx", ".gguf"])
def test_distribution_scanner_rejects_modern_weight_formats(
    tmp_path: Path,
    suffix: str,
) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(f"model/runtime{suffix}", b"synthetic-model-payload")

    result = subprocess.run(
        [sys.executable, str(SCANNER), str(archive)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "model weight files must not be distributed" in result.stdout
    assert f"runtime{suffix}" in result.stdout


def test_repository_scanner_rejects_files_above_the_scan_limit(
    tmp_path: Path,
) -> None:
    oversized = tmp_path / "opaque.dat"
    with oversized.open("wb") as stream:
        stream.seek(20 * 1024 * 1024)
        stream.write(b"x")

    result = subprocess.run(
        [sys.executable, str(SCANNER), "--repository", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "repository/opaque.dat" in result.stdout
    assert "file exceeds" in result.stdout
