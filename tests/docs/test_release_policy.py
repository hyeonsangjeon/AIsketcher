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
        if path.is_file()
        and (path.name.casefold() == "readme" or path.stem.casefold() == "readme")
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
    assert "gh-action-pypi-publish" not in workflow
    assert workflow.count(
        "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7"
    ) == 5
    assert workflow.count(
        "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97 # v7"
    ) == 5
    assert (
        "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a # v7"
        in workflow
    )
    assert workflow.count("persist-credentials: false") == 3


def test_pages_deployment_is_manual_only() -> None:
    workflow = (WORKFLOWS / "pages.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in workflow
    assert "push:" not in workflow
    assert "pull_request:" not in workflow


def test_pypi_uses_oidc_and_protected_environment() -> None:
    workflow = (WORKFLOWS / "publish-pypi.yml").read_text(encoding="utf-8")
    assert "release:\n    types: [published]" in workflow
    assert "workflow_dispatch:" in workflow
    assert (
        "github.event_name == 'release' || github.event_name == 'workflow_dispatch'"
        in workflow
    )
    assert "environment: pypi" in workflow
    assert "id-token: write" in workflow
    assert (
        "pypa/gh-action-pypi-publish@ba38be9e461d3875417946c167d0b5f3d385a247"
        " # v1.14.1"
    ) in workflow
    assert (
        "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7"
        in workflow
    )
    assert (
        "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97 # v7"
        in workflow
    )
    assert (
        "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a # v7"
        in workflow
    )
    assert workflow.count(
        "actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c # v8"
    ) == 2
    assert "needs: [build, publish]" in workflow
    assert "Verify matching published GitHub Release" in workflow
    assert 'gh api "repos/${GITHUB_REPOSITORY}/releases/tags/${expected_tag}"' in workflow
    assert '"${GITHUB_REF}" != "refs/tags/${expected_tag}"' in workflow
    assert 'release_draft="$(jq -r \'.draft\'' in workflow
    assert "gh release upload" in workflow
    assert "gh release download" in workflow
    assert "cmp -s" in workflow
    assert "already exists with different bytes" in workflow
    assert "Verify whether PyPI already has these exact distributions" in workflow
    assert "pypi_state.outputs.publish_needed == 'true'" in workflow
    assert "PyPI already contains the exact verified distributions." in workflow
    assert "PyPI distribution {path.name} exists with different bytes." in workflow
    assert "--clobber" not in workflow
    assert "DISPATCH_CONFIRM" in workflow
    assert "password:" not in workflow
    assert "api-token" not in workflow.lower()
    assert "EXPECTED_VERSION" in workflow
    assert 'refs/tags/v*' in workflow
    assert "persist-credentials: false" in workflow
    assert "wheel_smoke.py" in workflow
    assert "source archive must contain exactly one PKG-INFO file" in workflow
    assert "python -m pytest" in workflow
    assert "scan_distribution.py --repository . dist/*" in workflow


def test_pages_uses_reviewed_full_sha_actions() -> None:
    workflow = (WORKFLOWS / "pages.yml").read_text(encoding="utf-8")
    assert (
        "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7"
        in workflow
    )
    assert (
        "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97 # v7"
        in workflow
    )
    assert (
        "actions/configure-pages@45bfe0192ca1faeb007ade9deae92b16b8254a0d # v6"
        in workflow
    )
    assert (
        "actions/upload-pages-artifact@fc324d3547104276b827a68afc52ff2a11cc49c9 # v5"
        in workflow
    )
    assert (
        "actions/deploy-pages@cd2ce8fcbc39b97be8ca5fce6e763baed58fa128 # v5"
        in workflow
    )
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
    assert 'aisketcher[demo]==0.3.0' in readme
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
