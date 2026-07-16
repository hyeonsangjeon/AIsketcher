from __future__ import annotations

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
    assert "actions/checkout@v6" in workflow
    assert "actions/setup-python@v6" in workflow
    assert "actions/upload-artifact@v7" in workflow
    assert workflow.count("persist-credentials: false") == 3


@pytest.mark.parametrize("name", ["pages.yml", "publish-pypi.yml"])
def test_deployment_workflows_are_manual_only(name: str) -> None:
    workflow = (WORKFLOWS / name).read_text(encoding="utf-8")
    assert "workflow_dispatch:" in workflow
    assert "push:" not in workflow
    assert "pull_request:" not in workflow


def test_pypi_uses_oidc_and_protected_environment() -> None:
    workflow = (WORKFLOWS / "publish-pypi.yml").read_text(encoding="utf-8")
    assert "environment: pypi" in workflow
    assert "id-token: write" in workflow
    assert "pypa/gh-action-pypi-publish@v1.14.0" in workflow
    assert "password:" not in workflow
    assert "api-token" not in workflow.lower()
    assert "EXPECTED_VERSION" in workflow
    assert 'refs/tags/v${EXPECTED_VERSION}' in workflow
    assert "persist-credentials: false" in workflow
    assert "wheel_smoke.py" in workflow
    assert "python -m pytest" in workflow
    assert "scan_distribution.py --repository . dist/*" in workflow


def test_pages_uses_current_official_major_actions() -> None:
    workflow = (WORKFLOWS / "pages.yml").read_text(encoding="utf-8")
    assert "actions/configure-pages@v6" in workflow
    assert "actions/upload-pages-artifact@v5" in workflow
    assert "actions/deploy-pages@v5" in workflow
    assert "persist-credentials: false" in workflow


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
    notes = (ROOT / "docs/releases/0.2.0.md").read_text(encoding="utf-8")

    assert 'version = "0.2.0"' in pyproject
    assert '__version__ = "0.2.0"' in package
    assert notes.startswith("# AIsketcher 0.2.0\n")


def test_readme_exposes_the_packaged_first_run() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "aisketcher init && aisketcher studio" in readme
    assert "releases/download/v0.2.0/aisketcher-0.2.0-py3-none-any.whl" in readme
    assert 'AIsketcher[demo] @ https://' in readme


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
