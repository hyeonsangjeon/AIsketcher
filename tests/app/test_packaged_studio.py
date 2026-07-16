from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from examples.studio_app import AppController as LegacyAppController
from examples.studio_app.runtime import GuidedSampleCatalog as LegacyGuidedSampleCatalog

from aisketcher.studio_app import AppController, GuidedSampleCatalog
from aisketcher.studio_app.app import CSS_PATH


def test_packaged_studio_is_the_canonical_legacy_implementation() -> None:
    assert LegacyAppController is AppController
    assert LegacyGuidedSampleCatalog is GuidedSampleCatalog
    assert AppController.__module__ == "aisketcher.studio_app.runtime"


def test_default_guided_sample_comes_from_package_resources() -> None:
    expected = Path(
        str(files("aisketcher.studio_app").joinpath("assets", "pocket-kingdom"))
    )
    catalog = GuidedSampleCatalog()

    assert catalog.root.resolve() == expected.resolve()
    assert "docs" not in catalog.root.parts
    assert catalog.available is True
    sample = catalog.load()
    assert sample.root == expected.resolve()
    assert len(sample.candidates) == 4


def test_packaged_studio_static_files_are_local_to_the_package() -> None:
    package_root = Path(str(files("aisketcher.studio_app"))).resolve()

    assert CSS_PATH.resolve().is_relative_to(package_root)
    assert CSS_PATH.is_file()
    assert (package_root / "assets/pocket-kingdom/ARTWORK_NOTICE.md").is_file()
