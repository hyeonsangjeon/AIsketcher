from __future__ import annotations

import stat
from pathlib import Path

import pytest

from aisketcher.config import (
    CONFIG_ENV_VAR,
    CONFIG_SCHEMA_VERSION,
    AIsketcherConfig,
    default_project_config_path,
    default_user_config_path,
    load_config,
    load_config_file,
    save_config,
)
from aisketcher.errors import ValidationError
from aisketcher.models import SeedMode


def write_config(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"schema_version: {CONFIG_SCHEMA_VERSION}\n{body}", encoding="utf-8")


def test_defaults_are_valid_and_paths_are_platform_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "aisketcher.config.user_config_path",
        lambda *args, **kwargs: tmp_path / "user-config",
    )
    monkeypatch.chdir(tmp_path)

    config = AIsketcherConfig()

    assert config.preset == "flux2-klein-edit@1"
    assert config.seed_mode is SeedMode.SCOUT
    assert config.output_count == 4
    assert config.cache_path is None
    assert default_user_config_path() == tmp_path / "user-config" / "config.yaml"
    assert default_project_config_path() == tmp_path / "aisketcher.yaml"


def test_save_and_load_round_trip_is_stable_and_private(tmp_path: Path) -> None:
    path = tmp_path / "settings" / "config.yaml"
    expected = AIsketcherConfig(
        preset="sdxl-canny@1",
        device="mps",
        output_count=1,
        seed_mode="locked",
        seed=6764547109648557242,
        language="ko",
        cache_dir="~/Models/AIsketcher cache",
        allow_downloads=False,
    )

    assert save_config(expected, path) == path
    loaded = load_config_file(path)

    assert loaded == expected
    assert loaded.seed_mode is SeedMode.LOCKED
    assert loaded.seed == 6764547109648557242
    assert loaded.cache_path == Path("~/Models/AIsketcher cache").expanduser()
    document = path.read_text(encoding="utf-8")
    assert "schema_version: 1" in document
    assert 'cache_dir: "~/Models/AIsketcher cache"' in document
    assert "allow_downloads: false" in document
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_save_protects_existing_configuration(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    save_config(AIsketcherConfig(), path)

    with pytest.raises(FileExistsError, match="already exists"):
        save_config(AIsketcherConfig(language="ko"), path)

    save_config(AIsketcherConfig(language="ko"), path, overwrite=True)
    assert load_config_file(path).language == "ko"


def test_project_values_override_user_values(tmp_path: Path) -> None:
    user = tmp_path / "user.yaml"
    project = tmp_path / "project.yaml"
    write_config(user, 'language: "ko"\noutput_count: 8\ndevice: "cpu"\n')
    write_config(project, 'output_count: 1\npreset: "quality"\n')

    config = load_config(user_path=user, project_path=project)

    assert config.language == "ko"
    assert config.device == "cpu"
    assert config.output_count == 1
    assert config.preset == "sdxl-canny@1"


def test_environment_selects_project_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing_user = tmp_path / "missing-user.yaml"
    override = tmp_path / "override.yaml"
    write_config(override, 'language: "ko" # team default\n')
    monkeypatch.setenv(CONFIG_ENV_VAR, str(override))

    assert load_config(user_path=missing_user).language == "ko"

    monkeypatch.setenv(CONFIG_ENV_VAR, str(tmp_path / "missing.yaml"))
    with pytest.raises(ValidationError, match="does not exist"):
        load_config(user_path=missing_user)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("", "empty"),
        ("language: ko\n", "schema_version is required"),
        ("schema_version: 2\n", "schema_version must be 1"),
        ("schema_version: 1\nmagic: true\n", "Unknown configuration"),
        ("schema_version: 1\noutput_count: 3\n", "output_count must be 1, 4, or 8"),
        ("schema_version: 1\n  nested: true\n", "Nested YAML"),
        ("schema_version: 1\nlanguage: ko\nlanguage: en\n", "Duplicate setting"),
        ("schema_version: 1\ncache_dir: [unsafe]\n", "Only YAML scalar"),
    ],
)
def test_invalid_or_unsupported_yaml_is_actionable(
    tmp_path: Path, body: str, message: str
) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(body, encoding="utf-8")

    with pytest.raises(ValidationError, match=message):
        load_config_file(path)


def test_mapping_validation_rejects_wrong_types_and_unknown_keys() -> None:
    with pytest.raises(ValidationError, match="allow_downloads"):
        AIsketcherConfig.from_mapping({"allow_downloads": "yes"})
    with pytest.raises(ValidationError, match="Unknown configuration"):
        AIsketcherConfig.from_mapping({"not_a_setting": True})
    with pytest.raises(ValidationError, match="device"):
        AIsketcherConfig(device="tpu")
    with pytest.raises(ValidationError, match="seed"):
        AIsketcherConfig(seed=1 << 63)
    with pytest.raises(ValidationError, match="requires seed"):
        AIsketcherConfig(seed_mode="locked", output_count=1)
    with pytest.raises(ValidationError, match="output_count 1"):
        AIsketcherConfig(seed_mode="locked", seed=42, output_count=4)
    with pytest.raises(ValidationError, match="only valid"):
        AIsketcherConfig(seed=42)
