from __future__ import annotations

from pathlib import Path

import pytest

from aisketcher.cli import main
from aisketcher.config import AIsketcherConfig, load_config_file, save_config
from aisketcher.errors import AIsketcherError


def test_init_writes_a_versioned_settings_ledger(tmp_path: Path) -> None:
    destination = tmp_path / "settings.yaml"

    result = main(
        [
            "init",
            "--path",
            str(destination),
            "--language",
            "ko",
            "--device",
            "mps",
            "--outputs",
            "1",
            "--seed-mode",
            "locked",
            "--seed",
            "6764547109648557242",
            "--offline",
        ]
    )

    assert result == 0
    config = load_config_file(destination)
    assert config.schema_version == 1
    assert config.language == "ko"
    assert config.device == "mps"
    assert config.output_count == 1
    assert str(config.seed_mode) == "locked"
    assert config.seed == 6764547109648557242
    assert config.allow_downloads is False


def test_init_protects_existing_settings_unless_forced(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    destination = tmp_path / "settings.yaml"
    save_config(AIsketcherConfig(language="en"), destination)

    assert main(["init", "--path", str(destination), "--language", "ko"]) == 2
    assert "already exists" in capsys.readouterr().err
    assert load_config_file(destination).language == "en"

    assert main(["init", "--path", str(destination), "--language", "ko", "--force"]) == 0
    assert load_config_file(destination).language == "ko"


def test_studio_loads_explicit_config_and_cli_language_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "settings.yaml"
    save_config(AIsketcherConfig(language="en", output_count=8), destination)
    captured: dict[str, object] = {}

    def fake_launch(config: AIsketcherConfig, *, port: int | None = None) -> int:
        captured["config"] = config
        captured["port"] = port
        return 0

    monkeypatch.setattr("aisketcher.cli._launch_studio", fake_launch)

    assert (
        main(
            [
                "studio",
                "--config",
                str(destination),
                "--language",
                "ko",
                "--port",
                "7861",
            ]
        )
        == 0
    )
    config = captured["config"]
    assert isinstance(config, AIsketcherConfig)
    assert config.language == "ko"
    assert config.output_count == 8
    assert captured["port"] == 7861


def test_studio_rejects_missing_explicit_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["studio", "--config", str(tmp_path / "missing.yaml")]) == 2
    assert "does not exist" in capsys.readouterr().err


def test_studio_reports_missing_demo_extra_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def missing_demo(config: AIsketcherConfig, *, port: int | None = None) -> int:
        raise AIsketcherError(
            "The Studio example requires Gradio. Install AIsketcher with the 'demo' extra."
        )

    monkeypatch.setattr("aisketcher.cli._launch_studio", missing_demo)

    assert main(["studio"]) == 2
    captured = capsys.readouterr()
    assert "AIsketcher with the 'demo' extra" in captured.err
    assert "Traceback" not in captured.err
