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


@pytest.mark.parametrize(
    ("preset", "simple_model"),
    (
        ("flux2-klein-edit@1", "auto"),
        ("sdxl-canny-lite@1", "sdxl-canny-lite@1"),
    ),
)
def test_studio_uses_the_managed_cache_for_the_korean_translator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preset: str,
    simple_model: str,
) -> None:
    import aisketcher as package
    import aisketcher.studio_app as studio_app
    from aisketcher.cli import _launch_studio
    from aisketcher.prompt_normalization import MarianKoreanEnglishTranslator

    captured: dict[str, object] = {}
    cache = tmp_path / "models"

    class FakeManager:
        def __init__(
            self, cache_dir: str | Path | None, *, allow_downloads: bool
        ) -> None:
            assert cache_dir == cache
            assert allow_downloads is True
            self.cache_dir = Path(cache_dir)

    class FakeStudio:
        pass

    class FakeController:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def close(self) -> None:
            captured["closed"] = True

    class FakeDemo:
        _studio_launch_kwargs = {"server_name": "127.0.0.1"}

        def launch(self, **kwargs: object) -> None:
            captured["launch"] = kwargs

    monkeypatch.setattr(package, "PresetManager", FakeManager)
    monkeypatch.setattr(package, "Studio", FakeStudio)

    def fake_build_app(controller: object, **kwargs: object) -> FakeDemo:
        captured["build_controller"] = controller
        captured["build_kwargs"] = kwargs
        return FakeDemo()

    monkeypatch.setattr(studio_app, "AppController", FakeController)
    monkeypatch.setattr(studio_app, "build_app", fake_build_app)

    config = AIsketcherConfig(
        cache_dir=str(cache),
        preset=preset,
    )
    assert _launch_studio(config, port=7862) == 0

    translator = captured["prompt_translator"]
    assert isinstance(translator, MarianKoreanEnglishTranslator)
    assert translator.cache_dir == str(cache / "translation")
    build_kwargs = captured["build_kwargs"]
    assert isinstance(build_kwargs, dict)
    assert build_kwargs["default_preset"] == preset
    assert build_kwargs["default_simple_model"] == simple_model
    assert captured["launch"] == {
        "server_name": "127.0.0.1",
        "server_port": 7862,
    }
    assert captured["closed"] is True
