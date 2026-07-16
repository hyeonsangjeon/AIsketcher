from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from aisketcher.studio_app import AppController, build_app
from aisketcher.studio_app.app import (
    LITE_PRESET,
    QUALITY_PRESET,
    _generation_args_for_view,
    _lock_choices,
    _preset_selection,
    _response_values,
    _seed_output_state,
    _variation_choices,
)
from aisketcher.studio_app.runtime import AppResponse, AppState


def test_studio_package_is_import_safe_without_eager_gradio_import() -> None:
    # Importing aisketcher.studio_app above must work whether the optional demo
    # dependency is present or not.  The actual builder performs the lazy import.
    assert callable(build_app)
    assert AppController.__module__.endswith("studio_app.runtime")


def test_advanced_choices_map_only_to_supported_core_values() -> None:
    assert dict(_variation_choices("en"))["Moderate"] == "balanced"
    assert {value for _, value in _lock_choices("en")} == {"structure"}


def test_simple_generation_ignores_hidden_advanced_config() -> None:
    state = AppState.new("en").replace(view="simple").payload()
    values = (
        state,
        "source.png",
        "Paper art",
        "graphic_design",
        "faithful",
        QUALITY_PRESET,
        1,
        "locked",
        "6764547109648557242",
        False,
        50,
        12.0,
        ("structure", "composition"),
    )

    resolved = _generation_args_for_view(values)

    assert resolved[:5] == values[:5]
    assert resolved[5:] == (
        LITE_PRESET,
        4,
        "scout",
        "",
        True,
        30,
        5.0,
        ("structure",),
    )


def test_advanced_generation_keeps_visible_config() -> None:
    state = AppState.new("en").replace(view="advanced").payload()
    values = (
        state,
        "source.png",
        "Paper art",
        "graphic_design",
        "faithful",
        QUALITY_PRESET,
        8,
        "explicit",
        "1,2,3,4,5,6,7,8",
        True,
        40,
        7.0,
        ("structure",),
    )

    assert _generation_args_for_view(values) == values


def test_locked_seed_forces_one_output_and_restores_previous_selection() -> None:
    locked = _seed_output_state("scout", "locked", 8, 4)
    restored = _seed_output_state("locked", "scout", locked[0], locked[2])

    assert locked == (1, False, 8)
    assert restored == (8, True, 8)


def test_preset_selection_is_canonical_and_localized() -> None:
    selected, plan = _preset_selection("ko", QUALITY_PRESET)

    assert selected == QUALITY_PRESET
    assert plan.startswith("Quality는 SDXL Base")
    with pytest.raises(ValueError, match="packaged"):
        _preset_selection("en", "arbitrary-model")


def test_generation_response_does_not_open_gallery_preview_automatically() -> None:
    gr = SimpleNamespace(update=lambda **values: values)
    response = AppResponse(
        state={"session_id": "test"},
        source="source.png",
        selected="candidate-1.png",
        gallery=(("candidate-1.png", "Direction 1"),),
        recommendation="Recorded direction",
        status="Ready",
    )

    gallery_update = _response_values(gr, response)[3]

    assert gallery_update == {"value": [("candidate-1.png", "Direction 1")]}
    assert "selected_index" not in gallery_update


@pytest.mark.skipif(
    importlib.util.find_spec("gradio") is None, reason="Gradio demo extra is absent"
)
def test_build_app_has_private_serial_launch_defaults(tmp_path: Path) -> None:
    app = build_app(AppController(workspace_root=tmp_path))

    assert app._studio_launch_kwargs["server_name"] == "127.0.0.1"
    assert app._studio_launch_kwargs["share"] is False
    assert app._studio_launch_kwargs["max_file_size"] == 20 * 1024 * 1024
    assert app.enable_queue is True
    assert app._queue.default_concurrency_limit == 1

    elem_ids = [
        component.get("props", {}).get("elem_id")
        for component in app.config["components"]
        if component.get("props", {}).get("elem_id")
    ]
    assert len(elem_ids) == len(set(elem_ids))


@pytest.mark.skipif(
    importlib.util.find_spec("gradio") is None, reason="Gradio demo extra is absent"
)
def test_locked_seed_config_starts_with_disabled_single_output(tmp_path: Path) -> None:
    app = build_app(
        AppController(workspace_root=tmp_path),
        default_output_count=8,
        default_seed_mode="locked",
        default_seed=7,
    )
    output = next(
        component
        for component in app.config["components"]
        if component.get("props", {}).get("elem_id") == "output-control"
    )["props"]

    assert output["choices"] == [("1", 1)]
    assert output["value"] == 1
    assert output["interactive"] is False

    state = AppState.new("en").payload()
    locked = app._studio_update_seed_input(state, "locked", 8, "scout", 4)
    restored = app._studio_update_seed_input(state, "scout", 1, "locked", locked[2])
    assert locked[1]["choices"] == (("1", 1),)
    assert locked[1]["value"] == 1
    assert locked[1]["interactive"] is False
    assert restored[1]["value"] == 8
    assert restored[1]["interactive"] is True


@pytest.mark.skipif(
    importlib.util.find_spec("gradio") is None, reason="Gradio demo extra is absent"
)
def test_model_prepare_uses_generation_preset_component(tmp_path: Path) -> None:
    app = build_app(
        AppController(workspace_root=tmp_path),
        default_preset=QUALITY_PRESET,
        default_output_count=8,
    )
    components = {component["id"]: component for component in app.config["components"]}
    model_button_id = next(
        component_id
        for component_id, component in components.items()
        if component.get("props", {}).get("elem_id") == "model-action"
    )
    model_choice_id = next(
        component_id
        for component_id, component in components.items()
        if component.get("props", {}).get("elem_id") == "model-choice"
    )
    preset_id = next(
        component_id
        for component_id, component in components.items()
        if component["type"] == "dropdown"
        and component.get("props", {}).get("label") == "Preset"
    )
    prepare_dependency = next(
        dependency
        for dependency in app.config["dependencies"]
        if (model_button_id, "click") in dependency["targets"]
    )

    assert preset_id in prepare_dependency["inputs"]
    assert model_choice_id not in prepare_dependency["inputs"]

    state = AppState.new("ko").payload()
    model_update = app._studio_sync_model_preset(state, QUALITY_PRESET)
    generation_update = app._studio_sync_generation_preset(state, LITE_PRESET)
    assert model_update[0]["value"] == QUALITY_PRESET
    assert model_update[1]["value"].startswith("Quality는")
    assert generation_update[0]["value"] == LITE_PRESET
    assert generation_update[1]["value"].startswith("Lite는")

    reset = app._studio_clear_overrides(state)
    assert reset[1] == QUALITY_PRESET
    assert reset[4]["value"] == 8
    assert reset[10] == QUALITY_PRESET
    assert reset[11]["value"].startswith("Quality는")


@pytest.mark.skipif(
    importlib.util.find_spec("gradio") is None, reason="Gradio demo extra is absent"
)
def test_runtime_localization_updates_canny_info_and_model_plan(tmp_path: Path) -> None:
    controller = AppController(workspace_root=tmp_path)
    app = build_app(controller)

    updates = app._studio_localize(
        controller.initial_state("en"),
        "ko",
        QUALITY_PRESET,
        "scout",
    )

    assert updates[23]["info"] == "현재 SDXL 프리셋에 필요한 설정입니다."
    assert updates[19]["value"] == QUALITY_PRESET
    assert updates[32]["value"] == QUALITY_PRESET
    assert updates[33]["value"].startswith("Quality는 SDXL Base")
