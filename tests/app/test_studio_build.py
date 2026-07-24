from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import aisketcher.cli as cli_module
from aisketcher.presets import InstallItem, InstallPlan
from aisketcher.studio_app import AppController, build_app
from aisketcher.studio_app import app as app_module
from aisketcher.studio_app.app import (
    AUTO_MODEL,
    BROWSER_SESSION_STORAGE_KEY,
    BROWSER_TAB_SESSION_JS,
    FLUX_PRESET,
    LITE_PRESET,
    QUALITY_PRESET,
    _cancelled_response_values,
    _generation_args_for_view,
    _lock_choices,
    _preset_selection,
    _response_values,
    _seed_output_state,
    _variation_choices,
)
from aisketcher.studio_app.i18n import text
from aisketcher.studio_app.runtime import AppResponse, AppState


def test_studio_package_is_import_safe_without_eager_gradio_import() -> None:
    # Importing aisketcher.studio_app above must work whether the optional demo
    # dependency is present or not.  The actual builder performs the lazy import.
    assert callable(build_app)
    assert AppController.__module__.endswith("studio_app.runtime")


def test_module_entrypoint_delegates_to_the_configured_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...] | None] = []

    def fake_cli_main(argv: tuple[str, ...] | None = None) -> int:
        calls.append(argv)
        return 0

    monkeypatch.setattr(cli_module, "main", fake_cli_main)

    assert app_module.main() == 0
    assert calls == [("studio",)]


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
        AUTO_MODEL,
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
        FLUX_PRESET,
        4,
        "scout",
        "",
        False,
        4,
        1.0,
        ("structure",),
    )


def test_simple_legacy_selection_keeps_the_legacy_canny_recipe() -> None:
    state = AppState.new("ko").replace(view="simple").payload()
    values = (
        state,
        "source.png",
        "종이 공예",
        "graphic_design",
        "balanced",
        LITE_PRESET,
        FLUX_PRESET,
        8,
        "explicit",
        "1,2,3,4,5,6,7,8",
        False,
        4,
        1.0,
        (),
    )

    resolved = _generation_args_for_view(values)

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
        AUTO_MODEL,
        QUALITY_PRESET,
        8,
        "explicit",
        "1,2,3,4,5,6,7,8",
        True,
        40,
        7.0,
        ("structure",),
    )

    assert _generation_args_for_view(values) == values[:5] + values[6:]


def test_advanced_flux_generation_uses_the_validated_locked_recipe() -> None:
    state = AppState.new("en").replace(view="advanced").payload()
    values = (
        state,
        "source.png",
        "Paper art",
        "graphic_design",
        "faithful",
        AUTO_MODEL,
        FLUX_PRESET,
        4,
        "scout",
        "",
        False,
        50,
        12.0,
        ("structure",),
    )

    resolved = _generation_args_for_view(values)

    assert resolved[5] == FLUX_PRESET
    assert resolved[10:12] == (4, 1.0)


def test_locked_seed_forces_one_output_and_restores_previous_selection() -> None:
    locked = _seed_output_state("scout", "locked", 8, 4)
    restored = _seed_output_state("locked", "scout", locked[0], locked[2])

    assert locked == (1, False, 8)
    assert restored == (8, True, 8)


def test_preset_selection_is_canonical_and_localized() -> None:
    selected, plan = _preset_selection("ko", QUALITY_PRESET)

    assert selected == QUALITY_PRESET
    assert plan.startswith("**SDXL Canny Quality · 레거시**")
    auto_selected, auto_plan = _preset_selection("en", AUTO_MODEL)
    assert auto_selected == FLUX_PRESET
    assert "recommended" in auto_plan
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
    assert _response_values(gr, response)[6:] == ({}, {}, {})


def test_cancelled_response_preserves_visible_outputs() -> None:
    gr = SimpleNamespace(update=lambda **values: values)
    state = AppState.new("ko").payload()

    values = _cancelled_response_values(gr, state)

    assert values[0] == state
    assert values[1:5] == ({}, {}, {}, {})
    assert values[5].startswith("사용자가 작업을 중지했습니다.")
    assert values[6:] == ({}, {}, {})


def test_guided_response_populates_manifest_recipe_controls() -> None:
    gr = SimpleNamespace(update=lambda **values: values)
    response = AppResponse(
        state={"session_id": "test"},
        source="source.png",
        selected="candidate-1.png",
        gallery=(("candidate-1.png", "Direction 1"),),
        recommendation="Recorded direction",
        status="Ready",
        prompt="Alphabet A as an intricate fantasy kingdom.",
        profile="graphic_design",
        structure="balanced",
    )

    assert _response_values(gr, response)[6:] == (
        {"value": "Alphabet A as an intricate fantasy kingdom."},
        {"value": "graphic_design"},
        {"value": "balanced"},
    )


def test_recipe_sync_clears_stale_controls_when_manifest_has_no_recipe() -> None:
    gr = SimpleNamespace(update=lambda **values: values)
    response = AppResponse(
        state={"session_id": "test"},
        source="source.png",
        selected="candidate-1.png",
        gallery=(("candidate-1.png", "Direction 1"),),
        recommendation="Recorded direction",
        status="Ready",
        sync_recipe_controls=True,
    )

    assert _response_values(gr, response)[6:] == (
        {"value": ""},
        {"value": "graphic_design"},
        {"value": "balanced"},
    )


@pytest.mark.skipif(
    importlib.util.find_spec("gradio") is None, reason="Gradio demo extra is absent"
)
def test_build_app_has_private_serial_launch_defaults(tmp_path: Path) -> None:
    app = build_app(AppController(workspace_root=tmp_path))

    assert app._studio_launch_kwargs["server_name"] == "127.0.0.1"
    assert app._studio_launch_kwargs["share"] is False
    assert app._studio_launch_kwargs["max_file_size"] == 20 * 1024 * 1024
    assert "js" not in app._studio_launch_kwargs
    assert "head" not in app._studio_launch_kwargs
    recovery = next(
        component
        for component in app.config["components"]
        if component.get("props", {}).get("elem_id") == "connection-recovery-host"
    )
    assert "Could not parse server response" in recovery["props"]["js_on_load"]
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
    model_choice_id = next(
        component_id
        for component_id, component in components.items()
        if component.get("props", {}).get("elem_id") == "model-choice"
    )
    preset_id = next(
        component_id
        for component_id, component in components.items()
        if component["type"] == "dropdown" and component.get("props", {}).get("label") == "Preset"
    )
    prepare_dependency = next(
        dependency
        for dependency in app.config["dependencies"]
        if preset_id in dependency["inputs"]
        and any(
            components.get(output, {}).get("props", {}).get("elem_id") == "model-status"
            for output in dependency["outputs"]
        )
    )

    assert preset_id in prepare_dependency["inputs"]
    assert model_choice_id not in prepare_dependency["inputs"]

    state = AppState.new("ko").payload()
    model_update = app._studio_sync_model_preset(state, QUALITY_PRESET)
    generation_update = app._studio_sync_generation_preset(state, LITE_PRESET)
    assert model_update[0]["value"] == QUALITY_PRESET
    assert model_update[1]["value"].startswith("**SDXL Canny Quality · 레거시**")
    assert model_update[4]["value"] is True
    assert model_update[5]["value"] == 30
    assert generation_update[0]["value"] == LITE_PRESET
    assert generation_update[1]["value"].startswith("**SDXL Canny Lite · 레거시**")

    reset = app._studio_clear_overrides(state)
    assert reset[1] == QUALITY_PRESET
    assert reset[4]["value"] == 8
    assert reset[10] == QUALITY_PRESET
    assert reset[11]["value"].startswith("**SDXL Canny Quality · 레거시**")


@pytest.mark.skipif(
    importlib.util.find_spec("gradio") is None, reason="Gradio demo extra is absent"
)
def test_simple_model_selector_defaults_to_auto_and_confirms_with_button(
    tmp_path: Path,
) -> None:
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
        workspace_root=tmp_path,
        model_installer=installer,
        prompt_translator=Translator(),
    )
    app = build_app(controller)
    components = {component["id"]: component for component in app.config["components"]}
    by_elem_id = {
        component.get("props", {}).get("elem_id"): component
        for component in components.values()
        if component.get("props", {}).get("elem_id")
    }

    simple_model = by_elem_id["simple-model-choice"]
    assert simple_model["props"]["value"] == AUTO_MODEL
    assert [value for _, value in simple_model["props"]["choices"]] == [
        AUTO_MODEL,
        FLUX_PRESET,
        LITE_PRESET,
        QUALITY_PRESET,
    ]
    assert "about 16.2 GB" in by_elem_id["simple-model-plan"]["props"]["value"]
    assert "Current installer plan" not in by_elem_id["simple-model-plan"]["props"]["value"]
    assert by_elem_id["model-choice"]["props"]["choices"][0][1] == FLUX_PRESET
    assert by_elem_id["model-choice"]["props"]["value"] == FLUX_PRESET

    simple_model_id = simple_model["id"]
    simple_status_id = by_elem_id["simple-model-status"]["id"]
    prepare_dependency = next(
        dependency
        for dependency in app.config["dependencies"]
        if simple_model_id in dependency["inputs"] and dependency["outputs"] == [simple_status_id]
    )
    input_elem_ids = {
        components.get(component_id, {}).get("props", {}).get("elem_id")
        for component_id in prepare_dependency["inputs"]
    }
    assert "simple-model-choice" in input_elem_ids
    assert "model-choice" not in input_elem_ids

    state = AppState.new("ko").payload()
    plan_update, status_update = app._studio_sync_simple_model(state, AUTO_MODEL)
    assert "이미지 모델이 없으면 약 16.2 GB" in plan_update["value"]
    assert "한→영 도우미가 없으면 약 1.9 GB" in plan_update["value"]
    assert status_update == ""
    operation_id = controller.start_operation(state)
    assert (
        app._studio_prepare_simple_model(operation_id, state, AUTO_MODEL)
        == "로컬 모델 준비를 마쳤습니다."
    )
    assert installed[0][0] == FLUX_PRESET
    assert installed[0][1]["confirm"] is True
    assert len(translator_prepares) == 1
    assert translator_prepares[0]["confirm"] is True


@pytest.mark.skipif(
    importlib.util.find_spec("gradio") is None, reason="Gradio demo extra is absent"
)
def test_model_descriptions_render_current_installer_plan_in_every_callback(
    tmp_path: Path,
) -> None:
    class PlannedInstaller:
        def __init__(self) -> None:
            self.planned: list[str] = []
            self.verify_cache_values: list[bool] = []

        def plan_install(
            self,
            preset: str,
            *,
            verify_cache: bool = True,
        ) -> InstallPlan:
            self.planned.append(preset)
            self.verify_cache_values.append(verify_cache)
            cache = tmp_path / "managed cache"
            return InstallPlan(
                preset=preset,
                label=f"Plan for {preset}",
                items=(
                    InstallItem(
                        repo_id="black-forest-labs/FLUX.2-klein-4B",
                        revision="e7b7dc27f91deacad38e78976d1f2b499d76a294",
                        role="base",
                        destination=cache / f"{preset}-base",
                        installed=True,
                        allow_patterns=("*.safetensors",),
                        estimated_bytes=1_500_000_000,
                    ),
                    InstallItem(
                        repo_id="example/control-model",
                        revision="b" * 40,
                        role="control",
                        destination=cache / f"{preset}-control",
                        installed=False,
                        allow_patterns=("*.safetensors",),
                        estimated_bytes=250_000_000,
                    ),
                ),
                estimated_bytes=1_750_000_000,
                cache_dir=cache,
                license_notice="Review the pinned Example Model License.",
            )

        def install(self, preset: str, **kwargs: Any) -> None:
            del preset, kwargs

    installer = PlannedInstaller()
    controller = AppController(workspace_root=tmp_path, model_installer=installer)
    app = build_app(controller, default_preset=QUALITY_PRESET)
    by_elem_id = {
        component.get("props", {}).get("elem_id"): component.get("props", {})
        for component in app.config["components"]
        if component.get("props", {}).get("elem_id")
    }

    initial_simple = by_elem_id["simple-model-plan"]["value"]
    initial_advanced = by_elem_id["model-plan"]["value"]
    for rendered, preset in (
        (initial_simple, FLUX_PRESET),
        (initial_advanced, QUALITY_PRESET),
    ):
        assert '<details class="installer-plan-details">' in rendered
        assert "<details" in rendered and "<details open" not in rendered
        assert "Current installer plan" in rendered
        assert "Pinned Korean→English helper" in rendered
        assert "<code>facebook/m2m100_418M</code>" in rendered
        assert "<code>55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636</code>" in rendered
        assert "<strong>Transfer if missing:</strong> 1.9 GB" in rendered
        assert (
            "<strong>Upstream license:</strong> "
            '<a href="https://huggingface.co/facebook/m2m100_418M" '
            'target="_blank" rel="noreferrer">MIT</a>' in rendered
        )
        assert "Default Hugging Face cache" in rendered
        assert f"<code>{preset}</code>" in rendered
        assert (
            "<strong>Verified in this Studio process</strong> · "
            "<code>black-forest-labs/FLUX.2-klein-4B</code>" in rendered
        )
        assert (
            "<strong>Not yet verified · download if absent</strong> · "
            "<code>example/control-model</code>" in rendered
        )
        assert (
            "immutable revision <code>e7b7dc27f91deacad38e78976d1f2b499d76a294</code>" in rendered
        )
        assert f"immutable revision <code>{'b' * 40}</code>" in rendered
        assert "<strong>Maximum transfer if cache is absent:</strong> 250.0 MB" in rendered
        assert f"<code>{tmp_path / 'managed cache'}</code>" in rendered
        assert "Review the pinned Example Model License." in rendered
        assert (
            "upstream license "
            '<a href="https://www.apache.org/licenses/LICENSE-2.0" '
            'target="_blank" rel="noreferrer">apache-2.0</a>' in rendered
        )

    state = AppState.new("en").payload()
    simple_update, _ = app._studio_sync_simple_model(state, AUTO_MODEL)
    generation_update = app._studio_sync_generation_preset(state, LITE_PRESET)
    flux_generation_update = app._studio_sync_generation_preset(state, FLUX_PRESET)
    model_update = app._studio_sync_model_preset(state, QUALITY_PRESET)
    reset_update = app._studio_clear_overrides(state)
    localized = app._studio_localize(
        state,
        "ko",
        AUTO_MODEL,
        QUALITY_PRESET,
        "scout",
    )

    assert f"<code>{FLUX_PRESET}</code>" in simple_update["value"]
    assert f"<code>{LITE_PRESET}</code>" in generation_update[1]["value"]
    assert generation_update[5]["interactive"] is True
    assert generation_update[6]["interactive"] is True
    assert flux_generation_update[5]["value"] == 4
    assert flux_generation_update[5]["interactive"] is False
    assert flux_generation_update[6]["value"] == 1.0
    assert flux_generation_update[6]["interactive"] is False
    assert f"<code>{QUALITY_PRESET}</code>" in model_update[1]["value"]
    assert f"<code>{QUALITY_PRESET}</code>" in reset_update[11]["value"]
    assert "현재 설치 계획" in localized[9]["value"]
    assert (
        "<strong>현재 Studio 프로세스에서 검증됨</strong> · "
        "<code>black-forest-labs/FLUX.2-klein-4B</code>" in localized[9]["value"]
    )
    assert (
        "<strong>아직 미검증 · 없으면 다운로드</strong> · "
        "<code>example/control-model</code>" in localized[37]["value"]
    )
    assert localized[13]["interactive"] is True
    assert f"<code>{FLUX_PRESET}</code>" in localized[9]["value"]
    assert f"<code>{QUALITY_PRESET}</code>" in localized[37]["value"]
    assert set(installer.planned) >= {FLUX_PRESET, LITE_PRESET, QUALITY_PRESET}
    assert installer.verify_cache_values
    assert not any(installer.verify_cache_values)


@pytest.mark.skipif(
    importlib.util.find_spec("gradio") is None, reason="Gradio demo extra is absent"
)
def test_gallery_and_refinement_layer_use_the_recorded_ux_contract(tmp_path: Path) -> None:
    controller = AppController(workspace_root=tmp_path)
    app = build_app(controller)
    by_elem_id = {
        component.get("props", {}).get("elem_id"): component.get("props", {})
        for component in app.config["components"]
        if component.get("props", {}).get("elem_id")
    }

    gallery = by_elem_id["result-gallery"]
    assert gallery["columns"] == 4
    assert gallery.get("rows") is None
    assert gallery.get("height") is None
    assert gallery["object_fit"] == "contain"
    assert by_elem_id["refine-composer"]["visible"] is False
    assert by_elem_id["guided-refine-overlay"]["visible"] is False
    assert by_elem_id["stop-action"]["visible"] is False
    assert by_elem_id["simple-model-stop-action"]["visible"] is False
    assert by_elem_id["model-stop-action"]["visible"] is False
    assert by_elem_id["guided-action"]["interactive"] is False
    assert by_elem_id["steps-control"]["value"] == 4
    assert by_elem_id["steps-control"]["interactive"] is False
    assert "locked to 4 steps" in by_elem_id["steps-control"]["info"]
    assert by_elem_id["guidance-control"]["value"] == 1.0
    assert by_elem_id["guidance-control"]["interactive"] is False
    assert "locked to CFG 1" in by_elem_id["guidance-control"]["info"]
    assert "connection-recovery-layer" in by_elem_id["connection-recovery-host"]["value"]
    assert "This Studio session has ended" in by_elem_id["connection-recovery-host"]["value"]

    guided = controller.open_guided_sample(controller.initial_state("ko"))
    composer_update, overlay_update, status_update = app._studio_open_refinement(guided.state)
    assert composer_update["visible"] is False
    assert overlay_update["visible"] is True
    assert set(status_update) <= {"__type__"}

    restored = app._studio_restore_gallery_after_preview(guided.state)
    assert len(restored["value"]) == 4
    assert "selected_index" not in restored

    gallery_id = next(
        component["id"]
        for component in app.config["components"]
        if component.get("props", {}).get("elem_id") == "result-gallery"
    )
    assert any(
        (gallery_id, "preview_close") in dependency["targets"]
        for dependency in app.config["dependencies"]
    )


@pytest.mark.skipif(
    importlib.util.find_spec("gradio") is None, reason="Gradio demo extra is absent"
)
def test_browser_refresh_reconnects_to_the_same_running_session(tmp_path: Path) -> None:
    controller = AppController(workspace_root=tmp_path)
    app = build_app(controller)
    browser_state = next(
        component for component in app.config["components"] if component["type"] == "browserstate"
    )
    timer = next(
        component for component in app.config["components"] if component["type"] == "timer"
    )
    assert browser_state["props"]["storage_key"] == BROWSER_SESSION_STORAGE_KEY
    tab_bind = next(
        dependency
        for dependency in app.config["dependencies"]
        if dependency.get("js") == BROWSER_TAB_SESSION_JS
    )
    assert tab_bind["queue"] is False
    assert "sessionStorage" in tab_bind["js"]
    assert 'navigation !== "reload"' in tab_bind["js"]
    assert timer["props"]["value"] == 1.0
    assert timer["props"]["active"] is True

    state = controller.initial_state("ko")
    operation_event = controller.claim_operation(state)
    recovered = app._studio_recover_browser_session(state, True)

    assert recovered[0] == state
    assert recovered[5].startswith("이 브라우저 세션")
    assert recovered[9]["visible"] is True
    assert recovered[10]["interactive"] is False
    controller.cancel_operation(state)
    stopping = app._studio_recover_browser_session(state, False)
    assert stopping[5].startswith("다시 연결한 작업")
    assert stopping[9]["visible"] is True
    assert stopping[9]["interactive"] is False
    controller.finish_operation(state, operation_event)

    idle = app._studio_recover_browser_session(state, False, stopping[5])
    assert idle[5]["value"] == text("ko", "ready")

    preserved = app._studio_recover_browser_session(state, False, "Keep this status")
    assert "value" not in preserved[5]

    first_tab = app._studio_bind_browser_tab({**state, "session_id": "a" * 32})
    second_tab = app._studio_bind_browser_tab({**state, "session_id": "b" * 32})
    assert first_tab["session_id"] != second_tab["session_id"]
    assert app._studio_bind_browser_tab(first_tab)["session_id"] == first_tab["session_id"]


@pytest.mark.skipif(
    importlib.util.find_spec("gradio") is None, reason="Gradio demo extra is absent"
)
def test_stop_keeps_retry_controls_disabled_until_the_backend_is_idle(
    tmp_path: Path,
) -> None:
    controller = AppController(workspace_root=tmp_path)
    app = build_app(controller)
    state = controller.initial_state("en")
    ticket = controller.start_operation(state)
    operation_event = controller.claim_operation(state, ticket)

    stopped = app._studio_stop_generation(state)

    assert controller.operation_state(state) == "stopping"
    assert stopped[1] == {
        "visible": True,
        "interactive": False,
        "__type__": "update",
    }
    assert all(update["interactive"] is False for update in stopped[2:])

    controller.finish_operation(state, operation_event)
    recovered = app._studio_recover_browser_session(state, False, stopped[0])
    assert recovered[9]["visible"] is False
    assert all(recovered[index]["interactive"] is True for index in range(10, 15))


@pytest.mark.skipif(
    importlib.util.find_spec("gradio") is None, reason="Gradio demo extra is absent"
)
def test_model_stop_and_reconnect_keep_prepare_disabled_until_install_is_idle(
    tmp_path: Path,
) -> None:
    controller = AppController(workspace_root=tmp_path)
    app = build_app(controller)
    state = controller.initial_state("ko")
    started = app._studio_begin_model_download(state)
    ticket = started[0]
    operation_event = controller.claim_operation(state, ticket)

    reconnected = app._studio_recover_browser_session(state, True)
    assert reconnected[9]["visible"] is False
    assert all(reconnected[index]["interactive"] is False for index in range(10, 15))
    assert reconnected[15]["visible"] is True
    assert reconnected[15]["interactive"] is True
    assert reconnected[16]["interactive"] is False
    assert reconnected[17]["visible"] is True
    assert reconnected[17]["interactive"] is True
    assert reconnected[18]["interactive"] is False

    stopped = app._studio_stop_model_download(state)
    assert controller.operation_state(state) == "stopping"
    assert stopped[1]["visible"] is True
    assert stopped[1]["interactive"] is False
    assert stopped[2]["interactive"] is False
    stopping_reconnect = app._studio_recover_browser_session(state, True)
    assert stopping_reconnect[15]["interactive"] is False
    assert stopping_reconnect[16]["interactive"] is False
    assert stopping_reconnect[17]["interactive"] is False
    assert stopping_reconnect[18]["interactive"] is False

    controller.finish_operation(state, operation_event)
    idle = app._studio_recover_browser_session(state, False, stopped[0])
    assert idle[15]["visible"] is False
    assert idle[16]["interactive"] is True
    assert idle[17]["visible"] is False
    assert idle[18]["interactive"] is True


@pytest.mark.skipif(
    importlib.util.find_spec("gradio") is None, reason="Gradio demo extra is absent"
)
def test_runtime_localization_updates_canny_info_and_model_plan(tmp_path: Path) -> None:
    controller = AppController(workspace_root=tmp_path)
    app = build_app(controller)

    updates = app._studio_localize(
        controller.initial_state("en"),
        "ko",
        AUTO_MODEL,
        QUALITY_PRESET,
        "scout",
    )

    assert updates[27]["info"] == "현재 SDXL 프리셋에 필요한 설정입니다."
    assert updates[8]["value"] == AUTO_MODEL
    assert "이미지 모델이 없으면 약 16.2 GB" in updates[9]["value"]
    assert "한→영 도우미가 없으면 약 1.9 GB" in updates[9]["value"]
    assert updates[23]["value"] == QUALITY_PRESET
    assert updates[36]["value"] == QUALITY_PRESET
    assert updates[37]["value"].startswith("**SDXL Canny Quality · 레거시**")
