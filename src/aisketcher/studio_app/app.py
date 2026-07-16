"""Packaged Gradio 6 Studio for AIsketcher.

Run after installing the ``demo`` extra with::

    python -m aisketcher.studio_app.app

Gradio is imported inside :func:`build_app`, so importing the example helpers
does not pull the web or local-model dependency sets into the base SDK.
"""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..errors import AIsketcherError
from .i18n import navigation_choices, normalize_language, structure_choices, text
from .runtime import (
    MAX_UPLOAD_BYTES,
    AppController,
    AppResponse,
    AppState,
    StudioAppError,
)

CSS_PATH = Path(__file__).with_name("styles.css")
STUDIO_CSS = CSS_PATH.read_text(encoding="utf-8")

LITE_PRESET = "sdxl-canny-lite@1"
QUALITY_PRESET = "sdxl-canny@1"
DEFAULT_PROFILE = "graphic_design"
DEFAULT_LOCKS = ("structure",)
OUTPUT_CHOICES = (("1", 1), ("4", 4), ("8", 8))
LOCKED_OUTPUT_CHOICES = (("1", 1),)


def _heading(language: str) -> str:
    title = html.escape(text(language, "headline")).replace("\n", "<br>")
    subtitle = html.escape(text(language, "subhead"))
    return f"<h1>{title}</h1><p>{subtitle}</p>"


def _directions_heading(language: str) -> str:
    title = html.escape(text(language, "directions"))
    subtitle = html.escape(text(language, "directions_help"))
    return f"<h2>{title}</h2><p>{subtitle}</p>"


def _profile_choices(language: str) -> list[tuple[str, str]]:
    if normalize_language(language) == "ko":
        return [
            ("프로덕트 디자인", "product_design"),
            ("그래픽 디자인", "graphic_design"),
            ("스케치·일러스트", "sketch"),
        ]
    return [
        ("Product design", "product_design"),
        ("Graphic design", "graphic_design"),
        ("Sketch & illustration", "sketch"),
    ]


def _seed_choices(language: str) -> list[tuple[str, str]]:
    return [
        (text(language, "auto_seeds"), "scout"),
        (text(language, "locked_seed"), "locked"),
        (text(language, "custom_seeds"), "explicit"),
    ]


def _lock_choices(language: str) -> list[tuple[str, str]]:
    return [(text(language, "lock_structure"), "structure")]


def _model_choices(language: str) -> list[tuple[str, str]]:
    return [
        (text(language, "model_lite"), LITE_PRESET),
        (text(language, "model_quality"), QUALITY_PRESET),
    ]


def _variation_choices(language: str) -> list[tuple[str, str]]:
    if normalize_language(language) == "ko":
        return [("미세하게", "subtle"), ("적당하게", "balanced"), ("과감하게", "bold")]
    return [("Subtle", "subtle"), ("Moderate", "balanced"), ("Bold", "bold")]


def _model_plan(language: str, preset: str) -> str:
    key = "model_plan_quality" if preset == QUALITY_PRESET else "model_plan_lite"
    return text(language, key)


def _preset_selection(language: str, preset: str) -> tuple[str, str]:
    """Return the canonical preset and its localized download plan."""

    if preset not in {LITE_PRESET, QUALITY_PRESET}:
        raise ValueError("preset must be a packaged AIsketcher preset")
    return preset, _model_plan(language, preset)


def _seed_output_state(
    previous_mode: str,
    selected_mode: str,
    current_output: int,
    remembered_output: int,
) -> tuple[int, bool, int]:
    """Resolve output UI state while preserving the pre-lock selection."""

    valid_modes = {"scout", "locked", "explicit"}
    if previous_mode not in valid_modes or selected_mode not in valid_modes:
        raise ValueError("seed mode must be scout, locked, or explicit")
    if current_output not in {1, 4, 8} or remembered_output not in {1, 4, 8}:
        raise ValueError("output count must be 1, 4, or 8")

    if selected_mode == "locked":
        remembered = current_output if previous_mode != "locked" else remembered_output
        return 1, False, remembered
    if previous_mode == "locked":
        return remembered_output, True, remembered_output
    return current_output, True, current_output


def _generation_args_for_view(values: Sequence[Any]) -> tuple[Any, ...]:
    """Apply the fixed Simple recipe without mutating Advanced controls."""

    if len(values) != 13:
        raise ValueError("generation callback requires 13 values")
    resolved = list(values)
    state = AppState.from_payload(resolved[0])
    if state.view != "advanced":
        resolved[5] = LITE_PRESET
        resolved[6] = 4
        resolved[7] = "scout"
        resolved[8] = ""
        resolved[9] = True
        resolved[10] = 30
        resolved[11] = 5.0
        resolved[12] = DEFAULT_LOCKS
    return tuple(resolved)


def _response_values(gr: Any, response: AppResponse) -> tuple[Any, ...]:
    sync_recipe = response.sync_recipe_controls or any(
        value is not None
        for value in (response.prompt, response.profile, response.structure)
    )
    return (
        response.state,
        response.source,
        response.selected,
        gr.update(value=list(response.gallery)),
        response.recommendation,
        response.status,
        gr.update(value=response.prompt or "") if sync_recipe else gr.update(),
        gr.update(value=response.profile or DEFAULT_PROFILE) if sync_recipe else gr.update(),
        gr.update(value=response.structure or "balanced") if sync_recipe else gr.update(),
    )


def build_app(
    controller: AppController | None = None,
    *,
    language: str = "en",
    default_preset: str = LITE_PRESET,
    default_output_count: int = 4,
    default_seed_mode: str = "scout",
    default_seed: int | None = None,
) -> Any:
    """Build the Simple-first Gradio app.

    ``controller`` is injectable so documentation, tests, and downstream users
    can provide a FakeStudio or a custom backend without changing the UI.
    """

    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise AIsketcherError(
            "The Studio example requires Gradio. Install AIsketcher with the 'demo' extra."
        ) from exc

    language = normalize_language(language)
    if default_preset not in {LITE_PRESET, QUALITY_PRESET}:
        raise ValueError("preset must be a packaged AIsketcher preset")
    if default_output_count not in {1, 4, 8}:
        raise ValueError("output_count must be 1, 4, or 8")
    if default_seed_mode not in {"scout", "locked", "explicit"}:
        raise ValueError("seed_mode must be scout, locked, or explicit")
    if default_seed is not None and not 0 <= default_seed <= (1 << 63) - 1:
        raise ValueError("seed must be a non-negative 63-bit integer")
    advanced_output_count = 1 if default_seed_mode == "locked" else default_output_count
    unlocked_output_default = (
        default_output_count if default_seed_mode != "locked" else 4
    )
    controller = controller or AppController()
    guided_available = controller.guided.available
    initial_status = text(language, "ready") if guided_available else text(language, "unavailable")

    theme = gr.themes.Base(
        primary_hue="blue",
        secondary_hue="red",
        neutral_hue="slate",
    ).set(
        body_background_fill="#ffffff",
        body_background_fill_dark="#ffffff",
        body_text_color="#0d203d",
        body_text_color_dark="#0d203d",
        block_background_fill="#ffffff",
        block_background_fill_dark="#ffffff",
        block_border_color="#dce3ed",
        block_border_color_dark="#dce3ed",
        button_primary_background_fill="#0f5cf5",
        button_primary_background_fill_hover="#084ed9",
        button_primary_text_color="#ffffff",
    )

    with gr.Blocks(title="AIsketcher v2 Studio", fill_width=True) as demo:
        app_state = gr.State(controller.initial_state(language), time_to_live=3600)
        active_seed_mode = gr.State(default_seed_mode)
        unlocked_output_count = gr.State(unlocked_output_default)

        with gr.Row(elem_id="studio-header", equal_height=True):
            gr.HTML(
                '<div id="studio-brand">AIsketcher <span class="version">v2</span></div>',
                container=False,
            )
            view_nav = gr.Radio(
                navigation_choices(language),
                value="simple",
                show_label=False,
                container=False,
                elem_id="view-nav",
                scale=2,
                min_width=320,
            )
            language_nav = gr.Radio(
                [("EN", "en"), ("한국어", "ko")],
                value=language,
                show_label=False,
                container=False,
                elem_id="language-nav",
                min_width=150,
            )

        with gr.Row(elem_id="workbench") as workbench:
            with gr.Column(scale=4, min_width=330, elem_id="control-rail"):
                heading = gr.HTML(_heading(language), elem_id="studio-heading", container=False)
                override_badge = gr.Markdown(
                    text(language, "overrides"),
                    visible=False,
                    elem_id="override-badge",
                )
                reset_overrides = gr.Button(
                    text(language, "clear"),
                    visible=False,
                    size="sm",
                    elem_id="reset-action",
                )
                sketch = gr.Image(
                    type="filepath",
                    format="png",
                    image_mode="RGB",
                    buttons=["fullscreen"],
                    label=text(language, "sketch"),
                    height=170,
                    interactive=False,
                    elem_id="sketch-input",
                    elem_classes="studio-field",
                )
                upload_button = gr.UploadButton(
                    text(language, "upload"),
                    type="filepath",
                    file_count="single",
                    file_types=["image"],
                    size="md",
                    elem_id="upload-action",
                )
                brief = gr.Textbox(
                    label=text(language, "brief"),
                    placeholder=text(language, "brief_placeholder"),
                    lines=3,
                    max_lines=4,
                    max_length=600,
                    elem_classes="studio-field",
                )
                profile = gr.Dropdown(
                    _profile_choices(language),
                    value=DEFAULT_PROFILE,
                    label=text(language, "profile"),
                    allow_custom_value=False,
                    filterable=False,
                    elem_classes="studio-field",
                )
                structure = gr.Radio(
                    structure_choices(language),
                    value="balanced",
                    label=text(language, "structure"),
                    elem_id="structure-control",
                    elem_classes="studio-field",
                )
                explore_button = gr.Button(
                    text(language, "explore"),
                    variant="primary",
                    elem_id="primary-action",
                )
                guided_button = gr.Button(
                    text(language, "guided"),
                    interactive=guided_available,
                    elem_id="guided-action",
                )
                status = gr.Markdown(initial_status, elem_id="app-status")

            with gr.Column(scale=8, min_width=560, elem_id="visual-workspace"):
                with gr.Row(elem_id="compare-row", equal_height=True):
                    source_preview = gr.Image(
                        label=text(language, "source"),
                        interactive=False,
                        buttons=["fullscreen"],
                        height=350,
                        elem_id="source-preview",
                    )
                    selected_preview = gr.Image(
                        label=text(language, "selected"),
                        interactive=False,
                        buttons=["fullscreen", "download"],
                        height=350,
                        elem_id="selected-preview",
                    )
                directions_heading = gr.HTML(
                    _directions_heading(language),
                    elem_id="directions-heading",
                    container=False,
                )
                gallery = gr.Gallery(
                    value=[],
                    columns=4,
                    rows=2,
                    height=200,
                    object_fit="cover",
                    allow_preview=True,
                    buttons=["fullscreen"],
                    show_label=False,
                    elem_id="result-gallery",
                )
                recommendation = gr.Markdown(
                    text(language, "empty"),
                    elem_id="recommendation",
                )
                with gr.Row(elem_id="result-actions"):
                    refine_button = gr.Button(
                        text(language, "refine"),
                        variant="primary",
                        elem_id="refine-action",
                    )
                    retry_button = gr.Button(text(language, "again"), elem_id="retry-action")
                    export_button = gr.Button(text(language, "export"), elem_id="export-action")
                tip = gr.Markdown(text(language, "tip"), elem_id="app-tip")
                export_file = gr.File(
                    label="Export",
                    interactive=False,
                    visible=False,
                    elem_id="export-file",
                )

            with gr.Column(
                scale=4,
                min_width=310,
                visible=False,
                elem_id="advanced-rail",
            ) as advanced_rail:
                advanced_title = gr.Markdown(
                    f"## {text(language, 'advanced')}", elem_id="advanced-title"
                )
                preset = gr.Dropdown(
                    [("Lite", LITE_PRESET), ("Quality", QUALITY_PRESET)],
                    value=default_preset,
                    label=text(language, "preset"),
                    allow_custom_value=False,
                    filterable=False,
                    elem_classes="studio-field",
                )
                seed_mode = gr.Radio(
                    _seed_choices(language),
                    value=default_seed_mode,
                    label=text(language, "seed_plan"),
                    elem_id="seed-control",
                    elem_classes="studio-field",
                )
                custom_seeds = gr.Textbox(
                    value=str(default_seed) if default_seed is not None else "",
                    label=text(language, "custom_seeds"),
                    placeholder=(
                        text(language, "locked_seed_placeholder")
                        if default_seed_mode == "locked"
                        else text(language, "custom_seed_placeholder")
                    ),
                    visible=default_seed_mode != "scout",
                    elem_classes="studio-field",
                )
                output_count = gr.Radio(
                    LOCKED_OUTPUT_CHOICES
                    if default_seed_mode == "locked"
                    else OUTPUT_CHOICES,
                    value=advanced_output_count,
                    label=text(language, "outputs"),
                    interactive=default_seed_mode != "locked",
                    elem_id="output-control",
                    elem_classes="studio-field",
                )
                canny = gr.Checkbox(
                    True,
                    label=text(language, "canny"),
                    info=text(language, "canny_info"),
                    interactive=False,
                    elem_classes="studio-field",
                )
                steps = gr.Slider(
                    10,
                    50,
                    value=30,
                    step=1,
                    label=text(language, "steps"),
                    elem_classes="studio-field",
                )
                guidance = gr.Slider(
                    1.0,
                    12.0,
                    value=5.0,
                    step=0.5,
                    label=text(language, "guidance"),
                    elem_classes="studio-field",
                )
                variation = gr.Radio(
                    _variation_choices(language),
                    value="subtle",
                    label=text(language, "variation"),
                    elem_id="variation-control",
                    elem_classes="studio-field",
                )
                locks = gr.CheckboxGroup(
                    _lock_choices(language),
                    value=list(DEFAULT_LOCKS),
                    label=text(language, "locks"),
                    elem_id="lock-control",
                    elem_classes="studio-field",
                )

                manifest_title = gr.Markdown(
                    f"### {text(language, 'manifest')}", elem_classes="advanced-section-title"
                )
                manifest_file = gr.UploadButton(
                    text(language, "manifest_upload"),
                    file_types=[".json", ".zip"],
                    type="filepath",
                    size="md",
                    elem_classes="studio-field",
                )
                replay_button = gr.Button(text(language, "replay"), elem_id="replay-action")

                model_title = gr.Markdown(
                    f"### {text(language, 'model_setup')}", elem_classes="advanced-section-title"
                )
                model_choice = gr.Radio(
                    _model_choices(language),
                    value=default_preset,
                    show_label=False,
                    elem_id="model-choice",
                    elem_classes="studio-field",
                )
                model_plan = gr.Markdown(
                    _model_plan(language, default_preset), elem_id="model-plan"
                )
                model_confirm = gr.Checkbox(
                    False,
                    label=text(language, "model_confirm"),
                    elem_classes="studio-field",
                )
                model_button = gr.Button(text(language, "model_prepare"), elem_id="model-action")
                model_status = gr.Markdown("", elem_id="model-status")

        with gr.Column(visible=False, elem_id="guide-panel") as guide_panel:
            guide_content = gr.Markdown(
                text(language, "guide_body"),
                elem_id="guide-content",
            )

        response_outputs = [
            app_state,
            source_preview,
            selected_preview,
            gallery,
            recommendation,
            status,
            brief,
            profile,
            structure,
        ]

        upload_button.upload(
            lambda path: path,
            inputs=[upload_button],
            outputs=[sketch],
            queue=False,
            show_progress="hidden",
        )

        def show_view(state_value: Mapping[str, Any], view: str) -> tuple[Any, ...]:
            state = AppState.from_payload(state_value).replace(view=view)
            return (
                state.payload(),
                gr.update(visible=view != "guide"),
                gr.update(visible=view == "guide"),
            )

        view_change = view_nav.change(
            show_view,
            inputs=[app_state, view_nav],
            outputs=[app_state, workbench, guide_panel],
            queue=False,
            show_progress="hidden",
        )
        # Apply the nested inspector visibility after its parent workbench has
        # been restored.  Updating both in one Gradio event can cause the parent
        # visibility patch to re-show a child that should remain hidden.
        view_change.then(
            lambda view: gr.update(visible=view == "advanced"),
            inputs=[view_nav],
            outputs=[advanced_rail],
            queue=False,
            show_progress="hidden",
        )

        def run_explore(*values: Any) -> tuple[Any, ...]:
            try:
                response = controller.explore(*_generation_args_for_view(values))
            except StudioAppError as exc:
                raise gr.Error(str(exc)) from exc
            return _response_values(gr, response)

        generation_inputs = [
            app_state,
            sketch,
            brief,
            profile,
            structure,
            preset,
            output_count,
            seed_mode,
            custom_seeds,
            canny,
            steps,
            guidance,
            locks,
        ]
        explore_button.click(
            run_explore,
            inputs=generation_inputs,
            outputs=response_outputs,
            concurrency_id="generation",
            concurrency_limit=1,
        )

        def open_guided(state_value: Mapping[str, Any]) -> tuple[Any, ...]:
            try:
                response = controller.open_guided_sample(state_value)
            except StudioAppError as exc:
                raise gr.Error(str(exc)) from exc
            return _response_values(gr, response)

        guided_button.click(
            open_guided,
            inputs=[app_state],
            outputs=response_outputs,
            concurrency_id="generation",
            concurrency_limit=1,
        )

        def choose_candidate(state_value: Mapping[str, Any], event: Any) -> tuple[Any, ...]:
            index = event.index
            if isinstance(index, (tuple, list)):
                index = index[0]
            try:
                next_state, image, detail, message = controller.select_candidate(
                    state_value, int(index)
                )
            except StudioAppError as exc:
                raise gr.Error(str(exc)) from exc
            return next_state, image, detail, message, gr.update(selected_index=int(index))

        # Gradio recognizes event payloads by their concrete annotation.  The
        # module uses postponed annotations and imports Gradio lazily, so attach
        # the runtime type explicitly after defining the callback.
        choose_candidate.__annotations__["event"] = gr.SelectData
        gallery.select(
            choose_candidate,
            inputs=[app_state],
            outputs=[app_state, selected_preview, recommendation, status, gallery],
            queue=False,
            show_progress="hidden",
        )

        def refine_study(
            state_value: Mapping[str, Any], strength: str, lock_values: Sequence[str]
        ) -> tuple[Any, ...]:
            try:
                response = controller.refine(state_value, strength, lock_values)
            except StudioAppError as exc:
                raise gr.Error(str(exc)) from exc
            return _response_values(gr, response)

        refine_button.click(
            refine_study,
            inputs=[app_state, variation, locks],
            outputs=response_outputs,
            concurrency_id="generation",
            concurrency_limit=1,
        )

        def retry_study(state_value: Mapping[str, Any]) -> tuple[Any, ...]:
            try:
                response = controller.try_again(state_value)
            except StudioAppError as exc:
                raise gr.Error(str(exc)) from exc
            return _response_values(gr, response)

        retry_button.click(
            retry_study,
            inputs=[app_state],
            outputs=response_outputs,
            concurrency_id="generation",
            concurrency_limit=1,
        )

        def export_study(state_value: Mapping[str, Any]) -> tuple[Any, str]:
            try:
                path, message = controller.export(state_value)
            except StudioAppError as exc:
                raise gr.Error(str(exc)) from exc
            return gr.update(value=path, visible=True), message

        export_button.click(
            export_study,
            inputs=[app_state],
            outputs=[export_file, status],
            concurrency_id="generation",
            concurrency_limit=1,
        )

        def replay_study(
            state_value: Mapping[str, Any], path: str | None, preset_value: str
        ) -> tuple[Any, ...]:
            try:
                response = controller.replay_manifest(state_value, path, preset_value)
            except StudioAppError as exc:
                raise gr.Error(str(exc)) from exc
            return _response_values(gr, response)

        replay_button.click(
            replay_study,
            inputs=[app_state, manifest_file, preset],
            outputs=response_outputs,
            concurrency_id="generation",
            concurrency_limit=1,
        )

        def mark_override(state_value: Mapping[str, Any]) -> tuple[Any, ...]:
            state = AppState.from_payload(state_value).replace(advanced_overrides=True)
            return (
                state.payload(),
                gr.update(value=text(state.language, "overrides"), visible=True),
                gr.update(visible=True),
            )

        advanced_controls: list[Any] = [
            preset,
            model_choice,
            seed_mode,
            custom_seeds,
            output_count,
            steps,
            guidance,
            variation,
            locks,
        ]
        for control in advanced_controls:
            control.input(
                mark_override,
                inputs=[app_state],
                outputs=[app_state, override_badge, reset_overrides],
                queue=False,
                show_progress="hidden",
            )

        def update_seed_input(
            state_value: Mapping[str, Any],
            mode: str,
            current_output: int,
            previous_mode: str,
            remembered_output: int,
        ) -> tuple[Any, ...]:
            lang = AppState.from_payload(state_value).language
            value, interactive, remembered = _seed_output_state(
                previous_mode,
                mode,
                int(current_output),
                int(remembered_output),
            )
            return (
                gr.update(
                    visible=mode != "scout",
                    placeholder=(
                        text(lang, "locked_seed_placeholder")
                        if mode == "locked"
                        else text(lang, "custom_seed_placeholder")
                    ),
                ),
                gr.update(
                    choices=LOCKED_OUTPUT_CHOICES if mode == "locked" else OUTPUT_CHOICES,
                    value=value,
                    interactive=interactive,
                ),
                remembered,
                mode,
            )

        seed_mode.input(
            update_seed_input,
            inputs=[
                app_state,
                seed_mode,
                output_count,
                active_seed_mode,
                unlocked_output_count,
            ],
            outputs=[
                custom_seeds,
                output_count,
                unlocked_output_count,
                active_seed_mode,
            ],
            queue=False,
            show_progress="hidden",
        )

        def clear_overrides(state_value: Mapping[str, Any]) -> tuple[Any, ...]:
            state = AppState.from_payload(state_value).replace(advanced_overrides=False)
            return (
                state.payload(),
                default_preset,
                default_seed_mode,
                gr.update(
                    value=str(default_seed) if default_seed is not None else "",
                    visible=default_seed_mode != "scout",
                    placeholder=(
                        text(state.language, "locked_seed_placeholder")
                        if default_seed_mode == "locked"
                        else text(state.language, "custom_seed_placeholder")
                    ),
                ),
                gr.update(
                    choices=(
                        LOCKED_OUTPUT_CHOICES
                        if default_seed_mode == "locked"
                        else OUTPUT_CHOICES
                    ),
                    value=advanced_output_count,
                    interactive=default_seed_mode != "locked",
                ),
                True,
                30,
                5.0,
                "subtle",
                list(DEFAULT_LOCKS),
                default_preset,
                gr.update(value=_model_plan(state.language, default_preset)),
                False,
                "",
                unlocked_output_default,
                default_seed_mode,
                gr.update(visible=False),
                gr.update(visible=False),
            )

        reset_overrides.click(
            clear_overrides,
            inputs=[app_state],
            outputs=[
                app_state,
                preset,
                seed_mode,
                custom_seeds,
                output_count,
                canny,
                steps,
                guidance,
                variation,
                locks,
                model_choice,
                model_plan,
                model_confirm,
                model_status,
                unlocked_output_count,
                active_seed_mode,
                override_badge,
                reset_overrides,
            ],
            queue=False,
            show_progress="hidden",
        )

        def sync_generation_preset(
            state_value: Mapping[str, Any], value: str
        ) -> tuple[Any, ...]:
            lang = AppState.from_payload(state_value).language
            selected, plan = _preset_selection(lang, value)
            return gr.update(value=selected), gr.update(value=plan), False, ""

        preset.input(
            sync_generation_preset,
            inputs=[app_state, preset],
            outputs=[model_choice, model_plan, model_confirm, model_status],
            queue=False,
            show_progress="hidden",
        )

        def sync_model_preset(
            state_value: Mapping[str, Any], value: str
        ) -> tuple[Any, ...]:
            lang = AppState.from_payload(state_value).language
            selected, plan = _preset_selection(lang, value)
            return gr.update(value=selected), gr.update(value=plan), False, ""

        model_choice.input(
            sync_model_preset,
            inputs=[app_state, model_choice],
            outputs=[preset, model_plan, model_confirm, model_status],
            queue=False,
            show_progress="hidden",
        )

        def prepare_model(
            state_value: Mapping[str, Any], preset_value: str, confirmed: bool
        ) -> str:
            state = AppState.from_payload(state_value)
            try:
                return controller.install_model(preset_value, confirmed, state.language)
            except StudioAppError as exc:
                raise gr.Error(str(exc)) from exc

        model_button.click(
            prepare_model,
            inputs=[app_state, preset, model_confirm],
            outputs=[model_status],
            concurrency_id="model-download",
            concurrency_limit=1,
        )

        localized_outputs = [
            app_state,
            view_nav,
            heading,
            sketch,
            upload_button,
            brief,
            profile,
            structure,
            explore_button,
            guided_button,
            source_preview,
            selected_preview,
            directions_heading,
            gallery,
            refine_button,
            retry_button,
            export_button,
            tip,
            advanced_title,
            preset,
            seed_mode,
            custom_seeds,
            output_count,
            canny,
            steps,
            guidance,
            variation,
            locks,
            manifest_title,
            manifest_file,
            replay_button,
            model_title,
            model_choice,
            model_plan,
            model_confirm,
            model_button,
            guide_content,
            reset_overrides,
            override_badge,
            status,
            recommendation,
        ]

        def localize(
            state_value: Mapping[str, Any],
            selected_language: str,
            selected_preset: str,
            selected_seed_mode: str = "scout",
        ) -> tuple[Any, ...]:
            lang = normalize_language(selected_language)
            state = AppState.from_payload(state_value).replace(language=lang)
            active = controller.localize_active_run(state, lang)
            if active is not None:
                state = AppState.from_payload(active.state)
                gallery_update = gr.update(value=list(active.gallery))
                status_update = gr.update(value=active.status)
                recommendation_update = gr.update(value=active.recommendation)
            else:
                gallery_update = gr.update()
                status_update = gr.update(
                    value=text(lang, "ready") if guided_available else text(lang, "unavailable")
                )
                recommendation_update = gr.update(value=text(lang, "empty"))
            return (
                state.payload(),
                gr.update(choices=navigation_choices(lang)),
                gr.update(value=_heading(lang)),
                gr.update(label=text(lang, "sketch")),
                gr.update(label=text(lang, "upload")),
                gr.update(label=text(lang, "brief"), placeholder=text(lang, "brief_placeholder")),
                gr.update(label=text(lang, "profile"), choices=_profile_choices(lang)),
                gr.update(label=text(lang, "structure"), choices=structure_choices(lang)),
                gr.update(value=text(lang, "explore")),
                gr.update(value=text(lang, "guided")),
                gr.update(label=text(lang, "source")),
                gr.update(label=text(lang, "selected")),
                gr.update(value=_directions_heading(lang)),
                gallery_update,
                gr.update(value=text(lang, "refine")),
                gr.update(value=text(lang, "again")),
                gr.update(value=text(lang, "export")),
                gr.update(value=text(lang, "tip")),
                gr.update(value=f"## {text(lang, 'advanced')}"),
                gr.update(label=text(lang, "preset"), value=selected_preset),
                gr.update(label=text(lang, "seed_plan"), choices=_seed_choices(lang)),
                gr.update(
                    label=text(lang, "custom_seeds"),
                    placeholder=(
                        text(lang, "locked_seed_placeholder")
                        if selected_seed_mode == "locked"
                        else text(lang, "custom_seed_placeholder")
                    ),
                ),
                gr.update(label=text(lang, "outputs")),
                gr.update(label=text(lang, "canny"), info=text(lang, "canny_info")),
                gr.update(label=text(lang, "steps")),
                gr.update(label=text(lang, "guidance")),
                gr.update(label=text(lang, "variation"), choices=_variation_choices(lang)),
                gr.update(label=text(lang, "locks"), choices=_lock_choices(lang)),
                gr.update(value=f"### {text(lang, 'manifest')}"),
                gr.update(label=text(lang, "manifest_upload")),
                gr.update(value=text(lang, "replay")),
                gr.update(value=f"### {text(lang, 'model_setup')}"),
                gr.update(choices=_model_choices(lang), value=selected_preset),
                gr.update(value=_model_plan(lang, selected_preset)),
                gr.update(label=text(lang, "model_confirm")),
                gr.update(value=text(lang, "model_prepare")),
                gr.update(value=text(lang, "guide_body")),
                gr.update(value=text(lang, "clear")),
                gr.update(value=text(lang, "overrides")),
                status_update,
                recommendation_update,
            )

        language_nav.change(
            localize,
            inputs=[app_state, language_nav, preset, seed_mode],
            outputs=localized_outputs,
            queue=False,
            show_progress="hidden",
        )

    demo.queue(max_size=16, default_concurrency_limit=1)
    # Gradio 6 moved theme/css to launch().  Storing both on the Blocks object
    # preserves a conventional build_app() return value while main() applies the
    # exact launch configuration without deprecated constructor arguments.
    demo._studio_theme = theme
    demo._studio_css = STUDIO_CSS
    demo._studio_controller = controller
    demo._studio_clear_overrides = clear_overrides
    demo._studio_localize = localize
    demo._studio_update_seed_input = update_seed_input
    demo._studio_sync_generation_preset = sync_generation_preset
    demo._studio_sync_model_preset = sync_model_preset
    demo._studio_launch_kwargs = {
        "server_name": "127.0.0.1",
        "share": False,
        "max_file_size": MAX_UPLOAD_BYTES,
        "allowed_paths": [str(controller.workspace_root), str(controller.guided.root.resolve())],
        "footer_links": [],
        "show_error": False,
        "theme": theme,
        "css": STUDIO_CSS,
    }
    return demo


def main() -> None:
    """Launch privately on localhost with one queued generation at a time."""

    demo = build_app()
    demo.launch(**demo._studio_launch_kwargs)


if __name__ == "__main__":
    main()
