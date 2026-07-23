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
from ..prompt_normalization import (
    MARIAN_KO_EN_DOWNLOAD_BYTES,
    MARIAN_KO_EN_MODEL_ID,
    MARIAN_KO_EN_REVISION,
)
from .i18n import navigation_choices, normalize_language, structure_choices, text
from .runtime import (
    MAX_UPLOAD_BYTES,
    AppController,
    AppResponse,
    AppState,
    StudioAppError,
    StudioJobCancelled,
)

CSS_PATH = Path(__file__).with_name("styles.css")
STUDIO_CSS = CSS_PATH.read_text(encoding="utf-8")
STUDIO_JS = """
(() => {
  if (window.__aisketcherConnectionRecoveryInstalled) return;
  window.__aisketcherConnectionRecoveryInstalled = true;

  const showRecovery = (message) => {
    const normalized = String(message || "");
    const parseFailure =
      normalized.includes("Could not parse server response") ||
      (normalized.includes("Unexpected token") && normalized.includes("<")) ||
      normalized.includes("Failed to fetch") ||
      normalized.includes("Connection to the server was lost");
    if (!parseFailure) return;
    const layer = document.getElementById("connection-recovery-layer");
    if (layer) layer.hidden = false;
  };

  window.addEventListener("unhandledrejection", (event) => {
    const reason = event.reason;
    showRecovery(reason?.message || reason);
  });
  window.addEventListener("error", (event) => {
    showRecovery(event.error?.message || event.message);
  });
  const connectionObserver = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      showRecovery(mutation.target?.textContent || "");
      for (const node of mutation.addedNodes || []) {
        showRecovery(node?.textContent || "");
      }
    }
  });
  connectionObserver.observe(document.body, {
    childList: true,
    subtree: true,
    characterData: true,
  });
  document.addEventListener("click", (event) => {
    const action = event.target?.closest?.("[data-connection-action]")?.dataset
      ?.connectionAction;
    if (action === "reload") window.location.reload();
    if (action === "dismiss") {
      const layer = document.getElementById("connection-recovery-layer");
      if (layer) layer.hidden = true;
    }
  });
})();
"""

AUTO_MODEL = "auto"
FLUX_PRESET = "flux2-klein-edit@1"
LITE_PRESET = "sdxl-canny-lite@1"
QUALITY_PRESET = "sdxl-canny@1"
PACKAGED_PRESETS = frozenset({FLUX_PRESET, LITE_PRESET, QUALITY_PRESET})
DEFAULT_PROFILE = "graphic_design"
DEFAULT_LOCKS = ("structure",)
OUTPUT_CHOICES = (("1", 1), ("4", 4), ("8", 8))
LOCKED_OUTPUT_CHOICES = (("1", 1),)
BROWSER_SESSION_STORAGE_KEY = "aisketcher.v3.browser-session"


def _heading(language: str) -> str:
    title = html.escape(text(language, "headline")).replace("\n", "<br>")
    subtitle = html.escape(text(language, "subhead"))
    return f"<h1>{title}</h1><p>{subtitle}</p>"


def _directions_heading(language: str) -> str:
    title = html.escape(text(language, "directions"))
    subtitle = html.escape(text(language, "directions_help"))
    return f"<h2>{title}</h2><p>{subtitle}</p>"


def _connection_recovery_html(language: str) -> str:
    title = html.escape(text(language, "connection_title"))
    body = html.escape(text(language, "connection_body"))
    reload_label = html.escape(text(language, "connection_reload"))
    dismiss_label = html.escape(text(language, "connection_dismiss"))
    return f"""
<section id="connection-recovery-layer" role="alertdialog" aria-modal="true"
         aria-labelledby="connection-recovery-title" hidden>
  <div class="connection-recovery-card">
    <span class="connection-recovery-mark" aria-hidden="true">↻</span>
    <h2 id="connection-recovery-title">{title}</h2>
    <p>{body}</p>
    <div class="connection-recovery-actions">
      <button type="button" data-connection-action="reload">{reload_label}</button>
      <button type="button" data-connection-action="dismiss">{dismiss_label}</button>
    </div>
  </div>
</section>
"""


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
        (text(language, "model_flux"), FLUX_PRESET),
        (text(language, "model_lite"), LITE_PRESET),
        (text(language, "model_quality"), QUALITY_PRESET),
    ]


def _simple_model_choices(language: str) -> list[tuple[str, str]]:
    return [
        (text(language, "model_auto"), AUTO_MODEL),
        *_model_choices(language),
    ]


def _variation_choices(language: str) -> list[tuple[str, str]]:
    if normalize_language(language) == "ko":
        return [("미세하게", "subtle"), ("적당하게", "balanced"), ("과감하게", "bold")]
    return [("Subtle", "subtle"), ("Moderate", "balanced"), ("Bold", "bold")]


def _humanize_bytes(value: Any) -> str:
    """Render a non-negative byte count with compact decimal units."""

    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError("byte count must be a non-negative number")
    amount = float(value)
    units = ("B", "kB", "MB", "GB", "TB")
    unit = units[0]
    for unit in units:
        if amount < 1000 or unit == units[-1]:
            break
        amount /= 1000
    if unit == "B":
        return f"{int(amount)} {unit}"
    return f"{amount:.1f} {unit}"


def _html_code(value: Any) -> str:
    """Keep installer-provided metadata inert inside a compact HTML detail."""

    normalized = " ".join(str(value).split())
    return f"<code>{html.escape(normalized, quote=True)}</code>"


def _registry_license(repo_id: str, revision: str) -> tuple[str, str] | None:
    """Return audited license metadata only for an exact registry match."""

    try:
        from ..model_registry import MODEL_ARTIFACTS
    except (ImportError, AttributeError):
        return None
    for artifact in MODEL_ARTIFACTS.values():
        if artifact.model_id == repo_id and artifact.revision == revision:
            return artifact.license_id, artifact.license_url
    return None


def _render_install_plan(language: str, plan: Any) -> str:
    """Render a collapsed view of the stable ``plan_install`` subset."""

    canonical = str(plan.preset)
    cache_dir = plan.cache_dir
    remaining = _humanize_bytes(plan.download_bytes)
    license_notice = html.escape(
        " ".join(str(plan.license_notice).split()),
        quote=True,
    )
    items = tuple(plan.items)
    if not canonical or not items:
        raise ValueError("install plan must include a preset and at least one artifact")

    lines = [
        '<details class="installer-plan-details">',
        "<summary>",
        f"<span>{html.escape(text(language, 'install_plan_heading'))}</span>",
        f"<strong>{html.escape(remaining)}</strong>",
        "</summary>",
        '<div class="installer-plan-body">',
        (
            f"<p><strong>{html.escape(text(language, 'install_plan_preset'))}:</strong> "
            f"{_html_code(canonical)}</p>"
        ),
        (
            f"<p><strong>{html.escape(text(language, 'install_plan_remaining'))}:</strong> "
            f"{html.escape(remaining)}</p>"
        ),
        (
            f"<p><strong>{html.escape(text(language, 'install_plan_cache'))}:</strong> "
            f"{_html_code(cache_dir)}</p>"
        ),
        f"<p><strong>{html.escape(text(language, 'install_plan_artifacts'))}:</strong></p>",
        "<ul>",
    ]
    for item in items:
        status_key = "install_plan_cached" if bool(item.installed) else "install_plan_missing"
        artifact_license = _registry_license(str(item.repo_id), str(item.revision))
        license_suffix = ""
        if artifact_license is not None:
            license_id, license_url = artifact_license
            license_suffix = (
                f" · {html.escape(text(language, 'install_plan_upstream_license'))} "
                f'<a href="{html.escape(license_url, quote=True)}" '
                f'target="_blank" rel="noreferrer">{html.escape(license_id)}</a>'
            )
        lines.extend(
            [
                "<li>",
                (
                    f"<strong>{html.escape(text(language, status_key))}</strong> · "
                    f"{_html_code(item.repo_id)} · "
                    f"{html.escape(text(language, 'install_plan_revision'))} "
                    f"{_html_code(item.revision)} · "
                    f"{html.escape(text(language, 'install_plan_role'))} "
                    f"{_html_code(item.role)} · "
                    f"{html.escape(_humanize_bytes(item.estimated_bytes))}{license_suffix}"
                ),
                (
                    f"<small>{html.escape(text(language, 'install_plan_destination'))}: "
                    f"{_html_code(item.destination)}</small>"
                ),
                "</li>",
            ]
        )
    lines.extend(
        [
            "</ul>",
            (
                f"<p><strong>{html.escape(text(language, 'install_plan_license'))}:</strong> "
                f"{license_notice}</p>"
            ),
            f"<p>{html.escape(text(language, 'model_confirm'))}</p>",
            "</div>",
            "</details>",
        ]
    )
    return "\n".join(lines)


def _render_translator_plan(language: str, controller: AppController) -> str:
    """Describe the bundled translator without importing its runtime."""

    translator = getattr(controller, "prompt_translator", None)
    metadata = getattr(translator, "metadata", None)
    if (
        getattr(metadata, "model_id", None) != MARIAN_KO_EN_MODEL_ID
        or getattr(metadata, "revision", None) != MARIAN_KO_EN_REVISION
    ):
        return ""
    cache_dir = getattr(translator, "cache_dir", None)
    cache_label = (
        _html_code(cache_dir)
        if cache_dir
        else html.escape(text(language, "translator_plan_cache_default"))
    )
    return "\n".join(
        (
            '<details class="installer-plan-details translator-plan-details">',
            "<summary>",
            f"<span>{html.escape(text(language, 'translator_plan_heading'))}</span>",
            f"<strong>{_humanize_bytes(MARIAN_KO_EN_DOWNLOAD_BYTES)}</strong>",
            "</summary>",
            '<div class="installer-plan-body">',
            f"<p>{_html_code(MARIAN_KO_EN_MODEL_ID)}</p>",
            (
                f"<p><strong>{html.escape(text(language, 'install_plan_revision'))}:"
                f"</strong> {_html_code(MARIAN_KO_EN_REVISION)}</p>"
            ),
            (
                f"<p><strong>{html.escape(text(language, 'translator_plan_transfer'))}:"
                f"</strong> {_humanize_bytes(MARIAN_KO_EN_DOWNLOAD_BYTES)}</p>"
            ),
            (
                f"<p><strong>{html.escape(text(language, 'translator_plan_cache'))}:"
                f"</strong> {cache_label}</p>"
            ),
            (
                f"<p><strong>{html.escape(text(language, 'translator_plan_license'))}:"
                '</strong> <a href="https://www.apache.org/licenses/LICENSE-2.0" '
                'target="_blank" rel="noreferrer">Apache-2.0</a></p>'
            ),
            "</div>",
            "</details>",
        )
    )


def _model_plan(
    language: str,
    preset: str,
    controller: AppController | None = None,
) -> str:
    keys = {
        AUTO_MODEL: "model_plan_auto",
        FLUX_PRESET: "model_plan_flux",
        LITE_PRESET: "model_plan_lite",
        QUALITY_PRESET: "model_plan_quality",
    }
    try:
        key = keys[preset]
    except KeyError as exc:
        raise ValueError("model must be Auto or a packaged AIsketcher preset") from exc
    static_plan = text(language, key)
    if controller is None:
        return static_plan
    translator_plan = _render_translator_plan(language, controller)
    canonical = FLUX_PRESET if preset == AUTO_MODEL else preset
    plan_model_install = getattr(controller, "plan_model_install", None)
    if not callable(plan_model_install):
        return (
            f"{static_plan}\n\n---\n\n{translator_plan}"
            if translator_plan
            else static_plan
        )
    plan = plan_model_install(canonical)
    if plan is None:
        return (
            f"{static_plan}\n\n---\n\n{translator_plan}"
            if translator_plan
            else static_plan
        )
    try:
        current_plan = _render_install_plan(language, plan)
    except (AttributeError, TypeError, ValueError):
        return (
            f"{static_plan}\n\n---\n\n{translator_plan}"
            if translator_plan
            else static_plan
        )
    sections = (static_plan, current_plan, translator_plan)
    return "\n\n---\n\n".join(section for section in sections if section)


def _preset_selection(
    language: str,
    preset: str,
    controller: AppController | None = None,
) -> tuple[str, str]:
    """Return the concrete preset and the selection's localized download plan."""

    if preset == AUTO_MODEL:
        return FLUX_PRESET, _model_plan(language, AUTO_MODEL, controller)
    if preset not in PACKAGED_PRESETS:
        raise ValueError("model must be Auto or a packaged AIsketcher preset")
    return preset, _model_plan(language, preset, controller)


def _preset_generation_defaults(preset: str) -> tuple[bool, int, float]:
    """Return Canny, step, and CFG defaults for a concrete or Auto selection."""

    canonical, _ = _preset_selection("en", preset)
    if canonical == FLUX_PRESET:
        return False, 4, 1.0
    return True, 30, 5.0


def _canny_info(language: str, preset: str) -> str:
    canonical, _ = _preset_selection(language, preset)
    key = "canny_info_flux" if canonical == FLUX_PRESET else "canny_info"
    return text(language, key)


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
    """Resolve the Simple model recipe without mutating Advanced controls."""

    if len(values) != 14:
        raise ValueError("generation callback requires 14 values")
    resolved = list(values)
    state = AppState.from_payload(resolved[0])
    simple_model = str(resolved.pop(5))
    if state.view != "advanced":
        preset, _ = _preset_selection(state.language, simple_model)
        canny, steps, guidance = _preset_generation_defaults(preset)
        resolved[5] = preset
        resolved[6] = 4
        resolved[7] = "scout"
        resolved[8] = ""
        resolved[9] = canny
        resolved[10] = steps
        resolved[11] = guidance
        resolved[12] = DEFAULT_LOCKS
    return tuple(resolved)


def _response_values(gr: Any, response: AppResponse) -> tuple[Any, ...]:
    sync_recipe = response.sync_recipe_controls or any(
        value is not None for value in (response.prompt, response.profile, response.structure)
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


def _cancelled_response_values(
    gr: Any, state_value: Mapping[str, Any] | AppState
) -> tuple[Any, ...]:
    """Preserve the visible study when cooperative cancellation wins the race."""

    state = AppState.from_payload(state_value)
    return (
        state.payload(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        text(state.language, "status_stopped"),
        gr.update(),
        gr.update(),
        gr.update(),
    )


def build_app(
    controller: AppController | None = None,
    *,
    language: str = "en",
    default_preset: str = FLUX_PRESET,
    default_simple_model: str = AUTO_MODEL,
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
    if default_preset not in PACKAGED_PRESETS:
        raise ValueError("preset must be a packaged AIsketcher preset")
    if default_simple_model not in {AUTO_MODEL, *PACKAGED_PRESETS}:
        raise ValueError("simple model must be Auto or a packaged AIsketcher preset")
    if default_output_count not in {1, 4, 8}:
        raise ValueError("output_count must be 1, 4, or 8")
    if default_seed_mode not in {"scout", "locked", "explicit"}:
        raise ValueError("seed_mode must be scout, locked, or explicit")
    if default_seed is not None and not 0 <= default_seed <= (1 << 63) - 1:
        raise ValueError("seed must be a non-negative 63-bit integer")
    advanced_output_count = 1 if default_seed_mode == "locked" else default_output_count
    unlocked_output_default = default_output_count if default_seed_mode != "locked" else 4
    default_canny, default_steps, default_guidance = _preset_generation_defaults(default_preset)
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
        app_state = gr.BrowserState(
            controller.initial_state(language),
            storage_key=BROWSER_SESSION_STORAGE_KEY,
        )
        active_seed_mode = gr.State(default_seed_mode)
        unlocked_output_count = gr.State(unlocked_output_default)
        session_poll = gr.Timer(1.0, active=True)

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
                simple_model_choice = gr.Dropdown(
                    _simple_model_choices(language),
                    value=default_simple_model,
                    label=text(language, "simple_model"),
                    info=text(language, "simple_model_info"),
                    allow_custom_value=False,
                    filterable=False,
                    elem_id="simple-model-choice",
                    elem_classes="studio-field",
                )
                simple_model_plan = gr.Markdown(
                    _model_plan(language, default_simple_model, controller),
                    elem_id="simple-model-plan",
                )
                with gr.Row(elem_id="simple-model-actions"):
                    simple_model_button = gr.Button(
                        text(language, "model_prepare"),
                        size="sm",
                        elem_id="simple-model-action",
                    )
                    simple_model_stop_button = gr.Button(
                        text(language, "stop"),
                        variant="stop",
                        size="sm",
                        visible=False,
                        elem_id="simple-model-stop-action",
                    )
                simple_model_status = gr.Markdown("", elem_id="simple-model-status")
                with gr.Row(elem_id="primary-actions"):
                    explore_button = gr.Button(
                        text(language, "explore"),
                        variant="primary",
                        elem_id="primary-action",
                    )
                    stop_button = gr.Button(
                        text(language, "stop"),
                        variant="stop",
                        visible=False,
                        elem_id="stop-action",
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
                    rows=None,
                    height=None,
                    object_fit="contain",
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
                with gr.Column(visible=False, elem_id="refine-composer") as refine_composer:
                    refinement_instruction = gr.Textbox(
                        label=text(language, "refine_prompt"),
                        placeholder=text(language, "refine_prompt_placeholder"),
                        info=text(language, "refine_prompt_help"),
                        lines=2,
                        max_lines=4,
                        max_length=600,
                        elem_id="refine-instruction",
                        elem_classes="studio-field",
                    )
                    with gr.Row(elem_id="refine-composer-actions"):
                        refine_submit = gr.Button(
                            text(language, "refine_apply"),
                            variant="primary",
                            elem_id="refine-submit-action",
                        )
                        refine_cancel = gr.Button(
                            text(language, "refine_cancel"),
                            elem_id="refine-cancel-action",
                        )
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
                    _model_choices(language),
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
                    LOCKED_OUTPUT_CHOICES if default_seed_mode == "locked" else OUTPUT_CHOICES,
                    value=advanced_output_count,
                    label=text(language, "outputs"),
                    interactive=default_seed_mode != "locked",
                    elem_id="output-control",
                    elem_classes="studio-field",
                )
                canny = gr.Checkbox(
                    default_canny,
                    label=text(language, "canny"),
                    info=_canny_info(language, default_preset),
                    interactive=False,
                    elem_classes="studio-field",
                )
                steps = gr.Slider(
                    4,
                    50,
                    value=default_steps,
                    step=1,
                    label=text(language, "steps"),
                    elem_classes="studio-field",
                )
                guidance = gr.Slider(
                    1.0,
                    12.0,
                    value=default_guidance,
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
                    _model_plan(language, default_preset, controller), elem_id="model-plan"
                )
                model_confirm = gr.Checkbox(
                    True,
                    label=text(language, "model_confirm"),
                    visible=False,
                    elem_classes="studio-field",
                )
                with gr.Row(elem_id="model-actions"):
                    model_button = gr.Button(
                        text(language, "model_prepare"), elem_id="model-action"
                    )
                    model_stop_button = gr.Button(
                        text(language, "stop"),
                        variant="stop",
                        visible=False,
                        elem_id="model-stop-action",
                    )
                model_status = gr.Markdown("", elem_id="model-status")

        with gr.Column(visible=False, elem_id="guide-panel") as guide_panel:
            guide_content = gr.Markdown(
                text(language, "guide_body"),
                elem_id="guide-content",
            )

        with (
            gr.Column(visible=False, elem_id="guided-refine-overlay") as guided_refine_overlay,
            gr.Column(elem_id="guided-refine-dialog"),
        ):
            guided_refine_title = gr.Markdown(
                f"## {text(language, 'guided_refine_title')}",
                elem_id="guided-refine-title",
            )
            guided_refine_body = gr.Markdown(
                text(language, "guided_refine_body"),
                elem_id="guided-refine-body",
            )
            guided_refine_model = gr.Markdown(
                text(language, "guided_refine_model"),
                elem_id="guided-refine-model",
            )
            with gr.Row(elem_id="guided-refine-actions"):
                guided_choose_model = gr.Button(
                    text(language, "guided_choose_model"),
                    variant="primary",
                    elem_id="guided-choose-model-action",
                )
                guided_keep_exploring = gr.Button(
                    text(language, "guided_keep_exploring"),
                    elem_id="guided-keep-exploring-action",
                )

        connection_recovery = gr.HTML(
            _connection_recovery_html(language),
            js_on_load=STUDIO_JS,
            container=False,
            elem_id="connection-recovery-host",
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
            except StudioJobCancelled:
                return _cancelled_response_values(gr, values[0])
            except StudioAppError as exc:
                raise gr.Error(str(exc)) from exc
            return _response_values(gr, response)

        generation_ui_outputs = [
            status,
            stop_button,
            explore_button,
            refine_button,
            refine_submit,
            retry_button,
            replay_button,
        ]
        recovery_outputs = [
            app_state,
            source_preview,
            selected_preview,
            gallery,
            recommendation,
            status,
            brief,
            profile,
            structure,
            stop_button,
            explore_button,
            refine_button,
            refine_submit,
            retry_button,
            replay_button,
            language_nav,
            view_nav,
            workbench,
            advanced_rail,
            guide_panel,
        ]

        def recover_browser_session(
            state_value: Mapping[str, Any],
            force_restore: bool = False,
            current_status: str | None = None,
        ) -> tuple[Any, ...]:
            state = AppState.from_payload(state_value)
            operation = controller.operation_state(state)
            nav_updates = (
                gr.update(value=state.language),
                gr.update(value=state.view),
                gr.update(visible=state.view != "guide"),
                gr.update(visible=state.view == "advanced"),
                gr.update(visible=state.view == "guide"),
            )
            if not force_restore:
                nav_updates = (gr.update(),) * 5
            if operation != "idle":
                status_key = (
                    "status_reconnected_stopping"
                    if operation == "stopping"
                    else "status_reconnected_running"
                )
                return (
                    state.payload(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    text(state.language, status_key),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(visible=True, interactive=True),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    *nav_updates,
                )

            recovered = controller.recover_latest_run(state, state.language)
            should_restore = recovered is not None and (
                force_restore or recovered.state.get("run_id") != state.run_id
            )
            if should_restore and recovered is not None:
                return (
                    recovered.state,
                    recovered.source,
                    recovered.selected,
                    gr.update(value=list(recovered.gallery)),
                    recovered.recommendation,
                    recovered.status,
                    gr.update(value=recovered.prompt or ""),
                    gr.update(value=recovered.profile or DEFAULT_PROFILE),
                    gr.update(value=recovered.structure or "balanced"),
                    gr.update(visible=False),
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    *nav_updates,
                )
            stale_recovery_statuses = {
                text(language, status_key)
                for language in ("en", "ko")
                for status_key in (
                    "status_reconnected_running",
                    "status_reconnected_stopping",
                )
            }
            idle_status = gr.update()
            if current_status in stale_recovery_statuses:
                idle_status = gr.update(
                    value=(
                        recovered.status
                        if recovered is not None
                        else (
                            text(state.language, "ready")
                            if guided_available
                            else text(state.language, "unavailable")
                        )
                    )
                )
            return (
                state.payload(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                idle_status,
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(visible=False),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                *nav_updates,
            )

        def begin_generation(state_value: Mapping[str, Any]) -> tuple[Any, ...]:
            state = AppState.from_payload(state_value)
            controller.begin_operation(state)
            return (
                text(state.language, "status_generating"),
                gr.update(visible=True, interactive=True),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
            )

        def finish_generation(state_value: Mapping[str, Any]) -> tuple[Any, ...]:
            controller.clear_operation(state_value)
            return (
                gr.update(),
                gr.update(visible=False),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
            )

        generation_inputs = [
            app_state,
            sketch,
            brief,
            profile,
            structure,
            simple_model_choice,
            preset,
            output_count,
            seed_mode,
            custom_seeds,
            canny,
            steps,
            guidance,
            locks,
        ]
        explore_start = explore_button.click(
            begin_generation,
            inputs=[app_state],
            outputs=generation_ui_outputs,
            queue=False,
            show_progress="hidden",
        )
        explore_event = explore_start.then(
            run_explore,
            inputs=generation_inputs,
            outputs=response_outputs,
            concurrency_id="generation",
            concurrency_limit=1,
            show_progress="hidden",
        )
        explore_event.then(
            finish_generation,
            inputs=[app_state],
            outputs=generation_ui_outputs,
            queue=False,
            show_progress="hidden",
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

        def restore_gallery_after_preview(
            state_value: Mapping[str, Any],
        ) -> Any:
            state = AppState.from_payload(state_value)
            try:
                active = controller.localize_active_run(state, state.language)
            except StudioAppError:
                active = None
            if active is None:
                return gr.update()
            # Do not set ``selected_index`` here: Gradio interprets it as an
            # instruction to reopen the preview that the user just closed.
            return gr.update(value=list(active.gallery))

        # Gradio can leave its preview thumbnail rail mounted after fullscreen
        # closes. Re-applying the canonical gallery value forces the grid back
        # into its normal layout without changing the selected direction.
        gallery.preview_close(
            restore_gallery_after_preview,
            inputs=[app_state],
            outputs=[gallery],
            queue=False,
            show_progress="hidden",
        )

        def open_refinement(
            state_value: Mapping[str, Any],
        ) -> tuple[Any, ...]:
            state = AppState.from_payload(state_value)
            try:
                mode = controller.refinement_mode(state)
            except StudioAppError as exc:
                return (
                    gr.update(visible=False),
                    gr.update(visible=False),
                    str(exc),
                )
            return (
                gr.update(visible=mode == "live"),
                gr.update(visible=mode == "guided"),
                gr.update(),
            )

        refine_button.click(
            open_refinement,
            inputs=[app_state],
            outputs=[refine_composer, guided_refine_overlay, status],
            queue=False,
            show_progress="hidden",
        )

        refine_cancel.click(
            lambda: gr.update(visible=False),
            outputs=[refine_composer],
            queue=False,
            show_progress="hidden",
        )
        guided_keep_exploring.click(
            lambda: gr.update(visible=False),
            outputs=[guided_refine_overlay],
            queue=False,
            show_progress="hidden",
        )

        def choose_model_for_guided(
            state_value: Mapping[str, Any],
        ) -> tuple[Any, ...]:
            state = AppState.from_payload(state_value).replace(view="advanced")
            return (
                state.payload(),
                gr.update(value="advanced"),
                gr.update(visible=True),
                gr.update(visible=False),
                text(state.language, "guided_model_status"),
            )

        guided_choose_model.click(
            choose_model_for_guided,
            inputs=[app_state],
            outputs=[
                app_state,
                view_nav,
                advanced_rail,
                guided_refine_overlay,
                status,
            ],
            queue=False,
            show_progress="hidden",
        )

        def refine_study(
            state_value: Mapping[str, Any],
            strength: str,
            lock_values: Sequence[str],
            instruction: str,
        ) -> tuple[Any, ...]:
            try:
                response = controller.refine(
                    state_value,
                    strength,
                    lock_values,
                    instruction,
                )
            except StudioJobCancelled:
                return _cancelled_response_values(gr, state_value)
            except StudioAppError as exc:
                raise gr.Error(str(exc)) from exc
            return _response_values(gr, response)

        refine_start = refine_submit.click(
            begin_generation,
            inputs=[app_state],
            outputs=generation_ui_outputs,
            queue=False,
            show_progress="hidden",
        )
        refine_start.then(
            lambda: gr.update(visible=False),
            outputs=[refine_composer],
            queue=False,
            show_progress="hidden",
        )
        refine_event = refine_start.then(
            refine_study,
            inputs=[app_state, variation, locks, refinement_instruction],
            outputs=response_outputs,
            concurrency_id="generation",
            concurrency_limit=1,
            show_progress="hidden",
        )
        refine_event.success(
            lambda: "",
            outputs=[refinement_instruction],
            queue=False,
            show_progress="hidden",
        )
        refine_event.then(
            finish_generation,
            inputs=[app_state],
            outputs=generation_ui_outputs,
            queue=False,
            show_progress="hidden",
        )

        def retry_study(state_value: Mapping[str, Any]) -> tuple[Any, ...]:
            try:
                response = controller.try_again(state_value)
            except StudioJobCancelled:
                return _cancelled_response_values(gr, state_value)
            except StudioAppError as exc:
                raise gr.Error(str(exc)) from exc
            return _response_values(gr, response)

        retry_start = retry_button.click(
            begin_generation,
            inputs=[app_state],
            outputs=generation_ui_outputs,
            queue=False,
            show_progress="hidden",
        )
        retry_event = retry_start.then(
            retry_study,
            inputs=[app_state],
            outputs=response_outputs,
            concurrency_id="generation",
            concurrency_limit=1,
            show_progress="hidden",
        )
        retry_event.then(
            finish_generation,
            inputs=[app_state],
            outputs=generation_ui_outputs,
            queue=False,
            show_progress="hidden",
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
            except StudioJobCancelled:
                return _cancelled_response_values(gr, state_value)
            except StudioAppError as exc:
                raise gr.Error(str(exc)) from exc
            return _response_values(gr, response)

        replay_start = replay_button.click(
            begin_generation,
            inputs=[app_state],
            outputs=generation_ui_outputs,
            queue=False,
            show_progress="hidden",
        )
        replay_event = replay_start.then(
            replay_study,
            inputs=[app_state, manifest_file, preset],
            outputs=response_outputs,
            concurrency_id="generation",
            concurrency_limit=1,
            show_progress="hidden",
        )
        replay_event.then(
            finish_generation,
            inputs=[app_state],
            outputs=generation_ui_outputs,
            queue=False,
            show_progress="hidden",
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
                        LOCKED_OUTPUT_CHOICES if default_seed_mode == "locked" else OUTPUT_CHOICES
                    ),
                    value=advanced_output_count,
                    interactive=default_seed_mode != "locked",
                ),
                default_canny,
                default_steps,
                default_guidance,
                "subtle",
                list(DEFAULT_LOCKS),
                default_preset,
                gr.update(value=_model_plan(state.language, default_preset, controller)),
                True,
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

        def sync_simple_model(
            state_value: Mapping[str, Any], value: str
        ) -> tuple[Any, ...]:
            lang = AppState.from_payload(state_value).language
            _selected, plan = _preset_selection(lang, value, controller)
            return gr.update(value=plan), ""

        simple_model_choice.input(
            sync_simple_model,
            inputs=[app_state, simple_model_choice],
            outputs=[simple_model_plan, simple_model_status],
            queue=False,
            show_progress="hidden",
        )

        def sync_generation_preset(state_value: Mapping[str, Any], value: str) -> tuple[Any, ...]:
            lang = AppState.from_payload(state_value).language
            selected, plan = _preset_selection(lang, value, controller)
            canny_value, step_value, guidance_value = _preset_generation_defaults(selected)
            return (
                gr.update(value=selected),
                gr.update(value=plan),
                True,
                "",
                gr.update(value=canny_value, info=_canny_info(lang, selected)),
                gr.update(value=step_value),
                gr.update(value=guidance_value),
            )

        preset.input(
            sync_generation_preset,
            inputs=[app_state, preset],
            outputs=[
                model_choice,
                model_plan,
                model_confirm,
                model_status,
                canny,
                steps,
                guidance,
            ],
            queue=False,
            show_progress="hidden",
        )

        def sync_model_preset(state_value: Mapping[str, Any], value: str) -> tuple[Any, ...]:
            lang = AppState.from_payload(state_value).language
            selected, plan = _preset_selection(lang, value, controller)
            canny_value, step_value, guidance_value = _preset_generation_defaults(selected)
            return (
                gr.update(value=selected),
                gr.update(value=plan),
                True,
                "",
                gr.update(value=canny_value, info=_canny_info(lang, selected)),
                gr.update(value=step_value),
                gr.update(value=guidance_value),
            )

        model_choice.input(
            sync_model_preset,
            inputs=[app_state, model_choice],
            outputs=[
                preset,
                model_plan,
                model_confirm,
                model_status,
                canny,
                steps,
                guidance,
            ],
            queue=False,
            show_progress="hidden",
        )

        def prepare_model(
            state_value: Mapping[str, Any], preset_value: str, confirmed: bool
        ) -> str:
            state = AppState.from_payload(state_value)
            concrete_preset, _ = _preset_selection(state.language, preset_value)
            try:
                return controller.install_model(
                    concrete_preset,
                    confirmed,
                    state.language,
                    state,
                )
            except StudioJobCancelled:
                return text(state.language, "status_stopped")
            except StudioAppError as exc:
                raise gr.Error(str(exc)) from exc

        def begin_model_download(
            state_value: Mapping[str, Any],
        ) -> tuple[Any, ...]:
            state = AppState.from_payload(state_value)
            controller.begin_operation(state)
            return (
                text(state.language, "status_downloading"),
                gr.update(visible=True, interactive=True),
                gr.update(interactive=False),
            )

        def finish_model_download(
            state_value: Mapping[str, Any],
            preset_value: str,
        ) -> tuple[Any, ...]:
            controller.clear_operation(state_value)
            state = AppState.from_payload(state_value)
            _selected, current_plan = _preset_selection(
                state.language,
                preset_value,
                controller,
            )
            return (
                gr.update(),
                gr.update(visible=False),
                gr.update(interactive=True),
                gr.update(value=current_plan),
            )

        model_start = model_button.click(
            begin_model_download,
            inputs=[app_state],
            outputs=[model_status, model_stop_button, model_button],
            queue=False,
            show_progress="hidden",
        )
        model_event = model_start.then(
            prepare_model,
            inputs=[app_state, preset, model_confirm],
            outputs=[model_status],
            concurrency_id="model-download",
            concurrency_limit=1,
            show_progress="hidden",
        )
        model_event.then(
            finish_model_download,
            inputs=[app_state, preset],
            outputs=[model_status, model_stop_button, model_button, model_plan],
            queue=False,
            show_progress="hidden",
        )

        def prepare_simple_model(
            state_value: Mapping[str, Any], model_value: str
        ) -> str:
            # The button itself is the explicit size-and-license confirmation.
            return prepare_model(state_value, model_value, True)

        simple_model_start = simple_model_button.click(
            begin_model_download,
            inputs=[app_state],
            outputs=[
                simple_model_status,
                simple_model_stop_button,
                simple_model_button,
            ],
            queue=False,
            show_progress="hidden",
        )
        simple_model_event = simple_model_start.then(
            prepare_simple_model,
            inputs=[app_state, simple_model_choice],
            outputs=[simple_model_status],
            concurrency_id="model-download",
            concurrency_limit=1,
            show_progress="hidden",
        )
        simple_model_event.then(
            finish_model_download,
            inputs=[app_state, simple_model_choice],
            outputs=[
                simple_model_status,
                simple_model_stop_button,
                simple_model_button,
                simple_model_plan,
            ],
            queue=False,
            show_progress="hidden",
        )

        def stop_generation(
            state_value: Mapping[str, Any],
        ) -> tuple[Any, ...]:
            message = controller.cancel_operation(state_value)
            controller.clear_operation(state_value)
            return (
                message,
                gr.update(visible=False),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
            )

        stop_button.click(
            stop_generation,
            inputs=[app_state],
            outputs=generation_ui_outputs,
            cancels=[
                explore_event,
                refine_event,
                retry_event,
                replay_event,
            ],
            queue=False,
            show_progress="hidden",
        )

        def stop_model_download(
            state_value: Mapping[str, Any],
        ) -> tuple[Any, ...]:
            message = controller.cancel_operation(state_value)
            controller.clear_operation(state_value)
            return (
                message,
                gr.update(visible=False),
                gr.update(interactive=True),
            )

        model_stop_button.click(
            stop_model_download,
            inputs=[app_state],
            outputs=[model_status, model_stop_button, model_button],
            cancels=[model_event],
            queue=False,
            show_progress="hidden",
        )
        simple_model_stop_button.click(
            stop_model_download,
            inputs=[app_state],
            outputs=[
                simple_model_status,
                simple_model_stop_button,
                simple_model_button,
            ],
            cancels=[simple_model_event],
            queue=False,
            show_progress="hidden",
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
            simple_model_choice,
            simple_model_plan,
            simple_model_button,
            simple_model_stop_button,
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
            refinement_instruction,
            refine_submit,
            refine_cancel,
            stop_button,
            model_stop_button,
            guided_refine_title,
            guided_refine_body,
            guided_refine_model,
            guided_choose_model,
            guided_keep_exploring,
            connection_recovery,
        ]

        def localize(
            state_value: Mapping[str, Any],
            selected_language: str,
            selected_simple_model: str,
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
                gr.update(
                    label=text(lang, "simple_model"),
                    info=text(lang, "simple_model_info"),
                    choices=_simple_model_choices(lang),
                    value=selected_simple_model,
                ),
                gr.update(value=_model_plan(lang, selected_simple_model, controller)),
                gr.update(value=text(lang, "model_prepare")),
                gr.update(value=text(lang, "stop")),
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
                gr.update(
                    label=text(lang, "preset"),
                    choices=_model_choices(lang),
                    value=selected_preset,
                ),
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
                gr.update(
                    label=text(lang, "canny"),
                    info=_canny_info(lang, selected_preset),
                ),
                gr.update(label=text(lang, "steps")),
                gr.update(label=text(lang, "guidance")),
                gr.update(label=text(lang, "variation"), choices=_variation_choices(lang)),
                gr.update(label=text(lang, "locks"), choices=_lock_choices(lang)),
                gr.update(value=f"### {text(lang, 'manifest')}"),
                gr.update(label=text(lang, "manifest_upload")),
                gr.update(value=text(lang, "replay")),
                gr.update(value=f"### {text(lang, 'model_setup')}"),
                gr.update(choices=_model_choices(lang), value=selected_preset),
                gr.update(value=_model_plan(lang, selected_preset, controller)),
                gr.update(label=text(lang, "model_confirm")),
                gr.update(value=text(lang, "model_prepare")),
                gr.update(value=text(lang, "guide_body")),
                gr.update(value=text(lang, "clear")),
                gr.update(value=text(lang, "overrides")),
                status_update,
                recommendation_update,
                gr.update(
                    label=text(lang, "refine_prompt"),
                    placeholder=text(lang, "refine_prompt_placeholder"),
                    info=text(lang, "refine_prompt_help"),
                ),
                gr.update(value=text(lang, "refine_apply")),
                gr.update(value=text(lang, "refine_cancel")),
                gr.update(value=text(lang, "stop")),
                gr.update(value=text(lang, "stop")),
                gr.update(value=f"## {text(lang, 'guided_refine_title')}"),
                gr.update(value=text(lang, "guided_refine_body")),
                gr.update(value=text(lang, "guided_refine_model")),
                gr.update(value=text(lang, "guided_choose_model")),
                gr.update(value=text(lang, "guided_keep_exploring")),
                gr.update(value=_connection_recovery_html(lang)),
            )

        language_nav.change(
            localize,
            inputs=[
                app_state,
                language_nav,
                simple_model_choice,
                preset,
                seed_mode,
            ],
            outputs=localized_outputs,
            queue=False,
            show_progress="hidden",
        )

        browser_load = demo.load(
            lambda state_value: recover_browser_session(state_value, True),
            inputs=[app_state],
            outputs=recovery_outputs,
            queue=False,
            show_progress="hidden",
        )
        browser_load.then(
            localize,
            inputs=[
                app_state,
                language_nav,
                simple_model_choice,
                preset,
                seed_mode,
            ],
            outputs=localized_outputs,
            queue=False,
            show_progress="hidden",
        )
        session_poll.tick(
            lambda state_value, current_status: recover_browser_session(
                state_value,
                False,
                current_status,
            ),
            inputs=[app_state, status],
            outputs=recovery_outputs,
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
    demo._studio_sync_simple_model = sync_simple_model
    demo._studio_prepare_simple_model = prepare_simple_model
    demo._studio_sync_generation_preset = sync_generation_preset
    demo._studio_sync_model_preset = sync_model_preset
    demo._studio_open_refinement = open_refinement
    demo._studio_choose_model_for_guided = choose_model_for_guided
    demo._studio_begin_generation = begin_generation
    demo._studio_stop_generation = stop_generation
    demo._studio_restore_gallery_after_preview = restore_gallery_after_preview
    demo._studio_recover_browser_session = recover_browser_session
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


def main() -> int:
    """Launch through the configured CLI so every cache uses one ledger."""

    from ..cli import main as cli_main

    return cli_main(("studio",))


if __name__ == "__main__":
    raise SystemExit(main())
