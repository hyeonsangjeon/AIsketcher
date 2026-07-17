"""Versioned, pinned model presets and explicit installation planning."""

from __future__ import annotations

import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path

from .errors import ModelUnavailableError, OptionalDependencyError, ValidationError
from .models import (
    BackendCapabilities,
    CapabilityIssue,
    CapabilityReport,
    CapabilitySeverity,
    Intent,
    ModelReference,
    Recipe,
    ResolvedRecipe,
    StructureMode,
)

SDXL_BASE = ModelReference(
    repo_id="stabilityai/stable-diffusion-xl-base-1.0",
    revision="462165984030d82259a11f4367a4eed129e94a7b",
    role="base",
)
SDXL_CANNY_QUALITY = ModelReference(
    repo_id="diffusers/controlnet-canny-sdxl-1.0",
    revision="eb115a19a10d14909256db740ed109532ab1483c",
    role="controlnet",
)
SDXL_CANNY_LITE = ModelReference(
    repo_id="diffusers/controlnet-canny-sdxl-1.0-small",
    revision="edd85f64c5f87dfb6d73762949d9daca16389518",
    role="controlnet",
)


@dataclass(frozen=True, slots=True)
class PresetDefinition:
    name: str
    label: str
    models: tuple[ModelReference, ...]
    estimated_bytes: int
    width: int = 1024
    height: int = 1024
    steps: int = 30
    guidance_scale: float = 5.0
    scheduler: str = "unipc"
    negative_prompt: str = ""


PRESETS: dict[str, PresetDefinition] = {
    "sdxl-canny-lite@1": PresetDefinition(
        name="sdxl-canny-lite@1",
        label="SDXL Canny Lite",
        models=(SDXL_BASE, SDXL_CANNY_LITE),
        estimated_bytes=7_261_425_974,
    ),
    "sdxl-canny@1": PresetDefinition(
        name="sdxl-canny@1",
        label="SDXL Canny Quality",
        models=(SDXL_BASE, SDXL_CANNY_QUALITY),
        estimated_bytes=9_443_327_981,
    ),
}

_ALIASES = {
    "lite": "sdxl-canny-lite@1",
    "quality": "sdxl-canny@1",
    "sdxl-canny-quality@1": "sdxl-canny@1",
}

_STRUCTURE_STRENGTH = {
    StructureMode.LOOSE: 0.55,
    StructureMode.BALANCED: 0.75,
    StructureMode.FAITHFUL: 0.95,
}

_BASE_ALLOW_PATTERNS = (
    "model_index.json",
    "scheduler/*.json",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "text_encoder_2/config.json",
    "text_encoder_2/model.fp16.safetensors",
    "tokenizer/*",
    "tokenizer_2/*",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
)
_CONTROLNET_ALLOW_PATTERNS = (
    "config.json",
    "diffusion_pytorch_model.fp16.safetensors",
)
_MODEL_BYTES = {
    (SDXL_BASE.repo_id, SDXL_BASE.revision): 6_941_187_536,
    (SDXL_CANNY_LITE.repo_id, SDXL_CANNY_LITE.revision): 320_238_438,
    (SDXL_CANNY_QUALITY.repo_id, SDXL_CANNY_QUALITY.revision): 2_502_140_445,
}


def _allow_patterns(model: ModelReference) -> tuple[str, ...]:
    if model.role == "base":
        return _BASE_ALLOW_PATTERNS
    if model.role == "controlnet":
        return _CONTROLNET_ALLOW_PATTERNS
    raise ValidationError(f"No safe download policy exists for model role {model.role!r}")


def get_preset(name: str) -> PresetDefinition:
    canonical = _ALIASES.get(name, name)
    try:
        return PRESETS[canonical]
    except KeyError as exc:
        available = ", ".join(sorted(PRESETS))
        raise ValidationError(f"Unknown preset {name!r}. Available presets: {available}") from exc


def resolve_recipe(
    preset_name: str,
    intent: Intent,
    overrides: Recipe | None,
    *,
    backend_name: str,
    capabilities: BackendCapabilities,
) -> ResolvedRecipe:
    """Apply preset -> intent -> explicit overrides -> backend capabilities."""

    preset = get_preset(preset_name)
    recipe = overrides or Recipe()
    width = recipe.width or preset.width
    height = recipe.height or preset.height
    scheduler = recipe.scheduler or preset.scheduler
    negative_prompt = (
        recipe.negative_prompt
        if recipe.negative_prompt is not None
        else preset.negative_prompt
    )
    issues: list[CapabilityIssue] = []

    if "canny" not in capabilities.controls:
        issues.append(
            CapabilityIssue(
                setting="control",
                requested="canny",
                applied=None,
                severity=CapabilitySeverity.ERROR,
                message="This preset requires a Canny-capable backend.",
            )
        )
    if scheduler not in capabilities.schedulers:
        fallback = capabilities.schedulers[0] if capabilities.schedulers else None
        issues.append(
            CapabilityIssue(
                setting="scheduler",
                requested=scheduler,
                applied=fallback,
                severity=(
                    CapabilitySeverity.WARNING if fallback else CapabilitySeverity.ERROR
                ),
                message=(
                    f"Backend does not provide {scheduler!r}; using {fallback!r}."
                    if fallback
                    else "Backend provides no compatible scheduler."
                ),
            )
        )
        if fallback:
            scheduler = fallback
    if negative_prompt and not capabilities.supports_negative_prompt:
        issues.append(
            CapabilityIssue(
                setting="negative_prompt",
                requested=negative_prompt,
                applied="",
                severity=CapabilitySeverity.WARNING,
                message="Backend cannot apply a negative prompt; it was omitted.",
            )
        )
        negative_prompt = ""

    report = CapabilityReport(backend=backend_name, issues=tuple(issues))
    return ResolvedRecipe(
        preset=preset.name,
        prompt=intent.prompt,
        profile=intent.profile,
        structure=StructureMode(intent.structure),
        width=width,
        height=height,
        steps=recipe.steps or preset.steps,
        guidance_scale=(
            recipe.guidance_scale
            if recipe.guidance_scale is not None
            else preset.guidance_scale
        ),
        control_strength=(
            recipe.control_strength
            if recipe.control_strength is not None
            else _STRUCTURE_STRENGTH[StructureMode(intent.structure)]
        ),
        scheduler=scheduler,
        negative_prompt=negative_prompt,
        models=preset.models,
        capability_report=report,
    )


def _default_cache_dir() -> Path:
    configured = os.environ.get("AISKETCHER_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "aisketcher"
    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Caches" / "AIsketcher"
    return Path.home() / ".cache" / "aisketcher"


@dataclass(frozen=True, slots=True)
class InstallItem:
    repo_id: str
    revision: str
    role: str
    destination: Path
    installed: bool
    allow_patterns: tuple[str, ...]
    estimated_bytes: int


@dataclass(frozen=True, slots=True)
class InstallPlan:
    preset: str
    label: str
    items: tuple[InstallItem, ...]
    estimated_bytes: int
    cache_dir: Path
    license_notice: str = (
        "Review each pinned model repository's license before installation. "
        "Model weights are not distributed with AIsketcher."
    )

    @property
    def installed(self) -> bool:
        return all(item.installed for item in self.items)

    @property
    def download_bytes(self) -> int:
        """Estimated network transfer after accounting for installed items."""

        return sum(item.estimated_bytes for item in self.items if not item.installed)

    @property
    def cached_bytes(self) -> int:
        return sum(item.estimated_bytes for item in self.items if item.installed)


@dataclass(frozen=True, slots=True)
class InstallResult:
    preset: str
    paths: tuple[Path, ...]
    downloaded: tuple[str, ...]


class PresetManager:
    """Plans and explicitly installs only AIsketcher's curated pinned presets."""

    def __init__(self, cache_dir: str | Path | None = None, *, allow_downloads: bool = True):
        self.cache_dir = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
        self.allow_downloads = allow_downloads

    @staticmethod
    def available() -> tuple[PresetDefinition, ...]:
        return tuple(PRESETS[name] for name in sorted(PRESETS))

    def _destination(self, model: ModelReference) -> Path:
        safe_repo = model.repo_id.replace("/", "--")
        return self.cache_dir / "models" / f"{safe_repo}@{model.revision}"

    @staticmethod
    def _required_files_present(
        destination: Path, allow_patterns: tuple[str, ...]
    ) -> bool:
        root = destination.resolve()
        for pattern in allow_patterns:
            matches = []
            for candidate in destination.glob(pattern):
                if not candidate.is_file():
                    continue
                try:
                    candidate.resolve().relative_to(root)
                except ValueError:
                    continue
                matches.append(candidate)
            if not matches:
                return False
        unsafe_suffixes = {".bin", ".ckpt", ".pt", ".pth", ".pkl", ".pickle"}
        return not any(
            path.is_file() and path.suffix.lower() in unsafe_suffixes
            for path in destination.rglob("*")
        )

    def model_path(self, model: ModelReference) -> Path | None:
        destination = self._destination(model)
        marker = destination / ".aisketcher-model.json"
        if not marker.is_file():
            return None
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if (
            payload.get("repo_id") != model.repo_id
            or payload.get("revision") != model.revision
            or payload.get("download_policy") != "fp16-components-v1"
            or payload.get("safe_tensors_only") is not True
            or payload.get("allow_patterns") != list(_allow_patterns(model))
        ):
            return None
        if not self._required_files_present(destination, _allow_patterns(model)):
            return None
        return destination

    def plan_install(self, preset_name: str) -> InstallPlan:
        preset = get_preset(preset_name)
        items = tuple(
            InstallItem(
                repo_id=model.repo_id,
                revision=model.revision,
                role=model.role,
                destination=self._destination(model),
                installed=self.model_path(model) is not None,
                allow_patterns=_allow_patterns(model),
                estimated_bytes=_MODEL_BYTES[(model.repo_id, model.revision)],
            )
            for model in preset.models
        )
        return InstallPlan(
            preset=preset.name,
            label=preset.label,
            items=items,
            estimated_bytes=preset.estimated_bytes,
            cache_dir=self.cache_dir,
        )

    def require_installed(self, preset_name: str) -> InstallPlan:
        plan = self.plan_install(preset_name)
        if not plan.installed:
            missing = ", ".join(item.repo_id for item in plan.items if not item.installed)
            raise ModelUnavailableError(
                f"Preset {plan.preset!r} is not installed. Missing: {missing}. "
                "Call PresetManager.install(..., confirm=True) after reviewing plan_install()."
            )
        return plan

    def install(self, preset_name: str, *, confirm: bool = False) -> InstallResult:
        plan = self.plan_install(preset_name)
        if not confirm:
            raise ValidationError(
                "Model installation requires confirm=True after displaying plan_install()."
            )
        if not self.allow_downloads:
            raise ModelUnavailableError("Model downloads are disabled for this PresetManager")
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise OptionalDependencyError(
                "Model installation requires the 'local' extra: pip install 'aisketcher[local]'"
            ) from exc

        downloaded: list[str] = []
        paths: list[Path] = []
        for item in plan.items:
            item.destination.mkdir(parents=True, exist_ok=True)
            if not item.installed:
                snapshot_download(
                    repo_id=item.repo_id,
                    revision=item.revision,
                    local_dir=item.destination,
                    allow_patterns=list(item.allow_patterns),
                    ignore_patterns=[
                        "*.bin",
                        "*.ckpt",
                        "*.pt",
                        "*.pth",
                        "*.pkl",
                        "*.pickle",
                        "*.py",
                    ],
                )
                if not self._required_files_present(
                    item.destination, item.allow_patterns
                ):
                    raise ModelUnavailableError(
                        f"Downloaded snapshot for {item.repo_id!r} is incomplete or unsafe"
                    )
                marker = {
                    "repo_id": item.repo_id,
                    "revision": item.revision,
                    "role": item.role,
                    "download_policy": "fp16-components-v1",
                    "allow_patterns": list(item.allow_patterns),
                    "safe_tensors_only": True,
                    "trust_remote_code": False,
                }
                (item.destination / ".aisketcher-model.json").write_text(
                    json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                downloaded.append(item.repo_id)
            paths.append(item.destination)
        return InstallResult(
            preset=plan.preset,
            paths=tuple(paths),
            downloaded=tuple(downloaded),
        )


__all__ = [
    "InstallItem",
    "InstallPlan",
    "InstallResult",
    "PRESETS",
    "PresetDefinition",
    "PresetManager",
    "get_preset",
    "resolve_recipe",
]
