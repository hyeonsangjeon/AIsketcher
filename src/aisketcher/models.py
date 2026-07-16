"""Public value objects for recipes, preparation, generation and replay."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

from .errors import ValidationError

if TYPE_CHECKING:
    from .study import Study


MIN_GENERATION_DIMENSION = 64
MAX_GENERATION_DIMENSION = 4096
MAX_GENERATION_PIXELS = MAX_GENERATION_DIMENSION**2
MAX_SOURCE_PIXELS = 50_000_000


def _validate_dimension(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{name} must be an integer")
    if not MIN_GENERATION_DIMENSION <= value <= MAX_GENERATION_DIMENSION or value % 8:
        raise ValidationError(
            f"{name} must be {MIN_GENERATION_DIMENSION}..{MAX_GENERATION_DIMENSION} "
            "and divisible by 8"
        )


def _validate_generation_size(width: int, height: int, label: str) -> None:
    _validate_dimension(width, f"{label}.width")
    _validate_dimension(height, f"{label}.height")
    if width * height > MAX_GENERATION_PIXELS:
        raise ValidationError(
            f"{label} exceeds the {MAX_GENERATION_PIXELS:,}-pixel generation limit"
        )


def _validate_finite_range(value: float, low: float, high: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be numeric")
    if not math.isfinite(float(value)) or not low <= float(value) <= high:
        raise ValidationError(f"{name} must be finite and between {low:g} and {high:g}")


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return str(self.value)


class StructureMode(_StringEnum):
    LOOSE = "loose"
    BALANCED = "balanced"
    FAITHFUL = "faithful"


class VariationStrength(_StringEnum):
    SUBTLE = "subtle"
    BALANCED = "balanced"
    BOLD = "bold"


class ReplayMode(_StringEnum):
    STRICT = "strict"
    COMPATIBLE = "compatible"


class CapabilitySeverity(_StringEnum):
    WARNING = "warning"
    ERROR = "error"


class SeedMode(_StringEnum):
    SCOUT = "scout"
    LOCKED = "locked"
    EXPLICIT = "explicit"


@dataclass(frozen=True, slots=True)
class CannyConfig:
    """Parameters used to derive a Canny structure control image."""

    low: int = 100
    high: int = 200
    aperture_size: int = 3
    l2_gradient: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.low < self.high <= 255:
            raise ValidationError("Canny thresholds must satisfy 0 <= low < high <= 255")
        if self.aperture_size not in (3, 5, 7):
            raise ValidationError("Canny aperture_size must be 3, 5, or 7")

    def to_dict(self) -> dict[str, Any]:
        return {
            "low": self.low,
            "high": self.high,
            "aperture_size": self.aperture_size,
            "l2_gradient": self.l2_gradient,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CannyConfig:
        if not isinstance(value, Mapping):
            raise ValidationError("CannyConfig must be an object")
        return cls(
            low=int(value.get("low", 100)),
            high=int(value.get("high", 200)),
            aperture_size=int(value.get("aperture_size", 3)),
            l2_gradient=bool(value.get("l2_gradient", False)),
        )


@dataclass(frozen=True, slots=True)
class PreparationDiagnostics:
    """Transparent, model-independent measurements of a prepared sketch."""

    contrast: float
    edge_density: float
    component_count: int
    fragmentation: float
    border_edge_ratio: float
    low_contrast: bool
    edge_sparse: bool
    edge_dense: bool
    crop_risk: bool
    recommended_canny: CannyConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "contrast": round(self.contrast, 6),
            "edge_density": round(self.edge_density, 6),
            "component_count": self.component_count,
            "fragmentation": round(self.fragmentation, 6),
            "border_edge_ratio": round(self.border_edge_ratio, 6),
            "flags": {
                "low_contrast": self.low_contrast,
                "edge_sparse": self.edge_sparse,
                "edge_dense": self.edge_dense,
                "crop_risk": self.crop_risk,
            },
            "recommended_canny": self.recommended_canny.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PreparationDiagnostics:
        if not isinstance(value, Mapping):
            raise ValidationError("PreparationDiagnostics must be an object")
        flags = value.get("flags", {})
        if not isinstance(flags, Mapping):
            raise ValidationError("PreparationDiagnostics.flags must be an object")
        recommended = value.get("recommended_canny", {})
        if not isinstance(recommended, Mapping):
            raise ValidationError(
                "PreparationDiagnostics.recommended_canny must be an object"
            )
        return cls(
            contrast=float(value.get("contrast", 0.0)),
            edge_density=float(value.get("edge_density", 0.0)),
            component_count=int(value.get("component_count", 0)),
            fragmentation=float(value.get("fragmentation", 0.0)),
            border_edge_ratio=float(value.get("border_edge_ratio", 0.0)),
            low_contrast=bool(flags.get("low_contrast", False)),
            edge_sparse=bool(flags.get("edge_sparse", False)),
            edge_dense=bool(flags.get("edge_dense", False)),
            crop_risk=bool(flags.get("crop_risk", False)),
            recommended_canny=CannyConfig.from_dict(recommended),
        )


@dataclass(slots=True)
class PreparedSketch:
    """Normalized source image and its structure control.

    ``source_name`` is deliberately not retained. Only a content hash is kept so
    exported manifests cannot disclose local paths or original filenames.
    """

    image: Image.Image
    control: Image.Image
    original_size: tuple[int, int]
    prepared_size: tuple[int, int]
    source_sha256: str
    control_sha256: str
    canny: CannyConfig
    diagnostics: PreparationDiagnostics

    def __post_init__(self) -> None:
        if len(self.prepared_size) != 2:
            raise ValidationError("PreparedSketch.prepared_size must contain two dimensions")
        _validate_generation_size(
            self.prepared_size[0], self.prepared_size[1], "PreparedSketch.prepared_size"
        )
        if self.image.size != self.prepared_size or self.control.size != self.prepared_size:
            raise ValidationError(
                "PreparedSketch image and control sizes must match prepared_size"
            )
        if (
            len(self.original_size) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in self.original_size
            )
            or self.original_size[0] * self.original_size[1] > MAX_SOURCE_PIXELS
        ):
            raise ValidationError(
                f"PreparedSketch.original_size must be positive and at most "
                f"{MAX_SOURCE_PIXELS:,} pixels"
            )
        for name, value in (
            ("source_sha256", self.source_sha256),
            ("control_sha256", self.control_sha256),
        ):
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValidationError(f"PreparedSketch.{name} must be a SHA-256 hex digest")


@dataclass(frozen=True, slots=True)
class Intent:
    """A designer-facing request, independent of a model implementation."""

    prompt: str
    profile: str = "graphic_design"
    structure: StructureMode | str = StructureMode.BALANCED

    def __post_init__(self) -> None:
        prompt = self.prompt.strip()
        profile = self.profile.strip()
        if not prompt:
            raise ValidationError("Intent.prompt cannot be empty")
        if not profile:
            raise ValidationError("Intent.profile cannot be empty")
        if len(prompt) > 10_000:
            raise ValidationError("Intent.prompt cannot exceed 10,000 characters")
        if len(profile) > 100:
            raise ValidationError("Intent.profile cannot exceed 100 characters")
        try:
            structure = StructureMode(self.structure)
        except ValueError as exc:
            raise ValidationError(
                "Intent.structure must be loose, balanced, or faithful"
            ) from exc
        object.__setattr__(self, "prompt", prompt)
        object.__setattr__(self, "profile", profile)
        object.__setattr__(self, "structure", structure)


@dataclass(frozen=True, slots=True)
class Recipe:
    """Optional expert overrides applied after a preset and an ``Intent``."""

    width: int | None = None
    height: int | None = None
    steps: int | None = None
    guidance_scale: float | None = None
    control_strength: float | None = None
    scheduler: str | None = None
    negative_prompt: str | None = None

    def __post_init__(self) -> None:
        if (self.width is None) != (self.height is None):
            raise ValidationError("Recipe.width and Recipe.height must be set together")
        if self.width is not None and self.height is not None:
            _validate_generation_size(self.width, self.height, "Recipe")
        if self.steps is not None and (
            isinstance(self.steps, bool)
            or not isinstance(self.steps, int)
            or not 1 <= self.steps <= 150
        ):
            raise ValidationError("Recipe.steps must be between 1 and 150")
        if self.guidance_scale is not None:
            _validate_finite_range(
                self.guidance_scale, 0, 30, "Recipe.guidance_scale"
            )
        if self.control_strength is not None:
            _validate_finite_range(
                self.control_strength, 0, 2, "Recipe.control_strength"
            )
        if self.scheduler is not None and (
            not self.scheduler.strip() or len(self.scheduler) > 100
        ):
            raise ValidationError("Recipe.scheduler must be 1..100 characters")
        if self.negative_prompt is not None and len(self.negative_prompt) > 20_000:
            raise ValidationError("Recipe.negative_prompt cannot exceed 20,000 characters")


@dataclass(frozen=True, slots=True)
class ModelReference:
    repo_id: str
    revision: str
    role: str

    def __post_init__(self) -> None:
        for name, value, limit in (
            ("repo_id", self.repo_id, 300),
            ("revision", self.revision, 200),
            ("role", self.role, 50),
        ):
            if not value.strip() or len(value) > limit:
                raise ValidationError(
                    f"ModelReference.{name} must be 1..{limit} characters"
                )

    def to_dict(self) -> dict[str, str]:
        return {"repo_id": self.repo_id, "revision": self.revision, "role": self.role}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ModelReference:
        return cls(
            repo_id=str(value["repo_id"]),
            revision=str(value["revision"]),
            role=str(value["role"]),
        )


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """A backend's explicit contract with the resolver."""

    controls: tuple[str, ...] = ("canny",)
    supports_seed: bool = True
    supports_negative_prompt: bool = True
    supports_variation: bool = True
    schedulers: tuple[str, ...] = ("unipc",)
    max_outputs: int = 8


@dataclass(frozen=True, slots=True)
class CapabilityIssue:
    setting: str
    requested: Any
    applied: Any
    severity: CapabilitySeverity
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "setting": self.setting,
            "requested": self.requested,
            "applied": self.applied,
            "severity": self.severity.value,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class CapabilityReport:
    backend: str
    issues: tuple[CapabilityIssue, ...] = ()

    @property
    def supported(self) -> bool:
        return not any(issue.severity is CapabilitySeverity.ERROR for issue in self.issues)

    @property
    def warnings(self) -> tuple[CapabilityIssue, ...]:
        return tuple(
            issue for issue in self.issues if issue.severity is CapabilitySeverity.WARNING
        )

    @property
    def errors(self) -> tuple[CapabilityIssue, ...]:
        return tuple(
            issue for issue in self.issues if issue.severity is CapabilitySeverity.ERROR
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "supported": self.supported,
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True, slots=True)
class ResolvedRecipe:
    """Concrete, replayable parameters after preset and capability resolution."""

    preset: str
    prompt: str
    profile: str
    structure: StructureMode
    width: int
    height: int
    steps: int
    guidance_scale: float
    control_strength: float
    scheduler: str
    negative_prompt: str
    models: tuple[ModelReference, ...]
    capability_report: CapabilityReport
    variation_strength: VariationStrength | None = None
    locks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.preset.strip() or len(self.preset) > 200:
            raise ValidationError("ResolvedRecipe.preset must be 1..200 characters")
        if not self.prompt.strip() or len(self.prompt) > 10_000:
            raise ValidationError("ResolvedRecipe.prompt must be 1..10,000 characters")
        if not self.profile.strip() or len(self.profile) > 100:
            raise ValidationError("ResolvedRecipe.profile must be 1..100 characters")
        try:
            structure = StructureMode(self.structure)
        except ValueError as exc:
            raise ValidationError("ResolvedRecipe.structure is invalid") from exc
        object.__setattr__(self, "structure", structure)
        _validate_generation_size(self.width, self.height, "ResolvedRecipe")
        if (
            isinstance(self.steps, bool)
            or not isinstance(self.steps, int)
            or not 1 <= self.steps <= 150
        ):
            raise ValidationError("ResolvedRecipe.steps must be between 1 and 150")
        _validate_finite_range(
            self.guidance_scale, 0, 30, "ResolvedRecipe.guidance_scale"
        )
        _validate_finite_range(
            self.control_strength, 0, 2, "ResolvedRecipe.control_strength"
        )
        if not self.scheduler.strip() or len(self.scheduler) > 100:
            raise ValidationError("ResolvedRecipe.scheduler must be 1..100 characters")
        if len(self.negative_prompt) > 20_000:
            raise ValidationError(
                "ResolvedRecipe.negative_prompt cannot exceed 20,000 characters"
            )
        if not self.models or not all(
            isinstance(model, ModelReference) for model in self.models
        ):
            raise ValidationError(
                "ResolvedRecipe.models must contain at least one ModelReference"
            )
        variation = self.variation_strength
        if variation is not None:
            try:
                variation = VariationStrength(variation)
            except ValueError as exc:
                raise ValidationError(
                    "ResolvedRecipe.variation_strength is invalid"
                ) from exc
            object.__setattr__(self, "variation_strength", variation)
        unknown_locks = set(self.locks) - {"structure"}
        if unknown_locks:
            raise ValidationError(
                f"ResolvedRecipe has unknown lock(s): {', '.join(sorted(unknown_locks))}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "preset": self.preset,
            "prompt": self.prompt,
            "profile": self.profile,
            "structure": self.structure.value,
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "guidance_scale": self.guidance_scale,
            "control_strength": self.control_strength,
            "scheduler": self.scheduler,
            "negative_prompt": self.negative_prompt,
            "models": [model.to_dict() for model in self.models],
            "variation_strength": (
                self.variation_strength.value if self.variation_strength else None
            ),
            "locks": list(self.locks),
            "capability_report": self.capability_report.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        backend: str = "replay",
    ) -> ResolvedRecipe:
        report_value = value.get("capability_report", {})
        if not isinstance(report_value, Mapping):
            raise ValidationError("ResolvedRecipe.capability_report must be an object")
        for name in ("width", "height", "steps"):
            raw = value.get(name)
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise ValidationError(f"ResolvedRecipe.{name} must be an integer")
        for name in ("guidance_scale", "control_strength"):
            raw = value.get(name)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValidationError(f"ResolvedRecipe.{name} must be numeric")
        model_values = value.get("models")
        if (
            not isinstance(model_values, list)
            or not model_values
            or not all(isinstance(item, Mapping) for item in model_values)
        ):
            raise ValidationError("ResolvedRecipe.models must be a non-empty list")
        issue_values = report_value.get("issues", [])
        if not isinstance(issue_values, list) or not all(
            isinstance(issue, Mapping) for issue in issue_values
        ):
            raise ValidationError("ResolvedRecipe.capability_report.issues must be a list")
        try:
            issues = tuple(
                CapabilityIssue(
                    setting=str(issue.get("setting", "unknown")),
                    requested=issue.get("requested"),
                    applied=issue.get("applied"),
                    severity=CapabilitySeverity(issue.get("severity", "warning")),
                    message=str(issue.get("message", "")),
                )
                for issue in issue_values
            )
        except ValueError as exc:
            raise ValidationError(
                "ResolvedRecipe.capability_report contains an invalid severity"
            ) from exc
        variation = value.get("variation_strength")
        return cls(
            preset=str(value["preset"]),
            prompt=str(value["prompt"]),
            profile=str(value["profile"]),
            structure=StructureMode(value["structure"]),
            width=value["width"],
            height=value["height"],
            steps=value["steps"],
            guidance_scale=float(value["guidance_scale"]),
            control_strength=float(value["control_strength"]),
            scheduler=str(value["scheduler"]),
            negative_prompt=str(value.get("negative_prompt", "")),
            models=tuple(ModelReference.from_dict(item) for item in model_values),
            variation_strength=VariationStrength(variation) if variation else None,
            locks=tuple(str(lock) for lock in value.get("locks", [])),
            capability_report=CapabilityReport(
                backend=str(report_value.get("backend", backend)), issues=issues
            ),
        )


def _splitmix64(value: int) -> int:
    """Return a stable positive 32-bit seed without relying on Python's hash."""

    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    value ^= value >> 31
    return int(value & 0x7FFFFFFF)


@dataclass(frozen=True, slots=True)
class SeedPlan:
    """A deterministic seed policy that can be serialized and replayed."""

    mode: SeedMode
    base_seed: int = 0
    seeds: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        try:
            mode = SeedMode(self.mode)
        except ValueError as exc:
            raise ValidationError("Unknown seed mode") from exc
        max_seed = (1 << 63) - 1
        if isinstance(self.base_seed, bool) or not isinstance(self.base_seed, int):
            raise ValidationError("base_seed must be an integer")
        if not 0 <= self.base_seed <= max_seed:
            raise ValidationError(f"base_seed must be between 0 and {max_seed}")
        if any(
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not 0 <= seed <= max_seed
            for seed in self.seeds
        ):
            raise ValidationError(f"seeds must be integers between 0 and {max_seed}")
        if mode is SeedMode.EXPLICIT and not self.seeds:
            raise ValidationError("Explicit SeedPlan requires at least one seed")
        object.__setattr__(self, "mode", mode)

    @classmethod
    def scout(cls, count: int | None = None, *, base_seed: int = 0) -> SeedPlan:
        if count is not None and (
            isinstance(count, bool) or not isinstance(count, int) or count < 1
        ):
            raise ValidationError("count must be positive")
        if isinstance(base_seed, bool) or not isinstance(base_seed, int):
            raise ValidationError("base_seed must be an integer")
        seeds = tuple(_splitmix64(base_seed + index) for index in range(count or 0))
        return cls(SeedMode.SCOUT, base_seed=base_seed, seeds=seeds)

    @classmethod
    def locked(cls, seed: int) -> SeedPlan:
        return cls(SeedMode.LOCKED, base_seed=seed, seeds=(seed,))

    @classmethod
    def explicit(cls, seeds: Sequence[int]) -> SeedPlan:
        return cls(SeedMode.EXPLICIT, seeds=tuple(seeds))

    def resolve(self, count: int) -> tuple[int, ...]:
        if count < 1:
            raise ValidationError("count must be positive")
        if self.mode is SeedMode.LOCKED:
            return (self.base_seed,) * count
        if self.mode is SeedMode.EXPLICIT:
            if len(self.seeds) != count:
                raise ValidationError(
                    f"Explicit SeedPlan has {len(self.seeds)} seeds but {count} outputs were requested"
                )
            return self.seeds
        if self.seeds:
            if len(self.seeds) != count:
                raise ValidationError(
                    f"Scout SeedPlan was prepared for {len(self.seeds)} outputs, not {count}"
                )
            return self.seeds
        return tuple(_splitmix64(self.base_seed + index) for index in range(count))

    def to_dict(self, count: int) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "base_seed": self.base_seed,
            "seeds": list(self.resolve(count)),
        }


@dataclass(frozen=True, slots=True)
class TechnicalScores:
    """Auditable structural scores; these are not aesthetic judgments."""

    structure_similarity: float
    edge_cleanliness: float
    distinctiveness: float
    badges: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "structure_similarity": round(self.structure_similarity, 6),
            "edge_cleanliness": round(self.edge_cleanliness, 6),
            "distinctiveness": round(self.distinctiveness, 6),
            "badges": list(self.badges),
            "method": "edge-iou/density-distance/mean-pixel-distance-v1",
        }


@dataclass(slots=True)
class GenerationRequest:
    prepared: PreparedSketch
    recipe: ResolvedRecipe
    seeds: tuple[int, ...]
    init_image: Image.Image | None = None
    denoise_strength: float | None = None


@dataclass(slots=True)
class GenerationResult:
    image: Image.Image
    seed: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Candidate:
    id: str
    image: Image.Image
    seed: int
    recipe: ResolvedRecipe
    scores: TechnicalScores
    prepared: PreparedSketch = field(repr=False)
    parent_id: str | None = None
    backend_metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)


@dataclass(slots=True)
class ReplayReport:
    manifest_path: Path
    mode: ReplayMode
    verified_files: tuple[str, ...]
    drift: tuple[str, ...]
    warnings: tuple[str, ...]
    candidate_hash_matches: tuple[bool, ...] = ()
    study: Study | None = None

    @property
    def replayed(self) -> bool:
        return self.study is not None

    @property
    def exact_candidate_match(self) -> bool:
        """Whether all regenerated candidates match the exported PNG hashes."""

        return bool(self.candidate_hash_matches) and all(self.candidate_hash_matches)
