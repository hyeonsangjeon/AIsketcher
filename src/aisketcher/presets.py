"""Versioned, pinned model presets and explicit installation planning."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import platform
import shutil
import stat
import tempfile
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ModelUnavailableError, OptionalDependencyError, ValidationError
from .model_registry import (
    MODEL_ARTIFACTS,
    HashPolicy,
    ModelArtifact,
    VerifiedFile,
)
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
FLUX2_KLEIN_BASE = ModelReference(
    repo_id="black-forest-labs/FLUX.2-klein-4B",
    revision="e7b7dc27f91deacad38e78976d1f2b499d76a294",
    role="base-edit",
)
FLUX2_SMALL_DECODER = ModelReference(
    repo_id="black-forest-labs/FLUX.2-small-decoder",
    revision="a3efc24f613ef42d9428af62fdbd6f5fd8856c4a",
    role="decoder",
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
    required_control: str = "canny"
    max_dimension: int | None = None


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
    "flux2-klein-edit@1": PresetDefinition(
        name="flux2-klein-edit@1",
        label="FLUX.2 Klein Edit",
        models=(FLUX2_KLEIN_BASE, FLUX2_SMALL_DECODER),
        estimated_bytes=16_229_653_713,
        steps=4,
        guidance_scale=1.0,
        scheduler="flow-match-euler",
        negative_prompt="",
        required_control="reference-image",
        max_dimension=1024,
    ),
}

_ALIASES = {
    "auto": "flux2-klein-edit@1",
    "flux": "flux2-klein-edit@1",
    "flux2": "flux2-klein-edit@1",
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
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "text_encoder_2/config.json",
    "text_encoder_2/model.fp16.safetensors",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "tokenizer_2/merges.txt",
    "tokenizer_2/special_tokens_map.json",
    "tokenizer_2/tokenizer_config.json",
    "tokenizer_2/vocab.json",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
)
_CONTROLNET_ALLOW_PATTERNS = (
    "config.json",
    "diffusion_pytorch_model.fp16.safetensors",
)
_FLUX2_BASE_ALLOW_PATTERNS = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors.index.json",
    "text_encoder/model-00001-of-00002.safetensors",
    "text_encoder/model-00002-of-00002.safetensors",
    "tokenizer/added_tokens.json",
    "tokenizer/chat_template.jinja",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
)
_FLUX2_DECODER_ALLOW_PATTERNS = (
    "config.json",
    "diffusion_pytorch_model.safetensors",
)
_MODEL_BYTES = {
    (SDXL_BASE.repo_id, SDXL_BASE.revision): 6_941_187_536,
    (SDXL_CANNY_LITE.repo_id, SDXL_CANNY_LITE.revision): 320_238_438,
    (SDXL_CANNY_QUALITY.repo_id, SDXL_CANNY_QUALITY.revision): 2_502_140_445,
    (FLUX2_KLEIN_BASE.repo_id, FLUX2_KLEIN_BASE.revision): 15_980_131_531,
    (FLUX2_SMALL_DECODER.repo_id, FLUX2_SMALL_DECODER.revision): 249_522_182,
}

_MARKER_NAME = ".aisketcher-model.json"
_MARKER_SCHEMA = "aisketcher-model-cache"
_MARKER_SCHEMA_VERSION = 2
_HF_LOCAL_DIR_ROOT = (".cache", "huggingface")
_HF_LOCAL_DIR_CONTROL_FILES = {
    (*_HF_LOCAL_DIR_ROOT, ".gitignore"),
    (*_HF_LOCAL_DIR_ROOT, ".gitignore.lock"),
    (*_HF_LOCAL_DIR_ROOT, "CACHEDIR.TAG"),
}
_MARKER_POLICY_VERSION = 2
_MAX_MARKER_BYTES = 1_048_576
_HASH_CHUNK_SIZE = 8 * 1024 * 1024


class _CacheIntegrityError(RuntimeError):
    """An installed snapshot does not match the immutable registry."""


@dataclass(frozen=True, slots=True)
class _FileStatFingerprint:
    path: str
    size_bytes: int
    mtime_ns: int
    ctime_ns: int
    device: int
    inode: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
            "ctime_ns": self.ctime_ns,
            "device": self.device,
            "inode": self.inode,
        }


@dataclass(frozen=True, slots=True)
class _VerificationReceipt:
    """Process-local trust anchor for the stat-only cache fast path.

    The persistent marker is intentionally not an authentication token: an
    attacker or broken tool with write access to the cache can forge it. A new
    ``PresetManager`` therefore performs one full registry-backed hash pass
    before issuing this in-memory receipt. Subsequent checks in the same
    process may skip multi-GB rehashing only while both the exact marker bytes
    and every regular file's size/mtime/ctime/device/inode remain unchanged.
    A privileged attacker capable of changing process memory or forging kernel
    stat metadata is outside this local-cache corruption threat model.
    """

    marker_sha256: str
    file_stats: tuple[_FileStatFingerprint, ...]


def _role_allow_patterns(model: ModelReference) -> tuple[str, ...]:
    if model.role == "base":
        return _BASE_ALLOW_PATTERNS
    if model.role == "controlnet":
        return _CONTROLNET_ALLOW_PATTERNS
    if model.role == "base-edit":
        return _FLUX2_BASE_ALLOW_PATTERNS
    if model.role == "decoder":
        return _FLUX2_DECODER_ALLOW_PATTERNS
    raise ValidationError(f"No safe download policy exists for model role {model.role!r}")


def _download_policy(model: ModelReference) -> str:
    if model.role in {"base", "controlnet"}:
        return "fp16-components-v1"
    if model.role in {"base-edit", "decoder"}:
        return "safetensors-components-v1"
    raise ValidationError(f"No safe download policy exists for model role {model.role!r}")


def _artifact_for_model(model: ModelReference) -> ModelArtifact | None:
    """Return the one immutable registry artifact backing ``model``.

    A preset without an exact registry entry and payload digests is not
    installable. This deliberately fails closed instead of falling back to
    revision-only or file-existence checks.
    """

    matches = tuple(
        artifact
        for artifact in MODEL_ARTIFACTS.values()
        if artifact.model_id == model.repo_id and artifact.revision == model.revision
    )
    if len(matches) != 1:
        return None
    artifact = matches[0]
    if (
        artifact.hash_policy is not HashPolicy.PINNED_COMMIT_AND_RUNTIME_SHA256
        or not artifact.files
    ):
        return None
    role_patterns = _role_allow_patterns(model)
    artifact_paths = tuple(required.path for required in artifact.files)
    if set(artifact_paths) != set(role_patterns):
        return None
    return artifact


def _allow_patterns(model: ModelReference) -> tuple[str, ...]:
    """Return the exact reviewed runtime files for an installable model."""

    artifact = _artifact_for_model(model)
    if artifact is None:
        raise ValidationError(
            f"Model {model.repo_id!r}@{model.revision} has no complete "
            "registry-backed runtime SHA-256 policy"
        )
    return tuple(required.path for required in artifact.files)


def _artifact_fingerprint(artifact: ModelArtifact) -> list[dict[str, int | str]]:
    return [
        {
            "path": required.path,
            "size_bytes": required.size_bytes,
            "sha256": required.sha256,
        }
        for required in artifact.files
    ]


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _download_pattern_groups(
    allow_patterns: tuple[str, ...],
) -> tuple[tuple[str, ...], ...]:
    """Split a curated allowlist into deterministic cancellation boundaries.

    Each exact path or glob remains a separate ``snapshot_download`` request.
    Hugging Face reuses files already present in ``local_dir``, while
    AIsketcher gets a safe point to observe cancellation after every curated
    component group.
    """

    return tuple((pattern,) for pattern in allow_patterns)


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

    if preset.required_control not in capabilities.controls:
        control_label = (
            "Canny"
            if preset.required_control == "canny"
            else preset.required_control.replace("-", " ")
        )
        issues.append(
            CapabilityIssue(
                setting="control",
                requested=preset.required_control,
                applied=None,
                severity=CapabilitySeverity.ERROR,
                message=(
                    f"This preset requires a {control_label}-capable backend."
                ),
            )
        )
    if preset.max_dimension is not None and (
        width > preset.max_dimension or height > preset.max_dimension
    ):
        issues.append(
            CapabilityIssue(
                setting="dimensions",
                requested=f"{width}x{height}",
                applied=None,
                severity=CapabilitySeverity.ERROR,
                message=(
                    f"This preset supports dimensions up to "
                    f"{preset.max_dimension}x{preset.max_dimension}."
                ),
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
        model_prompt=intent.model_prompt,
        prompt_metadata=intent.prompt_metadata,
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


def _path_has_symlink_component(path: Path) -> bool:
    """Fail closed when any existing component of an absolute path is a symlink."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            continue
        except OSError:
            return True
        if stat.S_ISLNK(metadata.st_mode):
            return True
    return False


def _is_safe_hf_local_dir_bookkeeping(
    relative: Path, runtime_files: set[str], revision: str
) -> bool:
    """Allow only Hugging Face's non-runtime ``local_dir`` bookkeeping files."""

    if relative.parts in _HF_LOCAL_DIR_CONTROL_FILES:
        return True
    if (
        len(revision) == 40
        and all(character in "0123456789abcdef" for character in revision)
        and relative.parts
        == (*_HF_LOCAL_DIR_ROOT, "trees", f"{revision}.json")
    ):
        return True
    if relative.parts[:3] != (*_HF_LOCAL_DIR_ROOT, "download"):
        return False

    bookkeeping_name = Path(*relative.parts[3:]).as_posix()
    for runtime_name in runtime_files:
        metadata_name = f"{runtime_name}.metadata"
        lock_name = Path(metadata_name).with_suffix(".lock").as_posix()
        if bookkeeping_name in {metadata_name, lock_name}:
            return True
    return False


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
        configured_cache = (
            Path(cache_dir) if cache_dir is not None else _default_cache_dir()
        )
        self.cache_dir = Path(os.path.abspath(os.fspath(configured_cache)))
        self.allow_downloads = allow_downloads
        self._verification_receipts: dict[Path, _VerificationReceipt] = {}

    @staticmethod
    def available() -> tuple[PresetDefinition, ...]:
        return tuple(PRESETS[name] for name in sorted(PRESETS))

    def _cache_boundary_has_symlink(self) -> bool:
        return _path_has_symlink_component(self.cache_dir / "models")

    def _destination(self, model: ModelReference) -> Path:
        if _artifact_for_model(model) is None:
            raise ValidationError(
                f"Model {model.repo_id!r}@{model.revision} has no complete "
                "registry-backed runtime SHA-256 policy"
            )
        safe_repo = model.repo_id.replace("/", "--")
        return self.cache_dir / "models" / f"{safe_repo}@{model.revision}"

    def model_destination(self, model: ModelReference) -> Path:
        """Return the managed local directory without implying it is installed."""

        if not isinstance(model, ModelReference):
            raise TypeError("model must be a ModelReference")
        _allow_patterns(model)
        return self._destination(model)

    @staticmethod
    def _required_files_present(
        destination: Path,
        allow_patterns: tuple[str, ...],
        *,
        revision: str,
    ) -> bool:
        if destination.is_symlink() or not destination.is_dir():
            return False
        root = destination.resolve()
        try:
            descendants = tuple(destination.rglob("*"))
        except OSError:
            return False
        if any(path.is_symlink() for path in descendants):
            return False
        for pattern in allow_patterns:
            matches = []
            for candidate in destination.glob(pattern):
                if candidate.is_symlink() or not candidate.is_file():
                    continue
                try:
                    candidate.resolve().relative_to(root)
                except (OSError, ValueError):
                    continue
                matches.append(candidate)
            if not matches:
                return False
        runtime_files = set(allow_patterns)
        for path in descendants:
            if path.is_dir():
                continue
            if not path.is_file():
                return False
            try:
                relative = path.resolve().relative_to(root)
            except (OSError, ValueError):
                return False
            relative_name = relative.as_posix()
            if relative_name in runtime_files or relative_name == _MARKER_NAME:
                continue
            if _is_safe_hf_local_dir_bookkeeping(
                relative,
                runtime_files,
                revision,
            ):
                continue
            return False
        return True

    @staticmethod
    def _safe_verified_path(destination: Path, required: VerifiedFile) -> Path:
        if destination.is_symlink() or not destination.is_dir():
            raise _CacheIntegrityError("model destination is not a regular directory")
        root = destination.resolve()
        candidate = destination.joinpath(*required.path.split("/"))
        current = candidate
        while current != destination:
            if current.is_symlink():
                raise _CacheIntegrityError(
                    f"verified artifact path contains a symlink: {required.path}"
                )
            current = current.parent
        try:
            candidate.resolve(strict=True).relative_to(root)
            metadata = candidate.lstat()
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise _CacheIntegrityError(
                f"verified artifact is missing or escapes the cache: {required.path}"
            ) from exc
        if not stat.S_ISREG(metadata.st_mode):
            raise _CacheIntegrityError(
                f"verified artifact is not a regular file: {required.path}"
            )
        return candidate

    @staticmethod
    def _stat_fingerprint(
        required: VerifiedFile, metadata: os.stat_result
    ) -> _FileStatFingerprint:
        return _FileStatFingerprint(
            path=required.path,
            size_bytes=metadata.st_size,
            mtime_ns=metadata.st_mtime_ns,
            ctime_ns=metadata.st_ctime_ns,
            device=metadata.st_dev,
            inode=metadata.st_ino,
        )

    def _collect_file_stats(
        self, destination: Path, artifact: ModelArtifact
    ) -> tuple[_FileStatFingerprint, ...]:
        fingerprints: list[_FileStatFingerprint] = []
        for required in artifact.files:
            candidate = self._safe_verified_path(destination, required)
            metadata = candidate.lstat()
            if metadata.st_size != required.size_bytes:
                raise _CacheIntegrityError(
                    f"verified artifact size mismatch: {required.path}"
                )
            fingerprints.append(self._stat_fingerprint(required, metadata))
        return tuple(fingerprints)

    def _verify_required_artifacts(
        self,
        destination: Path,
        artifact: ModelArtifact,
        *,
        cancellation_token: Any = None,
        cancel_event: Any = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> tuple[_FileStatFingerprint, ...]:
        fingerprints: list[_FileStatFingerprint] = []
        for required in artifact.files:
            candidate = self._safe_verified_path(destination, required)
            flags = os.O_RDONLY
            if hasattr(os, "O_CLOEXEC"):
                flags |= os.O_CLOEXEC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            try:
                descriptor = os.open(candidate, flags)
            except OSError as exc:
                raise _CacheIntegrityError(
                    f"could not safely open verified artifact: {required.path}"
                ) from exc
            digest = hashlib.sha256()
            with os.fdopen(descriptor, "rb", closefd=True) as stream:
                before = os.fstat(stream.fileno())
                if (
                    not stat.S_ISREG(before.st_mode)
                    or before.st_size != required.size_bytes
                ):
                    raise _CacheIntegrityError(
                        f"verified artifact size/type mismatch: {required.path}"
                    )
                while True:
                    self._raise_if_install_cancelled(
                        cancellation_token=cancellation_token,
                        cancel_event=cancel_event,
                        should_cancel=should_cancel,
                    )
                    chunk = stream.read(_HASH_CHUNK_SIZE)
                    if not chunk:
                        break
                    digest.update(chunk)
                self._raise_if_install_cancelled(
                    cancellation_token=cancellation_token,
                    cancel_event=cancel_event,
                    should_cancel=should_cancel,
                )
                after = os.fstat(stream.fileno())
            if (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise _CacheIntegrityError(
                    f"verified artifact changed while hashing: {required.path}"
                )
            if not hmac.compare_digest(digest.hexdigest(), required.sha256):
                raise _CacheIntegrityError(
                    f"verified artifact SHA-256 mismatch: {required.path}"
                )
            current = self._safe_verified_path(destination, required).lstat()
            if (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ) != (
                current.st_dev,
                current.st_ino,
                current.st_size,
                current.st_mtime_ns,
                current.st_ctime_ns,
            ):
                raise _CacheIntegrityError(
                    f"verified artifact changed after hashing: {required.path}"
                )
            fingerprints.append(self._stat_fingerprint(required, current))
        return tuple(fingerprints)

    @staticmethod
    def _marker_payload(
        model: ModelReference,
        artifact: ModelArtifact,
        allow_patterns: tuple[str, ...],
        file_stats: tuple[_FileStatFingerprint, ...],
    ) -> dict[str, Any]:
        return {
            "schema": _MARKER_SCHEMA,
            "schema_version": _MARKER_SCHEMA_VERSION,
            "policy_version": _MARKER_POLICY_VERSION,
            "hash_policy": artifact.hash_policy.value,
            "artifact_id": artifact.artifact_id,
            "artifact_fingerprint": _artifact_fingerprint(artifact),
            "verified_file_stats": [item.to_dict() for item in file_stats],
            "repo_id": model.repo_id,
            "revision": model.revision,
            "role": model.role,
            "download_policy": _download_policy(model),
            "allow_patterns": list(allow_patterns),
            "safe_tensors_only": True,
            "trust_remote_code": False,
        }

    @staticmethod
    def _read_marker(marker: Path) -> tuple[dict[str, Any], bytes] | None:
        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(marker, flags)
            with os.fdopen(descriptor, "rb", closefd=True) as stream:
                before = os.fstat(stream.fileno())
                if (
                    not stat.S_ISREG(before.st_mode)
                    or before.st_size <= 0
                    or before.st_size > _MAX_MARKER_BYTES
                ):
                    return None
                raw = stream.read(_MAX_MARKER_BYTES + 1)
                after = os.fstat(stream.fileno())
            if len(raw) > _MAX_MARKER_BYTES or (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                return None
            payload = json.loads(raw)
        except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        return payload, raw

    @staticmethod
    def _marker_identifies_model(payload: dict[str, Any], model: ModelReference) -> bool:
        return (
            payload.get("repo_id") == model.repo_id
            and payload.get("revision") == model.revision
        )

    @classmethod
    def _current_marker_matches(
        cls,
        payload: dict[str, Any],
        model: ModelReference,
        artifact: ModelArtifact,
        allow_patterns: tuple[str, ...],
        file_stats: tuple[_FileStatFingerprint, ...],
    ) -> bool:
        expected = cls._marker_payload(model, artifact, allow_patterns, file_stats)
        return payload == expected

    @staticmethod
    def _write_marker_atomic(marker: Path, payload: dict[str, Any]) -> bytes:
        raw = _canonical_json_bytes(payload)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=marker.parent,
                prefix=f"{marker.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_name = temporary.name
                os.chmod(temporary.name, 0o600)
                temporary.write(raw)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, marker)
            temporary_name = None
        finally:
            if temporary_name is not None:
                with suppress(FileNotFoundError):
                    Path(temporary_name).unlink()
        return raw

    def _remember_verification(
        self,
        destination: Path,
        marker_raw: bytes,
        file_stats: tuple[_FileStatFingerprint, ...],
    ) -> None:
        self._verification_receipts[destination] = _VerificationReceipt(
            marker_sha256=hashlib.sha256(marker_raw).hexdigest(),
            file_stats=file_stats,
        )

    def model_path(
        self,
        model: ModelReference,
        *,
        verify_cache: bool = True,
        cancellation_token: Any = None,
        cancel_event: Any = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> Path | None:
        artifact = _artifact_for_model(model)
        if artifact is None:
            return None
        if self._cache_boundary_has_symlink():
            return None
        destination = self._destination(model)
        marker = destination / _MARKER_NAME
        marker_result = self._read_marker(marker)
        if marker_result is None:
            return None
        payload, marker_raw = marker_result
        if not self._marker_identifies_model(payload, model):
            return None
        allow_patterns = _allow_patterns(model)
        if not self._required_files_present(
            destination,
            allow_patterns,
            revision=model.revision,
        ):
            return None
        try:
            current_stats = self._collect_file_stats(destination, artifact)
        except _CacheIntegrityError:
            return None
        receipt = self._verification_receipts.get(destination)
        marker_digest = hashlib.sha256(marker_raw).hexdigest()
        if (
            receipt is not None
            and receipt.marker_sha256 == marker_digest
            and receipt.file_stats == current_stats
            and self._current_marker_matches(
                payload,
                model,
                artifact,
                allow_patterns,
                current_stats,
            )
        ):
            return destination
        if not verify_cache:
            return None
        try:
            verified_stats = self._verify_required_artifacts(
                destination,
                artifact,
                cancellation_token=cancellation_token,
                cancel_event=cancel_event,
                should_cancel=should_cancel,
            )
        except _CacheIntegrityError:
            self._verification_receipts.pop(destination, None)
            return None
        marker_payload = self._marker_payload(
            model,
            artifact,
            allow_patterns,
            verified_stats,
        )
        try:
            marker_raw = self._write_marker_atomic(marker, marker_payload)
        except OSError:
            self._verification_receipts.pop(destination, None)
            return None
        self._remember_verification(destination, marker_raw, verified_stats)
        return destination

    def plan_install(
        self,
        preset_name: str,
        *,
        verify_cache: bool = True,
        cancellation_token: Any = None,
        cancel_event: Any = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> InstallPlan:
        """Describe a preset without weakening cache-integrity semantics.

        ``verify_cache=False`` is intended for latency-sensitive display code.
        It never turns a persistent marker into trust: only a current
        process-local verification receipt may report an item as installed.
        A fresh process therefore reports existing snapshots as unverified
        until an explicit default/full plan or install performs SHA-256 checks.
        """

        preset = get_preset(preset_name)
        items = tuple(
            InstallItem(
                repo_id=model.repo_id,
                revision=model.revision,
                role=model.role,
                destination=self._destination(model),
                installed=(
                    self.model_path(
                        model,
                        verify_cache=verify_cache,
                        cancellation_token=cancellation_token,
                        cancel_event=cancel_event,
                        should_cancel=should_cancel,
                    )
                    is not None
                ),
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

    @staticmethod
    def _install_cancelled(
        *,
        cancellation_token: Any = None,
        cancel_event: Any = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> bool:
        for candidate in (cancellation_token, cancel_event):
            is_set = getattr(candidate, "is_set", None)
            if callable(is_set) and bool(is_set()):
                return True
        return bool(should_cancel is not None and should_cancel())

    @classmethod
    def _raise_if_install_cancelled(
        cls,
        *,
        cancellation_token: Any = None,
        cancel_event: Any = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> None:
        if cls._install_cancelled(
            cancellation_token=cancellation_token,
            cancel_event=cancel_event,
            should_cancel=should_cancel,
        ):
            raise ModelUnavailableError(
                "Model installation was cancelled safely."
            )

    def _remove_incomplete_destination(self, destination: Path) -> None:
        """Remove only the exact managed entry, never a symlink target."""

        models_root = self.cache_dir / "models"
        if destination.parent != models_root or destination == models_root:
            raise ValidationError("Refusing to remove a path outside the managed model cache")
        if self._cache_boundary_has_symlink():
            raise ValidationError(
                "Refusing to remove through a symlinked model cache boundary"
            )
        self._verification_receipts.pop(destination, None)
        if destination.is_symlink():
            destination.unlink(missing_ok=True)
        elif destination.exists():
            shutil.rmtree(destination)

    def install(
        self,
        preset_name: str,
        *,
        confirm: bool = False,
        cancellation_token: Any = None,
        cancel_event: Any = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> InstallResult:
        plan = self.plan_install(
            preset_name,
            cancellation_token=cancellation_token,
            cancel_event=cancel_event,
            should_cancel=should_cancel,
        )
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
            self._raise_if_install_cancelled(
                cancellation_token=cancellation_token,
                cancel_event=cancel_event,
                should_cancel=should_cancel,
            )
            if not item.installed:
                if self._cache_boundary_has_symlink():
                    raise ModelUnavailableError(
                        "Managed model cache boundary cannot contain a symlink"
                    )
                self._remove_incomplete_destination(item.destination)
                item.destination.mkdir(parents=True, exist_ok=False)
                try:
                    for allow_pattern_group in _download_pattern_groups(
                        item.allow_patterns
                    ):
                        self._raise_if_install_cancelled(
                            cancellation_token=cancellation_token,
                            cancel_event=cancel_event,
                            should_cancel=should_cancel,
                        )
                        snapshot_download(
                            repo_id=item.repo_id,
                            revision=item.revision,
                            local_dir=item.destination,
                            allow_patterns=list(allow_pattern_group),
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
                        self._raise_if_install_cancelled(
                            cancellation_token=cancellation_token,
                            cancel_event=cancel_event,
                            should_cancel=should_cancel,
                        )
                    if not self._required_files_present(
                        item.destination,
                        item.allow_patterns,
                        revision=item.revision,
                    ):
                        raise ModelUnavailableError(
                            f"Downloaded snapshot for {item.repo_id!r} is incomplete or unsafe"
                        )
                    model = ModelReference(
                        repo_id=item.repo_id,
                        revision=item.revision,
                        role=item.role,
                    )
                    artifact = _artifact_for_model(model)
                    if artifact is None:
                        raise ModelUnavailableError(
                            f"Model {item.repo_id!r}@{item.revision} has no complete "
                            "registry-backed SHA-256 policy"
                        )
                    try:
                        verified_stats = self._verify_required_artifacts(
                            item.destination,
                            artifact,
                            cancellation_token=cancellation_token,
                            cancel_event=cancel_event,
                            should_cancel=should_cancel,
                        )
                    except _CacheIntegrityError as exc:
                        raise ModelUnavailableError(
                            f"Downloaded snapshot for {item.repo_id!r} failed "
                            "registry-backed SHA-256 verification"
                        ) from exc
                    marker_payload = self._marker_payload(
                        model,
                        artifact,
                        item.allow_patterns,
                        verified_stats,
                    )
                    marker_raw = self._write_marker_atomic(
                        item.destination / _MARKER_NAME,
                        marker_payload,
                    )
                    self._remember_verification(
                        item.destination,
                        marker_raw,
                        verified_stats,
                    )
                except Exception:
                    self._remove_incomplete_destination(item.destination)
                    raise
                downloaded.append(item.repo_id)
            paths.append(item.destination)
        self._raise_if_install_cancelled(
            cancellation_token=cancellation_token,
            cancel_event=cancel_event,
            should_cancel=should_cancel,
        )
        return InstallResult(
            preset=plan.preset,
            paths=tuple(paths),
            downloaded=tuple(downloaded),
        )


__all__ = [
    "FLUX2_KLEIN_BASE",
    "FLUX2_SMALL_DECODER",
    "InstallItem",
    "InstallPlan",
    "InstallResult",
    "PRESETS",
    "PresetDefinition",
    "PresetManager",
    "get_preset",
    "resolve_recipe",
]
