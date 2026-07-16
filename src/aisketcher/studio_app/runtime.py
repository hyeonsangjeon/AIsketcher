"""Framework-independent runtime for the AIsketcher Studio example.

The Gradio layer is intentionally thin.  Keeping validation, state handling,
sample discovery, and SDK calls here makes the demo testable with a FakeStudio
and importable without Gradio, Torch, or Diffusers.
"""

from __future__ import annotations

import inspect
import json
import re
import secrets
import shutil
import stat
import tempfile
import threading
import time
import uuid
import weakref
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from ..manifest import canonical_sha256
from .i18n import normalize_language, text

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 50_000_000
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_REPLAY_ARCHIVE_BYTES = MAX_UPLOAD_BYTES
MAX_REPLAY_FILES = 64
MAX_REPLAY_FILE_BYTES = 50 * 1024 * 1024
MAX_REPLAY_UNCOMPRESSED_BYTES = 200 * 1024 * 1024
ALLOWED_IMAGE_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})
OUTPUT_COUNTS = frozenset({1, 4, 8})
DISPLAY_BADGE_KEYS = {
    "most distinct": "badge_most_distinct",
    "closest structure": "badge_closest_structure",
    "cleanest edges": "badge_cleanest_edges",
}


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class StudioAppError(RuntimeError):
    """A safe error that may be shown directly in the example UI."""


class StudioFactory(Protocol):
    """Callable used by :class:`ModelPool` to construct a backend studio."""

    def __call__(self, preset: str) -> Any: ...


@dataclass(frozen=True, slots=True)
class AppState:
    """Lightweight values kept in ``gr.State``.

    Models, prepared images, and Study instances live in the bounded server-side
    registry.  No credentials, image arrays, or absolute paths are serialized
    into browser/session state.
    """

    session_id: str
    run_id: str | None = None
    selected_index: int | None = None
    language: str = "en"
    view: str = "simple"
    advanced_overrides: bool = False
    guided: bool = False

    @classmethod
    def new(cls, language: str = "en") -> AppState:
        return cls(session_id=uuid.uuid4().hex, language=normalize_language(language))

    @classmethod
    def from_payload(cls, value: Mapping[str, Any] | AppState | None) -> AppState:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            return cls.new()
        session_id = str(value.get("session_id") or uuid.uuid4().hex)
        if not session_id.isalnum() or len(session_id) > 64:
            session_id = uuid.uuid4().hex
        run_id = value.get("run_id")
        run_id = str(run_id) if run_id else None
        selected = value.get("selected_index")
        if not isinstance(selected, int):
            selected = None
        view = str(value.get("view", "simple"))
        if view not in {"simple", "advanced", "guide"}:
            view = "simple"
        return cls(
            session_id=session_id,
            run_id=run_id,
            selected_index=selected,
            language=normalize_language(str(value.get("language", "en"))),
            view=view,
            advanced_overrides=bool(value.get("advanced_overrides", False)),
            guided=bool(value.get("guided", False)),
        )

    def payload(self) -> dict[str, str | int | bool | None]:
        return asdict(self)

    def replace(self, **changes: Any) -> AppState:
        values = asdict(self)
        values.update(changes)
        return AppState(**values)


@dataclass(frozen=True, slots=True)
class StudioIntent:
    """Fallback intent passed to a FakeStudio when the SDK is not installed."""

    prompt: str
    profile: str
    structure: str


@dataclass(frozen=True, slots=True)
class StudioSeedPlan:
    """Fallback seed plan passed to a FakeStudio when the SDK is not installed."""

    mode: str
    count: int
    seeds: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class StudioRecipe:
    """Fallback recipe overrides passed to a FakeStudio."""

    steps: int
    guidance_scale: float


@dataclass(frozen=True, slots=True)
class CandidateView:
    path: str
    label: str
    reason: str = ""
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class GuidedSample:
    root: Path
    manifest_path: Path
    source_path: Path
    candidates: tuple[CandidateView, ...]
    selected_index: int
    provenance: str
    prompt: str | None = None
    profile: str | None = None
    structure: str | None = None


def _recipe_control_values(
    manifest: Mapping[str, Any],
) -> tuple[str | None, str | None, str | None]:
    """Return only manifest-authenticated values the Simple UI can represent."""

    prompt = profile = structure = None
    recipe = manifest.get("recipe")
    recipe_hash = manifest.get("recipe_sha256")
    if not (
        isinstance(recipe, Mapping)
        and isinstance(recipe_hash, str)
        and re.fullmatch(r"[0-9a-fA-F]{64}", recipe_hash)
        and secrets.compare_digest(canonical_sha256(recipe), recipe_hash.lower())
    ):
        return prompt, profile, structure
    raw_prompt = recipe.get("prompt")
    if isinstance(raw_prompt, str) and raw_prompt.strip() and len(raw_prompt) <= 600:
        prompt = raw_prompt.strip()
    raw_profile = recipe.get("profile")
    if isinstance(raw_profile, str) and raw_profile in {
        "product_design",
        "graphic_design",
        "sketch",
    }:
        profile = raw_profile
    raw_structure = recipe.get("structure")
    if isinstance(raw_structure, str) and raw_structure in {
        "loose",
        "balanced",
        "faithful",
    }:
        structure = raw_structure
    return prompt, profile, structure


class GuidedSampleCatalog:
    """Discover a bundled sample only when its real manifest is complete."""

    def __init__(self, root: str | Path | None = None) -> None:
        if root is not None:
            self.root = Path(root)
            return

        bundled = files("aisketcher.studio_app").joinpath("assets", "pocket-kingdom")
        # Wheels are installed as ordinary directories by supported Python
        # installers.  Converting the Traversable here keeps all downstream
        # validation and Gradio allowed-path checks on resolved filesystem paths.
        self.root = Path(str(bundled))

    @staticmethod
    def _inside(root: Path, value: str) -> Path:
        candidate = (root / value).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError as exc:
            raise StudioAppError("Guided Sample manifest contains an unsafe path.") from exc
        if not candidate.is_file():
            raise StudioAppError(f"Guided Sample asset is missing: {value}")
        return candidate

    @staticmethod
    def _path_value(spec: Any) -> str:
        if isinstance(spec, str):
            return spec
        if isinstance(spec, Mapping) and isinstance(spec.get("path"), str):
            return str(spec["path"])
        raise StudioAppError("Guided Sample manifest has an invalid artifact entry.")

    def load(self) -> GuidedSample:
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise StudioAppError("Guided Sample manifest is not bundled.")
        if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
            raise StudioAppError("Guided Sample manifest is unexpectedly large.")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StudioAppError("Guided Sample manifest cannot be read.") from exc
        if not isinstance(manifest, Mapping):
            raise StudioAppError("Guided Sample manifest must be a JSON object.")
        schema = manifest.get("schema") or manifest.get("schema_version")
        if schema != "aisketcher.manifest/v1":
            raise StudioAppError("Guided Sample requires an aisketcher.manifest/v1 manifest.")
        files = manifest.get("files")
        if not isinstance(files, Mapping) or not files:
            raise StudioAppError("Guided Sample requires the canonical manifest file table.")
        verified: dict[str, Path] = {}
        for key, descriptor in files.items():
            if not isinstance(key, str) or not isinstance(descriptor, Mapping):
                raise StudioAppError("Guided Sample manifest has an invalid file descriptor.")
            artifact = self._inside(self.root, self._path_value(descriptor))
            expected = descriptor.get("sha256")
            if not isinstance(expected, str) or len(expected) != 64:
                raise StudioAppError("Guided Sample manifest has an invalid artifact hash.")
            if artifact.stat().st_size > MAX_REPLAY_FILE_BYTES:
                raise StudioAppError("Guided Sample artifact exceeds the 50 MB limit.")
            digest = _file_sha256(artifact)
            if digest != expected.lower():
                raise StudioAppError(f"Guided Sample artifact hash mismatch: {key}")
            verified[key] = artifact
        if "source" not in verified:
            raise StudioAppError("Guided Sample manifest has no source artifact.")
        source_spec: Any = files["source"]
        source = self._inside(self.root, self._path_value(source_spec))
        raw_candidates = manifest.get("candidates")
        if not isinstance(raw_candidates, Sequence) or isinstance(raw_candidates, (str, bytes)):
            raise StudioAppError("Guided Sample manifest has no candidate list.")
        if not raw_candidates:
            raise StudioAppError("Guided Sample manifest contains no candidate images.")
        candidates: list[CandidateView] = []
        for index, item in enumerate(raw_candidates):
            artifact_spec: Any = item
            if isinstance(item, Mapping) and isinstance(item.get("file"), str):
                if not isinstance(files, Mapping) or item["file"] not in files:
                    raise StudioAppError("Guided Sample candidate references a missing file entry.")
                artifact_spec = files[item["file"]]
            path = self._inside(self.root, self._path_value(artifact_spec))
            label = f"Direction {index + 1}"
            reason = ""
            seed = None
            if isinstance(item, Mapping):
                scores = item.get("scores")
                score_badges = scores.get("badges") if isinstance(scores, Mapping) else None
                score_badge = (
                    score_badges[0] if isinstance(score_badges, Sequence) and score_badges else None
                )
                label = str(item.get("label") or item.get("badge") or score_badge or label)
                reason = str(item.get("reason") or "")
                raw_seed = item.get("seed")
                seed = raw_seed if isinstance(raw_seed, int) else None
            candidates.append(CandidateView(str(path), label, reason, seed))
        selected = manifest.get(
            "selected_index", manifest.get("selected", manifest.get("selection", 0))
        )
        if isinstance(selected, str):
            selected = next(
                (
                    index
                    for index, item in enumerate(raw_candidates)
                    if isinstance(item, Mapping) and item.get("id") == selected
                ),
                0,
            )
        if not isinstance(selected, int) or not 0 <= selected < len(candidates):
            selected = 0
        provenance_value = manifest.get("provenance")
        if isinstance(provenance_value, Mapping):
            provenance = str(provenance_value.get("generator") or "manifest-declared sample")
        else:
            provenance = str(
                manifest.get("generator") or manifest.get("backend") or "manifest-declared sample"
            )
        prompt, profile, structure = _recipe_control_values(manifest)
        return GuidedSample(
            root=self.root.resolve(),
            manifest_path=manifest_path.resolve(),
            source_path=source,
            candidates=tuple(candidates),
            selected_index=selected,
            provenance=provenance,
            prompt=prompt,
            profile=profile,
            structure=structure,
        )

    @property
    def available(self) -> bool:
        try:
            self.load()
        except StudioAppError:
            return False
        return True


@dataclass(slots=True)
class RunRecord:
    run_id: str
    session_id: str
    workspace: Path
    source_path: Path
    candidates: tuple[CandidateView, ...]
    study: Any = None
    studio: Any = None
    prepared: Any = None
    request: dict[str, Any] = field(default_factory=dict)
    guided_sample: GuidedSample | None = None
    status_key: str = "status_generated"
    created_at: float = field(default_factory=time.monotonic)
    touched_at: float = field(default_factory=time.monotonic)


class RunRegistry:
    """Thread-safe, bounded registry for heavyweight run objects."""

    def __init__(self, *, max_runs: int = 32, ttl_seconds: int = 3600) -> None:
        self.max_runs = max_runs
        self.ttl_seconds = ttl_seconds
        self._records: dict[str, RunRecord] = {}
        self._lock = threading.RLock()

    def _prune(self) -> None:
        now = time.monotonic()
        expired = [
            key for key, value in self._records.items() if now - value.touched_at > self.ttl_seconds
        ]
        for key in expired:
            self._discard(key)
        if len(self._records) > self.max_runs:
            ordered = sorted(self._records.values(), key=lambda item: item.touched_at)
            for item in ordered[: len(self._records) - self.max_runs]:
                self._discard(item.run_id)

    def _discard(self, run_id: str) -> None:
        record = self._records.pop(run_id, None)
        if record is not None:
            shutil.rmtree(record.workspace, ignore_errors=True)

    def clear(self) -> None:
        """Forget every run and remove its ephemeral workspace."""

        with self._lock:
            for run_id in tuple(self._records):
                self._discard(run_id)

    def put(self, record: RunRecord) -> None:
        with self._lock:
            self._records[record.run_id] = record
            self._prune()

    def get(self, run_id: str | None, session_id: str) -> RunRecord:
        if not run_id:
            raise StudioAppError("No active study. Explore or open the Guided Sample first.")
        with self._lock:
            self._prune()
            record = self._records.get(run_id)
            if record is None or record.session_id != session_id:
                raise StudioAppError("This study is no longer available in the current session.")
            record.touched_at = time.monotonic()
            return record


class ModelPool:
    """Construct each local preset once and serialize cache changes."""

    def __init__(self, factory: StudioFactory) -> None:
        self.factory = factory
        self._models: dict[str, Any] = {}
        self._lock = threading.RLock()

    def get(self, preset: str) -> Any:
        with self._lock:
            if preset not in self._models:
                self._models[preset] = self.factory(preset)
            return self._models[preset]


@dataclass(frozen=True, slots=True)
class AppResponse:
    state: dict[str, Any]
    source: str | None
    selected: str | None
    gallery: tuple[tuple[str, str], ...]
    recommendation: str
    status: str
    prompt: str | None = None
    profile: str | None = None
    structure: str | None = None
    sync_recipe_controls: bool = False


def _safe_session_dir(root: Path, session_id: str) -> Path:
    target = (root / session_id).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:  # pragma: no cover - guarded by AppState validation
        raise StudioAppError("Invalid session workspace.") from exc
    target.mkdir(parents=True, exist_ok=True)
    return target


def sanitize_upload(
    source: str | Path,
    destination_dir: str | Path,
    *,
    max_bytes: int = MAX_UPLOAD_BYTES,
    max_pixels: int = MAX_IMAGE_PIXELS,
) -> Path:
    """Validate an uploaded raster and save a metadata-free RGB PNG."""

    source_path = Path(source)
    if not source_path.is_file():
        raise StudioAppError("Choose a valid image file.")
    if source_path.stat().st_size > max_bytes:
        raise StudioAppError("Image is larger than 20 MB.")
    try:
        from PIL import Image, ImageOps, UnidentifiedImageError
    except ImportError as exc:  # pragma: no cover - install failure path
        raise StudioAppError(
            "Image support is unavailable. Install the base package with Pillow."
        ) from exc
    try:
        with Image.open(source_path) as probe:
            image_format = probe.format
            width, height = probe.size
            if image_format not in ALLOWED_IMAGE_FORMATS:
                raise StudioAppError("Use a PNG, JPEG, or WebP image.")
            if width <= 0 or height <= 0 or width * height > max_pixels:
                raise StudioAppError("Image dimensions exceed the 50 megapixel limit.")
        with Image.open(source_path) as opened:
            image = ImageOps.exif_transpose(opened)
            image.load()
            if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
                rgba = image.convert("RGBA")
                background = Image.new("RGBA", rgba.size, "white")
                background.alpha_composite(rgba)
                clean = background.convert("RGB")
            else:
                clean = image.convert("RGB")
    except StudioAppError:
        raise
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise StudioAppError("The uploaded image cannot be decoded safely.") from exc
    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / f"source-{uuid.uuid4().hex}.png"
    clean.save(output, format="PNG", optimize=True)
    return output


def _safe_bundle_relative(value: Any, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise StudioAppError(f"{label} contains an unsafe path.")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
        or relative.parts[0].endswith(":")
    ):
        raise StudioAppError(f"{label} contains an unsafe path.")
    return relative


def _load_replay_bundle(
    manifest_path: Path,
    *,
    uploaded_json: bool,
) -> tuple[dict[str, Any], tuple[tuple[PurePosixPath, Path], ...]]:
    if not manifest_path.is_file() or manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
        raise StudioAppError("Choose a valid manifest smaller than 2 MB.")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StudioAppError("The manifest is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise StudioAppError("The manifest must be a JSON object.")
    if payload.get("schema") != "aisketcher.manifest/v1":
        raise StudioAppError("Replay requires an aisketcher.manifest/v1 manifest.")
    files = payload.get("files")
    if not isinstance(files, Mapping) or not files:
        raise StudioAppError("The manifest requires a canonical file table.")
    if len(files) > MAX_REPLAY_FILES:
        raise StudioAppError(f"The replay bundle may contain at most {MAX_REPLAY_FILES} files.")

    root = manifest_path.parent.resolve()
    artifacts: list[tuple[PurePosixPath, Path]] = []
    missing: list[str] = []
    total_size = manifest_path.stat().st_size
    for key, descriptor in files.items():
        if not isinstance(key, str) or not isinstance(descriptor, Mapping):
            raise StudioAppError("The manifest contains an invalid file descriptor.")
        relative = _safe_bundle_relative(
            descriptor.get("path"),
            label=f"Manifest artifact {key!r}",
        )
        if relative == PurePosixPath("manifest.json"):
            raise StudioAppError("The manifest cannot declare itself as an artifact.")
        unresolved_artifact = root / Path(*relative.parts)
        artifact = unresolved_artifact.resolve()
        try:
            artifact.relative_to(root)
        except ValueError as exc:
            raise StudioAppError(f"Manifest artifact {key!r} contains an unsafe path.") from exc
        contains_symlink = any(
            (root / Path(*relative.parts[:depth])).is_symlink()
            for depth in range(1, len(relative.parts) + 1)
        )
        if not artifact.is_file() or contains_symlink:
            missing.append(relative.as_posix())
            continue
        expected = descriptor.get("sha256")
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in expected)
        ):
            raise StudioAppError(f"Manifest artifact {key!r} has an invalid SHA-256 hash.")
        size = artifact.stat().st_size
        if size > MAX_REPLAY_FILE_BYTES:
            raise StudioAppError(
                f"Replay artifact {relative.as_posix()!r} exceeds the 50 MB limit."
            )
        total_size += size
        if total_size > MAX_REPLAY_UNCOMPRESSED_BYTES:
            raise StudioAppError("The replay bundle exceeds the 200 MB expanded-size limit.")
        if _file_sha256(artifact) != expected.lower():
            raise StudioAppError(f"Manifest artifact {key!r} failed SHA-256 verification.")
        artifacts.append((relative, artifact))
    if missing:
        names = ", ".join(missing[:3])
        if len(missing) > 3:
            names += f", and {len(missing) - 3} more"
        if uploaded_json:
            raise StudioAppError(
                f"Manifest JSON is missing sibling artifacts ({names}). "
                "Upload the canonical export ZIP instead."
            )
        raise StudioAppError(f"The export ZIP is missing declared artifacts ({names}).")
    return payload, tuple(artifacts)


def _extract_replay_archive(source: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=False)
    manifest_paths: list[Path] = []
    normalized_names: set[str] = set()
    expanded_size = 0
    with zipfile.ZipFile(source) as archive:
        entries = archive.infolist()
        if not entries or len(entries) > MAX_REPLAY_FILES:
            raise StudioAppError(
                f"The export ZIP must contain between 1 and {MAX_REPLAY_FILES} entries."
            )
        for info in entries:
            relative = _safe_bundle_relative(info.filename, label="The export ZIP")
            normalized = relative.as_posix()
            if normalized in normalized_names:
                raise StudioAppError("The export ZIP contains duplicate paths.")
            normalized_names.add(normalized)
            if info.flag_bits & 0x1:
                raise StudioAppError("Encrypted export ZIP files are not supported.")
            mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(mode):
                raise StudioAppError("The export ZIP cannot contain symbolic links.")
            if info.file_size < 0 or info.file_size > MAX_REPLAY_FILE_BYTES:
                raise StudioAppError("An export ZIP entry exceeds the 50 MB limit.")
            expanded_size += info.file_size
            if expanded_size > MAX_REPLAY_UNCOMPRESSED_BYTES:
                raise StudioAppError("The export ZIP exceeds the 200 MB expanded-size limit.")

            target = (destination / Path(*relative.parts)).resolve()
            try:
                target.relative_to(destination.resolve())
            except ValueError as exc:  # pragma: no cover - also guarded by relative path checks
                raise StudioAppError("The export ZIP contains an unsafe path.") from exc
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            written = 0
            with archive.open(info, "r") as reader, target.open("xb") as writer:
                while chunk := reader.read(1024 * 1024):
                    written += len(chunk)
                    if written > MAX_REPLAY_FILE_BYTES:
                        raise StudioAppError("An export ZIP entry exceeds the 50 MB limit.")
                    writer.write(chunk)
            if written != info.file_size:
                raise StudioAppError("The export ZIP contains an entry with an invalid size.")
            if relative.name == "manifest.json":
                manifest_paths.append(target)
    if len(manifest_paths) != 1:
        raise StudioAppError("The export ZIP must contain exactly one manifest.json file.")
    return manifest_paths[0]


def prepare_replay_input(source: str | Path, destination_dir: str | Path) -> Path:
    """Stage a canonical manifest or safely extract a canonical export ZIP.

    A manifest selected through a browser normally arrives without its sibling
    artifacts.  Such uploads are rejected with guidance to use the canonical
    ZIP.  Direct callers can still provide a manifest whose declared artifacts
    are present beside it; those files are copied into the bounded session
    workspace before strict replay.
    """

    source_path = Path(source)
    if not source_path.is_file():
        raise StudioAppError("Choose a valid manifest JSON or export ZIP.")
    destination = Path(destination_dir)
    try:
        is_archive = zipfile.is_zipfile(source_path)
    except OSError as exc:
        raise StudioAppError("The replay upload cannot be read.") from exc
    if source_path.suffix.lower() == ".zip" and not is_archive:
        raise StudioAppError("Choose a valid export ZIP.")

    if is_archive:
        if source_path.stat().st_size > MAX_REPLAY_ARCHIVE_BYTES:
            raise StudioAppError("The export ZIP is larger than 20 MB.")
        try:
            manifest_path = _extract_replay_archive(source_path, destination)
            _load_replay_bundle(manifest_path, uploaded_json=False)
        except StudioAppError:
            shutil.rmtree(destination, ignore_errors=True)
            raise
        except (OSError, RuntimeError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
            shutil.rmtree(destination, ignore_errors=True)
            raise StudioAppError("The export ZIP cannot be unpacked safely.") from exc
        return manifest_path

    if source_path.stat().st_size > MAX_MANIFEST_BYTES:
        raise StudioAppError("Choose a valid manifest smaller than 2 MB.")
    payload: dict[str, Any]
    artifacts: tuple[tuple[PurePosixPath, Path], ...]
    payload, artifacts = _load_replay_bundle(source_path, uploaded_json=True)
    del payload  # Validation is repeated by the core after staging.
    try:
        destination.mkdir(parents=True, exist_ok=False)
        staged_manifest = destination / "manifest.json"
        shutil.copyfile(source_path, staged_manifest)
        for relative, artifact in artifacts:
            target = destination / Path(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(artifact, target)
    except OSError as exc:
        shutil.rmtree(destination, ignore_errors=True)
        raise StudioAppError("The manifest bundle could not be staged for replay.") from exc
    return staged_manifest


def _call_supported(function: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Call a pluggable backend without forcing optional keyword support."""

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(*args, **kwargs)
    accepts_any = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_any:
        return function(*args, **kwargs)
    supported = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return function(*args, **supported)


def _default_studio_factory(preset: str) -> Any:
    try:
        from aisketcher import Studio
    except (ImportError, AttributeError) as exc:
        raise StudioAppError(
            "Local generation is unavailable. Install AIsketcher with the 'local' extra."
        ) from exc
    try:
        return Studio.from_preset(preset, device="auto")
    except Exception as exc:  # noqa: BLE001 - converted to a UI-safe boundary error
        raise StudioAppError(f"Could not load the local preset: {exc}") from exc


def _intent(prompt: str, profile: str, structure: str) -> Any:
    try:
        from aisketcher import Intent
    except (ImportError, AttributeError):
        return StudioIntent(prompt=prompt, profile=profile, structure=structure)
    return Intent(prompt=prompt, profile=profile, structure=structure)


def _seed_plan(
    mode: str,
    output_count: int,
    raw_seeds: str,
    *,
    scout_base_seed: int | None = None,
) -> Any:
    if mode == "locked" and output_count != 1:
        raise StudioAppError("Locked seed mode requires one output.")
    seeds: tuple[int, ...] = ()
    if mode in {"explicit", "locked"}:
        try:
            seeds = tuple(int(value.strip()) for value in raw_seeds.split(",") if value.strip())
        except ValueError as exc:
            raise StudioAppError("Custom seeds must be comma-separated integers.") from exc
        expected = 1 if mode == "locked" else output_count
        if len(seeds) != expected:
            if mode == "locked":
                raise StudioAppError("Enter exactly one locked seed.")
            raise StudioAppError(f"Enter exactly {output_count} custom seeds.")
        if any(value < 0 or value >= 2**63 for value in seeds):
            raise StudioAppError("Custom seeds must be non-negative 63-bit integers.")
    base_seed = (
        secrets.randbits(63) if mode == "scout" and scout_base_seed is None else scout_base_seed
    )
    try:
        from aisketcher import SeedPlan
    except (ImportError, AttributeError):
        fallback_seeds = seeds if mode != "scout" else ((base_seed or 0),)
        return StudioSeedPlan(mode=mode, count=output_count, seeds=fallback_seeds)
    if mode == "explicit":
        return SeedPlan.explicit(seeds)
    if mode == "locked":
        return SeedPlan.locked(seeds[0])
    return SeedPlan.scout(output_count, base_seed=base_seed or 0)


def _recipe(steps: int, guidance: float) -> Any:
    try:
        from aisketcher import Recipe
    except (ImportError, AttributeError):
        return StudioRecipe(steps=int(steps), guidance_scale=float(guidance))
    return Recipe(steps=int(steps), guidance_scale=float(guidance))


def _candidate_items(study: Any) -> list[Any]:
    values = getattr(study, "candidates", study)
    if callable(values):
        values = values()
    if isinstance(values, (Mapping, str, bytes, Path)):
        raise StudioAppError("The backend returned an invalid candidate collection.")
    try:
        items = list(values)
    except TypeError as exc:
        raise StudioAppError("The backend returned no iterable candidates.") from exc
    if not items:
        raise StudioAppError("The backend returned no candidates.")
    return items


def _image_value(candidate: Any) -> Any:
    if isinstance(candidate, (str, Path)):
        return candidate
    if isinstance(candidate, Mapping):
        for key in ("image", "image_path", "path", "artifact"):
            if candidate.get(key) is not None:
                return candidate[key]
    for name in ("image", "image_path", "path", "artifact"):
        value = getattr(candidate, name, None)
        if value is not None:
            return value
    raise StudioAppError("A candidate has no readable image artifact.")


def _candidate_text(candidate: Any, index: int) -> tuple[str, str, int | None]:
    def value(name: str) -> Any:
        if isinstance(candidate, Mapping):
            return candidate.get(name)
        return getattr(candidate, name, None)

    label = value("technical_badge") or value("badge") or value("label")
    reason = value("reason") or value("recommendation_reason") or ""
    seed = value("seed")
    scores = value("scores")
    badges = scores.get("badges") if isinstance(scores, Mapping) else getattr(scores, "badges", ())
    if not label and isinstance(badges, Sequence) and badges:
        label = badges[0]
    if not reason and scores is not None:
        structure_score = (
            scores.get("structure_similarity")
            if isinstance(scores, Mapping)
            else getattr(scores, "structure_similarity", None)
        )
        if isinstance(structure_score, (int, float)):
            reason = f"Recorded structure similarity: {float(structure_score):.3f}."
    return (
        str(label or f"Direction {index + 1}"),
        str(reason),
        seed if isinstance(seed, int) else None,
    )


def _materialize_candidate(value: Any, destination: Path) -> Path:
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:  # pragma: no cover - install failure path
        raise StudioAppError("Image support is unavailable. Install Pillow.") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        if isinstance(value, (str, Path)):
            path = Path(value)
            if not path.is_file():
                raise StudioAppError("A backend candidate artifact is missing.")
            with Image.open(path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                image.load()
        elif isinstance(value, Image.Image):
            image = value.convert("RGB")
        else:
            image = Image.fromarray(value).convert("RGB")
        image.save(destination, format="PNG", optimize=True)
    except StudioAppError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise StudioAppError("A backend candidate could not be converted to an image.") from exc
    return destination


def _materialize_study(study: Any, destination: Path) -> tuple[CandidateView, ...]:
    candidates: list[CandidateView] = []
    for index, candidate in enumerate(_candidate_items(study)):
        image_path = _materialize_candidate(
            _image_value(candidate), destination / f"candidate-{index + 1}.png"
        )
        label, reason, seed = _candidate_text(candidate, index)
        candidates.append(CandidateView(str(image_path), label, reason, seed))
    return tuple(candidates)


class AppController:
    """Application use cases shared by Gradio callbacks and tests."""

    def __init__(
        self,
        *,
        studio_factory: StudioFactory | None = None,
        model_installer: Any = None,
        workspace_root: str | Path | None = None,
        guided_root: str | Path | None = None,
        registry: RunRegistry | None = None,
    ) -> None:
        self._owns_workspace_root = workspace_root is None
        if workspace_root is None:
            workspace_root = tempfile.mkdtemp(prefix="aisketcher-studio-")
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self._workspace_finalizer = (
            weakref.finalize(self, shutil.rmtree, self.workspace_root, True)
            if self._owns_workspace_root
            else None
        )
        self.registry = registry or RunRegistry()
        self.pool = ModelPool(studio_factory or _default_studio_factory)
        self.model_installer = model_installer
        self.guided = GuidedSampleCatalog(guided_root)

    def close(self) -> None:
        """Remove run data and the auto-created Studio workspace, if any."""

        self.registry.clear()
        if self._owns_workspace_root:
            shutil.rmtree(self.workspace_root, ignore_errors=True)
            if self._workspace_finalizer is not None:
                self._workspace_finalizer.detach()

    def initial_state(self, language: str = "en") -> dict[str, Any]:
        return AppState.new(language).payload()

    @staticmethod
    def _candidate_label(label: str, language: str) -> str:
        """Localize known display labels without mutating canonical candidate data."""

        badge_key = DISPLAY_BADGE_KEYS.get(label.casefold())
        if badge_key:
            return text(language, badge_key)
        direction, separator, index = label.rpartition(" ")
        if separator and direction.casefold() == "direction" and index.isdigit():
            return text(language, "direction_label").format(index=int(index))
        return label

    @classmethod
    def _gallery(
        cls, candidates: Iterable[CandidateView], language: str
    ) -> tuple[tuple[str, str], ...]:
        return tuple(
            (item.path, cls._candidate_label(item.label, language)) for item in candidates
        )

    @classmethod
    def _recommendation(cls, candidate: CandidateView, language: str) -> str:
        prefix = text(language, "recommendation_heading")
        seed = (
            f" · {text(language, 'seed_label')} {candidate.seed}"
            if candidate.seed is not None
            else ""
        )
        detail = candidate.reason or text(language, "recommendation_detail")
        label = cls._candidate_label(candidate.label, language)
        return f"**{prefix}: {label}{seed}**  \n{detail}"

    @staticmethod
    def _status(record: RunRecord, state: AppState) -> str:
        values: dict[str, Any] = {
            "count": len(record.candidates),
            "index": (state.selected_index or 0) + 1,
            "provenance": (
                record.guided_sample.provenance if record.guided_sample is not None else ""
            ),
        }
        return text(state.language, record.status_key).format(**values)

    def localize_active_run(
        self,
        state_value: Mapping[str, Any] | AppState | None,
        language: str,
    ) -> AppResponse | None:
        """Re-render an active run in ``language`` while preserving canonical data."""

        state = AppState.from_payload(state_value).replace(
            language=normalize_language(language)
        )
        if not state.run_id:
            return None
        record = self.registry.get(state.run_id, state.session_id)
        selected = state.selected_index
        if selected is None or not 0 <= selected < len(record.candidates):
            selected = 0
            state = state.replace(selected_index=selected)
        candidate = record.candidates[selected]
        return AppResponse(
            state=state.payload(),
            source=str(record.source_path),
            selected=candidate.path,
            gallery=self._gallery(record.candidates, state.language),
            recommendation=self._recommendation(candidate, state.language),
            status=self._status(record, state),
        )

    def explore(
        self,
        state_value: Mapping[str, Any] | AppState | None,
        image_path: str | Path | None,
        brief: str,
        profile: str,
        structure: str,
        preset: str,
        output_count: int,
        seed_mode: str,
        custom_seeds: str,
        canny: bool,
        steps: int,
        guidance: float,
        locks: Sequence[str],
    ) -> AppResponse:
        state = AppState.from_payload(state_value)
        if not image_path:
            raise StudioAppError(
                "Upload a sketch before exploring."
                if state.language == "en"
                else "탐색 전에 스케치를 업로드하세요."
            )
        prompt = brief.strip()
        if not prompt:
            raise StudioAppError(
                "Add a creative brief before exploring."
                if state.language == "en"
                else "탐색 전에 크리에이티브 브리프를 입력하세요."
            )
        if int(output_count) not in OUTPUT_COUNTS:
            raise StudioAppError("Outputs must be 1, 4, or 8.")
        if structure not in {"loose", "balanced", "faithful"}:
            raise StudioAppError("Choose a valid structure setting.")
        session_dir = _safe_session_dir(self.workspace_root, state.session_id)
        run_id = uuid.uuid4().hex
        run_dir = session_dir / run_id
        try:
            source = sanitize_upload(image_path, run_dir)
            studio = self.pool.get(preset)
            prepared = studio.prepare(str(source))
            intent = _intent(prompt, profile, structure)
            seed_plan = _seed_plan(seed_mode, int(output_count), custom_seeds)
            recipe = _recipe(int(steps), float(guidance))
            overrides = {
                "control": "canny" if canny else None,
                "steps": int(steps),
                "guidance": float(guidance),
                "locks": tuple(locks),
            }
            study = _call_supported(
                studio.explore,
                prepared,
                intent=intent,
                outputs=int(output_count),
                seed_plan=seed_plan,
                recipe=recipe,
                overrides=overrides,
                recipe_overrides=overrides,
            )
            candidates = _materialize_study(study, run_dir / "results")
        except StudioAppError:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise
        except Exception as exc:  # noqa: BLE001 - optional backend boundary
            shutil.rmtree(run_dir, ignore_errors=True)
            raise StudioAppError(f"Generation failed: {exc}") from exc
        record = RunRecord(
            run_id=run_id,
            session_id=state.session_id,
            workspace=run_dir,
            source_path=source,
            candidates=candidates,
            study=study,
            studio=studio,
            prepared=prepared,
            request={
                "brief": prompt,
                "profile": profile,
                "structure": structure,
                "preset": preset,
                "output_count": int(output_count),
                "seed_mode": seed_mode,
                "custom_seeds": custom_seeds,
                "canny": bool(canny),
                "steps": int(steps),
                "guidance": float(guidance),
                "locks": tuple(locks),
            },
            status_key="status_generated",
        )
        self.registry.put(record)
        selected = 0
        state = state.replace(run_id=run_id, selected_index=selected, guided=False)
        return AppResponse(
            state=state.payload(),
            source=str(source),
            selected=candidates[selected].path,
            gallery=self._gallery(candidates, state.language),
            recommendation=self._recommendation(candidates[selected], state.language),
            status=self._status(record, state),
        )

    def open_guided_sample(self, state_value: Mapping[str, Any] | AppState | None) -> AppResponse:
        state = AppState.from_payload(state_value)
        try:
            sample = self.guided.load()
        except StudioAppError as exc:
            raise StudioAppError(text(state.language, "unavailable")) from exc
        session_dir = _safe_session_dir(self.workspace_root, state.session_id)
        run_id = uuid.uuid4().hex
        record = RunRecord(
            run_id=run_id,
            session_id=state.session_id,
            workspace=session_dir / run_id,
            source_path=sample.source_path,
            candidates=sample.candidates,
            guided_sample=sample,
            status_key="status_guided",
        )
        self.registry.put(record)
        state = state.replace(
            run_id=run_id,
            selected_index=sample.selected_index,
            guided=True,
            advanced_overrides=False,
        )
        chosen = sample.candidates[sample.selected_index]
        return AppResponse(
            state=state.payload(),
            source=str(sample.source_path),
            selected=chosen.path,
            gallery=self._gallery(sample.candidates, state.language),
            recommendation=self._recommendation(chosen, state.language),
            status=self._status(record, state),
            prompt=sample.prompt,
            profile=sample.profile,
            structure=sample.structure,
            sync_recipe_controls=True,
        )

    def select_candidate(
        self, state_value: Mapping[str, Any] | AppState | None, index: int
    ) -> tuple[dict[str, Any], str, str, str]:
        state = AppState.from_payload(state_value)
        record = self.registry.get(state.run_id, state.session_id)
        if not 0 <= index < len(record.candidates):
            raise StudioAppError("Choose a valid candidate.")
        state = state.replace(selected_index=index)
        candidate = record.candidates[index]
        record.status_key = "status_selected"
        return (
            state.payload(),
            candidate.path,
            self._recommendation(candidate, state.language),
            self._status(record, state),
        )

    def refine(
        self,
        state_value: Mapping[str, Any] | AppState | None,
        strength: str,
        locks: Sequence[str],
    ) -> AppResponse:
        state = AppState.from_payload(state_value)
        record = self.registry.get(state.run_id, state.session_id)
        if record.guided_sample is not None:
            raise StudioAppError(
                "Guided Sample is read-only. Prepare Lite or Quality to refine it."
                if state.language == "en"
                else "가이드 샘플은 읽기 전용입니다. 발전시키려면 Lite 또는 Quality 모델을 준비하세요."
            )
        if state.selected_index is None:
            raise StudioAppError("Select a direction first.")
        pick = getattr(record.study, "pick", None)
        vary = getattr(record.studio, "vary", None)
        if not callable(pick) or not callable(vary):
            raise StudioAppError("The active backend does not support pick-and-vary yet.")
        new_id = uuid.uuid4().hex
        destination = record.workspace.parent / new_id
        try:
            selected = pick(state.selected_index)
            varied = _call_supported(
                vary,
                selected,
                outputs=int(record.request.get("output_count", 4)),
                strength=strength,
                locks=tuple(locks),
            )
            destination.mkdir(parents=True, exist_ok=False)
            source = destination / "source.png"
            shutil.copy2(record.source_path, source)
            candidates = _materialize_study(varied, destination / "results")
        except StudioAppError:
            shutil.rmtree(destination, ignore_errors=True)
            raise
        except Exception as exc:  # noqa: BLE001 - optional backend boundary
            shutil.rmtree(destination, ignore_errors=True)
            raise StudioAppError(f"Variation failed: {exc}") from exc
        new_record = RunRecord(
            run_id=new_id,
            session_id=state.session_id,
            workspace=destination,
            source_path=source,
            candidates=candidates,
            study=varied,
            studio=record.studio,
            prepared=record.prepared,
            request=dict(record.request),
            status_key="status_variation",
        )
        self.registry.put(new_record)
        state = state.replace(run_id=new_id, selected_index=0, guided=False)
        return AppResponse(
            state=state.payload(),
            source=str(source),
            selected=candidates[0].path,
            gallery=self._gallery(candidates, state.language),
            recommendation=self._recommendation(candidates[0], state.language),
            status=self._status(new_record, state),
        )

    def try_again(self, state_value: Mapping[str, Any] | AppState | None) -> AppResponse:
        """Start another exploration from the recorded request."""

        state = AppState.from_payload(state_value)
        record = self.registry.get(state.run_id, state.session_id)
        if record.guided_sample is not None:
            raise StudioAppError(
                "Guided Sample is read-only. Prepare a local model to explore new directions."
                if state.language == "en"
                else "가이드 샘플은 읽기 전용입니다. 새 방향을 탐색하려면 로컬 모델을 준비하세요."
            )
        request = record.request
        if not request:
            raise StudioAppError("The original exploration request is unavailable.")
        return self.explore(
            state,
            record.source_path,
            str(request["brief"]),
            str(request["profile"]),
            str(request["structure"]),
            str(request["preset"]),
            int(request["output_count"]),
            str(request["seed_mode"]),
            str(request["custom_seeds"]),
            bool(request["canny"]),
            int(request["steps"]),
            float(request["guidance"]),
            tuple(request["locks"]),
        )

    def export(self, state_value: Mapping[str, Any] | AppState | None) -> tuple[str, str]:
        state = AppState.from_payload(state_value)
        record = self.registry.get(state.run_id, state.session_id)
        export_dir = record.workspace / "export"
        if record.guided_sample is not None:
            shutil.rmtree(export_dir, ignore_errors=True)
            export_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(record.guided_sample.manifest_path, export_dir / "manifest.json")
            _, artifacts = _load_replay_bundle(
                record.guided_sample.manifest_path,
                uploaded_json=False,
            )
            for relative, artifact in artifacts:
                target = export_dir / Path(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(artifact, target)
            _load_replay_bundle(export_dir / "manifest.json", uploaded_json=False)
        else:
            export_method = getattr(record.study, "export", None)
            if not callable(export_method):
                raise StudioAppError("The active backend does not provide a canonical export.")
            try:
                result = _call_supported(
                    export_method,
                    str(export_dir),
                    overwrite=export_dir.exists(),
                )
            except Exception as exc:  # noqa: BLE001 - optional backend boundary
                raise StudioAppError(f"Export failed: {exc}") from exc
            if isinstance(result, (str, Path)) and Path(result).is_file():
                result_path = Path(result)
                if result_path.suffix.lower() == ".zip":
                    record.status_key = "status_export"
                    return str(result_path), self._status(record, state)
        archive = record.workspace / "aisketcher-study.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(export_dir.rglob("*")):
                if path.is_file():
                    bundle.write(path, path.relative_to(export_dir))
        if archive.stat().st_size == 0:
            raise StudioAppError("The export did not contain any artifacts.")
        record.status_key = "status_export"
        return str(archive), self._status(record, state)

    def replay_manifest(
        self,
        state_value: Mapping[str, Any] | AppState | None,
        manifest_path: str | Path | None,
        preset: str,
    ) -> AppResponse:
        state = AppState.from_payload(state_value)
        if not manifest_path:
            raise StudioAppError("Choose a manifest JSON or export ZIP first.")
        session_dir = _safe_session_dir(self.workspace_root, state.session_id)
        run_id = uuid.uuid4().hex
        run_dir = session_dir / run_id
        upload_dir = run_dir / "replay-bundle"
        try:
            path = prepare_replay_input(manifest_path, upload_dir)
            payload, _ = _load_replay_bundle(path, uploaded_json=False)
            studio = self.pool.get(preset)
            replay = getattr(studio, "replay", None)
            if not callable(replay):
                raise StudioAppError("The active backend does not support replay.")
            # Replay is the integrity boundary: unlike optional generation
            # overrides, strict mode must never be dropped for compatibility.
            replay_result = replay(str(path), mode="strict")
            study = getattr(replay_result, "study", None) or replay_result
            if study is replay_result and hasattr(replay_result, "replayed"):
                raise StudioAppError(
                    "The manifest was verified, but this backend did not produce replayed candidates."
                )
            candidates = _materialize_study(study, run_dir / "results")
        except StudioAppError:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise
        except Exception as exc:  # noqa: BLE001 - optional backend boundary
            shutil.rmtree(run_dir, ignore_errors=True)
            raise StudioAppError(f"Strict replay failed: {exc}") from exc
        source = path
        files = payload.get("files")
        source_value = files.get("source") if isinstance(files, Mapping) else None
        if isinstance(source_value, Mapping):
            relative = _safe_bundle_relative(
                source_value.get("path"), label="Manifest source artifact"
            )
            possible = (path.parent / Path(*relative.parts)).resolve()
            if possible.is_file():
                source = possible
        record = RunRecord(
            run_id=run_id,
            session_id=state.session_id,
            workspace=run_dir,
            source_path=source,
            candidates=candidates,
            study=study,
            studio=studio,
            status_key="status_replay",
        )
        self.registry.put(record)
        state = state.replace(run_id=run_id, selected_index=0, guided=False)
        prompt, profile, structure = _recipe_control_values(payload)
        return AppResponse(
            state=state.payload(),
            source=str(source)
            if source.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            else None,
            selected=candidates[0].path,
            gallery=self._gallery(candidates, state.language),
            recommendation=self._recommendation(candidates[0], state.language),
            status=self._status(record, state),
            prompt=prompt,
            profile=profile,
            structure=structure,
            sync_recipe_controls=True,
        )

    def install_model(self, preset: str, confirmed: bool, language: str = "en") -> str:
        if not confirmed:
            raise StudioAppError(
                "Review the download size and license, then confirm."
                if normalize_language(language) == "en"
                else "다운로드 용량과 라이선스를 확인한 뒤 동의하세요."
            )
        installer = self.model_installer
        if installer is None:
            try:
                from aisketcher import PresetManager
            except (ImportError, AttributeError) as exc:
                raise StudioAppError(
                    "Model setup is unavailable. Install AIsketcher with the 'local' extra."
                ) from exc
            installer = PresetManager()
        install = installer if callable(installer) else getattr(installer, "install", None)
        if not callable(install):
            raise StudioAppError("The configured model installer is invalid.")
        try:
            _call_supported(
                install,
                preset,
                confirm=True,
                trust_remote_code=False,
                safe_tensors_only=True,
            )
        except Exception as exc:  # noqa: BLE001 - optional provider boundary
            raise StudioAppError(f"Model preparation failed: {exc}") from exc
        return (
            "Local model is ready."
            if normalize_language(language) == "en"
            else "로컬 모델 준비를 마쳤습니다."
        )


__all__ = [
    "ALLOWED_IMAGE_FORMATS",
    "AppController",
    "AppResponse",
    "AppState",
    "CandidateView",
    "GuidedSample",
    "GuidedSampleCatalog",
    "MAX_IMAGE_PIXELS",
    "MAX_MANIFEST_BYTES",
    "MAX_REPLAY_ARCHIVE_BYTES",
    "MAX_REPLAY_FILE_BYTES",
    "MAX_REPLAY_FILES",
    "MAX_REPLAY_UNCOMPRESSED_BYTES",
    "MAX_UPLOAD_BYTES",
    "ModelPool",
    "RunRegistry",
    "StudioAppError",
    "StudioIntent",
    "StudioRecipe",
    "StudioSeedPlan",
    "prepare_replay_input",
    "sanitize_upload",
]
