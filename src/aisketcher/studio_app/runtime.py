"""Framework-independent runtime for the AIsketcher Studio example.

The Gradio layer is intentionally thin.  Keeping validation, state handling,
sample discovery, and SDK calls here makes the demo testable with a FakeStudio
and importable without Gradio, Torch, or Diffusers.
"""

from __future__ import annotations

import gc
import importlib
import inspect
import json
import math
import os
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
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from hashlib import sha256
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, cast

from ..manifest import canonical_sha256
from ..prompt_normalization import (
    KoreanEnglishTranslator,
    M2M100KoreanEnglishTranslator,
    NormalizedPrompt,
    contains_hangul,
    normalize_prompt,
)
from .i18n import normalize_language, text

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 50_000_000
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_REPLAY_ARCHIVE_BYTES = MAX_UPLOAD_BYTES
MAX_REPLAY_FILES = 64
MAX_REPLAY_FILE_BYTES = 50 * 1024 * 1024
MAX_REPLAY_UNCOMPRESSED_BYTES = 200 * 1024 * 1024
MAX_PROMPT_CHARS = 10_000
MAX_PROMPT_METADATA_BYTES = 20_000
ALLOWED_IMAGE_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})
OUTPUT_COUNTS = frozenset({1, 4, 8})
DISPLAY_BADGE_KEYS = {
    "most distinct": "badge_most_distinct",
    "closest structure": "badge_closest_structure",
    "cleanest edges": "badge_cleanest_edges",
}

_PROCESS_GENERATION_LOCK = threading.Lock()
_PROCESS_GENERATION_STATE_LOCK = threading.RLock()
_PROCESS_ACTIVE_GENERATION_SESSIONS: set[str] = set()
DEFAULT_CROSS_PROCESS_LEASE_TIMEOUT_SECONDS = 60 * 60
FLUX2_KLEIN_PRESET_PREFIX = "flux2-klein-edit@"


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_host_lease_path() -> Path:
    """Return a stable per-user lock path shared by local Studio processes."""

    user_scope = str(os.getuid()) if hasattr(os, "getuid") else "default"
    return (
        Path(tempfile.gettempdir())
        / f"aisketcher-{user_scope}"
        / "leases"
        / "accelerator-and-model-cache.lock"
    )


class StudioAppError(RuntimeError):
    """A safe error that may be shown directly in the example UI."""


class StudioJobCancelled(StudioAppError):
    """A queued or active Studio operation was stopped by its owning session."""


class _CrossProcessFileLease:
    """Advisory host lease for accelerator use and shared model-cache writes.

    The lock file deliberately remains on disk after release. Its JSON payload
    is diagnostic metadata only; the operating-system lock is authoritative,
    so a process crash releases the lease and stale metadata never blocks the
    next Studio process.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @contextmanager
    def acquire(
        self,
        *,
        cancellation_event: threading.Event,
        timeout_seconds: float,
        session_id: str,
    ) -> Iterator[None]:
        try:
            lock_module = importlib.import_module("fcntl")
            lock_kind = "posix"
        except ImportError:
            try:
                lock_module = importlib.import_module("msvcrt")
                lock_kind = "windows"
            except ImportError as exc:  # pragma: no cover - unsupported platform
                raise StudioAppError(
                    "This platform has no supported cross-process file locking; "
                    "AIsketcher refuses concurrent GPU/cache work."
                ) from exc

        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDWR | os.O_CREAT
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except OSError as exc:
            raise StudioAppError(
                f"Could not open the AIsketcher host lease: {exc}"
            ) from exc
        handle = os.fdopen(descriptor, "r+", encoding="utf-8")
        acquired = False
        deadline = time.monotonic() + timeout_seconds
        try:
            if lock_kind == "windows":
                # msvcrt.locking() locks bytes rather than a whole descriptor.
                # Keep one byte present before attempting the non-blocking lock.
                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write("\0")
                    handle.flush()
            while not acquired:
                if cancellation_event.is_set():
                    raise StudioJobCancelled("Stopped by user.")
                try:
                    if lock_kind == "posix":
                        lock_module.flock(
                            handle.fileno(),
                            lock_module.LOCK_EX | lock_module.LOCK_NB,
                        )
                    else:
                        handle.seek(0)
                        lock_module.locking(
                            handle.fileno(),
                            lock_module.LK_NBLCK,
                            1,
                        )
                    acquired = True
                except OSError as exc:
                    if lock_kind == "posix" and not isinstance(exc, BlockingIOError):
                        raise StudioAppError(
                            f"Could not acquire the AIsketcher host lease: {exc}"
                        ) from exc
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            "Timed out waiting for another AIsketcher process."
                        ) from None
                    cancellation_event.wait(min(0.1, remaining))

            # Replacing stale owner metadata after acquiring the OS lock makes
            # crash recovery observable without trusting a stale PID.
            handle.seek(0)
            json.dump(
                {
                    "schema": "aisketcher.host-lease/v1",
                    "pid": os.getpid(),
                    "session_id": session_id,
                    "acquired_at_unix": time.time(),
                },
                handle,
                sort_keys=True,
            )
            handle.truncate()
            handle.flush()
            os.fsync(handle.fileno())
            yield
        finally:
            if acquired:
                with suppress(OSError):
                    if lock_kind == "posix":
                        lock_module.flock(handle.fileno(), lock_module.LOCK_UN)
                    else:
                        handle.seek(0)
                        lock_module.locking(
                            handle.fileno(),
                            lock_module.LK_UNLCK,
                            1,
                        )
            handle.close()


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
    model_prompt: str | None = None
    prompt_metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def generation_prompt(self) -> str:
        """Return the model-facing prompt while retaining the user's original."""

        return self.model_prompt or self.prompt


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

    def latest(self, session_id: str) -> RunRecord | None:
        """Return the newest retained run owned by ``session_id``."""

        with self._lock:
            self._prune()
            matching = [
                record for record in self._records.values() if record.session_id == session_id
            ]
            if not matching:
                return None
            record = max(matching, key=lambda item: item.created_at)
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

    def discard(self, preset: str, *, expected: Any | None = None) -> Any | None:
        """Evict a failed model so the next request can construct a clean one."""

        with self._lock:
            current = self._models.get(preset)
            if current is None or (expected is not None and current is not expected):
                return None
            return self._models.pop(preset)


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


def _reject_undeclared_replay_files(
    destination: Path,
    manifest_path: Path,
    artifacts: Sequence[tuple[PurePosixPath, Path]],
) -> None:
    """Require a canonical ZIP to contain only its manifest and declared files."""

    root = destination.resolve()
    expected = {manifest_path.resolve().relative_to(root).as_posix()}
    expected.update(
        artifact.resolve().relative_to(root).as_posix() for _relative, artifact in artifacts
    )
    actual = {
        path.resolve().relative_to(root).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    }
    extras = sorted(actual - expected)
    if extras:
        names = ", ".join(extras[:3])
        if len(extras) > 3:
            names += f", and {len(extras) - 3} more"
        raise StudioAppError(
            f"The export ZIP contains files not declared by its manifest ({names})."
        )


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
            _payload, archive_artifacts = _load_replay_bundle(
                manifest_path,
                uploaded_json=False,
            )
            _reject_undeclared_replay_files(
                destination,
                manifest_path,
                archive_artifacts,
            )
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


def _accepts_any_keyword(function: Callable[..., Any], names: Sequence[str]) -> bool:
    """Return whether a callable can receive at least one named refinement input."""

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return False
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return True
    keyword_kinds = {
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    }
    return any(
        name in signature.parameters and signature.parameters[name].kind in keyword_kinds
        for name in names
    )


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


def _intent(
    prompt: str,
    profile: str,
    structure: str,
    *,
    model_prompt: str | None = None,
    prompt_metadata: Mapping[str, Any] | None = None,
) -> Any:
    metadata = dict(prompt_metadata or {})
    try:
        from aisketcher import Intent
    except (ImportError, AttributeError):
        return StudioIntent(
            prompt=prompt,
            profile=profile,
            structure=structure,
            model_prompt=model_prompt,
            prompt_metadata=metadata,
        )
    return Intent(
        prompt=prompt,
        profile=profile,
        structure=structure,
        model_prompt=model_prompt,
        prompt_metadata=metadata,
    )


def _normalization_metadata(result: NormalizedPrompt) -> dict[str, Any]:
    """Return bounded audit metadata without duplicating the two prompt fields."""

    value = result.to_dict()
    return {
        "normalization": {
            "detected_language": value["detected_language"],
            "status": value["status"],
            "translator": value["translator"],
            "enhancement_applied": value["enhancement_applied"],
        }
    }


def _refined_prompt_metadata(
    value: Any,
    *,
    original_instruction: str,
    model_instruction: str,
    automatic: bool,
    normalization: NormalizedPrompt,
) -> dict[str, Any]:
    """Copy prompt metadata and append one replay-safe refinement event."""

    if isinstance(value, Mapping):
        try:
            copied = json.loads(json.dumps(dict(value), ensure_ascii=False))
        except (TypeError, ValueError):
            copied = {}
    else:
        copied = {}
    if not isinstance(copied, dict):
        copied = {}
    prior = copied.get("refinements")
    refinements = list(prior) if isinstance(prior, list) else []
    refinements.append(
        {
            "original_instruction": original_instruction,
            "model_instruction": model_instruction,
            "automatic": automatic,
            "normalization": _normalization_metadata(normalization)["normalization"],
        }
    )
    copied["refinements"] = refinements
    return copied


def _validate_refinement_payload(
    original_prompt: str,
    model_prompt: str,
    prompt_metadata: Mapping[str, Any],
) -> None:
    """Fail before generation when refinement would exceed recipe limits."""

    if len(original_prompt) > MAX_PROMPT_CHARS:
        raise StudioAppError(
            "Refinement would make the original prompt exceed the 10,000-character prompt limit."
        )
    if len(model_prompt) > MAX_PROMPT_CHARS:
        raise StudioAppError(
            "Refinement would make the model prompt exceed the 10,000-character prompt limit."
        )
    try:
        encoded_metadata = json.dumps(
            dict(prompt_metadata),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise StudioAppError("Refinement metadata must contain JSON-compatible values.") from exc
    if len(encoded_metadata) > MAX_PROMPT_METADATA_BYTES:
        raise StudioAppError("Refinement would exceed the 20,000-byte prompt metadata limit.")


def _translation_unavailable_message(language: str) -> str:
    if normalize_language(language) == "ko":
        return (
            "한국어 프롬프트를 감지했지만 로컬 한→영 번역 모델이 준비되지 "
            "않았습니다. aisketcher[translate]를 설치하고 고정된 번역 모델을 "
            "준비한 뒤 다시 시도하세요. 원문은 보존되었고 이미지 모델로 "
            "전송되지 않았습니다."
        )
    return (
        "Korean text was detected, but the local Korean-to-English translator "
        "is not ready. Install aisketcher[translate] and prepare the pinned "
        "translation model, then try again. The original prompt was preserved "
        "and was not sent to the image model."
    )


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


def _effective_generation_recipe(
    preset: str,
    steps: int,
    guidance: float,
) -> tuple[int, float]:
    """Return controls that truthfully match the selected backend profile.

    FLUX.2 Klein is a distilled four-step checkpoint. Its backend rejects any
    resolved recipe that diverges from the validated 4-step, CFG-1 profile, so
    crafted callbacks and stale browser state are normalized here as well as in
    the UI.
    """

    if preset.startswith(FLUX2_KLEIN_PRESET_PREFIX):
        return 4, 1.0
    return int(steps), float(guidance)


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


def _materialize_study(
    study: Any,
    destination: Path,
    *,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[CandidateView, ...]:
    candidates: list[CandidateView] = []
    for index, candidate in enumerate(_candidate_items(study)):
        if should_cancel is not None and should_cancel():
            raise StudioJobCancelled("Stopped by user.")
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
        prompt_translator: KoreanEnglishTranslator | None = None,
        workspace_root: str | Path | None = None,
        guided_root: str | Path | None = None,
        registry: RunRegistry | None = None,
        host_lease_path: str | Path | None = None,
        host_lease_timeout_seconds: float = DEFAULT_CROSS_PROCESS_LEASE_TIMEOUT_SECONDS,
    ) -> None:
        lease_timeout = float(host_lease_timeout_seconds)
        if not math.isfinite(lease_timeout) or lease_timeout <= 0:
            raise ValueError("host_lease_timeout_seconds must be a positive finite number")
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
        # Construction is dependency-lazy and cache-only.  No translation
        # dependency is imported and no model file is downloaded until Hangul
        # is actually submitted. Applications may inject another translator.
        self.prompt_translator = (
            prompt_translator if prompt_translator is not None else M2M100KoreanEnglishTranslator()
        )
        lease_path = (
            _default_host_lease_path()
            if host_lease_path is None
            else Path(host_lease_path).expanduser().resolve()
        )
        self._host_lease = _CrossProcessFileLease(lease_path)
        self._host_lease_timeout_seconds = lease_timeout
        self.guided = GuidedSampleCatalog(guided_root)
        self._operation_lock = threading.RLock()
        self._operation_tokens: dict[str, threading.Event] = {}
        self._operation_ids: dict[str, str] = {}
        self._operation_kinds: dict[str, str] = {}
        self._operation_targets: dict[str, Any] = {}
        self._claimed_operation_tokens: dict[str, threading.Event] = {}

    def _resolve_model_installer(self) -> Any:
        """Return the configured installer, creating the local default lazily."""

        installer = self.model_installer
        if installer is not None:
            return installer
        try:
            from aisketcher import PresetManager
        except (ImportError, AttributeError) as exc:
            raise StudioAppError(
                "Model setup is unavailable. Install AIsketcher with the 'local' extra."
            ) from exc
        installer = PresetManager()
        self.model_installer = installer
        return installer

    def plan_model_install(self, preset: str) -> Any | None:
        """Return a latency-safe display plan when the installer supports one.

        Custom callable installers predate the explicit planning API, so the
        Studio treats a missing or unusable ``plan_install`` method as a signal
        to retain its static, backwards-compatible model guidance. The bundled
        manager receives ``verify_cache=False`` here: opening Studio must never
        hide the page behind a multi-GB integrity pass. The explicit model
        preparation action performs the authoritative SHA-256 verification.
        """

        try:
            installer = self._resolve_model_installer()
        except StudioAppError:
            return None
        plan_install = getattr(installer, "plan_install", None)
        if not callable(plan_install):
            return None
        try:
            return _call_supported(plan_install, preset, verify_cache=False)
        except Exception:  # noqa: BLE001 - optional installer compatibility boundary
            return None

    def begin_operation(self, state_value: Mapping[str, Any] | AppState | None) -> threading.Event:
        """Create or reuse the per-session cooperative cancellation token."""

        _operation_id, operation_event = self._start_operation(state_value)
        return operation_event

    def start_operation(
        self,
        state_value: Mapping[str, Any] | AppState | None,
        *,
        kind: str = "generation",
    ) -> str:
        """Return a stable ticket for one queued browser operation.

        Gradio runs the immediate button callback before it dispatches the
        queued Python job.  The ticket crosses that gap so a Stop click can
        tombstone the exact queued request.  A later retry receives a new
        ticket; a delayed callback carrying the old ticket can therefore never
        claim the retry's fresh cancellation token.
        """

        operation_id, _operation_event = self._start_operation(state_value, kind=kind)
        return operation_id

    def _start_operation(
        self,
        state_value: Mapping[str, Any] | AppState | None,
        *,
        kind: str = "generation",
    ) -> tuple[str, threading.Event]:
        if kind not in {"generation", "model"}:
            raise ValueError("operation kind must be generation or model")
        state = AppState.from_payload(state_value)
        with self._operation_lock:
            operation_event = self._operation_tokens.get(state.session_id)
            claimed = state.session_id in self._claimed_operation_tokens
            if operation_event is None or (operation_event.is_set() and not claimed):
                operation_event = threading.Event()
                self._operation_tokens[state.session_id] = operation_event
                self._operation_ids[state.session_id] = uuid.uuid4().hex
                self._operation_kinds[state.session_id] = kind
            operation_id = self._operation_ids.get(state.session_id)
            if operation_id is None:
                operation_id = uuid.uuid4().hex
                self._operation_ids[state.session_id] = operation_id
            self._operation_kinds.setdefault(state.session_id, kind)
            return operation_id, operation_event

    def claim_operation(
        self,
        state_value: Mapping[str, Any] | AppState | None,
        operation_id: str | None = None,
    ) -> threading.Event:
        """Claim one server operation for a session, rejecting duplicate clicks."""

        state = AppState.from_payload(state_value)
        with self._operation_lock:
            operation_event = self._operation_tokens.get(state.session_id)
            current_id = self._operation_ids.get(state.session_id)
            if operation_id is not None and current_id != operation_id:
                raise StudioJobCancelled(text(state.language, "status_stopped"))
            if state.session_id in self._claimed_operation_tokens:
                raise StudioAppError(text(state.language, "generation_already_running"))
            if operation_id is not None:
                if (
                    operation_event is None
                    or operation_event.is_set()
                    or current_id != operation_id
                ):
                    raise StudioJobCancelled(text(state.language, "status_stopped"))
            elif operation_event is not None and operation_event.is_set():
                # A direct caller may use begin_operation()/cancel_operation()
                # without a browser ticket.  Preserve the cancelled token as a
                # tombstone instead of silently resurrecting that queued call.
                raise StudioJobCancelled(text(state.language, "status_stopped"))
            elif operation_event is None:
                operation_event = threading.Event()
                self._operation_tokens[state.session_id] = operation_event
                self._operation_ids[state.session_id] = uuid.uuid4().hex
                self._operation_kinds[state.session_id] = "generation"
            self._claimed_operation_tokens[state.session_id] = operation_event
            return operation_event

    def finish_operation(
        self,
        state_value: Mapping[str, Any] | AppState | None,
        operation_event: threading.Event,
    ) -> None:
        """Forget an event only when it is still the active session operation."""

        state = AppState.from_payload(state_value)
        with self._operation_lock:
            if self._operation_tokens.get(state.session_id) is operation_event:
                self._operation_tokens.pop(state.session_id, None)
                self._operation_ids.pop(state.session_id, None)
                self._operation_kinds.pop(state.session_id, None)
                self._operation_targets.pop(state.session_id, None)
            if self._claimed_operation_tokens.get(state.session_id) is operation_event:
                self._claimed_operation_tokens.pop(state.session_id, None)

    def clear_operation(
        self,
        state_value: Mapping[str, Any] | AppState | None,
        operation_id: str | None = None,
    ) -> None:
        """Drop UI bookkeeping after a queued event completes or is cancelled."""

        state = AppState.from_payload(state_value)
        with self._operation_lock:
            if (
                operation_id is not None
                and self._operation_ids.get(state.session_id) != operation_id
            ):
                return
            # A Stop click can cancel Gradio's queued event while the backend
            # thread is still unwinding. Keep that active claim and its token
            # until ``finish_operation`` runs so a quick second click cannot
            # overlap the first GPU job.
            if state.session_id not in self._claimed_operation_tokens:
                operation_event = self._operation_tokens.get(state.session_id)
                if (
                    operation_id is not None
                    or operation_event is None
                    or not operation_event.is_set()
                ):
                    self._operation_tokens.pop(state.session_id, None)
                    self._operation_ids.pop(state.session_id, None)
                    self._operation_kinds.pop(state.session_id, None)
                    self._operation_targets.pop(state.session_id, None)

    def operation_state(
        self,
        state_value: Mapping[str, Any] | AppState | None,
    ) -> str:
        """Return ``idle``, ``running``, or ``stopping`` for one browser session."""

        state = AppState.from_payload(state_value)
        with self._operation_lock:
            operation_event = self._operation_tokens.get(state.session_id)
            claimed = state.session_id in self._claimed_operation_tokens
            if operation_event is None and not claimed:
                return "idle"
            if operation_event is not None and operation_event.is_set():
                return "stopping" if claimed else "idle"
            return "running"

    def operation_kind(
        self,
        state_value: Mapping[str, Any] | AppState | None,
    ) -> str | None:
        """Return the active UI operation family for reconnect-safe controls."""

        state = AppState.from_payload(state_value)
        with self._operation_lock:
            return self._operation_kinds.get(state.session_id)

    def _set_operation_target(self, state: AppState, target: Any) -> None:
        with self._operation_lock:
            self._operation_targets[state.session_id] = target

    @contextmanager
    def _generation_slot(
        self,
        state: AppState,
        operation_event: threading.Event,
    ) -> Iterator[None]:
        """Serialize accelerator/cache work across local Studio processes."""

        with _PROCESS_GENERATION_STATE_LOCK:
            if state.session_id in _PROCESS_ACTIVE_GENERATION_SESSIONS:
                raise StudioAppError(text(state.language, "generation_already_running"))
            _PROCESS_ACTIVE_GENERATION_SESSIONS.add(state.session_id)
        acquired = False
        try:
            while not acquired:
                self._check_cancelled(operation_event)
                acquired = _PROCESS_GENERATION_LOCK.acquire(timeout=0.1)
            self._check_cancelled(operation_event)
            try:
                with self._host_lease.acquire(
                    cancellation_event=operation_event,
                    timeout_seconds=self._host_lease_timeout_seconds,
                    session_id=state.session_id,
                ):
                    self._check_cancelled(operation_event)
                    yield
            except StudioJobCancelled as exc:
                raise StudioJobCancelled(text(state.language, "status_stopped")) from exc
            except TimeoutError as exc:
                message = (
                    "Another AIsketcher process is still using this GPU or model cache. "
                    "Wait for it to finish, or stop that process and retry."
                    if state.language == "en"
                    else (
                        "다른 AIsketcher 프로세스가 이 GPU 또는 모델 캐시를 사용 중입니다. "
                        "작업이 끝날 때까지 기다리거나 해당 프로세스를 중지한 뒤 다시 시도하세요."
                    )
                )
                raise StudioAppError(message) from exc
        finally:
            if acquired:
                _PROCESS_GENERATION_LOCK.release()
            with _PROCESS_GENERATION_STATE_LOCK:
                _PROCESS_ACTIVE_GENERATION_SESSIONS.discard(state.session_id)

    @staticmethod
    def _is_accelerator_oom(exc: BaseException) -> bool:
        name = type(exc).__name__.casefold()
        message = str(exc).casefold()
        return (
            "outofmemory" in name
            or "cuda out of memory" in message
            or "cuda error: out of memory" in message
            or "mps backend out of memory" in message
        )

    @classmethod
    def _release_accelerator_memory(cls, target: Any) -> None:
        """Close a failed runtime and release allocator caches when available."""

        cls._request_cancel(target)
        backend = getattr(target, "backend", None)
        cls._request_cancel(backend)
        for candidate in (backend, target):
            close = getattr(candidate, "close", None)
            if callable(close):
                with suppress(Exception):
                    close()
        with suppress(Exception):
            import torch

            cuda = getattr(torch, "cuda", None)
            if cuda is not None and callable(getattr(cuda, "empty_cache", None)):
                cuda.empty_cache()
        gc.collect()

    def _recover_accelerator_failure(
        self,
        preset: str,
        studio: Any,
    ) -> None:
        evicted = self.pool.discard(preset, expected=studio)
        self._release_accelerator_memory(evicted or studio)

    @staticmethod
    def _check_cancelled(operation_event: threading.Event) -> None:
        if operation_event.is_set():
            raise StudioJobCancelled("Stopped by user.")

    @staticmethod
    def _request_cancel(target: Any) -> None:
        """Ask an optional backend/installer to stop without assuming its API."""

        if target is None:
            return
        for name in ("request_cancel", "cancel"):
            method = getattr(target, name, None)
            if callable(method):
                with suppress(Exception):
                    _call_supported(method)
                # Cancellation is best-effort at provider boundaries.  The
                # shared token remains the authoritative cooperative hook.
                return

    def cancel_operation(
        self,
        state_value: Mapping[str, Any] | AppState | None,
        operation_id: str | None = None,
    ) -> str:
        """Stop this session's queued/running work and notify optional providers."""

        state = AppState.from_payload(state_value)
        with self._operation_lock:
            if (
                operation_id is not None
                and self._operation_ids.get(state.session_id) != operation_id
            ):
                return text(state.language, "status_stopped")
            operation_event = self._operation_tokens.get(state.session_id)
            if operation_event is not None:
                operation_event.set()
            claimed_event = self._claimed_operation_tokens.get(state.session_id)
            active_target = (
                self._operation_targets.get(state.session_id)
                if operation_event is not None and claimed_event is operation_event
                else None
            )
        self._request_cancel(active_target)
        self._request_cancel(getattr(active_target, "backend", None))
        return text(state.language, "status_stopped")

    def refinement_mode(self, state_value: Mapping[str, Any] | AppState | None) -> str:
        """Return ``guided`` or ``live`` for the selected refinement target."""

        state = AppState.from_payload(state_value)
        record = self.registry.get(state.run_id, state.session_id)
        if state.selected_index is None:
            raise StudioAppError(text(state.language, "refine_select_first"))
        return "guided" if record.guided_sample is not None else "live"

    def close(self) -> None:
        """Remove run data and the auto-created Studio workspace, if any."""

        with self._operation_lock:
            for operation_event in self._operation_tokens.values():
                operation_event.set()
            self._operation_tokens.clear()
            self._operation_ids.clear()
            self._operation_kinds.clear()
            self._operation_targets.clear()
            self._claimed_operation_tokens.clear()
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
        return tuple((item.path, cls._candidate_label(item.label, language)) for item in candidates)

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

        state = AppState.from_payload(state_value).replace(language=normalize_language(language))
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

    def _normalize_model_prompt(
        self,
        prompt: str,
        *,
        language: str,
        enhance_for_design_edit: bool = False,
        operation_event: threading.Event | None = None,
    ) -> NormalizedPrompt:
        """Normalize a user prompt within the owning operation's lifetime.

        Translation providers are not required to implement cooperative
        cancellation.  Checking the same per-session token immediately before
        and after normalization guarantees that a Stop received while a
        blocking translator is running can never fall through into generation.
        """

        if operation_event is not None:
            self._check_cancelled(operation_event)
        try:
            result = normalize_prompt(
                prompt,
                translator=self.prompt_translator,
                enhance_for_design_edit=enhance_for_design_edit,
            )
        except Exception as exc:  # noqa: BLE001 - optional translator boundary
            if operation_event is not None and operation_event.is_set():
                raise StudioJobCancelled(text(language, "status_stopped")) from exc
            if contains_hangul(prompt):
                raise StudioAppError(_translation_unavailable_message(language)) from exc
            raise StudioAppError(f"The creative brief could not be prepared: {exc}") from exc
        if operation_event is not None:
            self._check_cancelled(operation_event)
        if not result.model_ready:
            raise StudioAppError(_translation_unavailable_message(language))
        return result

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
        operation_id: str | None = None,
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
        steps, guidance = _effective_generation_recipe(preset, steps, guidance)
        operation_event = self.claim_operation(state, operation_id)
        session_dir = _safe_session_dir(self.workspace_root, state.session_id)
        run_id = uuid.uuid4().hex
        run_dir = session_dir / run_id
        studio: Any = None
        try:
            normalized = self._normalize_model_prompt(
                prompt,
                language=state.language,
                enhance_for_design_edit=preset.startswith("flux2-klein-edit@"),
                operation_event=operation_event,
            )
            model_prompt = normalized.require_model_prompt()
            prompt_metadata = _normalization_metadata(normalized)
            with self._generation_slot(state, operation_event):
                self._check_cancelled(operation_event)
                source = sanitize_upload(image_path, run_dir)
                self._check_cancelled(operation_event)
                studio = self.pool.get(preset)
                self._set_operation_target(state, studio)
                prepared = _call_supported(
                    studio.prepare,
                    str(source),
                    cancellation_token=operation_event,
                    cancel_event=operation_event,
                    should_cancel=operation_event.is_set,
                )
                self._check_cancelled(operation_event)
                intent = _intent(
                    prompt,
                    profile,
                    structure,
                    model_prompt=model_prompt,
                    prompt_metadata=prompt_metadata,
                )
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
                    cancellation_token=operation_event,
                    cancel_event=operation_event,
                    should_cancel=operation_event.is_set,
                )
                self._check_cancelled(operation_event)
                candidates = _materialize_study(
                    study,
                    run_dir / "results",
                    should_cancel=operation_event.is_set,
                )
        except StudioJobCancelled:
            if studio is not None:
                self._recover_accelerator_failure(preset, studio)
            shutil.rmtree(run_dir, ignore_errors=True)
            raise
        except StudioAppError:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise
        except Exception as exc:  # noqa: BLE001 - optional backend boundary
            shutil.rmtree(run_dir, ignore_errors=True)
            if operation_event.is_set():
                if studio is not None:
                    self._recover_accelerator_failure(preset, studio)
                else:
                    self._release_accelerator_memory(None)
                raise StudioJobCancelled(text(state.language, "status_stopped")) from exc
            if self._is_accelerator_oom(exc):
                if studio is not None:
                    self._recover_accelerator_failure(preset, studio)
                else:
                    self._release_accelerator_memory(None)
                raise StudioAppError(text(state.language, "gpu_out_of_memory")) from exc
            raise StudioAppError(f"Generation failed: {exc}") from exc
        finally:
            self.finish_operation(state, operation_event)
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
                "model_prompt": model_prompt,
                "prompt_metadata": prompt_metadata,
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

    def recover_latest_run(
        self,
        state_value: Mapping[str, Any] | AppState | None,
        language: str,
    ) -> AppResponse | None:
        """Restore the newest retained run after a browser refresh."""

        state = AppState.from_payload(state_value).replace(language=normalize_language(language))
        record = self.registry.latest(state.session_id)
        if record is None:
            return None
        selected = state.selected_index if state.run_id == record.run_id else 0
        if selected is None or not 0 <= selected < len(record.candidates):
            selected = 0
        state = state.replace(
            run_id=record.run_id,
            selected_index=selected,
            guided=record.guided_sample is not None,
        )
        candidate = record.candidates[selected]
        return AppResponse(
            state=state.payload(),
            source=str(record.source_path),
            selected=candidate.path,
            gallery=self._gallery(record.candidates, state.language),
            recommendation=self._recommendation(candidate, state.language),
            status=self._status(record, state),
            prompt=(
                record.guided_sample.prompt
                if record.guided_sample is not None
                else str(record.request.get("brief") or "")
            ),
            profile=(
                record.guided_sample.profile
                if record.guided_sample is not None
                else str(record.request.get("profile") or "graphic_design")
            ),
            structure=(
                record.guided_sample.structure
                if record.guided_sample is not None
                else str(record.request.get("structure") or "balanced")
            ),
            sync_recipe_controls=True,
        )

    def select_candidate(
        self, state_value: Mapping[str, Any] | AppState | None, index: int
    ) -> tuple[dict[str, Any], str, str, str]:
        state = AppState.from_payload(state_value)
        record = self.registry.get(state.run_id, state.session_id)
        if not 0 <= index < len(record.candidates):
            raise StudioAppError("Choose a valid candidate.")
        pick = getattr(record.study, "pick", None)
        if callable(pick):
            try:
                pick(index)
            except Exception as exc:  # noqa: BLE001 - optional backend boundary
                raise StudioAppError(
                    f"The selected direction could not be recorded: {exc}"
                ) from exc
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
        additional_instruction: str = "",
        operation_id: str | None = None,
    ) -> AppResponse:
        state = AppState.from_payload(state_value)
        record = self.registry.get(state.run_id, state.session_id)
        if record.guided_sample is not None:
            raise StudioAppError(
                "Guided Sample is read-only. Open the model preparation layer and "
                "prepare Auto or FLUX.2 Klein to create a live refinement."
                if state.language == "en"
                else "가이드 샘플은 읽기 전용입니다. 모델 준비 레이어에서 Auto 또는 "
                "FLUX.2 Klein을 준비하면 실제 발전 결과를 만들 수 있습니다."
            )
        if state.selected_index is None:
            raise StudioAppError("Select a direction first.")
        instruction = additional_instruction.strip()
        if len(instruction) > 600:
            raise StudioAppError(text(state.language, "refine_instruction_too_long"))
        automatic_instruction = not instruction
        if automatic_instruction:
            instruction = (
                "Polish the selected direction while preserving its core idea and structure."
            )
        pick = getattr(record.study, "pick", None)
        if not callable(pick):
            raise StudioAppError("The active backend does not support pick-and-vary yet.")
        operation_event = self.claim_operation(state, operation_id)
        preset = str(record.request.get("preset") or "")
        new_id = uuid.uuid4().hex
        destination = record.workspace.parent / new_id
        owns_studio = False
        active_studio: Any = None
        try:
            normalized_instruction = self._normalize_model_prompt(
                instruction,
                language=state.language,
                operation_event=operation_event,
            )
            model_instruction = normalized_instruction.require_model_prompt()
            with self._generation_slot(state, operation_event):
                # A cancelled/OOM run evicts and closes the pooled runtime. Older
                # RunRecords intentionally retain their immutable Study data, but
                # must reacquire the current preset runtime before variation.
                # Otherwise one session's cancellation poisons every completed
                # run that still points at the evicted Studio instance.
                active_studio = self.pool.get(preset) if preset else record.studio
                vary = getattr(active_studio, "vary", None)
                if not callable(vary):
                    raise StudioAppError("The active backend does not support pick-and-vary yet.")
                self._set_operation_target(state, active_studio)
                owns_studio = True
                self._check_cancelled(operation_event)
                selected = pick(state.selected_index)
                original_recipe = getattr(selected, "recipe", None)
                original_prompt = getattr(original_recipe, "prompt", None)
                if not isinstance(original_prompt, str) or not original_prompt.strip():
                    original_prompt = str(record.request.get("brief") or "").strip()
                original_model_prompt = getattr(original_recipe, "model_prompt", None)
                if not isinstance(original_model_prompt, str) or not original_model_prompt.strip():
                    original_model_prompt = str(
                        record.request.get("model_prompt") or original_prompt
                    ).strip()
                prompt_metadata = _refined_prompt_metadata(
                    getattr(
                        original_recipe,
                        "prompt_metadata",
                        record.request.get("prompt_metadata", {}),
                    ),
                    original_instruction=instruction,
                    model_instruction=model_instruction,
                    automatic=automatic_instruction,
                    normalization=normalized_instruction,
                )
                user_instruction_label = (
                    "추가 발전 지시" if state.language == "ko" else "Refinement instruction"
                )
                composed_original_prompt = (
                    f"{original_prompt.rstrip()}\n\n{user_instruction_label}: {instruction}"
                    if original_prompt
                    else instruction
                )
                composed_model_prompt = (
                    f"{original_model_prompt.rstrip()}\n\n"
                    f"Refinement instruction: {model_instruction}"
                    if original_model_prompt
                    else model_instruction
                )
                _validate_refinement_payload(
                    composed_original_prompt,
                    composed_model_prompt,
                    prompt_metadata,
                )
                selected_for_variation = selected
                instruction_embedded = False
                if (
                    original_recipe is not None
                    and is_dataclass(original_recipe)
                    and is_dataclass(selected)
                    and original_prompt
                ):
                    try:
                        refined_recipe = replace(
                            original_recipe,
                            prompt=composed_original_prompt,
                            model_prompt=composed_model_prompt,
                            prompt_metadata=prompt_metadata,
                        )
                        selected_for_variation = replace(
                            cast(Any, selected),
                            recipe=refined_recipe,
                        )
                        instruction_embedded = True
                    except (TypeError, ValueError):
                        # Pluggable candidates may be dataclasses with stricter
                        # constructors. Such backends can consume the explicit
                        # keyword passed below instead.
                        selected_for_variation = selected
                if not instruction_embedded and not _accepts_any_keyword(
                    vary,
                    ("additional_instruction", "refinement_prompt"),
                ):
                    raise StudioAppError(
                        "The active backend cannot apply refinement instructions "
                        "safely. It must accept additional_instruction or "
                        "refinement_prompt, or expose replaceable dataclass recipes."
                    )
                self._check_cancelled(operation_event)
                varied = _call_supported(
                    vary,
                    selected_for_variation,
                    outputs=int(record.request.get("output_count", 4)),
                    strength=strength,
                    locks=tuple(locks),
                    additional_instruction=model_instruction,
                    refinement_prompt=composed_model_prompt,
                    cancellation_token=operation_event,
                    cancel_event=operation_event,
                    should_cancel=operation_event.is_set,
                )
                self._check_cancelled(operation_event)
                destination.mkdir(parents=True, exist_ok=False)
                source = destination / "source.png"
                shutil.copy2(record.source_path, source)
                candidates = _materialize_study(
                    varied,
                    destination / "results",
                    should_cancel=operation_event.is_set,
                )
        except StudioJobCancelled:
            if owns_studio:
                if preset:
                    self._recover_accelerator_failure(preset, active_studio)
                else:
                    self._release_accelerator_memory(active_studio)
            shutil.rmtree(destination, ignore_errors=True)
            raise
        except StudioAppError:
            shutil.rmtree(destination, ignore_errors=True)
            raise
        except Exception as exc:  # noqa: BLE001 - optional backend boundary
            shutil.rmtree(destination, ignore_errors=True)
            if operation_event.is_set():
                if owns_studio:
                    if preset:
                        self._recover_accelerator_failure(preset, active_studio)
                    else:
                        self._release_accelerator_memory(active_studio)
                raise StudioJobCancelled(text(state.language, "status_stopped")) from exc
            if self._is_accelerator_oom(exc):
                if preset:
                    self._recover_accelerator_failure(preset, active_studio)
                else:
                    self._release_accelerator_memory(active_studio)
                raise StudioAppError(text(state.language, "gpu_out_of_memory")) from exc
            raise StudioAppError(f"Variation failed: {exc}") from exc
        finally:
            self.finish_operation(state, operation_event)
        request = dict(record.request)
        request.update(
            {
                "refinement_instruction": instruction,
                "refinement_model_instruction": model_instruction,
                "refinement_instruction_automatic": automatic_instruction,
                "refinement_original_prompt": composed_original_prompt,
                "refinement_prompt": composed_model_prompt,
                "model_prompt": composed_model_prompt,
                "prompt_metadata": prompt_metadata,
                "parent_run_id": record.run_id,
            }
        )
        new_record = RunRecord(
            run_id=new_id,
            session_id=state.session_id,
            workspace=destination,
            source_path=source,
            candidates=candidates,
            study=varied,
            studio=active_studio,
            prepared=record.prepared,
            request=request,
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

    def try_again(
        self,
        state_value: Mapping[str, Any] | AppState | None,
        operation_id: str | None = None,
    ) -> AppResponse:
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
            operation_id,
        )

    def export(self, state_value: Mapping[str, Any] | AppState | None) -> tuple[str, str]:
        state = AppState.from_payload(state_value)
        record = self.registry.get(state.run_id, state.session_id)
        export_dir = record.workspace / "export"
        if record.guided_sample is not None:
            shutil.rmtree(export_dir, ignore_errors=True)
            export_dir.mkdir(parents=True, exist_ok=True)
            manifest, artifacts = _load_replay_bundle(
                record.guided_sample.manifest_path,
                uploaded_json=False,
            )
            for relative, artifact in artifacts:
                target = export_dir / Path(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(artifact, target)
            notice_source = record.guided_sample.root / "ARTWORK_NOTICE.md"
            if not notice_source.is_file():
                raise StudioAppError("Guided Sample artwork notice is not bundled.")
            notice_target = export_dir / "ARTWORK_NOTICE.md"
            shutil.copy2(notice_source, notice_target)
            manifest_files = manifest.get("files")
            if not isinstance(manifest_files, dict):
                raise StudioAppError("Guided Sample manifest has no mutable file table.")
            manifest_files["artwork_notice"] = {
                "path": "ARTWORK_NOTICE.md",
                "sha256": _file_sha256(notice_target),
            }
            (export_dir / "manifest.json").write_text(
                json.dumps(
                    manifest,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
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
            manifest_path = export_dir / "manifest.json"
            instruction = record.request.get("refinement_instruction")
            if isinstance(instruction, str) and instruction and manifest_path.is_file():
                try:
                    manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if not isinstance(manifest_value, dict):
                        raise ValueError("manifest is not an object")
                    manifest_value["refinement"] = {
                        "original_brief": str(record.request.get("brief") or ""),
                        "additional_instruction": instruction,
                        "automatic": bool(
                            record.request.get("refinement_instruction_automatic", False)
                        ),
                        "parent_run_id": str(record.request.get("parent_run_id") or ""),
                    }
                    manifest_path.write_text(
                        json.dumps(
                            manifest_value,
                            indent=2,
                            sort_keys=True,
                            ensure_ascii=False,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    raise StudioAppError(
                        "The refinement manifest could not be annotated safely."
                    ) from exc
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
        operation_id: str | None = None,
    ) -> AppResponse:
        state = AppState.from_payload(state_value)
        if not manifest_path:
            raise StudioAppError("Choose a manifest JSON or export ZIP first.")
        operation_event = self.claim_operation(state, operation_id)
        session_dir = _safe_session_dir(self.workspace_root, state.session_id)
        run_id = uuid.uuid4().hex
        run_dir = session_dir / run_id
        upload_dir = run_dir / "replay-bundle"
        studio: Any = None
        try:
            with self._generation_slot(state, operation_event):
                path = prepare_replay_input(manifest_path, upload_dir)
                payload, _ = _load_replay_bundle(path, uploaded_json=False)
                self._check_cancelled(operation_event)
                studio = self.pool.get(preset)
                self._set_operation_target(state, studio)
                replay = getattr(studio, "replay", None)
                if not callable(replay):
                    raise StudioAppError("The active backend does not support replay.")
                # Replay is the integrity boundary: unlike optional generation
                # overrides, strict mode must never be dropped for compatibility.
                replay_result = replay(str(path), mode="strict")
                self._check_cancelled(operation_event)
                study = getattr(replay_result, "study", None) or replay_result
                if study is replay_result and hasattr(replay_result, "replayed"):
                    raise StudioAppError(
                        "The manifest was verified, but this backend did not produce replayed candidates."
                    )
                candidates = _materialize_study(
                    study,
                    run_dir / "results",
                    should_cancel=operation_event.is_set,
                )
        except StudioJobCancelled:
            if studio is not None:
                self._recover_accelerator_failure(preset, studio)
            shutil.rmtree(run_dir, ignore_errors=True)
            raise
        except StudioAppError:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise
        except Exception as exc:  # noqa: BLE001 - optional backend boundary
            shutil.rmtree(run_dir, ignore_errors=True)
            if operation_event.is_set():
                if studio is not None:
                    self._recover_accelerator_failure(preset, studio)
                else:
                    self._release_accelerator_memory(None)
                raise StudioJobCancelled(text(state.language, "status_stopped")) from exc
            if self._is_accelerator_oom(exc):
                if studio is not None:
                    self._recover_accelerator_failure(preset, studio)
                else:
                    self._release_accelerator_memory(None)
                raise StudioAppError(text(state.language, "gpu_out_of_memory")) from exc
            raise StudioAppError(f"Strict replay failed: {exc}") from exc
        finally:
            self.finish_operation(state, operation_event)
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

    def install_model(
        self,
        preset: str,
        confirmed: bool,
        language: str = "en",
        state_value: Mapping[str, Any] | AppState | None = None,
        operation_id: str | None = None,
    ) -> str:
        if not confirmed:
            raise StudioAppError(
                "Review the download size and license, then confirm."
                if normalize_language(language) == "en"
                else "다운로드 용량과 라이선스를 확인한 뒤 동의하세요."
            )
        installer = self._resolve_model_installer()
        install = installer if callable(installer) else getattr(installer, "install", None)
        if not callable(install):
            raise StudioAppError("The configured model installer is invalid.")
        state = (
            AppState.from_payload(state_value).replace(language=normalize_language(language))
            if state_value is not None
            else AppState.new(language)
        )
        operation_event = self.claim_operation(state, operation_id)
        try:
            # Cache writes and provider initialization share the same process
            # slot as generation. A queued session therefore owns only its
            # cancellation token, never another session's shared installer or
            # translator object.
            with self._generation_slot(state, operation_event):
                self._set_operation_target(state, installer)
                self._check_cancelled(operation_event)
                _call_supported(
                    install,
                    preset,
                    confirm=True,
                    trust_remote_code=False,
                    safe_tensors_only=True,
                    cancellation_token=operation_event,
                    cancel_event=operation_event,
                    should_cancel=operation_event.is_set,
                )
                self._check_cancelled(operation_event)
                prepare_translator = getattr(self.prompt_translator, "prepare", None)
                if callable(prepare_translator):
                    self._set_operation_target(state, self.prompt_translator)
                    _call_supported(
                        prepare_translator,
                        confirm=True,
                        cancellation_token=operation_event,
                        cancel_event=operation_event,
                        should_cancel=operation_event.is_set,
                    )
                    self._check_cancelled(operation_event)
        except Exception as exc:  # noqa: BLE001 - optional provider boundary
            if isinstance(exc, StudioJobCancelled):
                raise
            if operation_event.is_set():
                self._request_cancel(installer)
                self._request_cancel(self.prompt_translator)
                raise StudioJobCancelled(text(state.language, "status_stopped")) from exc
            raise StudioAppError(f"Model preparation failed: {exc}") from exc
        finally:
            self.finish_operation(state, operation_event)
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
    "StudioJobCancelled",
    "StudioIntent",
    "StudioRecipe",
    "StudioSeedPlan",
    "prepare_replay_input",
    "sanitize_upload",
]
