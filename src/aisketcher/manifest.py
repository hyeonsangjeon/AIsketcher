"""Sanitized artifact export and integrity verification for manifest v1."""

from __future__ import annotations

import json
import math
import platform
import re
import shutil
import stat
from collections.abc import Mapping
from hashlib import sha256
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw
from PIL import __version__ as pillow_version

from .errors import IntegrityError, ReplayError, ValidationError
from .study import Study

MANIFEST_SCHEMA = "aisketcher.manifest/v1"
MAX_MANIFEST_BYTES = 5_000_000
MAX_ARTIFACTS = 32
MAX_ARTIFACT_BYTES = 100_000_000

_SENSITIVE_PATTERNS = (
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{16,}\b"),
    re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{16,}|github_pat_[A-Za-z0-9_]{16,})\b"),
    # Retain detection for the underscore form used by early integrations.
    re.compile(r"\bsk_[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"(?i)\b(?:access[_ -]?key|secret[_ -]?key|token|password)\s*[:=]\s*\S+"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{16,}"),
)

_EXPORTABLE_BACKEND_METADATA_KEYS = frozenset(
    {
        "algorithm",
        "backend",
        "device",
        "mps_isolated",
        "mps_retries",
        "sequential",
        "shared_pipeline_components",
        "vae_dtype",
    }
)
_SAFE_METADATA_TEXT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._@:+-]{0,127}\Z")


def _contains_sensitive_text(value: str) -> bool:
    return any(pattern.search(value) for pattern in _SENSITIVE_PATTERNS)


def _sanitize_backend_metadata(metadata_value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only bounded, non-sensitive primitives from built-in backends."""

    sanitized: dict[str, Any] = {}
    for key in sorted(_EXPORTABLE_BACKEND_METADATA_KEYS):
        if key not in metadata_value:
            continue
        value = metadata_value[key]
        if (
            value is None
            or isinstance(value, bool)
            or (isinstance(value, int) and -(1 << 63) <= value <= (1 << 63) - 1)
            or (isinstance(value, float) and math.isfinite(value))
            or (
                isinstance(value, str)
                and _SAFE_METADATA_TEXT.fullmatch(value)
                and not _contains_sensitive_text(value)
            )
        ):
            sanitized[key] = value
    return sanitized


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_clean_png(image: Image.Image, path: Path) -> None:
    # Creating a fresh RGB image guarantees EXIF and source metadata are discarded.
    clean = Image.fromarray(np.asarray(image.convert("RGB"), dtype=np.uint8).copy(), "RGB")
    clean.save(path, format="PNG", optimize=False)


def _contact_sheet(study: Study) -> Image.Image:
    thumbs: list[Image.Image] = []
    thumb_size = (384, 384)
    for candidate in study.candidates:
        thumb = Image.new("RGB", thumb_size, "#eeeae3")
        fitted = candidate.image.copy().convert("RGB")
        fitted.thumbnail((384, 350), Image.Resampling.LANCZOS)
        offset = ((384 - fitted.width) // 2, (350 - fitted.height) // 2)
        thumb.paste(fitted, offset)
        ImageDraw.Draw(thumb).text((12, 360), candidate.id, fill="#17243b")
        thumbs.append(thumb)
    columns = min(4, len(thumbs))
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 384, rows * 384), "#eeeae3")
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % columns) * 384, (index // columns) * 384))
    return sheet


def _read_bounded_json(path: Path) -> Any:
    """Read JSON without trusting a pre-read size check or following a symlink."""

    try:
        path_stat = path.lstat()
        if not stat.S_ISREG(path_stat.st_mode):
            raise ValidationError("Existing export manifest must be a regular file")
        with path.open("rb") as handle:
            encoded = handle.read(MAX_MANIFEST_BYTES + 1)
    except ValidationError:
        raise
    except OSError as exc:
        raise ValidationError("Cannot inspect the existing export manifest") from exc
    if len(encoded) > MAX_MANIFEST_BYTES:
        raise ValidationError(
            f"Existing export manifest exceeds the {MAX_MANIFEST_BYTES:,}-byte safety limit"
        )
    try:
        return json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError("Existing export manifest is not valid UTF-8 JSON") from exc


def _canonical_export_artifacts(manifest: Mapping[str, Any]) -> dict[str, str]:
    """Return the exact artifact layout emitted by manifest v1."""

    files = manifest.get("files")
    candidates = manifest.get("candidates")
    kind = manifest.get("kind")
    if (
        not isinstance(files, dict)
        or not isinstance(candidates, list)
        or not candidates
        or len(files) > MAX_ARTIFACTS
        or kind not in ("exploration", "variation")
    ):
        raise ValidationError("Existing manifest is not a canonical AIsketcher export")

    expected = {
        "source": "source.png",
        "control": "control.png",
        "contact_sheet": "contact-sheet.png",
    }
    if kind == "variation":
        expected["parent"] = "parent.png"

    candidate_ids: set[str] = set()
    for index, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            raise ValidationError("Existing manifest has an invalid candidate entry")
        candidate_id = candidate.get("id")
        if (
            not isinstance(candidate_id, str)
            or re.fullmatch(r"candidate-[0-9a-f]{12}", candidate_id) is None
            or candidate_id in candidate_ids
        ):
            raise ValidationError("Existing manifest has an invalid candidate id")
        candidate_ids.add(candidate_id)
        file_key = f"candidate:{candidate_id}"
        if candidate.get("file") != file_key:
            raise ValidationError("Existing manifest has a non-canonical candidate file")
        expected[file_key] = f"candidates/{index:02d}-{candidate_id}.png"

    if set(files) != set(expected):
        raise ValidationError("Existing manifest has a non-canonical artifact set")
    for key, relative in expected.items():
        descriptor = files[key]
        if (
            not isinstance(descriptor, dict)
            or descriptor.get("path") != relative
            or re.fullmatch(r"[0-9a-f]{64}", str(descriptor.get("sha256", ""))) is None
        ):
            raise ValidationError(f"Existing manifest has an invalid artifact descriptor: {key}")
    return expected


def _validate_owned_export_for_overwrite(output: Path, manifest_path: Path) -> None:
    """Prove an existing directory contains only one intact export before deleting it."""

    try:
        output_stat = output.lstat()
    except OSError as exc:
        raise ValidationError("Cannot inspect the existing export directory") from exc
    if not stat.S_ISDIR(output_stat.st_mode):
        raise ValidationError("overwrite=True requires a real AIsketcher export directory")

    value = _read_bounded_json(manifest_path)
    if not isinstance(value, dict) or value.get("schema") != MANIFEST_SCHEMA:
        raise ValidationError(
            f"overwrite=True requires an existing {MANIFEST_SCHEMA!r} export"
        )
    artifacts = _canonical_export_artifacts(value)

    expected_files = {"manifest.json", *artifacts.values()}
    expected_directories = {
        PurePosixPath(relative).parent.as_posix()
        for relative in artifacts.values()
        if PurePosixPath(relative).parent != PurePosixPath(".")
    }
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    try:
        for entry in output.rglob("*"):
            entry_stat = entry.lstat()
            relative = entry.relative_to(output).as_posix()
            if stat.S_ISLNK(entry_stat.st_mode):
                raise ValidationError("Existing export contains a symbolic link")
            if stat.S_ISDIR(entry_stat.st_mode):
                actual_directories.add(relative)
            elif stat.S_ISREG(entry_stat.st_mode):
                actual_files.add(relative)
            else:
                raise ValidationError("Existing export contains a non-regular filesystem entry")
    except ValidationError:
        raise
    except OSError as exc:
        raise ValidationError("Cannot inspect files in the existing export") from exc

    if actual_files != expected_files or actual_directories != expected_directories:
        raise ValidationError(
            "overwrite=True is only allowed when the directory contains exactly one "
            "canonical AIsketcher export"
        )

    files = value["files"]
    for key, relative in artifacts.items():
        artifact = output / Path(PurePosixPath(relative))
        try:
            artifact_stat = artifact.lstat()
        except OSError as exc:
            raise ValidationError(f"Cannot inspect existing artifact {key!r}") from exc
        if not stat.S_ISREG(artifact_stat.st_mode):
            raise ValidationError(f"Existing artifact {key!r} is not a regular file")
        if artifact_stat.st_size > MAX_ARTIFACT_BYTES:
            raise ValidationError(f"Existing artifact {key!r} exceeds the safety limit")
        if _file_sha256(artifact) != files[key]["sha256"]:
            raise ValidationError(f"Existing artifact {key!r} failed its integrity check")


def runtime_versions(backend: str) -> dict[str, str]:
    try:
        package_version = metadata.version("AIsketcher")
    except metadata.PackageNotFoundError:
        package_version = "0.2.0"
    versions = {
        "aisketcher": package_version,
        "python": platform.python_version(),
        "pillow": pillow_version,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "backend": backend,
    }
    if backend == "diffusers":
        for distribution in ("diffusers", "torch", "transformers"):
            try:
                versions[distribution] = metadata.version(distribution)
            except metadata.PackageNotFoundError:
                versions[distribution] = "unavailable"
    return versions


def export_study(study: Study, path: str | Path, *, overwrite: bool = False) -> Path:
    output = Path(path)
    manifest_path = output / "manifest.json"
    if output.is_symlink() or manifest_path.is_symlink():
        raise ValidationError("Export paths must not be symbolic links")
    if output.exists() and not overwrite:
        raise ValidationError(f"{output} already exists; pass overwrite=True to replace it")
    if _contains_sensitive_text(study.recipe.prompt) or _contains_sensitive_text(
        study.recipe.negative_prompt
    ):
        raise ValidationError(
            "Recipe text appears to contain a credential or token; refusing to export it"
        )
    if overwrite and output.exists():
        _validate_owned_export_for_overwrite(output, manifest_path)
        shutil.rmtree(output)
    # Claim the destination exclusively so a concurrent creator cannot turn this
    # into an implicit overwrite after the existence check above.
    output.mkdir(parents=True, exist_ok=False)
    candidates_dir = output / "candidates"
    candidates_dir.mkdir()

    source_path = output / "source.png"
    control_path = output / "control.png"
    contact_path = output / "contact-sheet.png"
    _save_clean_png(study.prepared.image, source_path)
    _save_clean_png(study.prepared.control, control_path)
    _save_clean_png(_contact_sheet(study), contact_path)

    files: dict[str, dict[str, str]] = {
        "source": {"path": "source.png", "sha256": _file_sha256(source_path)},
        "control": {"path": "control.png", "sha256": _file_sha256(control_path)},
        "contact_sheet": {
            "path": "contact-sheet.png",
            "sha256": _file_sha256(contact_path),
        },
    }
    if study.parent is not None:
        parent_path = output / "parent.png"
        _save_clean_png(study.parent.image, parent_path)
        files["parent"] = {"path": "parent.png", "sha256": _file_sha256(parent_path)}

    candidate_values: list[dict[str, Any]] = []
    for index, candidate in enumerate(study.candidates, start=1):
        relative = PurePosixPath("candidates") / f"{index:02d}-{candidate.id}.png"
        candidate_path = output / Path(relative)
        _save_clean_png(candidate.image, candidate_path)
        file_key = f"candidate:{candidate.id}"
        files[file_key] = {
            "path": relative.as_posix(),
            "sha256": _file_sha256(candidate_path),
        }
        candidate_values.append(
            {
                "id": candidate.id,
                "seed": candidate.seed,
                "file": file_key,
                "parent_id": candidate.parent_id,
                "scores": candidate.scores.to_dict(),
                "backend_metadata": _sanitize_backend_metadata(candidate.backend_metadata),
            }
        )

    recipe_value = study.recipe.to_dict()
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "run_id": study.id,
        "kind": study.kind,
        "backend": study.backend,
        "source": {
            "original_size": list(study.prepared.original_size),
            "prepared_size": list(study.prepared.prepared_size),
            "canny": study.prepared.canny.to_dict(),
            "diagnostics": study.prepared.diagnostics.to_dict(),
        },
        "recipe": recipe_value,
        "recipe_sha256": canonical_sha256(recipe_value),
        "seed_plan": {
            "mode": study.seed_mode.value,
            "seeds": [candidate.seed for candidate in study.candidates],
        },
        "candidates": candidate_values,
        "selection": study.selected_id,
        "lineage": {
            "parent_id": study.parent.id if study.parent is not None else None,
            "locks": list(study.recipe.locks),
            "variation_strength": (
                study.recipe.variation_strength.value
                if study.recipe.variation_strength is not None
                else None
            ),
        },
        "files": files,
        "runtime": runtime_versions(study.backend),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def load_manifest(path: str | Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    try:
        if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
            raise ReplayError(f"Manifest exceeds the {MAX_MANIFEST_BYTES:,}-byte safety limit")
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except ReplayError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReplayError(f"Cannot read manifest: {manifest_path}") from exc
    if not isinstance(value, dict):
        raise ReplayError("Manifest must be a JSON object")
    if value.get("schema") != MANIFEST_SCHEMA:
        raise ReplayError(
            f"Unsupported manifest schema {value.get('schema')!r}; expected {MANIFEST_SCHEMA!r}"
        )
    if not isinstance(value.get("files"), dict):
        raise ReplayError("Manifest files must be an object")
    if len(value["files"]) > MAX_ARTIFACTS:
        raise ReplayError(f"Manifest cannot reference more than {MAX_ARTIFACTS} artifacts")
    return manifest_path, value


def verify_manifest_files(manifest_path: Path, manifest: Mapping[str, Any]) -> tuple[str, ...]:
    root = manifest_path.parent.resolve()
    verified: list[str] = []
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise IntegrityError("Manifest files must be an object")
    for key, descriptor in files.items():
        if not isinstance(key, str) or not isinstance(descriptor, Mapping):
            raise IntegrityError("Manifest contains an invalid artifact descriptor")
        relative = PurePosixPath(str(descriptor.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise IntegrityError(f"Unsafe artifact path for {key!r}")
        path = (root / Path(relative)).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise IntegrityError(f"Artifact {key!r} escapes the export directory") from exc
        if not path.is_file():
            raise IntegrityError(f"Artifact {key!r} is missing: {relative.as_posix()}")
        if path.stat().st_size > MAX_ARTIFACT_BYTES:
            raise IntegrityError(
                f"Artifact {key!r} exceeds the {MAX_ARTIFACT_BYTES:,}-byte safety limit"
            )
        expected = str(descriptor.get("sha256", ""))
        actual = _file_sha256(path)
        if not re.fullmatch(r"[0-9a-f]{64}", expected) or actual != expected:
            raise IntegrityError(f"Artifact hash mismatch for {key!r}")
        verified.append(relative.as_posix())
    return tuple(verified)


__all__ = [
    "MANIFEST_SCHEMA",
    "canonical_sha256",
    "export_study",
    "load_manifest",
    "runtime_versions",
    "verify_manifest_files",
]
