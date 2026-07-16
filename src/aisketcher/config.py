"""Versioned, dependency-free user and project configuration.

The configuration format deliberately supports a small YAML scalar subset.  It
keeps the base package lightweight, makes hand-editing pleasant, and rejects
YAML features (objects, tags, anchors, and nested data) that are unnecessary for
the public settings contract.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from platformdirs import user_config_path

from .errors import ValidationError
from .models import SeedMode
from .presets import get_preset

CONFIG_SCHEMA_VERSION = 1
CONFIG_ENV_VAR = "AISKETCHER_CONFIG"
PROJECT_CONFIG_FILENAME = "aisketcher.yaml"
USER_CONFIG_FILENAME = "config.yaml"

_CONFIG_KEYS = frozenset(
    {
        "schema_version",
        "preset",
        "device",
        "output_count",
        "seed_mode",
        "seed",
        "language",
        "cache_dir",
        "allow_downloads",
    }
)
_DEVICES = frozenset({"auto", "cuda", "mps", "cpu"})
_OUTPUT_COUNTS = frozenset({1, 4, 8})
_LANGUAGES = frozenset({"en", "ko"})
_KEY_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_INTEGER_PATTERN = re.compile(r"[-+]?(?:0|[1-9][0-9]*)")


@dataclass(frozen=True, slots=True)
class AIsketcherConfig:
    """Stable settings shared by the CLI and Studio app.

    ``cache_dir=None`` delegates to the platform-specific model cache already
    used by :class:`~aisketcher.presets.PresetManager`.
    """

    schema_version: int = CONFIG_SCHEMA_VERSION
    preset: str = "sdxl-canny-lite@1"
    device: str = "auto"
    output_count: int = 4
    seed_mode: SeedMode | str = SeedMode.SCOUT
    seed: int | None = None
    language: str = "en"
    cache_dir: str | None = None
    allow_downloads: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != CONFIG_SCHEMA_VERSION
        ):
            raise ValidationError(
                f"schema_version must be {CONFIG_SCHEMA_VERSION}; "
                f"received {self.schema_version!r}"
            )

        preset = _require_string(self.preset, "preset")
        canonical_preset = get_preset(preset).name
        object.__setattr__(self, "preset", canonical_preset)

        device = _require_string(self.device, "device").lower()
        if device not in _DEVICES:
            raise ValidationError("device must be auto, cuda, mps, or cpu")
        object.__setattr__(self, "device", device)

        if isinstance(self.output_count, bool) or self.output_count not in _OUTPUT_COUNTS:
            raise ValidationError("output_count must be 1, 4, or 8")

        try:
            seed_mode = SeedMode(self.seed_mode)
        except (TypeError, ValueError) as exc:
            raise ValidationError("seed_mode must be scout, locked, or explicit") from exc
        object.__setattr__(self, "seed_mode", seed_mode)

        if self.seed is not None and (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed <= (1 << 63) - 1
        ):
            raise ValidationError("seed must be a non-negative 63-bit integer or null")
        if seed_mode is SeedMode.LOCKED:
            if self.seed is None:
                raise ValidationError("locked seed_mode requires seed")
            if self.output_count != 1:
                raise ValidationError("locked seed_mode requires output_count 1")
        elif self.seed is not None:
            raise ValidationError("seed is only valid when seed_mode is locked")

        language = _require_string(self.language, "language").lower()
        if language not in _LANGUAGES:
            raise ValidationError("language must be en or ko")
        object.__setattr__(self, "language", language)

        if self.cache_dir is not None:
            cache_dir = _require_string(self.cache_dir, "cache_dir")
            if "\x00" in cache_dir:
                raise ValidationError("cache_dir cannot contain a null byte")
            object.__setattr__(self, "cache_dir", cache_dir)

        if not isinstance(self.allow_downloads, bool):
            raise ValidationError("allow_downloads must be true or false")

    @property
    def cache_path(self) -> Path | None:
        """Return the expanded configured cache path, if one was supplied."""

        return Path(self.cache_dir).expanduser() if self.cache_dir is not None else None

    def to_dict(self) -> dict[str, object]:
        """Return a serialization-safe mapping in schema order."""

        return {
            "schema_version": self.schema_version,
            "preset": self.preset,
            "device": self.device,
            "output_count": self.output_count,
            "seed_mode": str(self.seed_mode),
            "seed": self.seed,
            "language": self.language,
            "cache_dir": self.cache_dir,
            "allow_downloads": self.allow_downloads,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> AIsketcherConfig:
        """Validate a complete or partial mapping against the current schema."""

        _validate_keys(value)
        return cls(**dict(value))


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} must be a non-empty string")
    return value.strip()


def default_user_config_path() -> Path:
    """Return the cross-platform per-user configuration path."""

    return user_config_path("AIsketcher", appauthor=False) / USER_CONFIG_FILENAME


def default_project_config_path(directory: str | Path | None = None) -> Path:
    """Return ``aisketcher.yaml`` below ``directory`` (or the current directory)."""

    base = Path.cwd() if directory is None else Path(directory)
    return base / PROJECT_CONFIG_FILENAME


def load_config(
    *,
    user_path: str | Path | None = None,
    project_path: str | Path | None = None,
) -> AIsketcherConfig:
    """Load defaults, then user settings, then a project override.

    ``AISKETCHER_CONFIG`` selects an explicit project override when
    ``project_path`` is omitted.  An explicitly selected environment path must
    exist; automatically discovered user and project files are optional.
    """

    config = AIsketcherConfig()
    resolved_user = default_user_config_path() if user_path is None else Path(user_path)
    if resolved_user.is_file():
        config = _merge_file(config, resolved_user)

    configured_project = os.environ.get(CONFIG_ENV_VAR) if project_path is None else None
    if project_path is not None:
        resolved_project = Path(project_path)
        project_is_required = True
    elif configured_project:
        resolved_project = Path(configured_project).expanduser()
        project_is_required = True
    else:
        resolved_project = default_project_config_path()
        project_is_required = False

    if resolved_project.is_file():
        if resolved_project.resolve() != resolved_user.resolve():
            config = _merge_file(config, resolved_project)
    elif project_is_required:
        raise ValidationError(f"Configuration file does not exist: {resolved_project}")
    return config


def load_config_file(path: str | Path) -> AIsketcherConfig:
    """Load one configuration file over the package defaults."""

    resolved = Path(path)
    if not resolved.is_file():
        raise ValidationError(f"Configuration file does not exist: {resolved}")
    return _merge_file(AIsketcherConfig(), resolved)


def save_config(
    config: AIsketcherConfig,
    path: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically save ``config`` as a user-readable YAML document.

    Existing settings are protected unless ``overwrite=True`` is explicit.
    Newly created temporary and destination files use owner-only permissions.
    """

    if not isinstance(config, AIsketcherConfig):
        raise TypeError("config must be an AIsketcherConfig")
    destination = default_user_config_path() if path is None else Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Configuration already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = _dump_yaml(config)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(document)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _merge_file(base: AIsketcherConfig, path: Path) -> AIsketcherConfig:
    try:
        document = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValidationError(f"Cannot read configuration file {path}: {exc}") from exc
    try:
        values = _parse_yaml(document)
        _validate_keys(values)
        if "schema_version" not in values:
            raise ValidationError("schema_version is required")
        version = values.pop("schema_version")
        if isinstance(version, bool) or version != CONFIG_SCHEMA_VERSION:
            raise ValidationError(
                f"schema_version must be {CONFIG_SCHEMA_VERSION}; received {version!r}"
            )
        merged = base.to_dict()
        merged.update(values)
        return AIsketcherConfig.from_mapping(merged)
    except ValidationError as exc:
        raise ValidationError(f"Invalid configuration in {path}: {exc}") from exc


def _validate_keys(value: Mapping[str, Any]) -> None:
    unknown = sorted(set(value) - _CONFIG_KEYS)
    if unknown:
        raise ValidationError(f"Unknown configuration setting(s): {', '.join(unknown)}")


def _dump_yaml(config: AIsketcherConfig) -> str:
    values = config.to_dict()
    lines = [
        "# AIsketcher user settings. Project settings may override these values.",
        "# Schema changes are explicit so future releases can migrate safely.",
    ]
    for key, value in values.items():
        lines.append(f"{key}: {_dump_scalar(value)}")
    return "\n".join(lines) + "\n"


def _dump_scalar(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    raise TypeError(f"Unsupported configuration value: {type(value).__name__}")


def _parse_yaml(document: str) -> dict[str, object]:
    values: dict[str, object] = {}
    for line_number, raw_line in enumerate(document.splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if raw_line.strip() in {"---", "..."}:
            continue
        if raw_line.startswith((" ", "\t")):
            raise ValidationError(
                f"Nested YAML is not supported (line {line_number})"
            )
        key, separator, raw_value = raw_line.partition(":")
        key = key.strip()
        if not separator or _KEY_PATTERN.fullmatch(key) is None:
            raise ValidationError(f"Expected 'setting: value' on line {line_number}")
        if key in values:
            raise ValidationError(f"Duplicate setting {key!r} on line {line_number}")
        values[key] = _parse_scalar(raw_value.strip(), line_number)
    if not values:
        raise ValidationError("Configuration file is empty")
    return values


def _parse_scalar(value: str, line_number: int) -> object:
    if not value:
        raise ValidationError(f"A scalar value is required on line {line_number}")
    if value.startswith(('"', "'")):
        return _parse_quoted_scalar(value, line_number)
    value = _strip_plain_comment(value)
    normalized = value.lower()
    if normalized in {"null", "~"}:
        return None
    if normalized in {"true", "false"}:
        return normalized == "true"
    if _INTEGER_PATTERN.fullmatch(value):
        try:
            return int(value)
        except ValueError as exc:  # pragma: no cover - guarded by the pattern
            raise ValidationError(f"Invalid integer on line {line_number}") from exc
    if value[0] in "[{&*!|>":
        raise ValidationError(f"Only YAML scalar values are supported (line {line_number})")
    return value


def _parse_quoted_scalar(value: str, line_number: int) -> str:
    if value.startswith('"'):
        try:
            parsed, end = json.JSONDecoder().raw_decode(value)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"Invalid quoted string on line {line_number}") from exc
        if not isinstance(parsed, str):
            raise ValidationError(f"Only string quotes are supported (line {line_number})")
        remainder = value[end:].strip()
        if remainder and not remainder.startswith("#"):
            raise ValidationError(f"Invalid quoted string on line {line_number}")
        return parsed
    characters: list[str] = []
    index = 1
    while index < len(value):
        character = value[index]
        if character != "'":
            characters.append(character)
            index += 1
            continue
        if index + 1 < len(value) and value[index + 1] == "'":
            characters.append("'")
            index += 2
            continue
        remainder = value[index + 1 :].strip()
        if remainder and not remainder.startswith("#"):
            raise ValidationError(f"Invalid quoted string on line {line_number}")
        return "".join(characters)
    raise ValidationError(f"Invalid quoted string on line {line_number}")


def _strip_plain_comment(value: str) -> str:
    marker = value.find(" #")
    if marker >= 0:
        value = value[:marker].rstrip()
    return value
