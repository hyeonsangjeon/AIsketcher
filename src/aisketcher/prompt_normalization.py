"""Explicit, dependency-lazy prompt normalization.

Korean text is not sent to an image model under an assumption that the model
understands it.  Applications may inject a translator, or explicitly construct
the local M2M100 adapter below.  Constructing either the normalizer or the
adapter never downloads model weights.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import re
import stat
import threading
import unicodedata
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .errors import (
    AIsketcherError,
    IntegrityError,
    ModelUnavailableError,
    OptionalDependencyError,
    ValidationError,
)
from .model_registry import VerifiedFile

M2M100_KO_EN_MODEL_ID = "facebook/m2m100_418M"
M2M100_KO_EN_REVISION = "55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636"
M2M100_KO_EN_FILES = (
    VerifiedFile(
        path="config.json",
        size_bytes=908,
        sha256="df0ae43e4e4b0d7e3c97b7f447857a70ef6b6a2aa1f145cedbcc730d95f67134",
    ),
    VerifiedFile(
        path="generation_config.json",
        size_bytes=233,
        sha256="aed76366507333ddbb8bd49960f23c82fe6446b3319a46a54befdb45324ccf61",
    ),
    VerifiedFile(
        path="pytorch_model.bin",
        size_bytes=1_935_796_948,
        sha256="d907ea45e4e4b9db163382a6674f6218b3c59566fe06d77f4055c208b4e87ed1",
    ),
    VerifiedFile(
        path="sentencepiece.bpe.model",
        size_bytes=2_423_393,
        sha256="d8f7c76ed2a5e0822be39f0a4f95a55eb19c78f4593ce609e2edbc2aea4d380a",
    ),
    VerifiedFile(
        path="special_tokens_map.json",
        size_bytes=1_140,
        sha256="c1a4f86c3874d279ae1b2a05162858db5dd6c61665d84223ed886cbcff08fda6",
    ),
    VerifiedFile(
        path="tokenizer_config.json",
        size_bytes=298,
        sha256="a53e6aa83da0b82565ed90c3849056307a9453843322ac5b8439ec4b9497fe48",
    ),
    VerifiedFile(
        path="vocab.json",
        size_bytes=3_708_092,
        sha256="b6e77e474aeea8f441363aca7614317c06381f3eacfe10fb9856d5081d1074cc",
    ),
)
M2M100_KO_EN_DOWNLOAD_BYTES = sum(
    required.size_bytes for required in M2M100_KO_EN_FILES
)
M2M100_KO_EN_WEIGHTS_SHA256 = next(
    required.sha256
    for required in M2M100_KO_EN_FILES
    if required.path == "pytorch_model.bin"
)
M2M100_KO_EN_MAX_INPUT_TOKENS = 1_024

_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_HANGUL_PATTERN = re.compile(
    "[\u1100-\u11ff\u3130-\u318f\ua960-\ua97f\uac00-\ud7af\ud7b0-\ud7ff]"
)
_MAX_PROMPT_LENGTH = 10_000
_SAFETENSORS_CONVERSION_ENV = "DISABLE_SAFETENSORS_CONVERSION"
_TRANSFORMERS_LOAD_ENV_LOCK = threading.Lock()
_WEIGHTS_HASH_CHUNK_BYTES = 8 * 1024 * 1024
_VERIFIED_TRANSLATOR_FILES: set[
    tuple[str, str, int, int, int, int, int, int]
] = set()
_WEIGHTS_VERIFICATION_LOCK = threading.Lock()

# Apply longer phrases first so a future shorter entry can never consume part
# of a more specific visual-design term. The source is segmented around these
# terms: M2M100 sees only Korean-source segments, while reviewed production
# vocabulary is assembled into the translated result without being tokenized
# under ``src_lang="ko"``.
_KOREAN_VISUAL_DESIGN_TERMINOLOGY = tuple(
    sorted(
        (
            ("겹겹이 자른 종이 공예", "layered cut-paper craft"),
            ("스튜디오 조명", "studio lighting"),
            ("미니어처", "miniature"),
            ("판타지", "fantasy"),
            ("마스코트", "mascot"),
            ("코발트", "cobalt"),
            ("주홍", "vermilion"),
            ("노랑", "yellow"),
            ("금색", "gold"),
            ("시안", "cyan"),
        ),
        key=lambda item: (-len(item[0]), item[0]),
    )
)
_KOREAN_VISUAL_DESIGN_TRANSLATIONS = dict(_KOREAN_VISUAL_DESIGN_TERMINOLOGY)
_KOREAN_VISUAL_DESIGN_PATTERN = re.compile(
    "|".join(
        re.escape(korean) for korean, _english in _KOREAN_VISUAL_DESIGN_TERMINOLOGY
    )
)
# A connective particle between two reviewed terms is grammatical glue, not a
# useful standalone sentence for M2M100. Keep this deliberately small: other
# Korean text still goes through the pinned translator instead of being
# presented as a hand-authored translation.
_KOREAN_REVIEWED_TERM_CONNECTORS = {
    "과": "and",
    "와": "and",
    "및": "and",
}

DESIGN_EDIT_PROMPT_TEMPLATE = (
    "Design image editing brief:\n"
    "{brief}\n\n"
    "Editing constraints: use the input image as the visual source. "
    "Preserve every unmentioned subject, identity, composition, and geometric "
    "relationship. Apply only changes explicitly requested in the brief. "
    "Do not introduce unrelated content."
)


@contextmanager
def _disable_transformers_conversion_pr() -> Iterator[None]:
    """Prevent Transformers from fetching an unpinned conversion PR.

    Current Transformers releases may start a background SafeTensors conversion
    download after loading a repository that publishes PyTorch weights at the
    requested commit. That helper resolves ``refs/pr/*`` independently of the
    caller's immutable revision and outlives ``from_pretrained``. The upstream
    opt-out environment variable is process-global, so guard the short load
    region and restore its previous value exactly.
    """

    with _TRANSFORMERS_LOAD_ENV_LOCK:
        previous = os.environ.get(_SAFETENSORS_CONVERSION_ENV)
        os.environ[_SAFETENSORS_CONVERSION_ENV] = "1"
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop(_SAFETENSORS_CONVERSION_ENV, None)
            else:
                os.environ[_SAFETENSORS_CONVERSION_ENV] = previous


class TranslationSetupCancelled(AIsketcherError):
    """Raised when explicit local translator preparation is stopped."""


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return str(self.value)


class PromptNormalizationStatus(_StringEnum):
    """Outcome of a prompt normalization attempt."""

    UNCHANGED = "unchanged"
    TRANSLATED = "translated"
    TRANSLATOR_UNAVAILABLE = "translator-unavailable"
    TRANSLATION_FAILED = "translation-failed"


@dataclass(frozen=True, slots=True)
class TranslatorMetadata:
    """Provider details recorded with a translated prompt."""

    provider: str
    model_id: str | None = None
    revision: str | None = None
    local_files_only: bool | None = None

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValidationError("TranslatorMetadata.provider cannot be empty")
        if self.model_id is not None and not self.model_id.strip():
            raise ValidationError("TranslatorMetadata.model_id cannot be empty")
        if self.revision is not None and not self.revision.strip():
            raise ValidationError("TranslatorMetadata.revision cannot be empty")
        if self.local_files_only is not None and not isinstance(
            self.local_files_only, bool
        ):
            raise ValidationError(
                "TranslatorMetadata.local_files_only must be a boolean or None"
            )

    def to_dict(self) -> dict[str, str | bool | None]:
        return {
            "provider": self.provider,
            "model_id": self.model_id,
            "revision": self.revision,
            "local_files_only": self.local_files_only,
        }


@runtime_checkable
class KoreanEnglishTranslator(Protocol):
    """Injectable Korean-to-English translator contract."""

    @property
    def metadata(self) -> TranslatorMetadata:
        """Return provider, model, revision, and network-policy metadata."""

    def translate(self, text: str) -> str:
        """Translate one non-empty Korean or mixed-language prompt to English."""


@dataclass(frozen=True, slots=True)
class NormalizedPrompt:
    """A transparent prompt-normalization result.

    ``normalized_prompt`` is ``None`` when Korean was detected but no verified
    translation could be produced.  This deliberately blocks callers from
    treating raw Korean as a model-ready English prompt while retaining the
    exact original text for the UI and audit trail.
    """

    original_prompt: str
    normalized_prompt: str | None
    detected_language: str
    status: PromptNormalizationStatus
    translator: TranslatorMetadata | None
    enhancement_applied: bool
    warning: str | None = None

    @property
    def model_ready(self) -> bool:
        return self.normalized_prompt is not None

    def require_model_prompt(self) -> str:
        """Return the model-ready prompt or raise an actionable error."""

        if self.normalized_prompt is None:
            raise ValidationError(
                self.warning
                or "The prompt could not be normalized into a model-ready form."
            )
        return self.normalized_prompt

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_prompt": self.original_prompt,
            "normalized_prompt": self.normalized_prompt,
            "detected_language": self.detected_language,
            "status": str(self.status),
            "translator": (
                self.translator.to_dict() if self.translator is not None else None
            ),
            "enhancement_applied": self.enhancement_applied,
            "model_ready": self.model_ready,
            "warning": self.warning,
        }


def contains_hangul(text: str) -> bool:
    """Return whether ``text`` contains modern or compatibility Hangul."""

    return bool(_HANGUL_PATTERN.search(text))


def enhance_design_edit_prompt(brief: str) -> str:
    """Add conservative design-edit guidance without inventing subject matter."""

    normalized_brief = _validated_prompt(brief, name="brief")
    enhanced = DESIGN_EDIT_PROMPT_TEMPLATE.format(brief=normalized_brief)
    if len(enhanced) > _MAX_PROMPT_LENGTH:
        raise ValidationError(
            f"Enhanced prompt cannot exceed {_MAX_PROMPT_LENGTH:,} characters"
        )
    return enhanced


def normalize_prompt(
    prompt: str,
    *,
    translator: KoreanEnglishTranslator | None = None,
    enhance_for_design_edit: bool = False,
) -> NormalizedPrompt:
    """Normalize a user prompt without implicit translation or network access.

    A translator is called only when Hangul is present.  Callers that want the
    bundled M2M100 implementation must instantiate
    :class:`M2M100KoreanEnglishTranslator` themselves, making the cache/network
    policy explicit.
    """

    original_prompt = prompt
    stripped_prompt = _validated_prompt(prompt, name="prompt")
    if not contains_hangul(stripped_prompt):
        normalized = (
            enhance_design_edit_prompt(stripped_prompt)
            if enhance_for_design_edit
            else stripped_prompt
        )
        return NormalizedPrompt(
            original_prompt=original_prompt,
            normalized_prompt=normalized,
            detected_language="und",
            status=PromptNormalizationStatus.UNCHANGED,
            translator=None,
            enhancement_applied=enhance_for_design_edit,
        )

    if translator is None:
        return NormalizedPrompt(
            original_prompt=original_prompt,
            normalized_prompt=None,
            detected_language="ko",
            status=PromptNormalizationStatus.TRANSLATOR_UNAVAILABLE,
            translator=None,
            enhancement_applied=False,
            warning=(
                "Korean text was detected, but no translator was configured. "
                "The original prompt was preserved and was not sent to the image "
                "model. Inject a translator or install aisketcher[translate] and "
                "explicitly load the pinned local M2M100 adapter."
            ),
        )

    metadata = translator.metadata
    try:
        translated = translator.translate(stripped_prompt)
    except Exception:
        return _failed_translation(original_prompt, metadata)
    if not isinstance(translated, str) or not translated.strip():
        return _failed_translation(original_prompt, metadata)

    normalized = translated.strip()
    if len(normalized) > _MAX_PROMPT_LENGTH:
        return _failed_translation(original_prompt, metadata)
    if enhance_for_design_edit:
        try:
            normalized = enhance_design_edit_prompt(normalized)
        except ValidationError:
            return _failed_translation(original_prompt, metadata)

    return NormalizedPrompt(
        original_prompt=original_prompt,
        normalized_prompt=normalized,
        detected_language="ko",
        status=PromptNormalizationStatus.TRANSLATED,
        translator=metadata,
        enhancement_applied=enhance_for_design_edit,
    )


class M2M100KoreanEnglishTranslator:
    """Lazy local M2M100 Korean-to-English adapter.

    The default is cache-only.  Passing ``local_files_only=False`` is an
    explicit opt-in to Hugging Face Hub access.  The revision must always be an
    immutable 40-character commit SHA.
    """

    def __init__(
        self,
        *,
        model_id: str = M2M100_KO_EN_MODEL_ID,
        revision: str = M2M100_KO_EN_REVISION,
        local_files_only: bool = True,
        cache_dir: str | None = None,
        max_new_tokens: int = 512,
    ) -> None:
        model_id = model_id.strip()
        revision = revision.strip().lower()
        if not model_id or "/" not in model_id:
            raise ValidationError(
                "M2M100 model_id must be a non-empty namespaced repository id"
            )
        if not _COMMIT_PATTERN.fullmatch(revision):
            raise ValidationError(
                "M2M100 revision must be an immutable 40-character commit SHA"
            )
        if not isinstance(local_files_only, bool):
            raise ValidationError("local_files_only must be a boolean")
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or not 1 <= max_new_tokens <= 2_048
        ):
            raise ValidationError("max_new_tokens must be an integer from 1 to 2,048")

        self.model_id = model_id
        self.revision = revision
        self.local_files_only = local_files_only
        if cache_dir is None:
            self.cache_dir = None
            self._managed_cache_root: Path | None = None
        else:
            managed_cache_root = Path(
                os.path.abspath(os.fspath(Path(cache_dir).expanduser()))
            )
            self.cache_dir = os.fspath(managed_cache_root)
            self._managed_cache_root = managed_cache_root
        self.max_new_tokens = max_new_tokens
        self._components: tuple[Any, Any] | None = None
        self._load_lock = threading.Lock()

    @property
    def metadata(self) -> TranslatorMetadata:
        return TranslatorMetadata(
            provider="huggingface-transformers",
            model_id=self.model_id,
            revision=self.revision,
            local_files_only=self.local_files_only,
        )

    def translate(self, text: str) -> str:
        source = _validated_prompt(text, name="text")
        tokenizer, model = self._load_components()
        encoded = _encode_m2m100_source(tokenizer, source)
        _enforce_m2m100_input_limit(encoded)
        target_language_id = tokenizer.get_lang_id("en")
        if isinstance(target_language_id, bool) or not isinstance(
            target_language_id, int
        ):
            raise ValidationError("M2M100 tokenizer returned an invalid English ID")

        segments = _segment_korean_visual_design_terminology(source)
        if len(segments) == 1 and segments[0] == (source, None):
            return _generate_m2m100_translation(
                tokenizer,
                model,
                encoded,
                max_new_tokens=self.max_new_tokens,
                target_language_id=target_language_id,
            )

        translated_segments: list[str] = []
        for index, (segment, reviewed_translation) in enumerate(segments):
            if reviewed_translation is not None:
                translated_segments.append(reviewed_translation)
                continue
            leading, translatable, trailing = _split_prompt_boundary_delimiters(
                segment
            )
            if not translatable or not contains_hangul(translatable):
                translated_segments.append(segment)
                continue
            reviewed_connector = _reviewed_term_connector(
                segments, index, translatable
            )
            if reviewed_connector is not None:
                translated_segments.append(
                    f"{leading}{reviewed_connector}{trailing}"
                )
                continue
            segment_encoded = _encode_m2m100_source(tokenizer, translatable)
            _enforce_m2m100_input_limit(segment_encoded)
            translated = _generate_m2m100_translation(
                tokenizer,
                model,
                segment_encoded,
                max_new_tokens=self.max_new_tokens,
                target_language_id=target_language_id,
            )
            translated_segments.append(f"{leading}{translated}{trailing}")

        translated_prompt = _join_translated_prompt_segments(translated_segments)
        if not translated_prompt:
            raise ValidationError("M2M100 translator returned an empty translation")
        return translated_prompt

    def prepare(
        self,
        *,
        confirm: bool = False,
        cancellation_token: threading.Event | None = None,
        cancel_event: threading.Event | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> TranslatorMetadata:
        """Explicitly prepare the pinned translator for later cache-only use.

        Construction and normal translation remain cache-only by default.
        Network access is enabled only for this confirmed first-use operation.
        The official repository contains a pinned PyTorch state dictionary
        rather than SafeTensors, so Transformers is forced to use its
        weights-only loader and remote model code stays disabled. Cancellation
        is cooperative at safe repository-file boundaries; already cached
        files remain available for a later retry.
        """

        if not confirm:
            raise ValidationError(
                "Translator setup requires confirm=True after displaying its "
                "download size and MIT license."
            )
        events = tuple(
            event
            for event in (cancellation_token, cancel_event)
            if event is not None
        )

        def cancellation_requested() -> bool:
            return any(event.is_set() for event in events) or bool(
                should_cancel is not None and should_cancel()
            )

        self._load_components(
            allow_download=True,
            should_cancel=cancellation_requested,
        )
        return self.metadata

    @staticmethod
    def _check_setup_cancelled(
        should_cancel: Callable[[], bool] | None,
    ) -> None:
        if should_cancel is not None and should_cancel():
            raise TranslationSetupCancelled("Korean translator setup was stopped.")

    def _load_components(
        self,
        *,
        allow_download: bool = False,
        should_cancel: Callable[[], bool] | None = None,
    ) -> tuple[Any, Any]:
        self._check_setup_cancelled(should_cancel)
        if self._components is not None:
            return self._components
        with self._load_lock:
            self._check_setup_cancelled(should_cancel)
            cached = self._cached_components()
            if cached is not None:
                return cached
            self._assert_managed_cache_boundary()
            try:
                transformers = importlib.import_module("transformers")
                huggingface_hub = importlib.import_module("huggingface_hub")
            except ImportError as exc:
                raise OptionalDependencyError(
                    "Local Korean translation requires the optional translation "
                    "dependencies. Install them with `pip install "
                    '"aisketcher[translate]"`.'
                ) from exc

            load_options: dict[str, Any] = {
                "revision": self.revision,
                "local_files_only": False if allow_download else self.local_files_only,
                "trust_remote_code": False,
            }
            if self.cache_dir is not None:
                load_options["cache_dir"] = self.cache_dir
            try:
                if (
                    self.model_id != M2M100_KO_EN_MODEL_ID
                    or self.revision != M2M100_KO_EN_REVISION
                ):
                    raise IntegrityError(
                        "No reviewed runtime-file integrity policy exists for "
                        f"{self.model_id}@{self.revision}."
                    )
                snapshot_paths: list[Path] = []
                for required in M2M100_KO_EN_FILES:
                    self._assert_managed_cache_boundary()
                    download_options: dict[str, Any] = {
                        "repo_id": self.model_id,
                        "filename": required.path,
                        "revision": self.revision,
                        "local_files_only": load_options["local_files_only"],
                    }
                    if self.cache_dir is not None:
                        download_options["cache_dir"] = self.cache_dir
                    downloaded_path = Path(
                        huggingface_hub.hf_hub_download(**download_options)
                    )
                    self._check_setup_cancelled(should_cancel)
                    downloaded_path = self._validated_managed_download_path(
                        downloaded_path,
                        required=required,
                    )
                    _verify_translator_file(
                        downloaded_path,
                        required=required,
                        should_cancel=should_cancel,
                    )
                    snapshot_paths.append(downloaded_path)
                snapshot_dir = snapshot_paths[0].parent
                if any(path.parent != snapshot_dir for path in snapshot_paths):
                    raise IntegrityError(
                        "Pinned M2M100 runtime files did not resolve to one "
                        "immutable snapshot."
                    )
                for required, downloaded_path in zip(
                    M2M100_KO_EN_FILES,
                    snapshot_paths,
                    strict=True,
                ):
                    self._validated_managed_download_path(
                        downloaded_path,
                        required=required,
                    )
                _assert_exact_m2m100_snapshot(
                    snapshot_dir,
                    M2M100_KO_EN_FILES,
                )
                with _disable_transformers_conversion_pr():
                    tokenizer = transformers.AutoTokenizer.from_pretrained(
                        os.fspath(snapshot_dir),
                        local_files_only=True,
                        trust_remote_code=False,
                        src_lang="ko",
                    )
                    self._check_setup_cancelled(should_cancel)
                    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
                        os.fspath(snapshot_dir),
                        local_files_only=True,
                        trust_remote_code=False,
                        use_safetensors=False,
                        weights_only=True,
                    )
                self._check_setup_cancelled(should_cancel)
            except OSError as exc:
                location = (
                    "the local Hugging Face cache"
                    if load_options["local_files_only"]
                    else "the pinned Hugging Face Hub revision"
                )
                raise ModelUnavailableError(
                    f"Could not load {self.model_id}@{self.revision} from {location}. "
                    "No fallback model or moving revision was used."
                ) from exc
            self._components = (tokenizer, model)
            return self._components

    def _cached_components(self) -> tuple[Any, Any] | None:
        return self._components

    def _assert_managed_cache_boundary(self) -> None:
        """Reject a configured translation cache with a symlinked boundary."""

        if self._managed_cache_root is None:
            return
        if _path_has_symlink_component(self._managed_cache_root):
            raise IntegrityError(
                "Managed M2M100 translation cache boundary cannot contain a symlink."
            )

    def _validated_managed_download_path(
        self,
        downloaded_path: Path,
        *,
        required: VerifiedFile,
    ) -> Path:
        """Keep Hub writes and loads below the configured translation cache.

        Hugging Face snapshots intentionally expose each file as a symlink to
        an immutable blob in the same repository cache.  The file symlink is
        therefore allowed, while the managed cache root, every directory on
        the snapshot path, and the resolved blob target must remain inside the
        configured cache.
        """

        if self._managed_cache_root is None:
            return downloaded_path

        self._assert_managed_cache_boundary()
        absolute_path = Path(os.path.abspath(os.fspath(downloaded_path)))
        try:
            absolute_path.relative_to(self._managed_cache_root)
        except ValueError as exc:
            raise IntegrityError(
                "Pinned M2M100 runtime file escaped the managed translation "
                f"cache: {required.path}."
            ) from exc
        if _path_has_symlink_component(absolute_path.parent):
            raise IntegrityError(
                "Managed M2M100 translation cache directory path cannot contain "
                f"a symlink: {required.path}."
            )
        try:
            resolved_path = absolute_path.resolve(strict=True)
            resolved_path.relative_to(self._managed_cache_root)
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise IntegrityError(
                "Pinned M2M100 runtime file symlink escaped the managed "
                f"translation cache: {required.path}."
            ) from exc
        return absolute_path


def _assert_exact_m2m100_snapshot(
    snapshot_dir: Path,
    required_files: tuple[VerifiedFile, ...],
) -> None:
    """Reject runtime entries that are not covered by the pinned hash policy."""

    if snapshot_dir.is_symlink() or not snapshot_dir.is_dir():
        raise IntegrityError("Pinned M2M100 snapshot is not a regular directory.")
    expected_files = {
        Path(*required.path.split("/")).as_posix() for required in required_files
    }
    expected_directories: set[str] = set()
    for expected_file in expected_files:
        parent = Path(expected_file).parent
        while parent != Path("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    try:
        entries = tuple(snapshot_dir.rglob("*"))
    except OSError as exc:
        raise IntegrityError("Pinned M2M100 snapshot could not be inspected.") from exc

    for entry in entries:
        try:
            relative_name = entry.relative_to(snapshot_dir).as_posix()
        except ValueError as exc:
            raise IntegrityError(
                "Pinned M2M100 snapshot entry escaped its directory."
            ) from exc
        if relative_name in expected_files:
            if entry.is_file():
                continue
            raise IntegrityError(
                "Pinned M2M100 runtime entry is not a regular file: "
                f"{relative_name}."
            )
        if (
            relative_name in expected_directories
            and entry.is_dir()
            and not entry.is_symlink()
        ):
            continue
        raise IntegrityError(
            f"Pinned M2M100 snapshot contains undeclared runtime entry: "
            f"{relative_name}."
        )


def _encode_m2m100_source(tokenizer: Any, source: str) -> Mapping[str, Any]:
    encoded = tokenizer(
        source,
        return_tensors="pt",
        truncation=False,
    )
    if not isinstance(encoded, Mapping):
        raise ValidationError("M2M100 tokenizer returned an invalid input batch")
    return encoded


def _enforce_m2m100_input_limit(encoded: Mapping[str, Any]) -> None:
    input_token_count = _single_input_token_count(encoded.get("input_ids"))
    if input_token_count > M2M100_KO_EN_MAX_INPUT_TOKENS:
        raise ValidationError(
            f"M2M100 input contains {input_token_count:,} tokens; the maximum "
            f"is {M2M100_KO_EN_MAX_INPUT_TOKENS:,}. Shorten the prompt before "
            "translation."
        )


def _generate_m2m100_translation(
    tokenizer: Any,
    model: Any,
    encoded: Mapping[str, Any],
    *,
    max_new_tokens: int,
    target_language_id: int,
) -> str:
    generated = model.generate(
        **dict(encoded),
        max_new_tokens=max_new_tokens,
        forced_bos_token_id=target_language_id,
    )
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    if (
        not isinstance(decoded, list)
        or not decoded
        or not isinstance(decoded[0], str)
        or not decoded[0].strip()
    ):
        raise ValidationError("M2M100 translator returned an empty translation")
    return decoded[0].strip()


def _validated_prompt(value: str, *, name: str) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValidationError(f"{name} cannot be empty")
    if len(normalized) > _MAX_PROMPT_LENGTH:
        raise ValidationError(
            f"{name} cannot exceed {_MAX_PROMPT_LENGTH:,} characters"
        )
    return normalized


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


def _segment_korean_visual_design_terminology(
    source: str,
) -> tuple[tuple[str, str | None], ...]:
    """Split Korean source text around reviewed visual-design terminology."""

    segments: list[tuple[str, str | None]] = []
    cursor = 0
    for match in _KOREAN_VISUAL_DESIGN_PATTERN.finditer(source):
        if match.start() > cursor:
            segments.append((source[cursor : match.start()], None))
        korean = match.group(0)
        segments.append((korean, _KOREAN_VISUAL_DESIGN_TRANSLATIONS[korean]))
        cursor = match.end()
    if cursor < len(source):
        segments.append((source[cursor:], None))
    return tuple(segments) or ((source, None),)


def _split_prompt_boundary_delimiters(source: str) -> tuple[str, str, str]:
    """Keep punctuation and whitespace out of isolated translation calls.

    M2M100 may omit a leading comma when translating a short Korean fragment.
    The delimiter came from the user's prompt, so preserve it deterministically
    and translate only the fragment's lexical core.
    """

    start = 0
    while start < len(source) and _is_prompt_boundary_delimiter(source[start]):
        start += 1
    end = len(source)
    while end > start and _is_prompt_boundary_delimiter(source[end - 1]):
        end -= 1
    return source[:start], source[start:end], source[end:]


def _is_prompt_boundary_delimiter(character: str) -> bool:
    return character.isspace() or unicodedata.category(character).startswith("P")


def _reviewed_term_connector(
    segments: tuple[tuple[str, str | None], ...],
    index: int,
    translatable: str,
) -> str | None:
    """Resolve only exact connectors occurring between two reviewed terms."""

    if index <= 0 or index >= len(segments) - 1:
        return None
    if segments[index - 1][1] is None or segments[index + 1][1] is None:
        return None
    return _KOREAN_REVIEWED_TERM_CONNECTORS.get(translatable)


def _join_translated_prompt_segments(segments: list[str]) -> str:
    """Join fragments without fusing adjacent translated English words."""

    joined = ""
    for segment in segments:
        if joined and segment and joined[-1].isalnum() and segment[0].isalnum():
            joined += " "
        joined += segment
    return joined.strip()


def _single_input_token_count(input_ids: Any) -> int:
    """Return a tokenizer batch's sole sequence length, failing closed.

    Hugging Face returns a rank-two tensor when ``return_tensors="pt"`` is
    requested.  Lightweight integrations and test doubles may return Python
    lists instead, so accept both representations without importing torch.
    The translator always tokenizes one prompt and therefore rejects any
    multi-item or otherwise ambiguous batch.
    """

    shape = getattr(input_ids, "shape", None)
    if shape is not None:
        try:
            dimensions = tuple(int(dimension) for dimension in shape)
        except (TypeError, ValueError):
            dimensions = ()
        if len(dimensions) == 2 and dimensions[0] == 1 and dimensions[1] > 0:
            return dimensions[1]
        raise ValidationError(
            "M2M100 tokenizer returned invalid input_ids for one prompt"
        )

    if not isinstance(input_ids, (list, tuple)) or not input_ids:
        raise ValidationError(
            "M2M100 tokenizer returned invalid input_ids for one prompt"
        )
    first = input_ids[0]
    if isinstance(first, (list, tuple)):
        if len(input_ids) != 1 or not first:
            raise ValidationError(
                "M2M100 tokenizer returned invalid input_ids for one prompt"
            )
        return len(first)
    raise ValidationError("M2M100 tokenizer returned invalid input_ids for one prompt")


def _verify_translator_file(
    file_path: str | os.PathLike[str],
    *,
    required: VerifiedFile,
    should_cancel: Callable[[], bool] | None = None,
) -> None:
    """Stream-verify one reviewed runtime file once per unchanged local inode."""

    path = Path(file_path).resolve(strict=True)
    before = path.stat()
    cache_key = (
        str(path),
        required.sha256,
        required.size_bytes,
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    with _WEIGHTS_VERIFICATION_LOCK:
        if cache_key in _VERIFIED_TRANSLATOR_FILES:
            return
        if before.st_size != required.size_bytes or not path.is_file():
            raise IntegrityError(
                "Pinned M2M100 translator runtime file failed size/type "
                f"verification: {required.path}."
            )
        digest = hashlib.sha256()
        with path.open("rb") as weights_file:
            for chunk in iter(
                lambda: weights_file.read(_WEIGHTS_HASH_CHUNK_BYTES),
                b"",
            ):
                if should_cancel is not None and should_cancel():
                    raise TranslationSetupCancelled(
                        "Korean translator setup was stopped."
                    )
                digest.update(chunk)
        after = path.stat()
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
            or digest.hexdigest() != required.sha256
        ):
            raise IntegrityError(
                "Pinned M2M100 translator runtime file failed SHA-256 "
                f"verification: {required.path}. The translator was not loaded."
            )
        _VERIFIED_TRANSLATOR_FILES.add(cache_key)


def _verify_translator_weights(
    weights_path: str | os.PathLike[str],
    *,
    expected_sha256: str,
) -> None:
    """Backward-compatible internal wrapper for direct checkpoint verification."""

    path = Path(weights_path).resolve(strict=True)
    _verify_translator_file(
        path,
        required=VerifiedFile(
            path="pytorch_model.bin",
            size_bytes=path.stat().st_size,
            sha256=expected_sha256,
        ),
    )


def _failed_translation(
    original_prompt: str, metadata: TranslatorMetadata
) -> NormalizedPrompt:
    return NormalizedPrompt(
        original_prompt=original_prompt,
        normalized_prompt=None,
        detected_language="ko",
        status=PromptNormalizationStatus.TRANSLATION_FAILED,
        translator=metadata,
        enhancement_applied=False,
        warning=(
            "Korean translation failed, so the original prompt was preserved and "
            "was not sent to the image model. Check the configured translator and "
            "its pinned local model."
        ),
    )
