"""Explicit, dependency-lazy prompt normalization.

Korean text is not sent to an image model under an assumption that the model
understands it.  Applications may inject a translator, or explicitly construct
the local Marian adapter below.  Constructing either the normalizer or the
adapter never downloads model weights.
"""

from __future__ import annotations

import importlib
import os
import re
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from .errors import (
    AIsketcherError,
    ModelUnavailableError,
    OptionalDependencyError,
    ValidationError,
)

MARIAN_KO_EN_MODEL_ID = "Helsinki-NLP/opus-mt-ko-en"
MARIAN_KO_EN_REVISION = "e42d1f41b66194e6d10512f8a27bebc1f4f5097e"
MARIAN_KO_EN_DOWNLOAD_BYTES = 315_464_658
MARIAN_KO_EN_WEIGHTS_SHA256 = (
    "4b2209fbd0c58a05e9c4b818b49b6b7e54406cdee93acdccf840752a94fe34f4"
)

_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_HANGUL_PATTERN = re.compile(
    "[\u1100-\u11ff\u3130-\u318f\ua960-\ua97f\uac00-\ud7af\ud7b0-\ud7ff]"
)
_MAX_PROMPT_LENGTH = 10_000
_SAFETENSORS_CONVERSION_ENV = "DISABLE_SAFETENSORS_CONVERSION"
_TRANSFORMERS_LOAD_ENV_LOCK = threading.Lock()

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
    download after loading a repository that only publishes PyTorch weights at
    the requested commit. That helper resolves ``refs/pr/*`` independently of
    the caller's immutable revision, doubles this Marian checkpoint's transfer,
    and outlives ``from_pretrained``. The upstream opt-out environment variable
    is process-global, so guard the short load region and restore its previous
    value exactly.
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
    bundled Marian implementation must instantiate
    :class:`MarianKoreanEnglishTranslator` themselves, making the cache/network
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
                "explicitly load the pinned local Marian adapter."
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


class MarianKoreanEnglishTranslator:
    """Lazy local Marian Korean-to-English adapter.

    The default is cache-only.  Passing ``local_files_only=False`` is an
    explicit opt-in to Hugging Face Hub access.  The revision must always be an
    immutable 40-character commit SHA.
    """

    def __init__(
        self,
        *,
        model_id: str = MARIAN_KO_EN_MODEL_ID,
        revision: str = MARIAN_KO_EN_REVISION,
        local_files_only: bool = True,
        cache_dir: str | None = None,
        max_new_tokens: int = 512,
    ) -> None:
        model_id = model_id.strip()
        revision = revision.strip().lower()
        if not model_id or "/" not in model_id:
            raise ValidationError(
                "Marian model_id must be a non-empty namespaced repository id"
            )
        if not _COMMIT_PATTERN.fullmatch(revision):
            raise ValidationError(
                "Marian revision must be an immutable 40-character commit SHA"
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
        self.cache_dir = cache_dir
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
        encoded = tokenizer(
            [source],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        if not isinstance(encoded, Mapping):
            raise ValidationError("Marian tokenizer returned an invalid input batch")
        generated = model.generate(
            **dict(encoded),
            max_new_tokens=self.max_new_tokens,
        )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        if (
            not isinstance(decoded, list)
            or not decoded
            or not isinstance(decoded[0], str)
            or not decoded[0].strip()
        ):
            raise ValidationError("Marian translator returned an empty translation")
        return decoded[0].strip()

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
                "download size and Apache-2.0 license."
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
            try:
                transformers = importlib.import_module("transformers")
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
                tokenizer = transformers.AutoTokenizer.from_pretrained(
                    self.model_id, **load_options
                )
                self._check_setup_cancelled(should_cancel)
                with _disable_transformers_conversion_pr():
                    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
                        self.model_id,
                        **load_options,
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
