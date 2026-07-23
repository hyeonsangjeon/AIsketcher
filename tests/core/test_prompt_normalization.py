from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from aisketcher import (
    ModelUnavailableError,
    OptionalDependencyError,
    TranslationSetupCancelled,
    ValidationError,
)
from aisketcher.prompt_normalization import (
    MARIAN_KO_EN_DOWNLOAD_BYTES,
    MARIAN_KO_EN_MODEL_ID,
    MARIAN_KO_EN_REVISION,
    MARIAN_KO_EN_WEIGHTS_SHA256,
    MarianKoreanEnglishTranslator,
    PromptNormalizationStatus,
    TranslatorMetadata,
    contains_hangul,
    enhance_design_edit_prompt,
    normalize_prompt,
)


@dataclass
class FakeTranslator:
    translated: str = "A cool fantasy background with knight attire."
    calls: int = 0

    @property
    def metadata(self) -> TranslatorMetadata:
        return TranslatorMetadata(
            provider="test-double",
            model_id="example/ko-en",
            revision="test-revision",
            local_files_only=True,
        )

    def translate(self, text: str) -> str:
        self.calls += 1
        assert text
        return self.translated


def test_hangul_detection_covers_syllables_jamo_and_mixed_prompts() -> None:
    assert contains_hangul("멋진 판타지 배경")
    assert contains_hangul("knight 갑옷")
    assert contains_hangul("\u3131")
    assert contains_hangul("\u1100")
    assert not contains_hangul("A detailed knight in a fantasy landscape")


def test_english_prompt_is_trimmed_without_calling_translator() -> None:
    translator = FakeTranslator()

    result = normalize_prompt("  A paper-cut kingdom.  ", translator=translator)

    assert result.original_prompt == "  A paper-cut kingdom.  "
    assert result.normalized_prompt == "A paper-cut kingdom."
    assert result.detected_language == "und"
    assert result.status is PromptNormalizationStatus.UNCHANGED
    assert result.translator is None
    assert result.model_ready
    assert translator.calls == 0


def test_korean_prompt_without_translator_is_preserved_but_not_model_ready() -> None:
    result = normalize_prompt("  귀여운 종이 왕국  ")

    assert result.original_prompt == "  귀여운 종이 왕국  "
    assert result.normalized_prompt is None
    assert result.detected_language == "ko"
    assert result.status is PromptNormalizationStatus.TRANSLATOR_UNAVAILABLE
    assert not result.model_ready
    assert result.translator is None
    assert result.warning is not None
    assert "was not sent" in result.warning
    with pytest.raises(ValidationError, match="no translator"):
        result.require_model_prompt()


def test_injected_translator_returns_prompt_and_auditable_metadata() -> None:
    translator = FakeTranslator()

    result = normalize_prompt("멋진 판타지 배경", translator=translator)

    assert result.original_prompt == "멋진 판타지 배경"
    assert result.require_model_prompt() == translator.translated
    assert result.status is PromptNormalizationStatus.TRANSLATED
    assert result.translator == translator.metadata
    assert result.to_dict()["translator"] == {
        "provider": "test-double",
        "model_id": "example/ko-en",
        "revision": "test-revision",
        "local_files_only": True,
    }
    assert translator.calls == 1


def test_translation_failure_never_falls_back_to_raw_korean() -> None:
    translator = FakeTranslator(translated=" ")

    result = normalize_prompt("귀여운 캐릭터", translator=translator)

    assert result.original_prompt == "귀여운 캐릭터"
    assert result.normalized_prompt is None
    assert result.status is PromptNormalizationStatus.TRANSLATION_FAILED
    assert not result.model_ready
    assert result.warning is not None
    assert "was not sent" in result.warning


def test_design_edit_enhancement_preserves_brief_and_adds_only_constraints() -> None:
    brief = "Make the armor modern while keeping the background."

    enhanced = enhance_design_edit_prompt(brief)

    assert brief in enhanced
    assert "Preserve every unmentioned subject" in enhanced
    assert "Apply only changes explicitly requested" in enhanced
    assert "unrelated content" in enhanced


def test_translation_happens_before_opt_in_design_edit_enhancement() -> None:
    translator = FakeTranslator(translated="Make the armor more modern.")

    result = normalize_prompt(
        "갑옷을 더 현대적으로 바꿔주세요.",
        translator=translator,
        enhance_for_design_edit=True,
    )

    assert result.status is PromptNormalizationStatus.TRANSLATED
    assert result.enhancement_applied
    assert result.normalized_prompt is not None
    assert translator.translated in result.normalized_prompt
    assert "갑옷" not in result.normalized_prompt


def test_marian_adapter_construction_is_lazy_and_cache_only_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_import(name: str) -> Any:
        raise AssertionError(f"construction imported {name}")

    monkeypatch.setattr(
        "aisketcher.prompt_normalization.importlib.import_module",
        unexpected_import,
    )

    translator = MarianKoreanEnglishTranslator()

    assert translator.metadata == TranslatorMetadata(
        provider="huggingface-transformers",
        model_id=MARIAN_KO_EN_MODEL_ID,
        revision=MARIAN_KO_EN_REVISION,
        local_files_only=True,
    )
    assert MARIAN_KO_EN_DOWNLOAD_BYTES == 315_464_658
    assert len(MARIAN_KO_EN_WEIGHTS_SHA256) == 64


def test_marian_adapter_uses_pinned_revision_and_reuses_lazy_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DISABLE_SAFETENSORS_CONVERSION", raising=False)
    load_calls: list[tuple[str, str, dict[str, Any]]] = []

    class FakeTokenizer:
        def __call__(self, texts: list[str], **kwargs: Any) -> dict[str, list[int]]:
            assert texts == ["귀여운 캐릭터"]
            assert kwargs == {
                "return_tensors": "pt",
                "padding": True,
                "truncation": True,
            }
            return {"input_ids": [1, 2, 3]}

        def batch_decode(self, generated: Any, **kwargs: Any) -> list[str]:
            assert generated == [[4, 5, 6]]
            assert kwargs == {"skip_special_tokens": True}
            return ["  a cute character  "]

    class FakeModel:
        def generate(self, **kwargs: Any) -> list[list[int]]:
            assert kwargs == {"input_ids": [1, 2, 3], "max_new_tokens": 512}
            return [[4, 5, 6]]

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> FakeTokenizer:
            load_calls.append(("tokenizer", model_id, kwargs))
            return FakeTokenizer()

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> FakeModel:
            assert os.environ["DISABLE_SAFETENSORS_CONVERSION"] == "1"
            load_calls.append(("model", model_id, kwargs))
            return FakeModel()

    fake_transformers = SimpleNamespace(
        AutoTokenizer=FakeAutoTokenizer,
        AutoModelForSeq2SeqLM=FakeAutoModel,
    )
    monkeypatch.setattr(
        "aisketcher.prompt_normalization.importlib.import_module",
        lambda name: fake_transformers if name == "transformers" else None,
    )
    translator = MarianKoreanEnglishTranslator(cache_dir="/tmp/translation-cache")

    assert translator.translate("귀여운 캐릭터") == "a cute character"
    assert translator.translate("귀여운 캐릭터") == "a cute character"
    assert load_calls == [
        (
            "tokenizer",
            MARIAN_KO_EN_MODEL_ID,
            {
                "revision": MARIAN_KO_EN_REVISION,
                "local_files_only": True,
                "cache_dir": "/tmp/translation-cache",
                "trust_remote_code": False,
            },
        ),
        (
            "model",
            MARIAN_KO_EN_MODEL_ID,
            {
                "revision": MARIAN_KO_EN_REVISION,
                "local_files_only": True,
                "cache_dir": "/tmp/translation-cache",
                "trust_remote_code": False,
                "use_safetensors": False,
                "weights_only": True,
            },
        ),
    ]
    assert "DISABLE_SAFETENSORS_CONVERSION" not in os.environ


def test_marian_load_restores_an_existing_conversion_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DISABLE_SAFETENSORS_CONVERSION", "company-policy")

    class Factory:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> object:
            del model_id, kwargs
            if cls.__name__ == "ModelFactory":
                assert os.environ["DISABLE_SAFETENSORS_CONVERSION"] == "1"
            return object()

    fake_transformers = SimpleNamespace(
        AutoTokenizer=type("TokenizerFactory", (Factory,), {}),
        AutoModelForSeq2SeqLM=type("ModelFactory", (Factory,), {}),
    )
    monkeypatch.setattr(
        "aisketcher.prompt_normalization.importlib.import_module",
        lambda name: fake_transformers,
    )

    MarianKoreanEnglishTranslator().prepare(confirm=True)

    assert os.environ["DISABLE_SAFETENSORS_CONVERSION"] == "company-policy"


def test_marian_prepare_is_the_only_explicit_network_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_calls: list[tuple[str, dict[str, Any]]] = []

    class Factory:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> object:
            assert model_id == MARIAN_KO_EN_MODEL_ID
            load_calls.append((cls.__name__, kwargs))
            return object()

    fake_transformers = SimpleNamespace(
        AutoTokenizer=type("TokenizerFactory", (Factory,), {}),
        AutoModelForSeq2SeqLM=type("ModelFactory", (Factory,), {}),
    )
    monkeypatch.setattr(
        "aisketcher.prompt_normalization.importlib.import_module",
        lambda name: fake_transformers,
    )
    translator = MarianKoreanEnglishTranslator()

    with pytest.raises(ValidationError, match="confirm=True"):
        translator.prepare()
    assert load_calls == []

    assert translator.prepare(confirm=True) == translator.metadata
    assert translator.prepare(confirm=True) == translator.metadata
    assert len(load_calls) == 2
    assert load_calls[0][1] == {
        "revision": MARIAN_KO_EN_REVISION,
        "local_files_only": False,
        "trust_remote_code": False,
    }
    assert load_calls[1][1] == {
        "revision": MARIAN_KO_EN_REVISION,
        "local_files_only": False,
        "trust_remote_code": False,
        "use_safetensors": False,
        "weights_only": True,
    }


def test_marian_prepare_honors_cancellation_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []
    cancelled = threading.Event()
    cancelled.set()
    monkeypatch.setattr(
        "aisketcher.prompt_normalization.importlib.import_module",
        lambda name: imported.append(name),
    )

    with pytest.raises(TranslationSetupCancelled, match="stopped"):
        MarianKoreanEnglishTranslator().prepare(
            confirm=True,
            cancellation_token=cancelled,
        )

    assert imported == []


def test_marian_prepare_honors_cancellation_between_repository_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancelled = threading.Event()
    model_loads: list[str] = []

    class TokenizerFactory:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> object:
            del kwargs
            cancelled.set()
            return object()

    class ModelFactory:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> object:
            del kwargs
            model_loads.append(model_id)
            return object()

    fake_transformers = SimpleNamespace(
        AutoTokenizer=TokenizerFactory,
        AutoModelForSeq2SeqLM=ModelFactory,
    )
    monkeypatch.setattr(
        "aisketcher.prompt_normalization.importlib.import_module",
        lambda name: fake_transformers,
    )

    with pytest.raises(TranslationSetupCancelled, match="stopped"):
        MarianKoreanEnglishTranslator().prepare(
            confirm=True,
            cancel_event=cancelled,
        )

    assert model_loads == []


def test_marian_adapter_missing_dependency_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_transformers(name: str) -> Any:
        raise ImportError(name)

    monkeypatch.setattr(
        "aisketcher.prompt_normalization.importlib.import_module",
        missing_transformers,
    )

    with pytest.raises(OptionalDependencyError, match=r"aisketcher\[translate\]"):
        MarianKoreanEnglishTranslator().translate("한국어")


def test_marian_cache_miss_does_not_try_an_unpinned_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MissingFactory:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> None:
            assert kwargs["revision"] == MARIAN_KO_EN_REVISION
            assert kwargs["local_files_only"] is True
            raise OSError(model_id)

    fake_transformers = SimpleNamespace(
        AutoTokenizer=MissingFactory,
        AutoModelForSeq2SeqLM=MissingFactory,
    )
    monkeypatch.setattr(
        "aisketcher.prompt_normalization.importlib.import_module",
        lambda name: fake_transformers,
    )

    with pytest.raises(ModelUnavailableError, match="No fallback model"):
        MarianKoreanEnglishTranslator().translate("한국어")


@pytest.mark.parametrize("revision", ["main", "v1", "a" * 39, "g" * 40])
def test_marian_adapter_rejects_moving_or_invalid_revisions(revision: str) -> None:
    with pytest.raises(ValidationError, match="immutable"):
        MarianKoreanEnglishTranslator(revision=revision)
