from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from aisketcher import (
    IntegrityError,
    ModelUnavailableError,
    OptionalDependencyError,
    TranslationSetupCancelled,
    ValidationError,
)
from aisketcher import prompt_normalization as prompt_normalization_module
from aisketcher.model_registry import VerifiedFile
from aisketcher.prompt_normalization import (
    M2M100_KO_EN_DOWNLOAD_BYTES,
    M2M100_KO_EN_FILES,
    M2M100_KO_EN_MAX_INPUT_TOKENS,
    M2M100_KO_EN_MODEL_ID,
    M2M100_KO_EN_REVISION,
    M2M100_KO_EN_WEIGHTS_SHA256,
    M2M100KoreanEnglishTranslator,
    PromptNormalizationStatus,
    TranslatorMetadata,
    contains_hangul,
    enhance_design_edit_prompt,
    normalize_prompt,
)


def _materialize_verified_m2m_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    corrupt_path: str | None = None,
) -> dict[str, Path]:
    """Create a tiny exact-file snapshot while exercising every integrity check."""

    snapshot = tmp_path / "snapshot"
    paths: dict[str, Path] = {}
    verified: list[VerifiedFile] = []
    for official in M2M100_KO_EN_FILES:
        reviewed = f"reviewed fixture for {official.path}".encode()
        payload = reviewed
        if official.path == corrupt_path:
            payload = reviewed[:-1] + bytes([reviewed[-1] ^ 1])
        path = snapshot / official.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        paths[official.path] = path
        verified.append(
            VerifiedFile(
                path=official.path,
                size_bytes=len(reviewed),
                sha256=hashlib.sha256(reviewed).hexdigest(),
            )
        )
    monkeypatch.setattr(
        prompt_normalization_module,
        "M2M100_KO_EN_FILES",
        tuple(verified),
    )
    return paths


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


def test_m2m100_adapter_construction_is_lazy_and_cache_only_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_import(name: str) -> Any:
        raise AssertionError(f"construction imported {name}")

    monkeypatch.setattr(
        "aisketcher.prompt_normalization.importlib.import_module",
        unexpected_import,
    )

    translator = M2M100KoreanEnglishTranslator()

    assert translator.metadata == TranslatorMetadata(
        provider="huggingface-transformers",
        model_id=M2M100_KO_EN_MODEL_ID,
        revision=M2M100_KO_EN_REVISION,
        local_files_only=True,
    )
    assert M2M100_KO_EN_MODEL_ID == "facebook/m2m100_418M"
    assert M2M100_KO_EN_REVISION == "55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636"
    assert M2M100_KO_EN_DOWNLOAD_BYTES == 1_941_931_012
    assert len(M2M100_KO_EN_FILES) == 7
    assert sum(file.size_bytes for file in M2M100_KO_EN_FILES) == (
        M2M100_KO_EN_DOWNLOAD_BYTES
    )
    assert {file.path for file in M2M100_KO_EN_FILES} == {
        "config.json",
        "generation_config.json",
        "pytorch_model.bin",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "vocab.json",
    }
    assert M2M100_KO_EN_MAX_INPUT_TOKENS == 1_024
    assert (
        M2M100_KO_EN_WEIGHTS_SHA256
        == "d907ea45e4e4b9db163382a6674f6218b3c59566fe06d77f4055c208b4e87ed1"
    )


def test_m2m100_adapter_uses_pinned_revision_and_reuses_lazy_components(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("DISABLE_SAFETENSORS_CONVERSION", raising=False)
    load_calls: list[tuple[str, str, dict[str, Any]]] = []
    hub_calls: list[dict[str, Any]] = []
    runtime_paths = _materialize_verified_m2m_snapshot(monkeypatch, tmp_path)
    snapshot_dir = next(iter(runtime_paths.values())).parent

    class FakeTokenizer:
        def get_lang_id(self, language: str) -> int:
            assert language == "en"
            return 12_345

        def __call__(self, text: str, **kwargs: Any) -> dict[str, list[list[int]]]:
            assert text == "귀여운 캐릭터"
            assert kwargs == {
                "return_tensors": "pt",
                "truncation": False,
            }
            return {"input_ids": [[1, 2, 3]]}

        def batch_decode(self, generated: Any, **kwargs: Any) -> list[str]:
            assert generated == [[4, 5, 6]]
            assert kwargs == {"skip_special_tokens": True}
            return ["  a cute character  "]

    class FakeModel:
        def generate(self, **kwargs: Any) -> list[list[int]]:
            assert kwargs == {
                "input_ids": [[1, 2, 3]],
                "max_new_tokens": 512,
                "forced_bos_token_id": 12_345,
            }
            return [[4, 5, 6]]

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> FakeTokenizer:
            assert os.environ["DISABLE_SAFETENSORS_CONVERSION"] == "1"
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

    def fake_hf_hub_download(**kwargs: Any) -> str:
        hub_calls.append(kwargs)
        return str(runtime_paths[kwargs["filename"]])

    fake_hub = SimpleNamespace(hf_hub_download=fake_hf_hub_download)
    monkeypatch.setattr(
        "aisketcher.prompt_normalization.importlib.import_module",
        lambda name: fake_transformers if name == "transformers" else fake_hub,
    )
    translator = M2M100KoreanEnglishTranslator(cache_dir=str(tmp_path))

    assert translator.translate("귀여운 캐릭터") == "a cute character"
    assert translator.translate("귀여운 캐릭터") == "a cute character"
    assert load_calls == [
        (
            "tokenizer",
            str(snapshot_dir),
            {
                "local_files_only": True,
                "trust_remote_code": False,
                "src_lang": "ko",
            },
        ),
        (
            "model",
            str(snapshot_dir),
            {
                "local_files_only": True,
                "trust_remote_code": False,
                "use_safetensors": False,
                "weights_only": True,
            },
        ),
    ]
    assert hub_calls == [
        {
            "repo_id": M2M100_KO_EN_MODEL_ID,
            "filename": required.path,
            "revision": M2M100_KO_EN_REVISION,
            "local_files_only": True,
            "cache_dir": str(tmp_path),
        }
        for required in M2M100_KO_EN_FILES
    ]
    assert "DISABLE_SAFETENSORS_CONVERSION" not in os.environ


@pytest.mark.parametrize("symlink_location", ["ancestor", "translation-root"])
def test_m2m100_rejects_symlinked_managed_translation_cache_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    symlink_location: str,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    if symlink_location == "ancestor":
        symlinked_parent = tmp_path / "cache-parent"
        symlinked_parent.symlink_to(outside, target_is_directory=True)
        cache_dir = symlinked_parent / "translation"
        cache_dir.mkdir()
    else:
        outside_translation = outside / "translation"
        outside_translation.mkdir()
        cache_dir = tmp_path / "translation"
        cache_dir.symlink_to(outside_translation, target_is_directory=True)
    sentinel = outside / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    imported: list[str] = []
    monkeypatch.setattr(
        "aisketcher.prompt_normalization.importlib.import_module",
        lambda name: imported.append(name),
    )

    with pytest.raises(IntegrityError, match="translation cache boundary"):
        M2M100KoreanEnglishTranslator(cache_dir=str(cache_dir)).prepare(
            confirm=True
        )

    assert imported == []
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_m2m100_allows_hugging_face_snapshot_symlinks_to_internal_blobs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "translation"
    repository_cache = cache_dir / "models--facebook--m2m100_418M"
    snapshot_dir = (
        repository_cache / "snapshots" / M2M100_KO_EN_REVISION
    )
    blobs_dir = repository_cache / "blobs"
    snapshot_dir.mkdir(parents=True)
    blobs_dir.mkdir()
    runtime_paths: dict[str, Path] = {}
    verified: list[VerifiedFile] = []
    for index, official in enumerate(M2M100_KO_EN_FILES):
        payload = f"reviewed internal blob {index}: {official.path}".encode()
        blob_path = blobs_dir / hashlib.sha256(payload).hexdigest()
        blob_path.write_bytes(payload)
        snapshot_path = snapshot_dir / official.path
        snapshot_path.symlink_to(
            Path(os.path.relpath(blob_path, start=snapshot_path.parent))
        )
        runtime_paths[official.path] = snapshot_path
        verified.append(
            VerifiedFile(
                path=official.path,
                size_bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
    monkeypatch.setattr(
        prompt_normalization_module,
        "M2M100_KO_EN_FILES",
        tuple(verified),
    )
    component_loads: list[str] = []

    class Factory:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> object:
            del kwargs
            assert model_id == str(snapshot_dir)
            assert all(path.is_symlink() for path in runtime_paths.values())
            component_loads.append(cls.__name__)
            return object()

    fake_transformers = SimpleNamespace(
        AutoTokenizer=type("TokenizerFactory", (Factory,), {}),
        AutoModelForSeq2SeqLM=type("ModelFactory", (Factory,), {}),
    )

    def fake_hf_hub_download(**kwargs: Any) -> str:
        assert kwargs["cache_dir"] == str(cache_dir)
        return str(runtime_paths[kwargs["filename"]])

    fake_hub = SimpleNamespace(hf_hub_download=fake_hf_hub_download)
    monkeypatch.setattr(
        "aisketcher.prompt_normalization.importlib.import_module",
        lambda name: fake_transformers if name == "transformers" else fake_hub,
    )

    M2M100KoreanEnglishTranslator(cache_dir=str(cache_dir)).prepare(confirm=True)

    assert component_loads == ["TokenizerFactory", "ModelFactory"]


def test_m2m100_rejects_undeclared_snapshot_runtime_before_transformers_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_paths = _materialize_verified_m2m_snapshot(monkeypatch, tmp_path)
    snapshot_dir = next(iter(runtime_paths.values())).parent
    (snapshot_dir / "tokenizer.json").write_text(
        '{"unreviewed": true}',
        encoding="utf-8",
    )
    component_loads: list[str] = []

    class UnexpectedFactory:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> object:
            del model_id, kwargs
            component_loads.append(cls.__name__)
            return object()

    fake_transformers = SimpleNamespace(
        AutoTokenizer=type("TokenizerFactory", (UnexpectedFactory,), {}),
        AutoModelForSeq2SeqLM=type("ModelFactory", (UnexpectedFactory,), {}),
    )
    fake_hub = SimpleNamespace(
        hf_hub_download=lambda **kwargs: str(runtime_paths[kwargs["filename"]]),
    )
    monkeypatch.setattr(
        "aisketcher.prompt_normalization.importlib.import_module",
        lambda name: fake_transformers if name == "transformers" else fake_hub,
    )

    with pytest.raises(
        IntegrityError,
        match=r"undeclared runtime entry: tokenizer\.json",
    ):
        M2M100KoreanEnglishTranslator().translate("한국어")

    assert component_loads == []


def test_m2m100_rejects_symlinked_repository_directory_below_translation_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "translation"
    cache_dir.mkdir()
    outside_repository = tmp_path / "outside-repository"
    snapshot_dir = (
        outside_repository / "snapshots" / M2M100_KO_EN_REVISION
    )
    snapshot_dir.mkdir(parents=True)
    sentinel = outside_repository / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    repository_link = cache_dir / "models--facebook--m2m100_418M"
    repository_link.symlink_to(outside_repository, target_is_directory=True)
    first_required = M2M100_KO_EN_FILES[0]
    first_path = snapshot_dir / first_required.path
    first_path.write_bytes(b"untrusted external cache")
    model_loads: list[str] = []

    class UnexpectedFactory:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> object:
            del kwargs
            model_loads.append(model_id)
            return object()

    fake_transformers = SimpleNamespace(
        AutoTokenizer=UnexpectedFactory,
        AutoModelForSeq2SeqLM=UnexpectedFactory,
    )
    fake_hub = SimpleNamespace(
        hf_hub_download=lambda **kwargs: str(
            repository_link
            / "snapshots"
            / M2M100_KO_EN_REVISION
            / kwargs["filename"]
        ),
    )
    monkeypatch.setattr(
        "aisketcher.prompt_normalization.importlib.import_module",
        lambda name: fake_transformers if name == "transformers" else fake_hub,
    )

    with pytest.raises(IntegrityError, match="directory path"):
        M2M100KoreanEnglishTranslator(cache_dir=str(cache_dir)).translate("한국어")

    assert model_loads == []
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_m2m100_preserves_canonical_terms_without_mixing_english_into_korean_source() -> None:
    canonical_prompt = (
        "알파벳 A 모양의 스케치 구조를 유지한 정교한 미니어처 판타지 왕국, "
        "귀여운 작은 마스코트, 선명한 시안·코발트·주홍·노랑·금색, "
        "겹겹이 자른 종이 공예, 드라마틱한 스튜디오 조명"
    )
    tokenized_sources: list[str] = []
    generate_calls: list[dict[str, Any]] = []
    source_by_token_id: dict[int, str] = {}
    translated_segments = {
        "알파벳 A 모양의 스케치 구조를 유지한 정교한": "A refined",
        "왕국, 귀여운 작은": "kingdom with a cute little",
        "선명한": "vivid",
        "드라마틱한": "and dramatic",
    }

    class FakeTokenizer:
        def get_lang_id(self, language: str) -> int:
            assert language == "en"
            return 41

        def __call__(self, text: str, **kwargs: Any) -> dict[str, list[list[int]]]:
            assert kwargs == {
                "return_tensors": "pt",
                "truncation": False,
            }
            tokenized_sources.append(text)
            token_id = len(tokenized_sources)
            source_by_token_id[token_id] = text
            return {"input_ids": [[token_id]]}

        def batch_decode(self, generated: Any, **kwargs: Any) -> list[str]:
            assert kwargs == {"skip_special_tokens": True}
            token_id = generated[0][0]
            return [translated_segments[source_by_token_id[token_id]]]

    class FakeModel:
        def generate(self, **kwargs: Any) -> list[list[int]]:
            generate_calls.append(kwargs)
            return [[len(generate_calls) + 1]]

    translator = M2M100KoreanEnglishTranslator()
    translator._components = (FakeTokenizer(), FakeModel())

    result = normalize_prompt(canonical_prompt, translator=translator)

    assert result.original_prompt == canonical_prompt
    assert result.status is PromptNormalizationStatus.TRANSLATED
    assert result.require_model_prompt() == (
        "A refined miniature fantasy kingdom with a cute little mascot, vivid "
        "cyan·cobalt·vermilion·yellow·gold, layered cut-paper craft, and dramatic "
        "studio lighting"
    )
    assert tokenized_sources[0] == canonical_prompt
    assert set(tokenized_sources[1:]) == set(translated_segments)
    for tokenized_source in tokenized_sources:
        assert all(
            english not in tokenized_source
            for english in (
                "miniature",
                "fantasy",
                "mascot",
                "cyan",
                "cobalt",
                "vermilion",
                "yellow",
                "gold",
                "layered cut-paper craft",
                "studio lighting",
            )
        )
    assert len(generate_calls) == len(translated_segments)
    assert all(call["max_new_tokens"] == 512 for call in generate_calls)
    assert all(call["forced_bos_token_id"] == 41 for call in generate_calls)


def test_m2m100_preserves_glossary_boundaries_for_azure_acceptance_prompt() -> None:
    source = (
        "판타지 미니어처 나라, 코발트와 금색, 겹겹이 자른 종이 공예, "
        "귀여운 마스코트, 스튜디오 조명"
    )
    tokenized_sources: list[str] = []
    source_by_token_id: dict[int, str] = {}
    translated_segments = {
        "나라": "country",
        "귀여운": "the cute",
    }

    class FakeTokenizer:
        def get_lang_id(self, language: str) -> int:
            assert language == "en"
            return 41

        def __call__(self, text: str, **kwargs: Any) -> dict[str, list[list[int]]]:
            assert kwargs == {
                "return_tensors": "pt",
                "truncation": False,
            }
            tokenized_sources.append(text)
            token_id = len(tokenized_sources)
            source_by_token_id[token_id] = text
            return {"input_ids": [[token_id]]}

        def batch_decode(self, generated: Any, **kwargs: Any) -> list[str]:
            assert kwargs == {"skip_special_tokens": True}
            return [translated_segments[source_by_token_id[generated[0][0]]]]

    class FakeModel:
        def generate(self, **kwargs: Any) -> list[list[int]]:
            return [[kwargs["input_ids"][0][0]]]

    translator = M2M100KoreanEnglishTranslator()
    translator._components = (FakeTokenizer(), FakeModel())

    result = normalize_prompt(source, translator=translator)

    assert result.original_prompt == source
    assert result.status is PromptNormalizationStatus.TRANSLATED
    assert result.translator == translator.metadata
    assert result.require_model_prompt() == (
        "fantasy miniature country, cobalt and gold, layered cut-paper craft, "
        "the cute mascot, studio lighting"
    )
    assert tokenized_sources == [source, "나라", "귀여운"]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("주홍과 노랑", "vermilion and yellow"),
        ("코발트와 금색", "cobalt and gold"),
        ("코발트및금색", "cobalt and gold"),
    ],
)
def test_m2m100_joins_reviewed_terms_with_exact_connectors(
    source: str,
    expected: str,
) -> None:
    tokenized_sources: list[str] = []

    class FakeTokenizer:
        def get_lang_id(self, language: str) -> int:
            assert language == "en"
            return 41

        def __call__(self, text: str, **kwargs: Any) -> dict[str, list[list[int]]]:
            assert kwargs == {
                "return_tensors": "pt",
                "truncation": False,
            }
            tokenized_sources.append(text)
            return {"input_ids": [[1]]}

        def batch_decode(self, generated: Any, **kwargs: Any) -> list[str]:
            del generated, kwargs
            raise AssertionError("reviewed connectors must not be generated")

    class FakeModel:
        def generate(self, **kwargs: Any) -> list[list[int]]:
            del kwargs
            raise AssertionError("reviewed connectors must not be generated")

    translator = M2M100KoreanEnglishTranslator()
    translator._components = (FakeTokenizer(), FakeModel())

    assert translator.translate(source) == expected
    assert tokenized_sources == [source]


def test_m2m100_all_glossary_prompt_preserves_meaning_without_generation() -> None:
    source = "미니어처 판타지 마스코트 시안·코발트·주홍·노랑·금색"
    tokenized_sources: list[str] = []

    class FakeTokenizer:
        def get_lang_id(self, language: str) -> int:
            assert language == "en"
            return 41

        def __call__(self, text: str, **kwargs: Any) -> dict[str, list[list[int]]]:
            assert kwargs == {
                "return_tensors": "pt",
                "truncation": False,
            }
            tokenized_sources.append(text)
            return {"input_ids": [[1]]}

        def batch_decode(self, generated: Any, **kwargs: Any) -> list[str]:
            del generated, kwargs
            raise AssertionError("an all-glossary prompt must not be generated")

    class FakeModel:
        def generate(self, **kwargs: Any) -> list[list[int]]:
            del kwargs
            raise AssertionError("an all-glossary prompt must not be generated")

    translator = M2M100KoreanEnglishTranslator()
    translator._components = (FakeTokenizer(), FakeModel())

    assert translator.translate(source) == (
        "miniature fantasy mascot cyan·cobalt·vermilion·yellow·gold"
    )
    assert tokenized_sources == [source]


@pytest.mark.parametrize("representation", ["list", "tensor"])
@pytest.mark.parametrize(
    ("token_count", "translation_allowed"),
    [
        (M2M100_KO_EN_MAX_INPUT_TOKENS, True),
        (M2M100_KO_EN_MAX_INPUT_TOKENS + 1, False),
    ],
)
def test_m2m100_enforces_pinned_input_token_limit_without_truncation(
    representation: str,
    token_count: int,
    translation_allowed: bool,
) -> None:
    generate_calls: list[dict[str, Any]] = []

    class FakeTensor:
        def __init__(self, sequence_length: int) -> None:
            self.shape = (1, sequence_length)

    input_ids: Any = (
        [[7] * token_count]
        if representation == "list"
        else FakeTensor(token_count)
    )

    class FakeTokenizer:
        def get_lang_id(self, language: str) -> int:
            assert language == "en"
            return 41

        def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
            assert text == "한국어 프롬프트"
            assert kwargs == {
                "return_tensors": "pt",
                "truncation": False,
            }
            return {"input_ids": input_ids}

        def batch_decode(self, generated: Any, **kwargs: Any) -> list[str]:
            assert generated == [[2]]
            assert kwargs == {"skip_special_tokens": True}
            return ["translated prompt"]

    class FakeModel:
        def generate(self, **kwargs: Any) -> list[list[int]]:
            generate_calls.append(kwargs)
            return [[2]]

    translator = M2M100KoreanEnglishTranslator()
    translator._components = (FakeTokenizer(), FakeModel())

    if translation_allowed:
        assert translator.translate("한국어 프롬프트") == "translated prompt"
        assert len(generate_calls) == 1
        assert generate_calls[0]["input_ids"] is input_ids
    else:
        with pytest.raises(
            ValidationError,
            match=r"1,025 tokens; the maximum is 1,024",
        ):
            translator.translate("한국어 프롬프트")
        assert generate_calls == []


def test_m2m100_oversized_input_preserves_original_and_fails_normalization() -> None:
    original_prompt = "가" * 10_000
    generate_calls: list[dict[str, Any]] = []

    class FakeTokenizer:
        def __call__(self, text: str, **kwargs: Any) -> dict[str, list[list[int]]]:
            assert text == original_prompt
            assert kwargs == {
                "return_tensors": "pt",
                "truncation": False,
            }
            return {"input_ids": [[8] * (M2M100_KO_EN_MAX_INPUT_TOKENS + 1)]}

    class FakeModel:
        def generate(self, **kwargs: Any) -> None:
            generate_calls.append(kwargs)

    translator = M2M100KoreanEnglishTranslator()
    translator._components = (FakeTokenizer(), FakeModel())

    result = normalize_prompt(original_prompt, translator=translator)

    assert result.original_prompt == original_prompt
    assert result.normalized_prompt is None
    assert result.status is PromptNormalizationStatus.TRANSLATION_FAILED
    assert not result.model_ready
    assert generate_calls == []


def test_m2m100_load_restores_an_existing_conversion_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DISABLE_SAFETENSORS_CONVERSION", "company-policy")
    runtime_paths = _materialize_verified_m2m_snapshot(monkeypatch, tmp_path)

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
    fake_hub = SimpleNamespace(
        hf_hub_download=lambda **kwargs: str(runtime_paths[kwargs["filename"]]),
    )
    monkeypatch.setattr(
        "aisketcher.prompt_normalization.importlib.import_module",
        lambda name: fake_transformers if name == "transformers" else fake_hub,
    )

    M2M100KoreanEnglishTranslator().prepare(confirm=True)

    assert os.environ["DISABLE_SAFETENSORS_CONVERSION"] == "company-policy"


def test_m2m100_prepare_is_the_only_explicit_network_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    load_calls: list[tuple[str, dict[str, Any]]] = []
    hub_calls: list[dict[str, Any]] = []
    runtime_paths = _materialize_verified_m2m_snapshot(monkeypatch, tmp_path)
    snapshot_dir = next(iter(runtime_paths.values())).parent

    class Factory:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> object:
            assert model_id == str(snapshot_dir)
            load_calls.append((cls.__name__, kwargs))
            return object()

    fake_transformers = SimpleNamespace(
        AutoTokenizer=type("TokenizerFactory", (Factory,), {}),
        AutoModelForSeq2SeqLM=type("ModelFactory", (Factory,), {}),
    )

    def fake_hf_hub_download(**kwargs: Any) -> str:
        hub_calls.append(kwargs)
        return str(runtime_paths[kwargs["filename"]])

    fake_hub = SimpleNamespace(hf_hub_download=fake_hf_hub_download)
    monkeypatch.setattr(
        "aisketcher.prompt_normalization.importlib.import_module",
        lambda name: fake_transformers if name == "transformers" else fake_hub,
    )
    translator = M2M100KoreanEnglishTranslator()

    with pytest.raises(ValidationError, match="confirm=True"):
        translator.prepare()
    assert load_calls == []

    assert translator.prepare(confirm=True) == translator.metadata
    assert translator.prepare(confirm=True) == translator.metadata
    assert len(load_calls) == 2
    assert load_calls[0][1] == {
        "local_files_only": True,
        "trust_remote_code": False,
        "src_lang": "ko",
    }
    assert load_calls[1][1] == {
        "local_files_only": True,
        "trust_remote_code": False,
        "use_safetensors": False,
        "weights_only": True,
    }
    assert hub_calls == [
        {
            "repo_id": M2M100_KO_EN_MODEL_ID,
            "filename": required.path,
            "revision": M2M100_KO_EN_REVISION,
            "local_files_only": False,
        }
        for required in M2M100_KO_EN_FILES
    ]


def test_m2m100_prepare_honors_cancellation_before_loading(
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
        M2M100KoreanEnglishTranslator().prepare(
            confirm=True,
            cancellation_token=cancelled,
        )

    assert imported == []


def test_m2m100_prepare_honors_cancellation_between_repository_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cancelled = threading.Event()
    model_loads: list[str] = []
    runtime_paths = _materialize_verified_m2m_snapshot(monkeypatch, tmp_path)

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
    fake_hub = SimpleNamespace(
        hf_hub_download=lambda **kwargs: str(runtime_paths[kwargs["filename"]]),
    )
    monkeypatch.setattr(
        "aisketcher.prompt_normalization.importlib.import_module",
        lambda name: fake_transformers if name == "transformers" else fake_hub,
    )

    with pytest.raises(TranslationSetupCancelled, match="stopped"):
        M2M100KoreanEnglishTranslator().prepare(
            confirm=True,
            cancel_event=cancelled,
        )

    assert model_loads == []


def test_m2m100_adapter_missing_dependency_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_transformers(name: str) -> Any:
        raise ImportError(name)

    monkeypatch.setattr(
        "aisketcher.prompt_normalization.importlib.import_module",
        missing_transformers,
    )

    with pytest.raises(OptionalDependencyError, match=r"aisketcher\[translate\]"):
        M2M100KoreanEnglishTranslator().translate("한국어")


def test_m2m100_refuses_checkpoint_with_wrong_sha256_before_model_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_paths = _materialize_verified_m2m_snapshot(
        monkeypatch,
        tmp_path,
        corrupt_path="pytorch_model.bin",
    )
    model_loads: list[str] = []

    class UnexpectedFactory:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> object:
            del kwargs
            model_loads.append(model_id)
            return object()

    fake_transformers = SimpleNamespace(
        AutoTokenizer=UnexpectedFactory,
        AutoModelForSeq2SeqLM=UnexpectedFactory,
    )
    fake_hub = SimpleNamespace(
        hf_hub_download=lambda **kwargs: str(runtime_paths[kwargs["filename"]]),
    )
    monkeypatch.setattr(
        "aisketcher.prompt_normalization.importlib.import_module",
        lambda name: fake_transformers if name == "transformers" else fake_hub,
    )

    with pytest.raises(IntegrityError, match="SHA-256"):
        M2M100KoreanEnglishTranslator().translate("한국어")

    assert model_loads == []


@pytest.mark.parametrize(
    "tampered_path",
    ["config.json", "sentencepiece.bpe.model", "tokenizer_config.json"],
)
def test_m2m100_refuses_tampered_configuration_and_tokenizer_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tampered_path: str,
) -> None:
    runtime_paths = _materialize_verified_m2m_snapshot(
        monkeypatch,
        tmp_path,
        corrupt_path=tampered_path,
    )
    model_loads: list[str] = []

    class UnexpectedFactory:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> object:
            del kwargs
            model_loads.append(model_id)
            return object()

    fake_transformers = SimpleNamespace(
        AutoTokenizer=UnexpectedFactory,
        AutoModelForSeq2SeqLM=UnexpectedFactory,
    )
    fake_hub = SimpleNamespace(
        hf_hub_download=lambda **kwargs: str(runtime_paths[kwargs["filename"]]),
    )
    monkeypatch.setattr(
        "aisketcher.prompt_normalization.importlib.import_module",
        lambda name: fake_transformers if name == "transformers" else fake_hub,
    )

    with pytest.raises(IntegrityError, match=tampered_path):
        M2M100KoreanEnglishTranslator().translate("한국어")
    assert model_loads == []


def test_m2m100_weights_are_rehashed_only_once_per_unchanged_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    weights_path = tmp_path / "pytorch_model.bin"
    weights_path.write_bytes(b"stable-process-local-checkpoint")
    expected = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    real_sha256 = hashlib.sha256
    hash_starts = 0

    def counting_sha256() -> Any:
        nonlocal hash_starts
        hash_starts += 1
        return real_sha256()

    monkeypatch.setattr(
        prompt_normalization_module.hashlib,
        "sha256",
        counting_sha256,
    )

    prompt_normalization_module._verify_translator_weights(
        weights_path,
        expected_sha256=expected,
    )
    prompt_normalization_module._verify_translator_weights(
        weights_path,
        expected_sha256=expected,
    )

    assert hash_starts == 1


def test_m2m100_rejects_same_size_tamper_even_when_mtime_is_restored(
    tmp_path: Path,
) -> None:
    weights_path = tmp_path / "pytorch_model.bin"
    original = b"reviewed-checkpoint"
    weights_path.write_bytes(original)
    original_stat = weights_path.stat()
    expected = hashlib.sha256(original).hexdigest()

    prompt_normalization_module._verify_translator_weights(
        weights_path,
        expected_sha256=expected,
    )
    weights_path.write_bytes(b"tampered-checkpoint")
    os.utime(
        weights_path,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )

    with pytest.raises(IntegrityError, match="SHA-256"):
        prompt_normalization_module._verify_translator_weights(
            weights_path,
            expected_sha256=expected,
        )


def test_m2m100_cache_miss_does_not_try_an_unpinned_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MissingFactory:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> None:
            assert kwargs["revision"] == M2M100_KO_EN_REVISION
            assert kwargs["local_files_only"] is True
            raise OSError(model_id)

    fake_transformers = SimpleNamespace(
        AutoTokenizer=MissingFactory,
        AutoModelForSeq2SeqLM=MissingFactory,
    )

    hub_calls: list[dict[str, Any]] = []

    def missing_runtime_file(**kwargs: Any) -> None:
        hub_calls.append(kwargs)
        assert kwargs["repo_id"] == M2M100_KO_EN_MODEL_ID
        assert kwargs["filename"] == M2M100_KO_EN_FILES[0].path
        assert kwargs["revision"] == M2M100_KO_EN_REVISION
        assert kwargs["local_files_only"] is True
        raise OSError("not cached")

    fake_hub = SimpleNamespace(hf_hub_download=missing_runtime_file)
    monkeypatch.setattr(
        "aisketcher.prompt_normalization.importlib.import_module",
        lambda name: fake_transformers if name == "transformers" else fake_hub,
    )

    with pytest.raises(ModelUnavailableError, match="No fallback model"):
        M2M100KoreanEnglishTranslator().translate("한국어")
    assert len(hub_calls) == 1


@pytest.mark.parametrize("revision", ["main", "v1", "a" * 39, "g" * 40])
def test_m2m100_adapter_rejects_moving_or_invalid_revisions(revision: str) -> None:
    with pytest.raises(ValidationError, match="immutable"):
        M2M100KoreanEnglishTranslator(revision=revision)
