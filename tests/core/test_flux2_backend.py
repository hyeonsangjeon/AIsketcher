from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from PIL import Image

from aisketcher import FakeBackend, Intent, Studio
from aisketcher.errors import ModelUnavailableError, ValidationError
from aisketcher.flux2_backend import (
    DEFAULT_FLUX2_SMALL_DECODER_REPO_ID,
    Flux2GenerationCancelled,
    Flux2KleinBackend,
    Flux2KleinModelConfig,
    Flux2Progress,
    _Flux2Runtime,
)
from aisketcher.models import GenerationRequest, VariationStrength
from aisketcher.presets import resolve_recipe


class FakeGenerator:
    def __init__(self, device: str) -> None:
        self.device = device
        self.seed = -1

    def manual_seed(self, seed: int) -> FakeGenerator:
        self.seed = seed
        return self


class FakeCUDA:
    synchronize_calls = 0
    empty_cache_calls = 0

    @staticmethod
    def is_available() -> bool:
        return True

    @classmethod
    def synchronize(cls) -> None:
        cls.synchronize_calls += 1

    @classmethod
    def empty_cache(cls) -> None:
        cls.empty_cache_calls += 1


class FakeTorch:
    cuda = FakeCUDA()
    Generator = FakeGenerator
    float16 = "float16"
    float32 = "float32"


class FakePipeline:
    load_calls: ClassVar[list[tuple[str, dict[str, Any]]]] = []

    def __init__(self) -> None:
        self.call_kwargs: list[dict[str, Any]] = []
        self.offload_calls = 0
        self.to_calls: list[str] = []
        self.free_hook_calls = 0
        self._interrupt = False

    @classmethod
    def from_pretrained(cls, repo_id: str, **kwargs: Any) -> FakePipeline:
        cls.load_calls.append((repo_id, kwargs))
        return cls()

    def enable_model_cpu_offload(self) -> None:
        self.offload_calls += 1

    def maybe_free_model_hooks(self) -> None:
        self.free_hook_calls += 1

    def to(self, device: str) -> FakePipeline:
        self.to_calls.append(device)
        return self

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        self.call_kwargs.append(kwargs)
        callback = kwargs["callback_on_step_end"]
        callback(self, 0, 999, {})
        callback(self, 1, 500, {})
        callback(self, 2, 250, {})
        callback(self, 3, 0, {})
        size = (kwargs["width"], kwargs["height"])
        seed = kwargs["generator"].seed
        return SimpleNamespace(images=[Image.new("RGB", size, (seed % 255, 30, 60))])


class FakeAutoencoder:
    load_calls: ClassVar[list[tuple[str, dict[str, Any]]]] = []
    instances: ClassVar[list[FakeAutoencoder]] = []

    def __init__(self) -> None:
        self.to_calls: list[str] = []
        self.__class__.instances.append(self)

    @classmethod
    def from_pretrained(cls, repo_id: str, **kwargs: Any) -> FakeAutoencoder:
        cls.load_calls.append((repo_id, kwargs))
        return cls()

    def to(self, device: str) -> FakeAutoencoder:
        self.to_calls.append(device)
        return self


class FailingPipeline(FakePipeline):
    @classmethod
    def from_pretrained(cls, repo_id: str, **kwargs: Any) -> FakePipeline:
        cls.load_calls.append((repo_id, kwargs))
        raise RuntimeError("synthetic pipeline load failure")


def runtime() -> _Flux2Runtime:
    return _Flux2Runtime(
        torch=FakeTorch,
        pipeline_cls=FakePipeline,
        autoencoder_cls=FakeAutoencoder,
    )


def test_runtime_import_uses_the_flux2_specific_decoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_diffusers = SimpleNamespace(
        Flux2KleinPipeline=FakePipeline,
        AutoencoderKLFlux2=FakeAutoencoder,
    )

    monkeypatch.setattr(
        "aisketcher.flux2_backend.importlib.import_module",
        lambda name: FakeTorch if name == "torch" else fake_diffusers,
    )

    imported = Flux2KleinBackend()._import_runtime()

    assert imported.pipeline_cls is FakePipeline
    assert imported.autoencoder_cls is FakeAutoencoder


def request(
    *,
    seeds: tuple[int, ...] = (11, 22),
    init_image: Image.Image | None = None,
) -> GenerationRequest:
    studio = Studio(FakeBackend())
    prepared = studio.prepare(Image.new("RGB", (96, 64), "white"), max_side=96)
    recipe = resolve_recipe(
        "lite",
        Intent("Turn this sketch into a paper-cut fantasy castle"),
        None,
        backend_name="fake",
        capabilities=FakeBackend().capabilities,
    )
    recipe = replace(
        recipe,
        width=96,
        height=64,
        steps=4,
        guidance_scale=1.0,
        scheduler="flow-match-euler",
    )
    return GenerationRequest(
        prepared=prepared,
        recipe=recipe,
        seeds=seeds,
        init_image=init_image,
        denoise_strength=0.25 if init_image is not None else None,
    )


def test_injected_pipeline_generates_sequential_fp16_image_edits() -> None:
    FakeCUDA.synchronize_calls = 0
    FakeCUDA.empty_cache_calls = 0
    pipeline = FakePipeline()
    events: list[Flux2Progress] = []
    backend = Flux2KleinBackend(
        device="cuda",
        pipeline=pipeline,
        runtime=runtime(),
        on_progress=events.append,
    )

    results = backend.generate(request())

    assert [result.seed for result in results] == [11, 22]
    assert [call["generator"].seed for call in pipeline.call_kwargs] == [11, 22]
    assert all(call["generator"].device == "cuda" for call in pipeline.call_kwargs)
    assert all(call["num_inference_steps"] == 4 for call in pipeline.call_kwargs)
    assert all(call["guidance_scale"] == 1.0 for call in pipeline.call_kwargs)
    assert all(call["num_images_per_prompt"] == 1 for call in pipeline.call_kwargs)
    assert all(call["image"].mode == "RGB" for call in pipeline.call_kwargs)
    assert all(call["image"].size == (96, 64) for call in pipeline.call_kwargs)
    assert all(
        call["prompt"] == request().recipe.generation_prompt
        for call in pipeline.call_kwargs
    )
    assert all("strength" not in call for call in pipeline.call_kwargs)
    assert all(result.metadata["dtype"] == "float16" for result in results)
    assert all(result.metadata["reference_source"] == "prepared-source" for result in results)
    assert all(result.metadata["sequential"] is True for result in results)
    assert all(result.metadata["variation_strength_applied"] is None for result in results)
    assert all(result.metadata["variation_strength_method"] is None for result in results)
    assert [event.phase for event in events].count("complete") == 2
    assert FakeCUDA.empty_cache_calls == 2


def test_explicit_init_image_replaces_prepared_source_and_is_not_mutated() -> None:
    pipeline = FakePipeline()
    init_image = Image.new("RGBA", (32, 48), (1, 2, 3, 128))
    original_mode = init_image.mode
    original_size = init_image.size
    backend = Flux2KleinBackend(
        device="cuda",
        pipeline=pipeline,
        runtime=runtime(),
    )

    result = backend.generate(request(seeds=(7,), init_image=init_image))[0]

    reference = pipeline.call_kwargs[0]["image"]
    assert reference.mode == "RGB"
    assert reference.size == (96, 64)
    assert reference.getpixel((0, 0)) == (1, 2, 3)
    assert init_image.mode == original_mode
    assert init_image.size == original_size
    assert result.metadata["reference_source"] == "explicit-init"
    assert result.metadata["denoise_strength_requested"] == 0.25
    assert result.metadata["denoise_strength_applied"] is None
    assert result.metadata["denoise_strength_method"] == "mapped-to-edit-prompt-v1"
    assert result.metadata["variation_strength_requested"] == "subtle"
    assert result.metadata["variation_strength_applied"] == "subtle"
    assert result.metadata["variation_strength_method"] == "deterministic-edit-prompt-v1"
    assert result.metadata["structure_lock_applied_to_prompt"] is False
    assert pipeline.call_kwargs[0]["prompt"].startswith(
        f"{request(seeds=(7,), init_image=init_image).recipe.generation_prompt}\n\n"
    )
    assert "AIsketcher edit directive (subtle)" in pipeline.call_kwargs[0]["prompt"]
    assert "Structure lock is not active" in pipeline.call_kwargs[0]["prompt"]
    assert "strength" not in pipeline.call_kwargs[0]


@pytest.mark.parametrize(
    ("denoise_strength", "variation_strength", "instruction_fragment"),
    [
        (0.25, VariationStrength.SUBTLE, "Apply a subtle edit."),
        (0.45, VariationStrength.BALANCED, "Apply a balanced edit."),
        (0.65, VariationStrength.BOLD, "Apply a bold edit."),
    ],
)
def test_variation_strength_is_applied_as_a_deterministic_locked_edit_prompt(
    denoise_strength: float,
    variation_strength: VariationStrength,
    instruction_fragment: str,
) -> None:
    pipeline = FakePipeline()
    backend = Flux2KleinBackend(
        device="cuda",
        pipeline=pipeline,
        runtime=runtime(),
    )
    base_request = request(
        seeds=(7,),
        init_image=Image.new("RGB", (96, 64), "navy"),
    )
    generation_request = replace(
        base_request,
        recipe=replace(
            base_request.recipe,
            variation_strength=variation_strength,
            locks=("structure",),
        ),
        denoise_strength=denoise_strength,
    )

    result = backend.generate(generation_request)[0]

    prompt = pipeline.call_kwargs[0]["prompt"]
    assert prompt.startswith(f"{generation_request.recipe.generation_prompt}\n\n")
    assert (
        f"AIsketcher edit directive ({variation_strength.value})" in prompt
    )
    assert instruction_fragment in prompt
    assert "Structure lock is active" in prompt
    assert "preserve the reference composition" in prompt
    assert "strength" not in pipeline.call_kwargs[0]
    assert result.metadata["denoise_strength_requested"] == denoise_strength
    assert result.metadata["denoise_strength_applied"] is None
    assert result.metadata["denoise_strength_method"] == "mapped-to-edit-prompt-v1"
    assert result.metadata["variation_strength_requested"] == variation_strength.value
    assert result.metadata["variation_strength_applied"] == variation_strength.value
    assert result.metadata["variation_strength_method"] == "deterministic-edit-prompt-v1"
    assert result.metadata["structure_lock_applied_to_prompt"] is True


@pytest.mark.parametrize(
    ("denoise_strength", "expected"),
    [
        (0.0, VariationStrength.SUBTLE),
        (0.35, VariationStrength.SUBTLE),
        (0.350001, VariationStrength.BALANCED),
        (0.55, VariationStrength.BALANCED),
        (0.550001, VariationStrength.BOLD),
        (1.0, VariationStrength.BOLD),
    ],
)
def test_custom_denoise_values_use_stable_variation_boundaries(
    denoise_strength: float,
    expected: VariationStrength,
) -> None:
    assert (
        Flux2KleinBackend._variation_strength_from_denoise(denoise_strength)
        is expected
    )


@pytest.mark.parametrize("denoise_strength", [-0.01, 1.01, float("nan"), float("inf")])
def test_invalid_flux2_denoise_strength_is_rejected(denoise_strength: float) -> None:
    pipeline = FakePipeline()
    backend = Flux2KleinBackend(
        device="cuda",
        pipeline=pipeline,
        runtime=runtime(),
    )
    generation_request = replace(
        request(
            seeds=(7,),
            init_image=Image.new("RGB", (96, 64), "navy"),
        ),
        denoise_strength=denoise_strength,
    )

    with pytest.raises(ValidationError, match="denoise_strength"):
        backend.generate(generation_request)

    assert pipeline.call_kwargs == []


def test_mismatched_recipe_and_numeric_variation_strength_is_rejected() -> None:
    pipeline = FakePipeline()
    backend = Flux2KleinBackend(
        device="cuda",
        pipeline=pipeline,
        runtime=runtime(),
    )
    base_request = request(
        seeds=(7,),
        init_image=Image.new("RGB", (96, 64), "navy"),
    )
    generation_request = replace(
        base_request,
        recipe=replace(
            base_request.recipe,
            variation_strength=VariationStrength.BOLD,
        ),
        denoise_strength=0.25,
    )

    with pytest.raises(ValidationError, match="does not match"):
        backend.generate(generation_request)

    assert pipeline.call_kwargs == []


def test_generation_uses_model_prompt_without_discarding_original_prompt() -> None:
    pipeline = FakePipeline()
    backend = Flux2KleinBackend(
        device="cuda",
        pipeline=pipeline,
        runtime=runtime(),
    )
    original = request(seeds=(7,))
    translated = replace(
        original,
        recipe=replace(
            original.recipe,
            prompt="귀여운 종이 왕국",
            model_prompt="A cute paper kingdom.",
            prompt_metadata={"status": "translated", "detected_language": "ko"},
        ),
    )

    backend.generate(translated)

    assert translated.recipe.prompt == "귀여운 종이 왕국"
    assert pipeline.call_kwargs[0]["prompt"] == "A cute paper kingdom."


def test_real_loader_requires_registry_pinned_revision() -> None:
    backend = Flux2KleinBackend(device="cuda", runtime=runtime())

    with pytest.raises(ModelUnavailableError, match="immutable revision"):
        backend.generate(request(seeds=(7,)))

    assert FakePipeline.load_calls == []


def test_loader_is_fp16_safe_and_does_not_enable_remote_code() -> None:
    FakePipeline.load_calls.clear()
    FakeAutoencoder.load_calls.clear()
    backend = Flux2KleinBackend(
        model=Flux2KleinModelConfig(revision="a" * 40),
        device="cuda",
        local_files_only=False,
        runtime=runtime(),
    )

    result = backend.generate(request(seeds=(7,)))[0]

    assert result.seed == 7
    assert len(FakePipeline.load_calls) == 1
    repo_id, kwargs = FakePipeline.load_calls[0]
    assert repo_id == "black-forest-labs/FLUX.2-klein-4B"
    assert kwargs["revision"] == "a" * 40
    assert kwargs["torch_dtype"] == "float16"
    assert kwargs["use_safetensors"] is True
    assert kwargs["trust_remote_code"] is False
    assert kwargs["local_files_only"] is False
    assert "vae" not in kwargs
    assert FakeAutoencoder.load_calls == []
    assert backend._pipeline.offload_calls == 1
    assert result.metadata["cpu_offload"] is True
    assert result.metadata["decoder_source"] == "pipeline-default"


def test_pinned_small_decoder_is_loaded_lazily_and_injected_into_pipeline() -> None:
    FakePipeline.load_calls.clear()
    FakeAutoencoder.load_calls.clear()
    FakeAutoencoder.instances.clear()
    backend = Flux2KleinBackend(
        model=Flux2KleinModelConfig(
            revision="a" * 40,
            decoder_model_id=DEFAULT_FLUX2_SMALL_DECODER_REPO_ID,
            decoder_revision="c" * 40,
        ),
        device="cuda",
        local_files_only=False,
        runtime=runtime(),
    )

    result = backend.generate(request(seeds=(7,)))[0]

    assert len(FakeAutoencoder.load_calls) == 1
    decoder_id, decoder_kwargs = FakeAutoencoder.load_calls[0]
    assert decoder_id == DEFAULT_FLUX2_SMALL_DECODER_REPO_ID
    assert decoder_kwargs["revision"] == "c" * 40
    assert decoder_kwargs["torch_dtype"] == "float16"
    assert decoder_kwargs["use_safetensors"] is True
    assert decoder_kwargs["trust_remote_code"] is False
    assert decoder_kwargs["local_files_only"] is False
    assert FakePipeline.load_calls[0][1]["vae"] is FakeAutoencoder.instances[0]
    assert result.metadata["decoder_model_id"] == DEFAULT_FLUX2_SMALL_DECODER_REPO_ID
    assert result.metadata["decoder_revision"] == "c" * 40
    assert result.metadata["decoder_source"] == "pinned"

    decoder = FakeAutoencoder.instances[0]
    backend.close()
    assert decoder.to_calls[-1] == "cpu"


def test_managed_model_directories_are_loaded_without_leaking_paths_to_metadata(
    tmp_path: Path,
) -> None:
    FakePipeline.load_calls.clear()
    FakeAutoencoder.load_calls.clear()
    backend = Flux2KleinBackend(
        model=Flux2KleinModelConfig(
            revision="a" * 40,
            base_path=tmp_path / "base",
            decoder_model_id=DEFAULT_FLUX2_SMALL_DECODER_REPO_ID,
            decoder_revision="c" * 40,
            decoder_path=tmp_path / "decoder",
        ),
        device="cuda",
        local_files_only=True,
        runtime=runtime(),
    )

    result = backend.generate(request(seeds=(7,)))[0]

    base_location, base_kwargs = FakePipeline.load_calls[0]
    decoder_location, decoder_kwargs = FakeAutoencoder.load_calls[0]
    assert base_location == str(tmp_path / "base")
    assert decoder_location == str(tmp_path / "decoder")
    assert "revision" not in base_kwargs
    assert "revision" not in decoder_kwargs
    assert result.metadata["model_repo_id"] == "black-forest-labs/FLUX.2-klein-4B"
    assert result.metadata["model_revision"] == "a" * 40
    assert result.metadata["base_source"] == "managed-local-dir"
    assert result.metadata["decoder_source"] == "managed-local-dir"
    assert str(tmp_path) not in str(result.metadata)


def test_injected_small_decoder_bypasses_decoder_loader() -> None:
    FakePipeline.load_calls.clear()
    FakeAutoencoder.load_calls.clear()
    decoder = FakeAutoencoder()
    backend = Flux2KleinBackend(
        model=Flux2KleinModelConfig(revision="a" * 40),
        device="cuda",
        decoder=decoder,
        runtime=runtime(),
    )

    result = backend.generate(request(seeds=(7,)))[0]

    assert FakeAutoencoder.load_calls == []
    assert FakePipeline.load_calls[0][1]["vae"] is decoder
    assert result.metadata["decoder_model_id"] is None
    assert result.metadata["decoder_revision"] is None
    assert result.metadata["decoder_source"] == "injected"

    backend.close()
    assert decoder.to_calls == []


def test_loaded_small_decoder_is_released_when_pipeline_load_fails() -> None:
    FakeAutoencoder.load_calls.clear()
    FakeAutoencoder.instances.clear()
    failing_runtime = _Flux2Runtime(
        torch=FakeTorch,
        pipeline_cls=FailingPipeline,
        autoencoder_cls=FakeAutoencoder,
    )
    backend = Flux2KleinBackend(
        model=Flux2KleinModelConfig(
            revision="a" * 40,
            decoder_model_id=DEFAULT_FLUX2_SMALL_DECODER_REPO_ID,
            decoder_revision="c" * 40,
        ),
        device="cuda",
        runtime=failing_runtime,
    )

    with pytest.raises(ModelUnavailableError, match="Could not load pinned FLUX.2 Klein"):
        backend.generate(request(seeds=(7,)))

    assert backend._decoder is None
    assert FakeAutoencoder.instances[0].to_calls == ["cpu"]


@pytest.mark.parametrize(
    "config",
    [
        {"decoder_model_id": DEFAULT_FLUX2_SMALL_DECODER_REPO_ID},
        {"decoder_revision": "c" * 40},
        {
            "decoder_model_id": DEFAULT_FLUX2_SMALL_DECODER_REPO_ID,
            "decoder_revision": "main",
        },
    ],
)
def test_small_decoder_requires_complete_immutable_configuration(
    config: dict[str, str],
) -> None:
    with pytest.raises(ValidationError, match="decoder"):
        Flux2KleinModelConfig(**config)


def test_cancellation_stops_inside_the_denoising_callback() -> None:
    pipeline = FakePipeline()
    polls = 0
    events: list[Flux2Progress] = []

    def should_cancel() -> bool:
        nonlocal polls
        polls += 1
        return polls >= 3

    backend = Flux2KleinBackend(
        device="cuda",
        pipeline=pipeline,
        runtime=runtime(),
        should_cancel=should_cancel,
        on_progress=events.append,
    )

    with pytest.raises(Flux2GenerationCancelled, match="cancelled"):
        backend.generate(request(seeds=(7,)))

    assert len(pipeline.call_kwargs) == 1
    assert pipeline._interrupt is False
    assert events[-1].phase == "cancelled"


def test_request_cancel_integrates_with_provider_cancellation_contract() -> None:
    pipeline = FakePipeline()
    backend = Flux2KleinBackend(
        device="cuda",
        pipeline=pipeline,
        runtime=runtime(),
    )
    backend.request_cancel()

    with pytest.raises(Flux2GenerationCancelled, match="cancelled"):
        backend.generate(request(seeds=(7,)))

    assert pipeline.call_kwargs == []
    assert pipeline._interrupt is False

    # The signal is consumed by the cancelled call, so the pooled backend can
    # serve a later request without reconstruction.
    assert backend.generate(request(seeds=(8,)))[0].seed == 8


def test_close_releases_pipeline_and_cuda_cache() -> None:
    backend = Flux2KleinBackend(
        model=Flux2KleinModelConfig(revision="b" * 40),
        device="cuda",
        runtime=runtime(),
    )
    backend.generate(request(seeds=(7,)))
    pipeline = backend._pipeline
    previous_empty_cache_calls = FakeCUDA.empty_cache_calls

    backend.close()

    assert backend._pipeline is None
    assert pipeline is not None
    assert pipeline.free_hook_calls == 1
    assert pipeline.to_calls[-1] == "cpu"
    assert FakeCUDA.empty_cache_calls == previous_empty_cache_calls + 1
