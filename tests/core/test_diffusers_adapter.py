from __future__ import annotations

from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest
from PIL import Image, ImageDraw

from aisketcher import DiffusersBackend, GenerationError, Intent, SeedPlan, Studio
from aisketcher.backends import DiffusersGenerationCancelled, _DiffusersRuntime


class FakeGenerator:
    def __init__(self, device: str) -> None:
        self.device = device
        self.seed = -1

    def manual_seed(self, seed: int) -> FakeGenerator:
        self.seed = seed
        return self


class FakeMPS:
    @staticmethod
    def is_available() -> bool:
        return True


class FakeMPSRuntime:
    synchronize_calls = 0
    empty_cache_calls = 0

    @classmethod
    def synchronize(cls) -> None:
        cls.synchronize_calls += 1

    @classmethod
    def empty_cache(cls) -> None:
        cls.empty_cache_calls += 1


class FakeCUDA:
    @staticmethod
    def is_available() -> bool:
        return False


class AvailableCUDA:
    @staticmethod
    def is_available() -> bool:
        return True


class FakeTorch:
    cuda = FakeCUDA()
    backends = SimpleNamespace(mps=FakeMPS())
    Generator = FakeGenerator
    mps = FakeMPSRuntime
    float16 = "float16"
    float32 = "float32"

    @staticmethod
    def isfinite(tensor: FakeTensor) -> FakeFinite:
        return FakeFinite(bool(np.isfinite(tensor.values).all()))

    @staticmethod
    def no_grad() -> FakeNoGrad:
        return FakeNoGrad()


class FakeCUDATorch(FakeTorch):
    cuda = AvailableCUDA()


class FakeFinite:
    def __init__(self, value: bool) -> None:
        self.value = value

    def all(self) -> FakeFinite:
        return self

    def item(self) -> bool:
        return self.value


class FakeNoGrad:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_args: object) -> None:
        return None


class FakeTensor:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values
        self.device: str | None = None
        self.dtype: str | None = None

    def to(self, *, device: str, dtype: str) -> FakeTensor:
        self.device = device
        self.dtype = dtype
        return self

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    def __truediv__(self, value: float) -> FakeTensor:
        return FakeTensor(self.values / value)

    def __mul__(self, value: float) -> FakeTensor:
        return FakeTensor(self.values * value)

    def __sub__(self, other: FakeTensor) -> FakeTensor:
        return FakeTensor(self.values - other.values)


class FakeScheduler:
    def __init__(self) -> None:
        self.config = {"name": "original"}


class FakeSchedulerClass:
    @staticmethod
    def from_config(config: dict[str, str]) -> FakeScheduler:
        assert config["name"] in ("original", "configured")
        scheduler = FakeScheduler()
        scheduler.config = {"name": "configured"}
        return scheduler


class FakePipeline:
    load_calls: ClassVar[list[tuple[str, str | None, dict[str, Any]]]] = []

    def __init__(self) -> None:
        self.scheduler = FakeScheduler()
        self.call_kwargs: list[dict[str, Any]] = []
        self.device: str | None = None
        self.attention_slicing = False

    @classmethod
    def from_pretrained(
        cls, location: str, revision: str | None = None, **kwargs: Any
    ) -> FakePipeline:
        cls.load_calls.append((location, revision, kwargs))
        return cls()

    def enable_attention_slicing(self) -> None:
        self.attention_slicing = True

    def to(self, device: str) -> FakePipeline:
        self.device = device
        return self

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        self.call_kwargs.append(kwargs)
        seed = kwargs["generator"].seed
        size = (kwargs["width"], kwargs["height"])
        return SimpleNamespace(images=[Image.new("RGB", size, (seed % 255, 20, 30))])


class InterruptiblePipeline(FakePipeline):
    def __init__(self) -> None:
        super().__init__()
        self.before_step: Any = None
        self._interrupt = False

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        self.call_kwargs.append(kwargs)
        if callable(self.before_step):
            self.before_step()
        kwargs["callback_on_step_end"](self, 0, 999, {})
        size = (kwargs["width"], kwargs["height"])
        return SimpleNamespace(images=[Image.new("RGB", size, (40, 50, 60))])


class FakeVAE:
    def __init__(self, *, finite: bool = True) -> None:
        self.device = "mps"
        self.dtype = FakeTorch.float16
        self.to_calls: list[dict[str, str]] = []
        self.encode_inputs: list[FakeTensor] = []
        self.sample_generators: list[FakeGenerator] = []
        self.finite = finite
        self.config = SimpleNamespace(
            latents_mean=None,
            latents_std=None,
            scaling_factor=0.13025,
            force_upcast=True,
        )

    def to(self, *, device: str, dtype: str) -> FakeVAE:
        self.device = device
        self.dtype = dtype
        self.to_calls.append({"device": device, "dtype": dtype})
        return self

    def decode(self, _latents: FakeTensor, *, return_dict: bool) -> tuple[FakeTensor]:
        assert return_dict is False
        fill = 1.0 if self.finite else np.nan
        return (FakeTensor(np.full((1, 3, 64, 96), fill, dtype=np.float32)),)

    def encode(self, image: FakeTensor) -> SimpleNamespace:
        self.encode_inputs.append(image)
        height, width = image.shape[-2:]
        return SimpleNamespace(
            latent_dist=FakeLatentDistribution(self, (1, 4, height // 8, width // 8))
        )


class FakeLatentDistribution:
    def __init__(self, vae: FakeVAE, shape: tuple[int, ...]) -> None:
        self.vae = vae
        self.shape = shape

    def sample(self, *, generator: FakeGenerator) -> FakeTensor:
        self.vae.sample_generators.append(generator)
        return FakeTensor(np.full(self.shape, generator.seed % 17, dtype=np.float32))


class FakeImageProcessor:
    def __init__(self) -> None:
        self.preprocess_inputs: list[Image.Image] = []

    def preprocess(self, image: Image.Image, *, height: int, width: int) -> FakeTensor:
        self.preprocess_inputs.append(image)
        return FakeTensor(np.ones((1, 3, height, width), dtype=np.float32))

    def postprocess(self, _decoded: FakeTensor, *, output_type: str) -> list[Image.Image]:
        assert output_type == "pil"
        return [Image.new("RGB", (96, 64), (80, 100, 120))]


class StableMPSPipeline(FakePipeline):
    def __init__(self, *, finite_decode: bool = True, finite_latents: bool = True) -> None:
        super().__init__()
        self.vae = FakeVAE(finite=finite_decode)
        self.image_processor = FakeImageProcessor()
        self.watermark = None
        self.finite_latents = finite_latents

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        self.call_kwargs.append(kwargs)
        assert kwargs["output_type"] == "latent"
        fill = 1.0 if self.finite_latents else np.nan
        latents = FakeTensor(np.full((1, 4, 8, 12), fill, dtype=np.float32))
        callback = kwargs["callback_on_step_end"]
        callback_kwargs = callback(self, 0, None, {"latents": latents})
        return SimpleNamespace(images=callback_kwargs["latents"])


class ReloadingBackend(DiffusersBackend):
    def __init__(self, first: StableMPSPipeline, replacements: list[StableMPSPipeline]) -> None:
        super().__init__(device="mps", pipeline=first)
        # This subclass models pipelines owned by the backend rather than a
        # caller-injected test double, so production retry behavior is enabled.
        self._pipeline_was_injected = False
        self.replacements = replacements
        self.reload_count = 0

    def _load_pipelines(self, _request: Any) -> None:
        if self._pipeline is not None:
            return
        self._pipeline = self.replacements[self.reload_count]
        self.reload_count += 1


class BlackPipeline(FakePipeline):
    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        self.call_kwargs.append(kwargs)
        size = (kwargs["width"], kwargs["height"])
        return SimpleNamespace(images=[Image.new("RGB", size, "black")])


class FakeControlNet:
    load_calls: ClassVar[list[tuple[str, str | None, dict[str, Any]]]] = []

    @classmethod
    def from_pretrained(cls, location: str, revision: str | None = None, **kwargs: Any) -> object:
        cls.load_calls.append((location, revision, kwargs))
        return object()


class FakeAutoImage:
    sources: ClassVar[list[FakePipeline]] = []

    @classmethod
    def from_pipe(cls, source: FakePipeline) -> FakePipeline:
        cls.sources.append(source)
        return FakePipeline()


class FakeAutoText(FakeAutoImage):
    pass


def runtime(torch: Any = FakeTorch) -> _DiffusersRuntime:
    return _DiffusersRuntime(
        torch=torch,
        controlnet_cls=FakeControlNet,
        txt2img_cls=FakePipeline,
        img2img_cls=FakePipeline,
        auto_txt2img_cls=FakeAutoText,
        auto_img2img_cls=FakeAutoImage,
        scheduler_cls=FakeSchedulerClass,
    )


def sketch() -> Image.Image:
    image = Image.new("RGB", (96, 64), "white")
    ImageDraw.Draw(image).line((5, 55, 48, 5, 90, 55), fill="black", width=3)
    return image


def test_mps_generation_is_sequential_and_variation_shares_components() -> None:
    text_pipeline = FakePipeline()
    backend = DiffusersBackend(device="mps", pipeline=text_pipeline)
    backend._runtime = runtime()
    studio = Studio(backend)
    study = studio.explore(
        studio.prepare(sketch(), max_side=96),
        intent=Intent("paper castle"),
        outputs=2,
    )
    assert len(text_pipeline.call_kwargs) == 2
    assert [call["generator"].device for call in text_pipeline.call_kwargs] == [
        "cpu",
        "cpu",
    ]
    assert len({call["generator"].seed for call in text_pipeline.call_kwargs}) == 2

    variants = studio.vary(study[0], outputs=2)
    assert len(variants) == 2
    assert FakeAutoImage.sources[-1] is text_pipeline
    variation_pipeline = backend._variation_pipeline
    assert variation_pipeline.device == "mps"
    assert variation_pipeline.attention_slicing
    assert all("control_image" in call for call in variation_pipeline.call_kwargs)
    assert all(call["strength"] == 0.25 for call in variation_pipeline.call_kwargs)


def test_loader_fetches_only_fp16_safetensors_for_requested_pipeline() -> None:
    FakePipeline.load_calls.clear()
    FakeControlNet.load_calls.clear()
    backend = DiffusersBackend(device="mps", local_files_only=False)
    backend._runtime = runtime()
    studio = Studio(backend)
    studio.explore(
        studio.prepare(sketch(), max_side=96),
        intent=Intent("paper castle"),
        outputs=1,
    )
    assert len(FakeControlNet.load_calls) == 1
    assert len(FakePipeline.load_calls) == 1
    control_kwargs = FakeControlNet.load_calls[0][2]
    pipeline_kwargs = FakePipeline.load_calls[0][2]
    for kwargs in (control_kwargs, pipeline_kwargs):
        assert kwargs["variant"] == "fp16"
        assert kwargs["use_safetensors"] is True
        assert kwargs["trust_remote_code"] is False
    assert backend._variation_pipeline is None


def test_mps_uses_explicit_fp32_vae_decode_and_clears_cache() -> None:
    FakeMPSRuntime.synchronize_calls = 0
    FakeMPSRuntime.empty_cache_calls = 0
    pipeline = StableMPSPipeline()
    backend = DiffusersBackend(device="mps", pipeline=pipeline)
    backend._runtime = runtime()
    study = Studio(backend).explore(
        Studio(backend).prepare(sketch(), max_side=96),
        intent=Intent("paper castle"),
        outputs=1,
    )
    assert len(study) == 1
    assert pipeline.call_kwargs[0]["output_type"] == "latent"
    assert pipeline.call_kwargs[0]["callback_on_step_end_tensor_inputs"] == ["latents"]
    assert pipeline.vae.dtype == FakeTorch.float32
    assert pipeline.vae.device == "cpu"
    assert all(call["dtype"] == FakeTorch.float32 for call in pipeline.vae.to_calls)
    assert [call["device"] for call in pipeline.vae.to_calls] == [
        "mps",
        "cpu",
        "mps",
        "cpu",
    ]
    assert FakeMPSRuntime.synchronize_calls >= 2
    assert FakeMPSRuntime.empty_cache_calls >= 2
    assert study[0].backend_metadata["vae_dtype"] == "float32"
    assert study[0].backend_metadata["mps_retries"] == 0
    assert study[0].backend_metadata["mps_isolated"] is False


def test_mps_variation_preencodes_image_on_cpu_with_the_generation_rng() -> None:
    text_pipeline = FakePipeline()
    variation_pipeline = StableMPSPipeline()
    backend = DiffusersBackend(
        device="mps",
        pipeline=text_pipeline,
        variation_pipeline=variation_pipeline,
    )
    backend._runtime = runtime()
    studio = Studio(backend)
    source = studio.explore(
        studio.prepare(sketch(), max_side=96),
        intent=Intent("paper castle"),
        outputs=1,
        seed_plan=SeedPlan.explicit((11,)),
    )

    variants = studio.vary(
        source[0],
        outputs=1,
        strength="subtle",
        seed_plan=SeedPlan.explicit((518939554,)),
    )

    call = variation_pipeline.call_kwargs[0]
    latent_image = call["image"]
    assert isinstance(latent_image, FakeTensor)
    assert latent_image.shape == (1, 4, 8, 12)
    assert latent_image.device == "cpu"
    assert latent_image.dtype == FakeTorch.float32
    assert variation_pipeline.vae.encode_inputs[0].device == "cpu"
    assert variation_pipeline.vae.encode_inputs[0].dtype == FakeTorch.float32
    assert variation_pipeline.vae.sample_generators == [call["generator"]]
    assert call["generator"].seed == 518939554
    assert variants[0].backend_metadata["vae_dtype"] == "float32"


def test_owned_mps_pipeline_is_reloaded_between_seeds_and_calls() -> None:
    first = StableMPSPipeline()
    second = StableMPSPipeline()
    third = StableMPSPipeline()
    next_call = StableMPSPipeline()
    backend = ReloadingBackend(first, [second, third, next_call])
    backend._runtime = runtime()
    studio = Studio(backend)
    seeds = (101, 202, 303)

    study = studio.explore(
        studio.prepare(sketch(), max_side=96),
        intent=Intent("paper castle"),
        outputs=3,
        seed_plan=SeedPlan.explicit(seeds),
    )

    assert backend.reload_count == 2
    assert [pipeline.call_kwargs[0]["generator"].seed for pipeline in (first, second, third)] == [
        101,
        202,
        303,
    ]
    assert all(len(pipeline.call_kwargs) == 1 for pipeline in (first, second, third))
    assert all(candidate.backend_metadata["mps_isolated"] is True for candidate in study)
    assert all(candidate.backend_metadata["mps_retries"] == 0 for candidate in study)

    later = studio.explore(
        studio.prepare(sketch(), max_side=96),
        intent=Intent("paper castle"),
        outputs=1,
        seed_plan=SeedPlan.explicit((404,)),
    )
    assert backend.reload_count == 3
    assert third.device == "cpu"
    assert next_call.call_kwargs[0]["generator"].seed == 404
    assert later[0].backend_metadata["mps_isolated"] is True


@pytest.mark.parametrize(
    ("device", "allow_cpu", "torch"),
    [("cuda", False, FakeCUDATorch), ("cpu", True, FakeTorch)],
)
def test_cuda_and_cpu_keep_standard_pipeline_path(device: str, allow_cpu: bool, torch: Any) -> None:
    pipeline = FakePipeline()
    variation_pipeline = FakePipeline()
    backend = DiffusersBackend(
        device=device,
        pipeline=pipeline,
        variation_pipeline=variation_pipeline,
        allow_cpu=allow_cpu,
    )
    backend._runtime = runtime(torch)
    studio = Studio(backend)

    study = studio.explore(
        studio.prepare(sketch(), max_side=96),
        intent=Intent("paper castle"),
        outputs=1,
    )

    assert pipeline.call_kwargs[0]["generator"].device == device
    assert "output_type" not in pipeline.call_kwargs[0]
    assert callable(pipeline.call_kwargs[0]["callback_on_step_end"])
    assert backend._pipeline is pipeline
    assert study[0].backend_metadata["mps_isolated"] is False

    variants = studio.vary(study[0], outputs=1, strength="subtle")
    variation_call = variation_pipeline.call_kwargs[0]
    assert isinstance(variation_call["image"], Image.Image)
    assert "output_type" not in variation_call
    assert variants[0].backend_metadata["mps_isolated"] is False


def test_request_cancel_stops_an_active_diffusers_step_and_is_consumed() -> None:
    pipeline = InterruptiblePipeline()
    backend = DiffusersBackend(
        device="cuda",
        pipeline=pipeline,
        variation_pipeline=FakePipeline(),
    )
    backend._runtime = runtime(FakeCUDATorch)
    studio = Studio(backend)
    prepared = studio.prepare(sketch(), max_side=96)
    pipeline.before_step = backend.request_cancel

    with pytest.raises(DiffusersGenerationCancelled, match="cancelled"):
        studio.explore(
            prepared,
            intent=Intent("paper castle"),
            outputs=1,
        )

    assert pipeline._interrupt is False
    pipeline.before_step = None
    recovered = studio.explore(
        prepared,
        intent=Intent("paper castle"),
        outputs=1,
    )
    assert len(recovered) == 1


def test_mps_invalid_latents_reload_pipeline_and_retry_same_seed_once() -> None:
    failed = StableMPSPipeline(finite_latents=False)
    recovered = StableMPSPipeline()
    backend = ReloadingBackend(failed, [recovered])
    backend._runtime = runtime()
    studio = Studio(backend)

    study = studio.explore(
        studio.prepare(sketch(), max_side=96),
        intent=Intent("paper castle"),
        outputs=1,
        seed_plan=SeedPlan.explicit((570858588,)),
    )

    assert backend.reload_count == 1
    assert failed.device == "cpu"
    assert failed.call_kwargs[0]["generator"].seed == 570858588
    assert recovered.call_kwargs[0]["generator"].seed == 570858588
    assert study[0].seed == 570858588
    assert study[0].backend_metadata["mps_retries"] == 1


def test_mps_retry_is_limited_to_one_full_reload() -> None:
    failed = StableMPSPipeline(finite_latents=False)
    still_invalid = StableMPSPipeline(finite_latents=False)
    backend = ReloadingBackend(failed, [still_invalid])
    backend._runtime = runtime()
    studio = Studio(backend)

    with pytest.raises(GenerationError, match="after one full pipeline reload"):
        studio.explore(
            studio.prepare(sketch(), max_side=96),
            intent=Intent("paper castle"),
            outputs=1,
            seed_plan=SeedPlan.explicit((570858588,)),
        )

    assert backend.reload_count == 1
    assert len(failed.call_kwargs) == 1
    assert len(still_invalid.call_kwargs) == 1


def test_mps_nonfinite_vae_decode_fails_instead_of_exporting_black() -> None:
    pipeline = StableMPSPipeline(finite_decode=False)
    backend = DiffusersBackend(device="mps", pipeline=pipeline)
    backend._runtime = runtime()
    studio = Studio(backend)
    with pytest.raises(GenerationError, match="non-finite pixels"):
        studio.explore(
            studio.prepare(sketch(), max_side=96),
            intent=Intent("paper castle"),
            outputs=1,
        )


def test_collapsed_black_pipeline_output_is_rejected() -> None:
    pipeline = BlackPipeline()
    backend = DiffusersBackend(device="mps", pipeline=pipeline)
    backend._runtime = runtime()
    studio = Studio(backend)
    with pytest.raises(GenerationError, match="near-black"):
        studio.explore(
            studio.prepare(sketch(), max_side=96),
            intent=Intent("paper castle"),
            outputs=1,
        )
