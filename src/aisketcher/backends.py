"""Generation backend protocol and built-in adapters.

This module deliberately performs no Torch or Diffusers import at module import
time. The optional runtime is loaded only after a local-generation call.
"""

from __future__ import annotations

import gc
import importlib
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
from PIL import Image, ImageEnhance

from .errors import (
    GenerationError,
    ModelUnavailableError,
    OptionalDependencyError,
    ValidationError,
)
from .models import (
    BackendCapabilities,
    GenerationRequest,
    GenerationResult,
    ModelReference,
)
from .presets import PresetManager


@runtime_checkable
class Backend(Protocol):
    """Minimal extension point for local, cloud or in-house generation systems."""

    name: str

    @property
    def capabilities(self) -> BackendCapabilities: ...

    def generate(self, request: GenerationRequest) -> list[GenerationResult]: ...


class FakeBackend:
    """Deterministic, network-free backend for tutorials, tests and UI previews."""

    name = "fake"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            controls=("canny",),
            supports_seed=True,
            supports_negative_prompt=True,
            supports_variation=True,
            schedulers=("unipc",),
            max_outputs=8,
        )

    def generate(self, request: GenerationRequest) -> list[GenerationResult]:
        width, height = request.recipe.width, request.recipe.height
        source = request.prepared.image.resize((width, height), Image.Resampling.LANCZOS)
        source_array = np.asarray(source, dtype=np.float32)
        edge = (
            np.asarray(
                request.prepared.control.resize((width, height), Image.Resampling.NEAREST).convert(
                    "L"
                ),
                dtype=np.uint8,
            )
            > 0
        )
        results: list[GenerationResult] = []

        for seed in request.seeds:
            rng = np.random.default_rng(seed)
            palette = rng.integers(45, 225, size=3).astype(np.float32)
            accent = rng.integers(0, 80, size=3).astype(np.float32)
            y_gradient = np.linspace(0.85, 1.12, height, dtype=np.float32)[:, None, None]
            paper = np.broadcast_to(palette[None, None, :], (height, width, 3)) * y_gradient
            mixed = source_array * 0.40 + paper * 0.60
            noise = rng.normal(0.0, 4.0, size=(height, width, 1))
            mixed = np.clip(mixed + noise, 0, 255)
            mixed[edge] = accent
            generated = Image.fromarray(mixed.astype(np.uint8), "RGB")

            if request.init_image is not None:
                strength = (
                    request.denoise_strength if request.denoise_strength is not None else 0.45
                )
                init = request.init_image.convert("RGB").resize(
                    (width, height), Image.Resampling.LANCZOS
                )
                generated = Image.blend(init, generated, float(strength))
            generated = ImageEnhance.Contrast(generated).enhance(1.08)
            results.append(
                GenerationResult(
                    image=generated,
                    seed=seed,
                    metadata={"backend": self.name, "algorithm": "guided-preview-v1"},
                )
            )
        return results


@dataclass(slots=True)
class _DiffusersRuntime:
    torch: Any
    controlnet_cls: Any
    txt2img_cls: Any
    img2img_cls: Any
    auto_txt2img_cls: Any
    auto_img2img_cls: Any
    scheduler_cls: Any


class DiffusersBackend:
    """Lazy, pinned SDXL ControlNet adapter.

    By default it loads only models installed through ``PresetManager``. Setting
    ``local_files_only=False`` still permits only the curated, immutable model
    references embedded in a resolved recipe.
    """

    name = "diffusers"

    def __init__(
        self,
        *,
        device: str = "auto",
        preset_manager: PresetManager | None = None,
        local_files_only: bool = True,
        pipeline: Any | None = None,
        variation_pipeline: Any | None = None,
        allow_cpu: bool = False,
    ) -> None:
        if device not in ("auto", "cuda", "mps", "cpu"):
            raise ValidationError("device must be auto, cuda, mps, or cpu")
        self.device = device
        self.preset_manager = preset_manager or PresetManager(allow_downloads=not local_files_only)
        if not local_files_only and not self.preset_manager.allow_downloads:
            raise ValidationError(
                "local_files_only=False conflicts with a PresetManager that disables downloads"
            )
        self.local_files_only = local_files_only
        self.allow_cpu = allow_cpu
        self._pipeline = pipeline
        self._variation_pipeline = variation_pipeline
        self._pipeline_was_injected = pipeline is not None or variation_pipeline is not None
        self._mps_pipeline_consumed = False
        self._loaded_preset: str | None = None
        self._runtime: _DiffusersRuntime | None = None
        self._resolved_device: str | None = None

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            controls=("canny",),
            supports_seed=True,
            supports_negative_prompt=True,
            supports_variation=True,
            schedulers=("unipc",),
            max_outputs=8,
        )

    def _import_runtime(self) -> _DiffusersRuntime:
        if self._runtime is not None:
            return self._runtime
        try:
            torch = importlib.import_module("torch")
            diffusers = importlib.import_module("diffusers")
        except ImportError as exc:
            raise OptionalDependencyError(
                "Local generation requires: pip install 'aisketcher[local]'"
            ) from exc
        self._runtime = _DiffusersRuntime(
            torch=torch,
            controlnet_cls=diffusers.ControlNetModel,
            txt2img_cls=diffusers.StableDiffusionXLControlNetPipeline,
            img2img_cls=diffusers.StableDiffusionXLControlNetImg2ImgPipeline,
            auto_txt2img_cls=diffusers.AutoPipelineForText2Image,
            auto_img2img_cls=diffusers.AutoPipelineForImage2Image,
            scheduler_cls=diffusers.UniPCMultistepScheduler,
        )
        return self._runtime

    def _select_device(self, runtime: _DiffusersRuntime) -> str:
        if self._resolved_device is not None:
            return self._resolved_device
        if self.device == "auto":
            if runtime.torch.cuda.is_available():
                selected = "cuda"
            elif (
                getattr(runtime.torch.backends, "mps", None)
                and runtime.torch.backends.mps.is_available()
            ):
                selected = "mps"
            else:
                selected = "cpu"
        else:
            selected = self.device
        if selected == "cuda" and not runtime.torch.cuda.is_available():
            raise ModelUnavailableError("CUDA was requested but is unavailable")
        if selected == "mps" and not (
            getattr(runtime.torch.backends, "mps", None)
            and runtime.torch.backends.mps.is_available()
        ):
            raise ModelUnavailableError("Apple MPS was requested but is unavailable")
        if selected == "cpu" and not self.allow_cpu:
            raise ModelUnavailableError(
                "CPU live generation is disabled because SDXL is impractical on CPU. "
                "Use Guided Sample/FakeBackend or pass allow_cpu=True explicitly."
            )
        self._resolved_device = selected
        return selected

    def _model_location(self, model: ModelReference) -> tuple[str, str | None]:
        installed = self.preset_manager.model_path(model)
        if installed is not None:
            return str(installed), None
        if self.local_files_only:
            self.preset_manager.require_installed(self._current_preset)
        return model.repo_id, model.revision

    @property
    def _current_preset(self) -> str:
        if self._pending_preset is None:
            raise RuntimeError("No recipe is being loaded")
        return self._pending_preset

    _pending_preset: str | None = None

    def _configure_pipeline(self, pipeline: Any, runtime: _DiffusersRuntime, device: str) -> Any:
        pipeline.scheduler = runtime.scheduler_cls.from_config(pipeline.scheduler.config)
        pipeline.enable_attention_slicing()
        pipeline.to(device)
        if device == "mps":
            self._stabilize_mps_vae(pipeline, runtime)
        return pipeline

    @staticmethod
    def _supports_explicit_vae_decode(pipeline: Any) -> bool:
        return all(hasattr(pipeline, attribute) for attribute in ("vae", "image_processor"))

    @staticmethod
    def _stabilize_mps_vae(pipeline: Any, runtime: _DiffusersRuntime) -> None:
        """Keep SDXL's VAE in fp32 on MPS instead of repeatedly half-casting it."""

        if DiffusersBackend._supports_explicit_vae_decode(pipeline):
            pipeline.vae.to(device="mps", dtype=runtime.torch.float32)

    @staticmethod
    def _offload_mps_vae_callback(runtime: _DiffusersRuntime) -> Any:
        offloaded = False

        def offload(
            pipeline: Any,
            _step_index: int,
            _timestep: Any,
            callback_kwargs: dict[str, Any],
        ) -> dict[str, Any]:
            nonlocal offloaded
            if not offloaded and DiffusersBackend._supports_explicit_vae_decode(pipeline):
                pipeline.vae.to(device="cpu", dtype=runtime.torch.float32)
                offloaded = True
            return callback_kwargs

        return offload

    @staticmethod
    def _cleanup_mps(runtime: _DiffusersRuntime) -> None:
        mps = getattr(runtime.torch, "mps", None)
        if mps is not None:
            synchronize = getattr(mps, "synchronize", None)
            if callable(synchronize):
                synchronize()
            empty_cache = getattr(mps, "empty_cache", None)
            if callable(empty_cache):
                empty_cache()
        gc.collect()

    @staticmethod
    def _tensor_is_finite(tensor: Any, runtime: _DiffusersRuntime) -> bool:
        finite = runtime.torch.isfinite(tensor).all()
        return bool(finite.item() if hasattr(finite, "item") else finite)

    def _decode_mps_latents(
        self, pipeline: Any, latents: Any, runtime: _DiffusersRuntime
    ) -> Image.Image:
        """Decode MPS latents explicitly with an fp32 VAE and finite checks."""

        self._stabilize_mps_vae(pipeline, runtime)
        decoded: Any | None = None
        images: Any | None = None
        try:
            latents = latents.to(device="mps", dtype=runtime.torch.float32)
            if not self._tensor_is_finite(latents, runtime):
                raise GenerationError("MPS diffusion produced non-finite latents")

            config = pipeline.vae.config
            has_mean = getattr(config, "latents_mean", None) is not None
            has_std = getattr(config, "latents_std", None) is not None
            if has_mean and has_std:
                mean = (
                    runtime.torch.tensor(config.latents_mean)
                    .view(1, 4, 1, 1)
                    .to(device="mps", dtype=runtime.torch.float32)
                )
                std = (
                    runtime.torch.tensor(config.latents_std)
                    .view(1, 4, 1, 1)
                    .to(device="mps", dtype=runtime.torch.float32)
                )
                latents = latents * std / config.scaling_factor + mean
            else:
                latents = latents / config.scaling_factor

            with runtime.torch.no_grad():
                decoded = pipeline.vae.decode(latents, return_dict=False)[0]
            if not self._tensor_is_finite(decoded, runtime):
                raise GenerationError(
                    "MPS VAE decode produced non-finite pixels; refusing a black result"
                )
            watermark = getattr(pipeline, "watermark", None)
            if watermark is not None:
                decoded = watermark.apply_watermark(decoded)
            images = pipeline.image_processor.postprocess(decoded, output_type="pil")
            if not images:
                raise GenerationError("MPS VAE decode returned no image")
            image = images[0]
            if not isinstance(image, Image.Image):
                raise GenerationError("MPS VAE decode did not return a PIL image")
            return image.convert("RGB")
        finally:
            latents = None
            decoded = None
            images = None
            pipeline.vae.to(device="cpu", dtype=runtime.torch.float32)

    def _encode_mps_init_latents(
        self,
        pipeline: Any,
        image: Image.Image,
        runtime: _DiffusersRuntime,
        generator: Any,
        *,
        width: int,
        height: int,
    ) -> Any:
        """Encode an img2img source on CPU while preserving generator state.

        Diffusers accepts a four-channel latent tensor as ``image`` and then
        performs its normal strength/timestep noise step. Sampling the VAE
        distribution with the same generator here preserves the random stream
        that its stock ``prepare_latents`` implementation would consume.
        """

        preprocessed: Any | None = None
        encoded: Any | None = None
        latents: Any | None = None
        mean: Any | None = None
        std: Any | None = None
        pipeline.vae.to(device="cpu", dtype=runtime.torch.float32)
        self._cleanup_mps(runtime)
        try:
            preprocessed = pipeline.image_processor.preprocess(
                image, height=height, width=width
            ).to(device="cpu", dtype=runtime.torch.float32)
            if not self._tensor_is_finite(preprocessed, runtime):
                raise GenerationError("MPS variation source contains non-finite pixels")

            with runtime.torch.no_grad():
                encoded = pipeline.vae.encode(preprocessed)
            latent_distribution = getattr(encoded, "latent_dist", None)
            if latent_distribution is not None:
                latents = latent_distribution.sample(generator=generator)
            else:
                latents = getattr(encoded, "latents", None)
            if latents is None:
                raise GenerationError("MPS CPU VAE encode returned no latent tensor")
            latents = latents.to(device="cpu", dtype=runtime.torch.float32)
            if not self._tensor_is_finite(latents, runtime):
                raise GenerationError("MPS CPU VAE encode produced non-finite latents")

            config = pipeline.vae.config
            has_mean = getattr(config, "latents_mean", None) is not None
            has_std = getattr(config, "latents_std", None) is not None
            if has_mean and has_std:
                mean = (
                    runtime.torch.tensor(config.latents_mean)
                    .view(1, 4, 1, 1)
                    .to(device="cpu", dtype=runtime.torch.float32)
                )
                std = (
                    runtime.torch.tensor(config.latents_std)
                    .view(1, 4, 1, 1)
                    .to(device="cpu", dtype=runtime.torch.float32)
                )
                latents = (latents - mean) * config.scaling_factor / std
            else:
                latents = latents * config.scaling_factor
            if not self._tensor_is_finite(latents, runtime):
                raise GenerationError("MPS CPU VAE latent normalization was non-finite")
            return latents.to(device="cpu", dtype=runtime.torch.float32)
        finally:
            preprocessed = None
            encoded = None
            latents = None
            mean = None
            std = None
            gc.collect()

    @staticmethod
    def _validate_output_image(
        image: Image.Image, *, expected_size: tuple[int, int], seed: int
    ) -> None:
        if image.size != expected_size:
            raise GenerationError(
                f"Backend returned {image.size[0]}x{image.size[1]} for seed {seed}; "
                f"expected {expected_size[0]}x{expected_size[1]}"
            )
        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if pixels.size == 0 or (int(pixels.max()) <= 1 and float(pixels.std()) < 0.5):
            raise GenerationError(f"Backend returned a collapsed near-black image for seed {seed}")

    def _reset_mps_pipelines(self, runtime: _DiffusersRuntime) -> None:
        pipelines = [self._pipeline, self._variation_pipeline]
        seen: set[int] = set()
        for candidate in pipelines:
            if candidate is None or id(candidate) in seen:
                continue
            seen.add(id(candidate))
            with suppress(AttributeError, RuntimeError):
                candidate.to("cpu")
        self._pipeline = None
        self._variation_pipeline = None
        self._mps_pipeline_consumed = False
        self._loaded_preset = None
        pipelines.clear()
        self._cleanup_mps(runtime)

    @staticmethod
    def _is_retryable_mps_error(error: Exception) -> bool:
        if isinstance(error, GenerationError):
            return True
        if isinstance(error, RuntimeError):
            message = str(error).lower()
            return any(
                marker in message
                for marker in ("out of memory", "non-finite", "nan", "mps backend")
            )
        return False

    def _load_pipelines(self, request: GenerationRequest) -> None:
        runtime = self._import_runtime()
        device = self._select_device(runtime)
        wants_variation = request.init_image is not None

        if wants_variation and self._variation_pipeline is not None:
            return
        if not wants_variation and self._pipeline is not None:
            return

        # Diffusers' from_pipe conversion reuses the same model components. This
        # avoids holding two SDXL weight sets in memory on 16 GB Apple Silicon.
        if wants_variation and self._pipeline is not None:
            self._variation_pipeline = runtime.auto_img2img_cls.from_pipe(self._pipeline)
            self._configure_pipeline(self._variation_pipeline, runtime, device)
            return
        if not wants_variation and self._variation_pipeline is not None:
            self._pipeline = runtime.auto_txt2img_cls.from_pipe(self._variation_pipeline)
            self._configure_pipeline(self._pipeline, runtime, device)
            return

        self._pending_preset = request.recipe.preset
        try:
            base = next(model for model in request.recipe.models if model.role == "base")
            control = next(model for model in request.recipe.models if model.role == "controlnet")
            base_location, base_revision = self._model_location(base)
            control_location, control_revision = self._model_location(control)
        finally:
            self._pending_preset = None

        dtype = runtime.torch.float16 if device in ("cuda", "mps") else runtime.torch.float32
        controlnet = runtime.controlnet_cls.from_pretrained(
            control_location,
            revision=control_revision,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16",
            local_files_only=self.local_files_only,
            trust_remote_code=False,
        )
        common = {
            "controlnet": controlnet,
            "torch_dtype": dtype,
            "use_safetensors": True,
            "variant": "fp16",
            "local_files_only": self.local_files_only,
            "trust_remote_code": False,
        }
        if wants_variation:
            self._variation_pipeline = runtime.img2img_cls.from_pretrained(
                base_location, revision=base_revision, **common
            )
            self._configure_pipeline(self._variation_pipeline, runtime, device)
        else:
            self._pipeline = runtime.txt2img_cls.from_pretrained(
                base_location, revision=base_revision, **common
            )
            self._configure_pipeline(self._pipeline, runtime, device)
        self._loaded_preset = request.recipe.preset

    def _generator(self, runtime: _DiffusersRuntime, device: str, seed: int) -> Any:
        # CPU generators are the stable Diffusers path for Apple MPS.
        generator_device = "cpu" if device == "mps" else device
        return runtime.torch.Generator(device=generator_device).manual_seed(seed)

    def _generate_one(
        self,
        *,
        request: GenerationRequest,
        pipeline: Any,
        runtime: _DiffusersRuntime,
        device: str,
        seed: int,
        control: Image.Image,
    ) -> tuple[Image.Image, bool]:
        """Generate one candidate while keeping accelerator references short-lived."""

        generator = self._generator(runtime, device, seed)
        stable_mps_decode = device == "mps" and self._supports_explicit_vae_decode(pipeline)
        kwargs: dict[str, Any] = {
            "prompt": request.recipe.prompt,
            "negative_prompt": request.recipe.negative_prompt or None,
            "num_inference_steps": request.recipe.steps,
            "guidance_scale": request.recipe.guidance_scale,
            "controlnet_conditioning_scale": request.recipe.control_strength,
            "generator": generator,
            "width": request.recipe.width,
            "height": request.recipe.height,
        }
        if request.init_image is None:
            kwargs["image"] = control
        else:
            init_image = request.init_image.convert("RGB").resize(
                (request.recipe.width, request.recipe.height), Image.Resampling.LANCZOS
            )
            if stable_mps_decode:
                kwargs["image"] = self._encode_mps_init_latents(
                    pipeline,
                    init_image,
                    runtime,
                    generator,
                    width=request.recipe.width,
                    height=request.recipe.height,
                )
            else:
                kwargs["image"] = init_image
            kwargs["control_image"] = control
            kwargs["strength"] = (
                request.denoise_strength if request.denoise_strength is not None else 0.45
            )
        if stable_mps_decode:
            # The VAE is needed before denoising by img2img, but not during the
            # denoising loop. Move it off MPS after the first completed step and
            # decode the final latent ourselves in fp32.
            if request.init_image is None:
                self._stabilize_mps_vae(pipeline, runtime)
            kwargs["output_type"] = "latent"
            kwargs["callback_on_step_end"] = self._offload_mps_vae_callback(runtime)
            kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

        output: Any | None = None
        output_images: Any | None = None
        try:
            output = pipeline(**kwargs)
            output_images = getattr(output, "images", None)
            if output_images is None or (
                isinstance(output_images, (list, tuple)) and not output_images
            ):
                raise GenerationError(f"Backend returned no image for seed {seed}")
            if stable_mps_decode:
                image = self._decode_mps_latents(pipeline, output_images, runtime)
            else:
                candidate = output_images[0]
                if not isinstance(candidate, Image.Image):
                    raise GenerationError(f"Backend did not return a PIL image for seed {seed}")
                image = candidate.convert("RGB")
            self._validate_output_image(
                image,
                expected_size=(request.recipe.width, request.recipe.height),
                seed=seed,
            )
            return image, stable_mps_decode
        finally:
            # Diffusers outputs retain the full device tensor batch. Clear both
            # aliases before asking MPS to reclaim cached allocations.
            output_images = None
            output = None
            kwargs.clear()
            if device == "mps":
                self._cleanup_mps(runtime)

    def generate(self, request: GenerationRequest) -> list[GenerationResult]:
        if self._loaded_preset not in (None, request.recipe.preset):
            self._pipeline = None
            self._variation_pipeline = None

        # Injected pipelines are intentionally supported for tests, while the
        # genuine runtime still remains dependency-lazy until this call.
        runtime = self._import_runtime()
        device = self._select_device(runtime)
        isolate_mps = device == "mps" and not self._pipeline_was_injected
        if isolate_mps and self._mps_pipeline_consumed:
            # A previous generate() call may have left a successfully-used
            # pipeline resident. Never carry that MPS state into another seed.
            self._reset_mps_pipelines(runtime)
        self._load_pipelines(request)
        pipeline = self._variation_pipeline if request.init_image is not None else self._pipeline
        if pipeline is None:
            raise ModelUnavailableError("No compatible Diffusers pipeline is configured")

        control = request.prepared.control.resize(
            (request.recipe.width, request.recipe.height), Image.Resampling.LANCZOS
        )
        results: list[GenerationResult] = []
        # Sequential generation is deliberate: it gives every output its own
        # generator and is the memory-safe path on Apple Silicon.
        for seed_index, seed in enumerate(request.seeds):
            if device == "mps":
                self._cleanup_mps(runtime)
            retry_reason: str | None = None
            try:
                image, stable_mps_decode = self._generate_one(
                    request=request,
                    pipeline=pipeline,
                    runtime=runtime,
                    device=device,
                    seed=seed,
                    control=control,
                )
            except Exception as error:
                if (
                    device != "mps"
                    or self._pipeline_was_injected
                    or not self._is_retryable_mps_error(error)
                ):
                    raise
                retry_reason = str(error)
                # The traceback owns the failed helper frame and, through it,
                # potentially large MPS tensors. Release it before reloading.
                error.__traceback__ = None

            mps_retries = 0
            if retry_reason is not None:
                mps_retries = 1
                pipeline = None
                self._reset_mps_pipelines(runtime)
                try:
                    self._load_pipelines(request)
                except Exception as reload_error:
                    raise GenerationError(
                        f"MPS generation failed for seed {seed} ({retry_reason}); "
                        f"the pipeline reload also failed: {reload_error}"
                    ) from reload_error
                pipeline = (
                    self._variation_pipeline if request.init_image is not None else self._pipeline
                )
                if pipeline is None:
                    raise ModelUnavailableError(
                        "No compatible Diffusers pipeline was available after MPS reload"
                    )
                try:
                    image, stable_mps_decode = self._generate_one(
                        request=request,
                        pipeline=pipeline,
                        runtime=runtime,
                        device=device,
                        seed=seed,
                        control=control,
                    )
                except Exception as retry_error:
                    retry_error.__traceback__ = None
                    raise GenerationError(
                        f"MPS seed {seed} remained invalid after one full pipeline "
                        f"reload: {retry_error}"
                    ) from retry_error
            results.append(
                GenerationResult(
                    image=image,
                    seed=seed,
                    metadata={
                        "backend": self.name,
                        "device": device,
                        "sequential": True,
                        "shared_pipeline_components": True,
                        "vae_dtype": "float32" if stable_mps_decode else "pipeline-default",
                        "mps_retries": mps_retries,
                        "mps_isolated": isolate_mps,
                    },
                )
            )
            if isolate_mps:
                self._mps_pipeline_consumed = True
                if seed_index + 1 < len(request.seeds):
                    next_seed = request.seeds[seed_index + 1]
                    pipeline = None
                    self._reset_mps_pipelines(runtime)
                    try:
                        self._load_pipelines(request)
                    except Exception as reload_error:
                        raise GenerationError(
                            "MPS seed isolation could not reload the pipeline before "
                            f"seed {next_seed}: {reload_error}"
                        ) from reload_error
                    pipeline = (
                        self._variation_pipeline
                        if request.init_image is not None
                        else self._pipeline
                    )
                    if pipeline is None:
                        raise ModelUnavailableError(
                            "No compatible Diffusers pipeline was available for the "
                            f"isolated MPS seed {next_seed}"
                        )
        self._loaded_preset = request.recipe.preset
        return results


__all__ = ["Backend", "DiffusersBackend", "FakeBackend"]
