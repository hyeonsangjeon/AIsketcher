"""Dependency-lazy FLUX.2 [klein] 4B image-edit backend.

The backend is intentionally isolated from the preset and model registries. A
caller must supply an immutable model revision before real weights can be
loaded; tests and applications may instead inject an already constructed
pipeline.

FLUX.2 [klein] is a unified generation/editing pipeline rather than a
ControlNet. AIsketcher's normalized source image is therefore used as the
reference image unless an explicit ``GenerationRequest.init_image`` is present.
"""

from __future__ import annotations

import gc
import importlib
import math
import re
import threading
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageOps

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
    VariationStrength,
)

DEFAULT_FLUX2_KLEIN_REPO_ID = "black-forest-labs/FLUX.2-klein-4B"
"""Official Apache-2.0 FLUX.2 [klein] 4B repository."""

DEFAULT_FLUX2_SMALL_DECODER_REPO_ID = "black-forest-labs/FLUX.2-small-decoder"
"""Optional decoder used by the validated low-memory T4 profile."""


@dataclass(frozen=True, slots=True)
class Flux2KleinModelConfig:
    """Model location selected by AIsketcher's curated registry.

    ``revision`` deliberately has no default. The Hub's moving ``main`` branch
    is not reproducible, so a real load is rejected until the registry supplies
    an immutable commit hash.
    """

    repo_id: str = DEFAULT_FLUX2_KLEIN_REPO_ID
    revision: str | None = None
    cache_dir: Path | None = None
    base_path: Path | None = None
    license_id: str = "apache-2.0"
    decoder_model_id: str | None = None
    decoder_revision: str | None = None
    decoder_path: Path | None = None

    def __post_init__(self) -> None:
        repo_id = self.repo_id.strip()
        if not repo_id or len(repo_id) > 300:
            raise ValidationError("Flux2KleinModelConfig.repo_id must be 1..300 characters")
        revision = self.revision
        if revision is not None:
            revision = revision.strip()
            if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
                raise ValidationError(
                    "Flux2KleinModelConfig.revision must be an immutable 40-character "
                    "Hub commit hash or null"
                )
            revision = revision.lower()
        license_id = self.license_id.strip()
        if not license_id or len(license_id) > 100:
            raise ValidationError("Flux2KleinModelConfig.license_id must be 1..100 characters")
        decoder_model_id = self.decoder_model_id
        decoder_revision = self.decoder_revision
        if (decoder_model_id is None) != (decoder_revision is None):
            raise ValidationError(
                "decoder_model_id and decoder_revision must be configured together"
            )
        if decoder_model_id is not None and decoder_revision is not None:
            decoder_model_id = decoder_model_id.strip()
            if not decoder_model_id or len(decoder_model_id) > 300:
                raise ValidationError("decoder_model_id must be 1..300 characters")
            decoder_revision = decoder_revision.strip()
            if not re.fullmatch(r"[0-9a-fA-F]{40}", decoder_revision):
                raise ValidationError(
                    "decoder_revision must be an immutable 40-character Hub commit hash"
                )
            decoder_revision = decoder_revision.lower()
        object.__setattr__(self, "repo_id", repo_id)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "license_id", license_id)
        object.__setattr__(self, "decoder_model_id", decoder_model_id)
        object.__setattr__(self, "decoder_revision", decoder_revision)
        if self.cache_dir is not None:
            object.__setattr__(self, "cache_dir", Path(self.cache_dir).expanduser())
        if self.base_path is not None:
            object.__setattr__(self, "base_path", Path(self.base_path).expanduser())
        if self.decoder_path is not None:
            if decoder_model_id is None:
                raise ValidationError(
                    "decoder_path requires decoder_model_id and decoder_revision"
                )
            object.__setattr__(
                self,
                "decoder_path",
                Path(self.decoder_path).expanduser(),
            )


@dataclass(frozen=True, slots=True)
class Flux2KleinSettings:
    """Runtime-safe defaults for the distilled 4B checkpoint."""

    num_inference_steps: int = 4
    guidance_scale: float = 1.0
    cpu_offload: bool = True
    cleanup_between_outputs: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.num_inference_steps, bool)
            or not isinstance(self.num_inference_steps, int)
            or not 1 <= self.num_inference_steps <= 50
        ):
            raise ValidationError("num_inference_steps must be between 1 and 50")
        if (
            isinstance(self.guidance_scale, bool)
            or not isinstance(self.guidance_scale, (int, float))
            or not 0 <= float(self.guidance_scale) <= 30
        ):
            raise ValidationError("guidance_scale must be between 0 and 30")


@dataclass(frozen=True, slots=True)
class Flux2Progress:
    """Small, UI-neutral progress event emitted by the backend."""

    phase: Literal["loading", "generating", "complete", "cancelled"]
    output_index: int
    output_count: int
    seed: int | None
    step: int
    total_steps: int


class Flux2GenerationCancelled(GenerationError):
    """Raised when a caller-provided cancellation hook stops generation."""


@dataclass(slots=True)
class _Flux2Runtime:
    torch: Any
    pipeline_cls: Any
    autoencoder_cls: Any | None = None


class _CancelSignal(Exception):
    """Internal control-flow exception raised from a Diffusers step callback."""


_VARIATION_PROMPT_METHOD = "deterministic-edit-prompt-v1"
_DENOISE_PROMPT_METHOD = "mapped-to-edit-prompt-v1"

_VARIATION_EDIT_INSTRUCTIONS: dict[VariationStrength, str] = {
    VariationStrength.SUBTLE: (
        "Apply a subtle edit. Keep the result very close to the reference image; "
        "make only restrained changes to styling, color, material, lighting, and "
        "fine details."
    ),
    VariationStrength.BALANCED: (
        "Apply a balanced edit. Keep the reference clearly recognizable while "
        "allowing noticeable changes to styling, color, material, lighting, and "
        "secondary details."
    ),
    VariationStrength.BOLD: (
        "Apply a bold edit. Make substantial changes to styling, color, material, "
        "lighting, and secondary forms while keeping the result recognizably "
        "derived from the reference image."
    ),
}

_STRUCTURE_LOCK_INSTRUCTION = (
    "Structure lock is active: preserve the reference composition, camera framing, "
    "subject placement, silhouettes, proportions, and major geometry."
)
_STRUCTURE_UNLOCKED_INSTRUCTION = (
    "Structure lock is not active: composition and major geometry may evolve when "
    "the requested edit benefits from it."
)


@dataclass(frozen=True, slots=True)
class _VariationPromptPlan:
    prompt: str
    requested: VariationStrength
    applied: VariationStrength
    structure_locked: bool


class Flux2KleinBackend:
    """Native FLUX.2 [klein] image-edit adapter optimized for a 16 GB T4.

    Torch and Diffusers are imported only when generation is first requested.
    CUDA uses FP16 because NVIDIA T4 does not provide native BF16 acceleration.
    Outputs are generated sequentially so each seed has its own generator and
    accelerator peak memory does not scale with the requested output count.
    """

    name = "flux2-klein"

    def __init__(
        self,
        *,
        model: Flux2KleinModelConfig | None = None,
        settings: Flux2KleinSettings | None = None,
        device: str = "auto",
        local_files_only: bool = True,
        allow_cpu: bool = False,
        pipeline: Any | None = None,
        decoder: Any | None = None,
        runtime: _Flux2Runtime | None = None,
        should_cancel: Callable[[], bool] | None = None,
        on_progress: Callable[[Flux2Progress], None] | None = None,
    ) -> None:
        if device not in {"auto", "cuda", "cpu"}:
            raise ValidationError("device must be auto, cuda, or cpu")
        self.model = model or Flux2KleinModelConfig()
        self.settings = settings or Flux2KleinSettings()
        self.device = device
        self.local_files_only = local_files_only
        self.allow_cpu = allow_cpu
        self._pipeline = pipeline
        self._pipeline_was_injected = pipeline is not None
        self._owns_pipeline = pipeline is None
        self._decoder = decoder
        self._decoder_was_injected = decoder is not None
        self._owns_decoder = decoder is None
        self._runtime = runtime
        self._resolved_device: str | None = None
        self._should_cancel = should_cancel
        self._on_progress = on_progress
        self._cancel_event = threading.Event()

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            controls=("reference-image",),
            supports_seed=True,
            supports_negative_prompt=False,
            supports_variation=True,
            schedulers=("flow-match-euler",),
            max_outputs=8,
        )

    def _import_runtime(self) -> _Flux2Runtime:
        if self._runtime is not None:
            return self._runtime
        try:
            torch = importlib.import_module("torch")
            diffusers = importlib.import_module("diffusers")
            pipeline_cls = diffusers.Flux2KleinPipeline
            autoencoder_cls = diffusers.AutoencoderKLFlux2
        except (ImportError, AttributeError) as exc:
            raise OptionalDependencyError(
                "FLUX.2 Klein generation requires a compatible local runtime: "
                "pip install 'aisketcher[local]'"
            ) from exc
        self._runtime = _Flux2Runtime(
            torch=torch,
            pipeline_cls=pipeline_cls,
            autoencoder_cls=autoencoder_cls,
        )
        return self._runtime

    def _select_device(self, runtime: _Flux2Runtime) -> str:
        if self._resolved_device is not None:
            return self._resolved_device
        cuda_available = bool(runtime.torch.cuda.is_available())
        selected = "cuda" if self.device == "auto" and cuda_available else self.device
        if selected == "auto":
            selected = "cpu"
        if selected == "cuda" and not cuda_available:
            raise ModelUnavailableError("CUDA was requested but is unavailable")
        if selected == "cpu" and not self.allow_cpu:
            raise ModelUnavailableError(
                "CPU live generation is disabled because FLUX.2 Klein is impractical on CPU. "
                "Use a CUDA GPU or pass allow_cpu=True explicitly."
            )
        self._resolved_device = selected
        return selected

    def _report(self, event: Flux2Progress) -> None:
        if self._on_progress is None:
            return
        try:
            self._on_progress(event)
        except Exception as exc:
            raise GenerationError(f"FLUX.2 progress callback failed: {exc}") from exc

    def _cancel_requested(self) -> bool:
        if self._cancel_event.is_set():
            return True
        if self._should_cancel is None:
            return False
        try:
            return bool(self._should_cancel())
        except Exception as exc:
            raise GenerationError(f"FLUX.2 cancellation callback failed: {exc}") from exc

    def request_cancel(self) -> None:
        """Cooperatively stop the active or next queued generation call."""

        self._cancel_event.set()
        pipeline = self._pipeline
        if pipeline is not None:
            with suppress(AttributeError):
                pipeline._interrupt = True

    def cancel(self) -> None:
        """Alias used by providers that expose a generic cancellation method."""

        self.request_cancel()

    def _raise_if_cancelled(
        self,
        *,
        output_index: int,
        output_count: int,
        seed: int | None,
        step: int = 0,
    ) -> None:
        if not self._cancel_requested():
            return
        self._report(
            Flux2Progress(
                phase="cancelled",
                output_index=output_index,
                output_count=output_count,
                seed=seed,
                step=step,
                total_steps=self.settings.num_inference_steps,
            )
        )
        raise Flux2GenerationCancelled("FLUX.2 generation was cancelled")

    def _load_pipeline(
        self, runtime: _Flux2Runtime, device: str, *, output_count: int
    ) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        if self.model.revision is None:
            raise ModelUnavailableError(
                "FLUX.2 Klein has no immutable revision configured. "
                "Pass the pinned revision selected by AIsketcher's model registry."
            )
        self._raise_if_cancelled(output_index=0, output_count=output_count, seed=None)
        self._report(
            Flux2Progress(
                phase="loading",
                output_index=0,
                output_count=output_count,
                seed=None,
                step=0,
                total_steps=self.settings.num_inference_steps,
            )
        )
        dtype = runtime.torch.float16 if device == "cuda" else runtime.torch.float32
        load_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "use_safetensors": True,
            "local_files_only": self.local_files_only,
            "trust_remote_code": False,
            "low_cpu_mem_usage": True,
        }
        model_location = (
            str(self.model.base_path)
            if self.model.base_path is not None
            else self.model.repo_id
        )
        if self.model.base_path is None:
            load_kwargs["revision"] = self.model.revision
        if self.model.cache_dir is not None and self.model.base_path is None:
            load_kwargs["cache_dir"] = str(self.model.cache_dir)
        decoder = self._load_decoder(runtime, dtype=dtype)
        if decoder is not None:
            load_kwargs["vae"] = decoder
        pipeline: Any | None = None
        try:
            pipeline = runtime.pipeline_cls.from_pretrained(
                model_location,
                **load_kwargs,
            )
            if device == "cuda" and self.settings.cpu_offload:
                try:
                    pipeline.enable_model_cpu_offload()
                except (AttributeError, RuntimeError) as exc:
                    raise ModelUnavailableError(
                        "The installed Diffusers runtime cannot enable model CPU offload"
                    ) from exc
            else:
                pipeline.to(device)
        except (GenerationError, ModelUnavailableError):
            self._release_pipeline_candidate(pipeline)
            self._release_decoder()
            raise
        except Exception as exc:
            self._release_pipeline_candidate(pipeline)
            self._release_decoder()
            location = (
                "the managed local model directory"
                if self.model.base_path is not None
                else ("local cache" if self.local_files_only else "Hugging Face Hub")
            )
            raise ModelUnavailableError(
                f"Could not load pinned FLUX.2 Klein weights from {location}: {exc}"
            ) from exc
        self._pipeline = pipeline
        return pipeline

    @staticmethod
    def _release_pipeline_candidate(pipeline: Any | None) -> None:
        if pipeline is None:
            return
        with suppress(AttributeError, RuntimeError):
            pipeline.maybe_free_model_hooks()
        with suppress(AttributeError, RuntimeError):
            pipeline.to("cpu")

    def _load_decoder(self, runtime: _Flux2Runtime, *, dtype: Any) -> Any | None:
        if self._decoder is not None:
            return self._decoder
        if self.model.decoder_model_id is None:
            return None
        if self.model.decoder_revision is None:
            # Flux2KleinModelConfig normally prevents this, but keep the model
            # boundary safe if a custom deserializer bypasses dataclass init.
            raise ModelUnavailableError(
                "FLUX.2 small decoder has no immutable revision configured"
            )
        if runtime.autoencoder_cls is None:
            raise OptionalDependencyError(
                "The installed Diffusers runtime does not provide AutoencoderKLFlux2"
            )
        load_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "use_safetensors": True,
            "local_files_only": self.local_files_only,
            "trust_remote_code": False,
            "low_cpu_mem_usage": True,
        }
        decoder_location = (
            str(self.model.decoder_path)
            if self.model.decoder_path is not None
            else self.model.decoder_model_id
        )
        if self.model.decoder_path is None:
            load_kwargs["revision"] = self.model.decoder_revision
        if self.model.cache_dir is not None and self.model.decoder_path is None:
            load_kwargs["cache_dir"] = str(self.model.cache_dir)
        try:
            decoder = runtime.autoencoder_cls.from_pretrained(
                decoder_location,
                **load_kwargs,
            )
        except Exception as exc:
            location = (
                "the managed local decoder directory"
                if self.model.decoder_path is not None
                else ("local cache" if self.local_files_only else "Hugging Face Hub")
            )
            raise ModelUnavailableError(
                f"Could not load pinned FLUX.2 small decoder from {location}: {exc}"
            ) from exc
        self._decoder = decoder
        return decoder

    def _release_decoder(self) -> None:
        decoder = self._decoder
        self._decoder = None
        if decoder is not None and self._owns_decoder:
            with suppress(AttributeError, RuntimeError):
                decoder.to("cpu")

    def _generator(self, runtime: _Flux2Runtime, device: str, seed: int) -> Any:
        return runtime.torch.Generator(device=device).manual_seed(seed)

    @staticmethod
    def _variation_strength_from_denoise(
        denoise_strength: float,
    ) -> VariationStrength:
        """Map the public 0..1 edit amount to AIsketcher's three UX levels.

        FLUX.2 Klein's unified edit pipeline has no numeric ``strength`` argument.
        The boundaries are the midpoints between Studio's canonical 0.25, 0.45,
        and 0.65 values, making custom requests deterministic as well.
        """

        if (
            isinstance(denoise_strength, bool)
            or not isinstance(denoise_strength, (int, float))
            or not math.isfinite(float(denoise_strength))
            or not 0 <= float(denoise_strength) <= 1
        ):
            raise ValidationError("FLUX.2 denoise_strength must be between 0 and 1")
        value = float(denoise_strength)
        if value <= 0.35:
            return VariationStrength.SUBTLE
        if value <= 0.55:
            return VariationStrength.BALANCED
        return VariationStrength.BOLD

    @classmethod
    def _variation_prompt_plan(
        cls,
        request: GenerationRequest,
    ) -> _VariationPromptPlan | None:
        """Build the explicit edit directive used in place of denoise strength."""

        if request.denoise_strength is None:
            return None
        mapped_strength = cls._variation_strength_from_denoise(
            request.denoise_strength
        )
        requested_strength = request.recipe.variation_strength or mapped_strength
        if requested_strength != mapped_strength:
            raise ValidationError(
                "The resolved variation strength does not match denoise_strength "
                f"({requested_strength.value} != {mapped_strength.value})"
            )
        structure_locked = "structure" in request.recipe.locks
        structure_instruction = (
            _STRUCTURE_LOCK_INSTRUCTION
            if structure_locked
            else _STRUCTURE_UNLOCKED_INSTRUCTION
        )
        edit_instruction = _VARIATION_EDIT_INSTRUCTIONS[mapped_strength]
        prompt = (
            f"{request.recipe.generation_prompt}\n\n"
            f"AIsketcher edit directive ({mapped_strength.value}): "
            f"{edit_instruction} {structure_instruction}"
        )
        return _VariationPromptPlan(
            prompt=prompt,
            requested=requested_strength,
            applied=mapped_strength,
            structure_locked=structure_locked,
        )

    @staticmethod
    def _reference_image(request: GenerationRequest) -> tuple[Image.Image, str]:
        source = request.init_image if request.init_image is not None else request.prepared.image
        source_kind = "explicit-init" if request.init_image is not None else "prepared-source"
        image = ImageOps.exif_transpose(source).convert("RGB")
        image = image.resize(
            (request.recipe.width, request.recipe.height),
            Image.Resampling.LANCZOS,
        )
        return image, source_kind

    def _cleanup_accelerator(self, runtime: _Flux2Runtime, device: str) -> None:
        gc.collect()
        if device == "cuda":
            with suppress(AttributeError, RuntimeError):
                runtime.torch.cuda.synchronize()
            with suppress(AttributeError, RuntimeError):
                runtime.torch.cuda.empty_cache()

    def _step_callback(
        self,
        *,
        output_index: int,
        output_count: int,
        seed: int,
    ) -> Callable[[Any, int, Any, dict[str, Any]], dict[str, Any]]:
        def callback(
            pipeline: Any,
            step_index: int,
            _timestep: Any,
            callback_kwargs: dict[str, Any],
        ) -> dict[str, Any]:
            completed_steps = step_index + 1
            self._report(
                Flux2Progress(
                    phase="generating",
                    output_index=output_index,
                    output_count=output_count,
                    seed=seed,
                    step=completed_steps,
                    total_steps=self.settings.num_inference_steps,
                )
            )
            if self._cancel_requested():
                with suppress(AttributeError):
                    pipeline._interrupt = True
                raise _CancelSignal
            return callback_kwargs

        return callback

    def _generate_one(
        self,
        *,
        request: GenerationRequest,
        pipeline: Any,
        runtime: _Flux2Runtime,
        device: str,
        seed: int,
        output_index: int,
        output_count: int,
        reference: Image.Image,
        generation_prompt: str,
    ) -> Image.Image:
        self._raise_if_cancelled(
            output_index=output_index,
            output_count=output_count,
            seed=seed,
        )
        generator = self._generator(runtime, device, seed)
        output: Any | None = None
        images: Any | None = None
        try:
            output = pipeline(
                image=reference,
                prompt=generation_prompt,
                height=request.recipe.height,
                width=request.recipe.width,
                num_inference_steps=self.settings.num_inference_steps,
                guidance_scale=float(self.settings.guidance_scale),
                num_images_per_prompt=1,
                generator=generator,
                output_type="pil",
                callback_on_step_end=self._step_callback(
                    output_index=output_index,
                    output_count=output_count,
                    seed=seed,
                ),
            )
            images = getattr(output, "images", None)
            if not isinstance(images, (list, tuple)) or not images:
                raise GenerationError(f"FLUX.2 returned no image for seed {seed}")
            image = images[0]
            if not isinstance(image, Image.Image):
                raise GenerationError(f"FLUX.2 did not return a PIL image for seed {seed}")
            image = image.convert("RGB")
            expected_size = (request.recipe.width, request.recipe.height)
            if image.size != expected_size:
                raise GenerationError(
                    f"FLUX.2 returned {image.size[0]}x{image.size[1]} for seed {seed}; "
                    f"expected {expected_size[0]}x{expected_size[1]}"
                )
            return image
        except _CancelSignal as exc:
            self._report(
                Flux2Progress(
                    phase="cancelled",
                    output_index=output_index,
                    output_count=output_count,
                    seed=seed,
                    step=0,
                    total_steps=self.settings.num_inference_steps,
                )
            )
            raise Flux2GenerationCancelled("FLUX.2 generation was cancelled") from exc
        finally:
            with suppress(AttributeError):
                pipeline._interrupt = False
            images = None
            output = None

    def _validate_request(self, request: GenerationRequest) -> None:
        if request.recipe.steps != self.settings.num_inference_steps:
            raise ValidationError(
                "The resolved FLUX.2 recipe steps do not match the backend settings "
                f"({request.recipe.steps} != {self.settings.num_inference_steps})"
            )
        if not math.isclose(
            request.recipe.guidance_scale,
            float(self.settings.guidance_scale),
            rel_tol=0,
            abs_tol=1e-9,
        ):
            raise ValidationError(
                "The resolved FLUX.2 recipe guidance scale does not match the backend "
                f"settings ({request.recipe.guidance_scale:g} != "
                f"{float(self.settings.guidance_scale):g})"
            )
        if request.recipe.negative_prompt:
            raise ValidationError("FLUX.2 Klein does not support a string negative prompt")

    def _generate(self, request: GenerationRequest) -> list[GenerationResult]:
        self._validate_request(request)
        output_count = len(request.seeds)
        if not 1 <= output_count <= self.capabilities.max_outputs:
            raise ValidationError(
                f"FLUX.2 requires 1..{self.capabilities.max_outputs} seeds"
            )
        max_seed = (1 << 63) - 1
        if any(
            isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= max_seed
            for seed in request.seeds
        ):
            raise ValidationError("FLUX.2 seeds must be non-negative 63-bit integers")

        variation_plan = self._variation_prompt_plan(request)
        runtime = self._import_runtime()
        device = self._select_device(runtime)
        pipeline = self._load_pipeline(runtime, device, output_count=output_count)
        reference, source_kind = self._reference_image(request)
        generation_prompt = (
            variation_plan.prompt
            if variation_plan is not None
            else request.recipe.generation_prompt
        )
        dtype_name = "float16" if device == "cuda" else "float32"
        results: list[GenerationResult] = []

        for output_index, seed in enumerate(request.seeds, start=1):
            image = self._generate_one(
                request=request,
                pipeline=pipeline,
                runtime=runtime,
                device=device,
                seed=seed,
                output_index=output_index,
                output_count=output_count,
                reference=reference,
                generation_prompt=generation_prompt,
            )
            results.append(
                GenerationResult(
                    image=image,
                    seed=seed,
                    metadata={
                        "backend": self.name,
                        "model_repo_id": self.model.repo_id,
                        "model_revision": self.model.revision,
                        "model_license": self.model.license_id,
                        "decoder_model_id": self.model.decoder_model_id,
                        "decoder_revision": self.model.decoder_revision,
                        "decoder_source": (
                            "injected"
                            if self._decoder_was_injected
                            else (
                                "managed-local-dir"
                                if self.model.decoder_path is not None
                                else "pinned"
                                if self.model.decoder_model_id is not None
                                else "pipeline-default"
                            )
                        ),
                        "base_source": (
                            "managed-local-dir"
                            if self.model.base_path is not None
                            else "pinned"
                        ),
                        "device": device,
                        "dtype": dtype_name,
                        "num_inference_steps": self.settings.num_inference_steps,
                        "guidance_scale": float(self.settings.guidance_scale),
                        "sequential": True,
                        "reference_source": source_kind,
                        "denoise_strength_requested": request.denoise_strength,
                        "denoise_strength_applied": None,
                        "denoise_strength_method": (
                            _DENOISE_PROMPT_METHOD
                            if variation_plan is not None
                            else None
                        ),
                        "variation_strength_requested": (
                            variation_plan.requested.value
                            if variation_plan is not None
                            else None
                        ),
                        "variation_strength_applied": (
                            variation_plan.applied.value
                            if variation_plan is not None
                            else None
                        ),
                        "variation_strength_method": (
                            _VARIATION_PROMPT_METHOD
                            if variation_plan is not None
                            else None
                        ),
                        "structure_lock_applied_to_prompt": (
                            variation_plan.structure_locked
                            if variation_plan is not None
                            else False
                        ),
                        "cpu_offload": bool(
                            device == "cuda"
                            and self.settings.cpu_offload
                            and not self._pipeline_was_injected
                        ),
                    },
                )
            )
            self._report(
                Flux2Progress(
                    phase="complete",
                    output_index=output_index,
                    output_count=output_count,
                    seed=seed,
                    step=self.settings.num_inference_steps,
                    total_steps=self.settings.num_inference_steps,
                )
            )
            if self.settings.cleanup_between_outputs:
                self._cleanup_accelerator(runtime, device)
        return results

    def generate(self, request: GenerationRequest) -> list[GenerationResult]:
        try:
            return self._generate(request)
        finally:
            self._cancel_event.clear()
            pipeline = self._pipeline
            if pipeline is not None:
                with suppress(AttributeError):
                    pipeline._interrupt = False

    def close(self) -> None:
        """Release owned pipeline references and reclaim accelerator cache."""

        self.request_cancel()
        pipeline = self._pipeline
        self._pipeline = None
        runtime = self._runtime
        if pipeline is not None and self._owns_pipeline:
            with suppress(AttributeError, RuntimeError):
                pipeline.maybe_free_model_hooks()
            with suppress(AttributeError, RuntimeError):
                pipeline.to("cpu")
        self._release_decoder()
        if runtime is not None:
            self._cleanup_accelerator(runtime, self._resolved_device or "cpu")
        self._resolved_device = None

    def __enter__(self) -> Flux2KleinBackend:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


__all__ = [
    "DEFAULT_FLUX2_KLEIN_REPO_ID",
    "DEFAULT_FLUX2_SMALL_DECODER_REPO_ID",
    "Flux2GenerationCancelled",
    "Flux2KleinBackend",
    "Flux2KleinModelConfig",
    "Flux2KleinSettings",
    "Flux2Progress",
]
