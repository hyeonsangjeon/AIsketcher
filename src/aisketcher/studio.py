"""High-level structured exploration workflow."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any

from PIL import Image

from .backends import Backend, DiffusersBackend
from .controls import image_sha256, rescore_candidates
from .controls import prepare as prepare_sketch
from .errors import ReplayError, UnsupportedCapabilityError, ValidationError
from .flux2_backend import (
    Flux2KleinBackend,
    Flux2KleinModelConfig,
    Flux2KleinSettings,
)
from .manifest import (
    canonical_sha256,
    load_manifest,
    runtime_versions,
    verify_manifest_files,
)
from .models import (
    MAX_GENERATION_DIMENSION,
    MAX_GENERATION_PIXELS,
    MAX_SOURCE_PIXELS,
    Candidate,
    CannyConfig,
    GenerationRequest,
    Intent,
    PreparationDiagnostics,
    PreparedSketch,
    Recipe,
    ReplayMode,
    ReplayReport,
    ResolvedRecipe,
    SeedMode,
    SeedPlan,
    TechnicalScores,
    VariationStrength,
)
from .presets import PresetManager, get_preset, resolve_recipe
from .study import Study

_DENOISE_STRENGTH = {
    VariationStrength.SUBTLE: 0.25,
    VariationStrength.BALANCED: 0.45,
    VariationStrength.BOLD: 0.65,
}


class Studio:
    """Coordinates preparation, recipe resolution, exploration and replay."""

    def __init__(
        self,
        backend: Backend,
        *,
        preset: str = "sdxl-canny-lite@1",
        preset_manager: PresetManager | None = None,
    ) -> None:
        if not all(
            hasattr(backend, attribute)
            for attribute in ("name", "generate", "capabilities")
        ):
            raise ValidationError("backend must implement the AIsketcher Backend protocol")
        self.backend = backend
        self.preset = get_preset(preset).name
        backend_manager = getattr(backend, "preset_manager", None)
        self.preset_manager: PresetManager = (
            preset_manager
            if preset_manager is not None
            else (
                backend_manager
                if isinstance(backend_manager, PresetManager)
                else PresetManager()
            )
        )

    @classmethod
    def from_preset(
        cls,
        preset: str = "flux2-klein-edit@1",
        *,
        device: str = "auto",
        backend: Backend | None = None,
        preset_manager: PresetManager | None = None,
        local_files_only: bool = True,
    ) -> Studio:
        selected_preset = get_preset(preset)
        manager = preset_manager or PresetManager(
            allow_downloads=not local_files_only
        )
        selected_backend = backend
        if selected_backend is None:
            if selected_preset.name == "flux2-klein-edit@1":
                base_model = next(
                    model for model in selected_preset.models if model.role == "base-edit"
                )
                decoder_model = next(
                    model for model in selected_preset.models if model.role == "decoder"
                )
                selected_backend = Flux2KleinBackend(
                    model=Flux2KleinModelConfig(
                        repo_id=base_model.repo_id,
                        revision=base_model.revision,
                        cache_dir=manager.cache_dir,
                        base_path=manager.model_destination(base_model),
                        decoder_model_id=decoder_model.repo_id,
                        decoder_revision=decoder_model.revision,
                        decoder_path=manager.model_destination(decoder_model),
                    ),
                    settings=Flux2KleinSettings(
                        num_inference_steps=selected_preset.steps,
                        guidance_scale=selected_preset.guidance_scale,
                    ),
                    device=device,
                    local_files_only=local_files_only,
                )
            else:
                selected_backend = DiffusersBackend(
                    device=device,
                    preset_manager=manager,
                    local_files_only=local_files_only,
                )
        return cls(
            selected_backend,
            preset=selected_preset.name,
            preset_manager=manager,
        )

    def prepare(
        self,
        source: str | Path | Image.Image,
        *,
        max_side: int = 1024,
        canny: CannyConfig | None = None,
        upscale: bool = True,
        max_pixels: int = 50_000_000,
    ) -> PreparedSketch:
        return prepare_sketch(
            source,
            max_side=max_side,
            canny=canny,
            upscale=upscale,
            max_pixels=max_pixels,
        )

    def resolve(
        self,
        intent: Intent,
        recipe: Recipe | None = None,
        *,
        prepared: PreparedSketch | None = None,
    ) -> ResolvedRecipe:
        resolved = resolve_recipe(
            self.preset,
            intent,
            recipe,
            backend_name=self.backend.name,
            capabilities=self.backend.capabilities,
        )
        # A 1024 preset denotes the longest edge. Preserve the sketch aspect
        # ratio unless Advanced explicitly supplied width and height.
        if prepared is not None and (recipe is None or recipe.width is None):
            resolved = replace(
                resolved,
                width=prepared.prepared_size[0],
                height=prepared.prepared_size[1],
            )
        if not resolved.capability_report.supported:
            messages = "; ".join(
                issue.message for issue in resolved.capability_report.errors
            )
            raise UnsupportedCapabilityError(messages)
        return resolved

    def _validate_outputs(self, outputs: int) -> None:
        if not 1 <= outputs <= self.backend.capabilities.max_outputs:
            raise UnsupportedCapabilityError(
                f"Backend {self.backend.name!r} supports 1.."
                f"{self.backend.capabilities.max_outputs} outputs per study"
            )

    @staticmethod
    def _stable_id(
        kind: str,
        prepared: PreparedSketch,
        recipe: ResolvedRecipe,
        seeds: tuple[int, ...],
        parent_id: str | None,
    ) -> str:
        payload = {
            "kind": kind,
            "source": prepared.source_sha256,
            "control": prepared.control_sha256,
            "recipe": recipe.to_dict(),
            "seeds": list(seeds),
            "parent": parent_id,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return sha256(encoded).hexdigest()[:16]

    def _generate_study(
        self,
        *,
        kind: str,
        prepared: PreparedSketch,
        recipe: ResolvedRecipe,
        seed_plan: SeedPlan,
        outputs: int,
        parent: Candidate | None = None,
        init_image: Image.Image | None = None,
        denoise_strength: float | None = None,
    ) -> Study:
        self._validate_outputs(outputs)
        if isinstance(self.backend, Flux2KleinBackend):
            # FLUX receives explicit managed-directory paths rather than asking
            # Diffusers to resolve a Hub id.  Revalidate those directories at
            # the last boundary before generation so an unmarked, incomplete,
            # or newly tampered cache can never bypass PresetManager's pinned
            # allowlist contract.
            self.preset_manager.require_installed(self.preset)
        seeds = seed_plan.resolve(outputs)
        request = GenerationRequest(
            prepared=prepared,
            recipe=recipe,
            seeds=seeds,
            init_image=init_image,
            denoise_strength=denoise_strength,
        )
        results = self.backend.generate(request)
        if len(results) != outputs:
            raise ReplayError(
                f"Backend returned {len(results)} results for {outputs} requested outputs"
            )
        if tuple(result.seed for result in results) != seeds:
            raise ReplayError("Backend returned results with different or reordered seeds")

        study_id = self._stable_id(
            kind, prepared, recipe, seeds, parent.id if parent is not None else None
        )
        candidates = [
            Candidate(
                id=f"candidate-{sha256(f'{study_id}:{index}:{result.seed}'.encode()).hexdigest()[:12]}",
                image=result.image.convert("RGB"),
                seed=result.seed,
                recipe=recipe,
                scores=TechnicalScores(0.0, 0.0, 0.0),
                prepared=prepared,
                parent_id=parent.id if parent is not None else None,
                backend_metadata=dict(result.metadata),
            )
            for index, result in enumerate(results)
        ]
        candidates = rescore_candidates(candidates)
        return Study(
            id=f"study-{study_id}",
            kind=kind,
            prepared=prepared,
            recipe=recipe,
            candidates=candidates,
            backend=self.backend.name,
            seed_mode=seed_plan.mode,
            parent=parent,
        )

    def explore(
        self,
        prepared: PreparedSketch,
        *,
        intent: Intent,
        outputs: int = 4,
        seed_plan: SeedPlan | None = None,
        recipe: Recipe | None = None,
    ) -> Study:
        resolved = self.resolve(intent, recipe, prepared=prepared)
        return self._generate_study(
            kind="exploration",
            prepared=prepared,
            recipe=resolved,
            seed_plan=seed_plan or SeedPlan.scout(outputs),
            outputs=outputs,
        )

    def vary(
        self,
        candidate: Candidate,
        *,
        outputs: int = 4,
        strength: VariationStrength | str = VariationStrength.SUBTLE,
        locks: tuple[str, ...] = ("structure",),
        seed_plan: SeedPlan | None = None,
    ) -> Study:
        if not self.backend.capabilities.supports_variation:
            raise UnsupportedCapabilityError(
                f"Backend {self.backend.name!r} does not support variations"
            )
        try:
            resolved_strength = VariationStrength(strength)
        except ValueError as exc:
            raise ValidationError("strength must be subtle, balanced, or bold") from exc
        allowed_locks = {"structure"}
        unknown_locks = set(locks) - allowed_locks
        if unknown_locks:
            raise ValidationError(
                f"Unknown variation lock(s): {', '.join(sorted(unknown_locks))}"
            )

        if "structure" in locks:
            prepared = candidate.prepared
        else:
            prepared = prepare_sketch(
                candidate.image,
                max_side=max(candidate.recipe.width, candidate.recipe.height),
            )
        recipe = replace(
            candidate.recipe,
            width=prepared.prepared_size[0],
            height=prepared.prepared_size[1],
            variation_strength=resolved_strength,
            locks=tuple(locks),
        )
        return self._generate_study(
            kind="variation",
            prepared=prepared,
            recipe=recipe,
            seed_plan=seed_plan
            or SeedPlan.scout(outputs, base_seed=(candidate.seed + 1) % (1 << 63)),
            outputs=outputs,
            parent=candidate,
            init_image=candidate.image,
            denoise_strength=_DENOISE_STRENGTH[resolved_strength],
        )

    @staticmethod
    def _artifact_path(
        manifest_path: Path, manifest: Mapping[str, Any], key: str
    ) -> Path:
        try:
            relative = manifest["files"][key]["path"]
        except (KeyError, TypeError) as exc:
            raise ReplayError(f"Manifest is missing artifact {key!r}") from exc
        return manifest_path.parent / Path(relative)

    @staticmethod
    def _load_replay_image(path: Path, label: str) -> Image.Image:
        try:
            with Image.open(path) as opened:
                width, height = opened.size
                if (
                    not 64 <= width <= MAX_GENERATION_DIMENSION
                    or not 64 <= height <= MAX_GENERATION_DIMENSION
                    or width % 8
                    or height % 8
                    or width * height > MAX_GENERATION_PIXELS
                ):
                    raise ReplayError(
                        f"Manifest {label} image has unsafe dimensions {width}x{height}"
                    )
                return opened.convert("RGB").copy()
        except ReplayError:
            raise
        except (OSError, Image.DecompressionBombError) as exc:
            raise ReplayError(f"Manifest {label} image is invalid") from exc

    def replay(
        self,
        manifest: str | Path,
        *,
        mode: ReplayMode | str = ReplayMode.STRICT,
    ) -> ReplayReport:
        try:
            replay_mode = ReplayMode(mode)
        except ValueError as exc:
            raise ValidationError("mode must be strict or compatible") from exc
        manifest_path, value = load_manifest(manifest)
        verified = verify_manifest_files(manifest_path, value)
        drift: list[str] = []
        warnings = [
            "Replay preserves recipe and seeds, but hardware/runtime differences may change pixels."
        ]

        kind = value.get("kind")
        if kind not in ("exploration", "variation"):
            raise ReplayError("Manifest kind must be exploration or variation")

        recorded_runtime = value.get("runtime", {})
        current_runtime = runtime_versions(self.backend.name)
        if isinstance(recorded_runtime, dict):
            for key, recorded in recorded_runtime.items():
                current = current_runtime.get(str(key))
                if current is not None and current != recorded:
                    warnings.append(
                        f"runtime {key!r} changed from {recorded!r} to {current!r}"
                    )

        recorded_backend = str(value.get("backend", ""))
        if recorded_backend != self.backend.name:
            message = (
                f"backend changed from {recorded_backend!r} to {self.backend.name!r}"
            )
            if replay_mode is ReplayMode.STRICT:
                raise ReplayError(message)
            drift.append(message)

        recipe_value = value.get("recipe")
        if not isinstance(recipe_value, dict):
            raise ReplayError("Manifest recipe must be an object")
        expected_recipe_hash = value.get("recipe_sha256")
        actual_recipe_hash = canonical_sha256(recipe_value)
        if expected_recipe_hash != actual_recipe_hash:
            message = "resolved recipe differs from its exported recipe hash"
            if replay_mode is ReplayMode.STRICT:
                raise ReplayError(message)
            drift.append(message)
        try:
            recorded_recipe = ResolvedRecipe.from_dict(recipe_value)
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            raise ReplayError("Manifest resolved recipe is invalid") from exc
        current_preset = get_preset(recorded_recipe.preset)
        if recorded_recipe.models != current_preset.models:
            message = "model revision differs from the immutable preset definition"
            if replay_mode is ReplayMode.STRICT:
                raise ReplayError(message)
            drift.append(message)
            recorded_recipe = replace(recorded_recipe, models=current_preset.models)

        capabilities = self.backend.capabilities
        if current_preset.required_control not in capabilities.controls:
            raise UnsupportedCapabilityError(
                f"Replay requires {current_preset.required_control} control support"
            )
        if recorded_recipe.scheduler not in capabilities.schedulers:
            message = f"scheduler {recorded_recipe.scheduler!r} is unsupported"
            if replay_mode is ReplayMode.STRICT or not capabilities.schedulers:
                raise ReplayError(message)
            replacement = capabilities.schedulers[0]
            drift.append(f"{message}; substituted {replacement!r}")
            recorded_recipe = replace(recorded_recipe, scheduler=replacement)
        if kind == "variation" and not capabilities.supports_variation:
            raise UnsupportedCapabilityError("Replay requires variation support")

        seed_value = value.get("seed_plan", {})
        candidate_values = value.get("candidates")
        try:
            raw_seeds = seed_value["seeds"]
            if not isinstance(raw_seeds, list) or not raw_seeds:
                raise ValueError("seeds must be a non-empty list")
            if len(raw_seeds) > capabilities.max_outputs:
                raise ValueError("too many seeds")
            if any(
                isinstance(seed, bool)
                or not isinstance(seed, int)
                or not 0 <= seed <= (1 << 63) - 1
                for seed in raw_seeds
            ):
                raise ValueError("seed is outside the supported integer range")
            seeds = tuple(raw_seeds)
            seed_mode = SeedMode(seed_value["mode"])
            if not isinstance(candidate_values, list) or len(candidate_values) != len(seeds):
                raise ValueError("candidate count does not match seeds")
            if not all(isinstance(candidate, Mapping) for candidate in candidate_values):
                raise ValueError("candidate entry must be an object")
        except (KeyError, TypeError, ValueError) as exc:
            raise ReplayError("Manifest seed plan or candidate list is invalid") from exc

        source_descriptor = value.get("source", {})
        try:
            if not isinstance(source_descriptor, Mapping):
                raise ValueError("source must be an object")
            source_image = self._load_replay_image(
                self._artifact_path(manifest_path, value, "source"), "source"
            )
            control_image = self._load_replay_image(
                self._artifact_path(manifest_path, value, "control"), "control"
            )
            original_value = source_descriptor["original_size"]
            prepared_value = source_descriptor["prepared_size"]
            if (
                not isinstance(original_value, list)
                or len(original_value) != 2
                or any(
                    isinstance(part, bool) or not isinstance(part, int) or part < 1
                    for part in original_value
                )
                or original_value[0] * original_value[1] > MAX_SOURCE_PIXELS
            ):
                raise ValueError("original_size is invalid")
            if (
                not isinstance(prepared_value, list)
                or len(prepared_value) != 2
                or tuple(prepared_value) != source_image.size
                or control_image.size != source_image.size
            ):
                raise ValueError("prepared_size does not match source/control artifacts")
            original_size = (original_value[0], original_value[1])
            prepared_size = (prepared_value[0], prepared_value[1])
            canny = CannyConfig.from_dict(source_descriptor["canny"])
            diagnostics = PreparationDiagnostics.from_dict(
                source_descriptor["diagnostics"]
            )
        except (KeyError, TypeError, ValueError, OSError, ValidationError) as exc:
            raise ReplayError("Manifest source metadata is invalid") from exc
        prepared = PreparedSketch(
            image=source_image,
            control=control_image,
            original_size=(original_size[0], original_size[1]),
            prepared_size=(prepared_size[0], prepared_size[1]),
            source_sha256=image_sha256(source_image),
            control_sha256=image_sha256(control_image),
            canny=canny,
            diagnostics=diagnostics,
        )

        seed_plan = SeedPlan.explicit(seeds)
        # Preserve the recorded UI mode in the replayed Study while ensuring the
        # exact recorded seed list is used for generation.

        parent: Candidate | None = None
        init_image: Image.Image | None = None
        if kind == "variation":
            lineage = value.get("lineage", {})
            if not isinstance(lineage, Mapping):
                raise ReplayError("Manifest variation lineage must be an object")
            init_image = self._load_replay_image(
                self._artifact_path(manifest_path, value, "parent"), "variation parent"
            )
            parent_id = lineage.get("parent_id") or "replayed-parent"
            parent = Candidate(
                id=str(parent_id),
                image=init_image,
                seed=0,
                recipe=recorded_recipe,
                scores=TechnicalScores(0.0, 0.0, 0.0),
                prepared=prepared,
            )
        denoise = (
            _DENOISE_STRENGTH.get(recorded_recipe.variation_strength)
            if recorded_recipe.variation_strength
            else None
        )
        study = self._generate_study(
            kind=kind,
            prepared=prepared,
            recipe=recorded_recipe,
            seed_plan=seed_plan,
            outputs=len(seeds),
            parent=parent,
            init_image=init_image,
            denoise_strength=denoise,
        )
        study.seed_mode = seed_mode
        selected_id = value.get("selection")
        if selected_id:
            recorded_candidates = candidate_values
            selected_index = next(
                (
                    index
                    for index, candidate_value in enumerate(recorded_candidates)
                    if candidate_value.get("id") == selected_id
                ),
                None,
            )
            if selected_index is not None and selected_index < len(study.candidates):
                study.pick(selected_index)

        candidate_hash_matches: list[bool] = []
        for index, candidate in enumerate(study.candidates):
            try:
                file_key = candidate_values[index]["file"]
                expected_hash = value["files"][file_key]["sha256"]
            except (IndexError, KeyError, TypeError) as exc:
                raise ReplayError("Manifest candidate artifact mapping is invalid") from exc
            candidate_hash_matches.append(image_sha256(candidate.image) == expected_hash)
        if not all(candidate_hash_matches):
            warnings.append(
                "One or more regenerated candidates differ from the exported PNG hashes."
            )

        return ReplayReport(
            manifest_path=manifest_path,
            mode=replay_mode,
            verified_files=verified,
            drift=tuple(drift),
            warnings=tuple(warnings),
            candidate_hash_matches=tuple(candidate_hash_matches),
            study=study,
        )


__all__ = ["Studio"]
