from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from aisketcher import (
    BackendCapabilities,
    CannyConfig,
    Intent,
    Recipe,
    SeedMode,
    SeedPlan,
    StructureMode,
    ValidationError,
)
from aisketcher.presets import resolve_recipe


def test_base_import_does_not_load_heavy_optional_dependencies() -> None:
    src = Path(__file__).parents[2] / "src"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src)
    code = """
import json, sys
import aisketcher
print(json.dumps([name for name in ('torch','diffusers','gradio','huggingface_hub') if name in sys.modules]))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == []


def test_intent_normalizes_structure_without_hidden_prompt() -> None:
    intent = Intent("  a paper kingdom  ", profile=" graphic_design ", structure="faithful")
    assert intent.prompt == "a paper kingdom"
    assert intent.profile == "graphic_design"
    assert intent.structure is StructureMode.FAITHFUL


@pytest.mark.parametrize("prompt", ["", "   "])
def test_intent_requires_a_prompt(prompt: str) -> None:
    with pytest.raises(ValidationError):
        Intent(prompt)


def test_recipe_validates_paired_latent_dimensions() -> None:
    with pytest.raises(ValidationError, match="set together"):
        Recipe(width=768)
    with pytest.raises(ValidationError, match="divisible"):
        Recipe(width=769, height=768)
    with pytest.raises(ValidationError, match="4096"):
        Recipe(width=4104, height=4096)
    with pytest.raises(ValidationError, match="finite"):
        Recipe(guidance_scale=float("nan"))


def test_canny_validates_threshold_order() -> None:
    with pytest.raises(ValidationError):
        CannyConfig(low=200, high=100)


def test_seed_plans_are_stable_and_explicit() -> None:
    first = SeedPlan.scout(4, base_seed=19)
    second = SeedPlan.scout(4, base_seed=19)
    assert first.mode is SeedMode.SCOUT
    assert first.resolve(4) == second.resolve(4)
    assert len(set(first.resolve(4))) == 4
    assert SeedPlan.locked(42).resolve(3) == (42, 42, 42)
    assert SeedPlan.explicit((1, 2)).resolve(2) == (1, 2)
    with pytest.raises(ValidationError, match="2 seeds"):
        SeedPlan.explicit((1, 2)).resolve(3)


def test_recipe_resolution_priority_and_capability_report() -> None:
    capabilities = BackendCapabilities(schedulers=("euler",))
    resolved = resolve_recipe(
        "sdxl-canny-lite@1",
        Intent("castle", structure="faithful"),
        Recipe(steps=12, guidance_scale=0.0, scheduler="unipc"),
        backend_name="limited",
        capabilities=capabilities,
    )
    assert resolved.steps == 12
    assert resolved.guidance_scale == 0.0
    assert resolved.control_strength == pytest.approx(0.95)
    assert resolved.scheduler == "euler"
    assert resolved.prompt == "castle"
    assert resolved.capability_report.supported
    assert resolved.capability_report.warnings[0].setting == "scheduler"
    assert resolved.capability_report.warnings[0].applied == "euler"


def test_resolved_recipe_round_trip() -> None:
    resolved = resolve_recipe(
        "sdxl-canny-lite@1",
        Intent("castle"),
        None,
        backend_name="fake",
        capabilities=BackendCapabilities(),
    )
    restored = type(resolved).from_dict(resolved.to_dict())
    assert restored == resolved
    assert replace(restored, prompt="new").prompt == "new"


def test_resolved_recipe_rejects_non_object_capability_issues() -> None:
    resolved = resolve_recipe(
        "sdxl-canny-lite@1",
        Intent("castle"),
        None,
        backend_name="fake",
        capabilities=BackendCapabilities(),
    )
    value = resolved.to_dict()
    value["capability_report"]["issues"] = [1]

    with pytest.raises(ValidationError, match="issues must be a list"):
        type(resolved).from_dict(value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("width", 65),
        ("width", 4104),
        ("steps", 9999),
        ("guidance_scale", 999.0),
        ("guidance_scale", float("nan")),
        ("control_strength", 999.0),
    ],
)
def test_resolved_recipe_rejects_unsafe_manifest_values(
    field: str, value: object
) -> None:
    resolved = resolve_recipe(
        "sdxl-canny-lite@1",
        Intent("castle"),
        None,
        backend_name="fake",
        capabilities=BackendCapabilities(),
    )
    payload = resolved.to_dict()
    payload[field] = value
    with pytest.raises(ValidationError):
        type(resolved).from_dict(payload)


def test_resolved_recipe_does_not_coerce_fractional_dimensions() -> None:
    resolved = resolve_recipe(
        "sdxl-canny-lite@1",
        Intent("castle"),
        None,
        backend_name="fake",
        capabilities=BackendCapabilities(),
    )
    payload = resolved.to_dict()
    payload["width"] = 64.9
    with pytest.raises(ValidationError, match="integer"):
        type(resolved).from_dict(payload)
