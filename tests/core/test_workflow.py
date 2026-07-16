from __future__ import annotations

from dataclasses import replace

import pytest
from PIL import Image, ImageDraw

from aisketcher import (
    BackendCapabilities,
    FakeBackend,
    GenerationRequest,
    GenerationResult,
    Intent,
    SeedPlan,
    Studio,
    UnsupportedCapabilityError,
    ValidationError,
)
from aisketcher.controls import image_sha256


def sketch() -> Image.Image:
    image = Image.new("RGB", (240, 160), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 45, 210, 140), outline="black", width=4)
    draw.polygon(((30, 45), (120, 10), (210, 45)), outline="black")
    return image


def test_full_fake_workflow_is_deterministic_and_auditable() -> None:
    studio = Studio(FakeBackend())
    prepared = studio.prepare(sketch(), max_side=256)
    source_before = image_sha256(prepared.image)
    control_before = image_sha256(prepared.control)
    plan = SeedPlan.scout(4, base_seed=7)
    first = studio.explore(
        prepared, intent=Intent("paper castle"), outputs=4, seed_plan=plan
    )
    second = studio.explore(
        prepared, intent=Intent("paper castle"), outputs=4, seed_plan=plan
    )
    assert first.id == second.id
    assert tuple(item.seed for item in first) == plan.resolve(4)
    assert [image_sha256(item.image) for item in first] == [
        image_sha256(item.image) for item in second
    ]
    assert image_sha256(prepared.image) == source_before
    assert image_sha256(prepared.control) == control_before
    assert sum("closest structure" in item.scores.badges for item in first) == 1
    assert sum("cleanest edges" in item.scores.badges for item in first) == 1
    assert sum("most distinct" in item.scores.badges for item in first) == 1
    assert all(0 <= item.scores.structure_similarity <= 1 for item in first)


def test_pick_and_vary_preserve_lineage() -> None:
    studio = Studio(FakeBackend())
    study = studio.explore(
        studio.prepare(sketch(), max_side=128),
        intent=Intent("paper castle"),
        outputs=2,
    )
    choice = study.pick(1)
    assert study.selected is choice
    variants = studio.vary(choice, outputs=2, strength="bold", locks=("structure",))
    assert variants.kind == "variation"
    assert variants.parent is choice
    assert variants.recipe.variation_strength.value == "bold"
    assert variants.recipe.locks == ("structure",)
    assert all(item.parent_id == choice.id for item in variants)
    assert all(item.prepared is choice.prepared for item in variants)


def test_vary_rejects_the_unimplemented_composition_lock() -> None:
    studio = Studio(FakeBackend())
    study = studio.explore(
        studio.prepare(sketch(), max_side=128),
        intent=Intent("paper castle"),
        outputs=1,
    )

    with pytest.raises(ValidationError, match="Unknown variation lock"):
        studio.vary(study[0], locks=("composition",))


def test_vary_wraps_a_maximum_63_bit_parent_seed() -> None:
    studio = Studio(FakeBackend())
    study = studio.explore(
        studio.prepare(sketch(), max_side=128),
        intent=Intent("paper castle"),
        outputs=1,
        seed_plan=SeedPlan.locked((1 << 63) - 1),
    )

    variants = studio.vary(study[0], outputs=1)

    assert variants[0].seed == SeedPlan.scout(1, base_seed=0).resolve(1)[0]


def test_pick_rejects_foreign_candidate_and_index() -> None:
    studio = Studio(FakeBackend())
    prepared = studio.prepare(sketch(), max_side=128)
    one = studio.explore(prepared, intent=Intent("one"), outputs=1)
    two = studio.explore(prepared, intent=Intent("two"), outputs=1)
    with pytest.raises(ValidationError, match="does not belong"):
        one.pick(two[0])
    with pytest.raises(ValidationError, match="out of range"):
        one.pick(2)


class NoControlBackend(FakeBackend):
    name = "no-control"

    @property
    def capabilities(self) -> BackendCapabilities:
        return replace(super().capabilities, controls=())


def test_unsupported_control_fails_before_generation() -> None:
    studio = Studio(NoControlBackend())
    prepared = studio.prepare(sketch(), max_side=128)
    with pytest.raises(UnsupportedCapabilityError, match="Canny"):
        studio.explore(prepared, intent=Intent("castle"), outputs=1)


class ReorderingBackend(FakeBackend):
    name = "reordering"

    def generate(self, request: GenerationRequest) -> list[GenerationResult]:
        return list(reversed(super().generate(request)))


def test_backend_may_not_reorder_seeds() -> None:
    studio = Studio(ReorderingBackend())
    prepared = studio.prepare(sketch(), max_side=128)
    with pytest.raises(Exception, match="reordered seeds"):
        studio.explore(prepared, intent=Intent("castle"), outputs=2)


def test_output_limit_is_reported() -> None:
    studio = Studio(FakeBackend())
    with pytest.raises(UnsupportedCapabilityError, match="1..8"):
        studio.explore(
            studio.prepare(sketch(), max_side=128),
            intent=Intent("castle"),
            outputs=9,
        )
