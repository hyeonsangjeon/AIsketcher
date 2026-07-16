"""Exploration result objects and designer-facing selection helpers."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from .errors import ValidationError
from .models import Candidate, PreparedSketch, ResolvedRecipe, SeedMode


@dataclass(slots=True)
class Study:
    """A reproducible set of related candidates."""

    id: str
    kind: str
    prepared: PreparedSketch
    recipe: ResolvedRecipe
    candidates: list[Candidate]
    backend: str
    seed_mode: SeedMode
    selected_id: str | None = None
    parent: Candidate | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.kind not in ("exploration", "variation"):
            raise ValidationError("Study.kind must be exploration or variation")
        if not self.candidates:
            raise ValidationError("Study requires at least one candidate")

    def __len__(self) -> int:
        return len(self.candidates)

    def __iter__(self) -> Iterator[Candidate]:
        return iter(self.candidates)

    def __getitem__(self, index: int) -> Candidate:
        return self.candidates[index]

    @property
    def selected(self) -> Candidate | None:
        if self.selected_id is None:
            return None
        return next(
            (candidate for candidate in self.candidates if candidate.id == self.selected_id),
            None,
        )

    def pick(self, candidate: int | str | Candidate) -> Candidate:
        """Select by zero-based index, candidate id, or candidate object."""

        selected: Candidate | None
        if isinstance(candidate, int):
            try:
                selected = self.candidates[candidate]
            except IndexError as exc:
                raise ValidationError(f"Candidate index {candidate} is out of range") from exc
        elif isinstance(candidate, Candidate):
            selected = next(
                (item for item in self.candidates if item.id == candidate.id), None
            )
            if selected is None:
                raise ValidationError("Candidate does not belong to this study")
        else:
            selected = next((item for item in self.candidates if item.id == candidate), None)
            if selected is None:
                raise ValidationError(f"Unknown candidate id {candidate!r}")
        self.selected_id = selected.id
        return selected

    def export(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Export sanitized artifacts, a contact sheet and manifest v1."""

        from .manifest import export_study

        return export_study(self, path, overwrite=overwrite)


__all__ = ["Study"]
