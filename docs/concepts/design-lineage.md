# Design lineage

An image file shows only an outcome. A design decision also needs its source,
settings, alternatives, selected parent, and subsequent variations.
AIsketcher calls that connected record a **study**.

```text
PreparedSketch
  └─ Study (scout)
       ├─ Candidate 0
       ├─ Candidate 1  ← picked
       │    └─ Study (variation)
       │         ├─ Candidate 0
       │         ├─ Candidate 1  ← final
       │         ├─ Candidate 2
       │         └─ Candidate 3
       ├─ Candidate 2
       └─ Candidate 3
```

## Intent is not a backend call

`Intent` captures what the designer means: prompt, work profile, and desired
structure adherence. A versioned preset and backend capabilities resolve that
intent into a concrete `ResolvedRecipe`.

Resolution follows a visible order:

1. versioned preset defaults;
2. intent and profile;
3. explicit Advanced overrides;
4. backend capability validation or documented approximation.

Unsupported behavior is never silently discarded. The capability report either
records an approximation or rejects the recipe.

## Selection is human; badges are evidence

Candidates keep their creation order. A badge may describe the candidate with
the closest structure, cleanest edge behavior, or greatest distinctness. Those
are technical signals, not an aesthetic ranking and not a replacement for a
designer’s choice.

## Variation preserves ancestry

`vary()` starts from one selected candidate and records its parent ID. Constraint
locks state what should remain stable; strength states how far the new study may
move. The original scout is never overwritten.

## Replay reports honesty, not magic

A manifest pins assets, recipes, and model revisions. Strict replay refuses
drift. Compatible replay can substitute a supported component but lists every
substitution. Different hardware and dependency kernels may still prevent
pixel-identical output, so replay reports the observed environment as well.
