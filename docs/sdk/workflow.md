# Complete SDK workflow

## 1. Prepare

```python
prepared = studio.prepare("sketch.jpg")
```

Preparation corrects EXIF orientation, converts to RGB, selects a model-safe
size, extracts the Canny control, and returns diagnostics. It does not modify
the source file. Review low contrast, edge density, fragmentation, and crop-risk
signals before generation.

See [Prepare](prepare.md).

## 2. Explore

```python
study = studio.explore(
    prepared,
    intent=intent,
    outputs=4,
    seed_plan=SeedPlan.scout(4),
)
```

`explore()` resolves intent against the preset and backend, generates each
candidate with an independent generator, and preserves creation order. The
resolved recipe and all actual seeds belong to the study.

## 3. Pick

```python
selected = study.pick(1)
```

Picking adds a selection record; it does not remove the other candidates. Use a
stable candidate ID when integrating with another interface, and treat the
index as a display convenience.

## 4. Vary

```python
variations = studio.vary(
    selected,
    outputs=4,
    strength="subtle",
    locks=("structure",),
)
```

The new study points back to the selected candidate. A lock expresses intent,
but the capability report remains the source of truth for how a backend applied
it.

See [Explore, pick, and vary](explore-refine.md).

## 5. Export

```python
variations.export("design-study")
```

The export is a directory with normalized inputs, controls, candidate images,
a contact sheet, and `manifest.json`. File hashes make later mutation visible.

## 6. Replay

```python
report = studio.replay("design-study/manifest.json", mode="strict")
```

Inspect the returned report even on success. It describes resolved resources,
runtime differences, and whether the generated artifact hashes match.

See [Export and replay](export-replay.md).
