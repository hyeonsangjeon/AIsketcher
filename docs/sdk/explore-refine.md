# Explore, pick, and vary

## Explore a deliberate set

The default scout contains four candidates. This is large enough to expose
meaningful direction changes while remaining readable as a contact sheet and
reasonable for local hardware.

```python
study = studio.explore(
    prepared,
    intent=intent,
    outputs=4,
    seed_plan=SeedPlan.scout(4),
)
```

Advanced workflows may use 1, 4, or 8 outputs. Each image receives an independent
generator and a recorded actual seed so batching changes do not silently reuse
one random stream.

## Read recommendation badges correctly

- **Closest structure** compares control and output structure.
- **Cleanest edges** looks for excessive or missing edge behavior.
- **Most distinct** identifies a candidate least redundant with the set.

Badges never reorder candidates and never say “best.” A candidate can have more
than one technical badge, and a designer can select one with none.

## Pick without deleting alternatives

```python
selected = study.pick(1)
```

The candidate set remains intact. The study records the selection without
deleting or reordering alternatives, and the selected candidate becomes the
parent record for future work.

## Vary with an explicit radius

```python
variations = studio.vary(
    selected,
    outputs=4,
    strength="subtle",
    locks=("structure",),
)
```

Use `subtle` to refine one direction, `balanced` to reopen composition choices,
and `bold` to explore a wider neighborhood. Locks are model-independent requests;
check `CapabilityReport` for the backend’s exact support or approximation.

For seed-plan choices, see [Seeds and output counts](../guides/seeds.md).
