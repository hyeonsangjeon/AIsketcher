# Quick start

Version 0.2.1 is published on
[PyPI](https://pypi.org/project/AIsketcher/0.2.1/). The commands below pin the
public release so a later package update cannot silently change the setup.

## Install the layer you need

=== "Core SDK"

    ```bash
    python -m pip install "AIsketcher==0.2.1"
    ```

    This installs preparation, recipe, lineage, export, and replay types. It
    does not install a generation runtime.

=== "Local generation"

    ```bash
    python -m pip install "AIsketcher[local]==0.2.1"
    ```

    Local presets download their pinned model files only after confirmation.

=== "Studio"

    ```bash
    python -m pip install "AIsketcher[demo]==0.2.1"
    aisketcher init
    aisketcher studio
    ```

    `init` writes a versioned per-user settings file and never downloads a
    model. Guided Sample then works from its bundled, hash-verified fixture
    without a network or GPU. Install `AIsketcher[local,demo]` instead for live
    local generation.

For the complete model-free first run:

```bash
python -m pip install "AIsketcher[demo]==0.2.1" && aisketcher init && aisketcher studio
```

`aisketcher init` protects an existing settings file. Use `--path` for a
project-specific file or `--force` only after reviewing what will be replaced.
See the [configuration reference](reference/configuration.md) for resolution
order and every supported key.

## Run a study

```python
from aisketcher import FakeBackend, Intent, PresetManager, SeedPlan, Studio

preset = "sdxl-canny-lite@1"
models = PresetManager()
plan = models.plan_install(preset)
print(plan.license_notice, plan.estimated_bytes, plan.items)

# Continue only after reviewing the plan and the upstream model licenses.
if not plan.installed:
    models.install(preset, confirm=True)

studio = Studio.from_preset(preset, device="auto", preset_manager=models)
prepared = studio.prepare("sketch.jpg")

study = studio.explore(
    prepared,
    intent=Intent(
        prompt="A friendly paper-cut character collection",
        profile="graphic_design",
        structure="balanced",
    ),
    outputs=4,
    seed_plan=SeedPlan.scout(4),
)

selected = study.pick(1)  # Stable zero-based index: the second candidate.
variations = studio.vary(
    selected,
    outputs=4,
    strength="subtle",
    locks=("structure",),
)
variations.export("design-study")
```

To test the API and lineage without downloading a model, construct
`Studio(FakeBackend(), preset="sdxl-canny-lite@1")`. The fake backend is for
deterministic tests and Guided Sample plumbing; it does not represent a creative
model result.

`study.pick(1)` uses the candidate’s stable index, not its visual rank. Inspect
the candidate ID and actual seed in the exported manifest before handing the
study off.

## Choose a first mode

- Start with **Guided Sample** to learn the flow with no model download once the
  canonical fixture status reports Ready.
- Use **Lite** for the lower-memory SDXL Canny preset.
- Use **Quality** when the additional local memory and download size are
  acceptable.

CPU live generation is intentionally disabled. CUDA is supported. Apple
Silicon MPS uses sequential generation and memory-saving behavior and is marked
experimental.

!!! info "Guided fixture"

    The canonical bundle contains its anonymous source, exact prepared input,
    Canny control, four real local SDXL directions, selected result, and a
    hash-verified study manifest. Guided Sample opens that fixture read-only and
    requires no model download.

## Next

- Understand the [design-lineage model](concepts/design-lineage.md).
- Read each stage in the [complete SDK workflow](sdk/workflow.md).
- Learn what [strict and compatible replay](sdk/export-replay.md) guarantee.
- Check [configuration](reference/configuration.md) and
  [troubleshooting](guides/troubleshooting.md) before a live model install.
