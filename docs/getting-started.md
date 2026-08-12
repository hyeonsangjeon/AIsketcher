# Quick start

PyPI names are case-insensitive, but the install identifier, Python import,
and command are shown consistently as lowercase `aisketcher`.

## 1. Open a real study with no model

```bash
python -m pip install "aisketcher==0.4.0"
aisketcher try
```

This is the shortest useful first run. It opens a bilingual local tour of the
bundled, hash-verified study and requires no GPU, internet connection, Torch, Diffusers, or
Gradio. Select each direction to inspect its seed and recorded evidence. Press
`Ctrl+C` when finished.

## 2. Choose the layer you need

=== "Core SDK"

    ```bash
    python -m pip install "aisketcher==0.4.0"
    ```

    Preparation, studies, lineage, export, replay, and the zero-download tour.

=== "Studio"

    ```bash
    python -m pip install "aisketcher[demo]==0.4.0"
    aisketcher init
    aisketcher studio
    ```

    Adds Gradio and the full local interface. The Guided Study still needs no
    model.

=== "Local generation"

    ```bash
    python -m pip install "aisketcher[local,demo]==0.4.0"
    aisketcher init
    aisketcher studio
    ```

    Adds Torch, Diffusers, and the pinned local runtimes. Model files are not
    bundled and are downloaded only after review and confirmation.

`aisketcher init` creates a versioned YAML settings ledger and protects an
existing file. Use `--path` for a project file or `--force` only after reviewing
what will be replaced. See [Configuration](reference/configuration.md).

## 3. Choose a concrete model role

- **Fast Edit · FLUX.2 Klein** is for photo restyling, flexible sketch
  interpretation, and instruction edits. It is four-step and T4-validated, but
  it is reference-image editing—not Canny—and cannot promise exact line locks.
- **Structure Lock · SDXL Canny Lite** is the lower-memory legacy fallback when
  preserving line/Canny structure matters more than modern edit quality.
- **Structure Lock+ · SDXL Canny** uses the full legacy ControlNet.

The former `Auto` label did not classify inputs: every route selected FLUX.2.
Version 0.4 therefore exposes the concrete model instead. Modern structure and
Pro candidates are tracked in [Choose a model](models/choosing-a-model.md) and
must pass the documented benchmark before promotion.

Before a model transfer, the packaged CLI checks the configured device,
available CUDA VRAM, and free cache space. Unsupported CPU/MPS combinations
stop before a multi-GB download. English setup downloads only the image model;
Korean Studio also prepares its separately pinned Korean→English helper.

## 4. Run a study in Python

```python
from aisketcher import Intent, PresetManager, SeedPlan, Studio

preset = "flux2-klein-edit@1"
models = PresetManager()
plan = models.plan_install(preset)
print(plan.download_bytes, plan.items, plan.license_notice)

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

selected = study.pick(1)
variations = studio.vary(
    selected,
    outputs=4,
    strength="subtle",
    locks=("structure",),
)
variations.export("design-study")
```

For a network- and model-free API test, use
`Studio(FakeBackend(), preset="sdxl-canny-lite@1")`. The fake backend is a
deterministic test double, not a creative model result.

## First-run behavior

- Guided Study and `aisketcher try` do not download or generate anything.
- Preparing a live model shows repositories, immutable revisions, transfer
  size, cache destination, and licenses first.
- A new process may need to verify cached model hashes before loading them.
- **Stop** cancels queued work immediately and running work at the next safe
  backend or file boundary. Refreshing the browser does not cancel GPU work.
- `42.3 / 107.6 s` means elapsed time versus an estimate, not a timeout.
- The packaged Studio binds to `127.0.0.1` and is a local single-user tool, not
  a public multi-user service.

## Next

- Learn the [design-lineage model](concepts/design-lineage.md).
- Read the [complete SDK workflow](sdk/workflow.md).
- Understand [strict and compatible replay](sdk/export-replay.md).
- Check [troubleshooting](guides/troubleshooting.md) before a live model setup.
