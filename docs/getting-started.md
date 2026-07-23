# Quick start

Version 0.3.0 is published on
[PyPI](https://pypi.org/project/AIsketcher/0.3.0/). The commands below pin the
public release so a later package update cannot silently change the setup.
PyPI names are case-insensitive; install `aisketcher` in lowercase to match the
Python import and command-line interface.

For the latest release:

```bash
pip install aisketcher
```

Use one of the pinned profiles below when you need a reproducible environment.

## Install the layer you need

=== "Core SDK"

    ```bash
    python -m pip install "aisketcher==0.3.0"
    ```

    This installs preparation, recipe, lineage, export, and replay types. It
    does not install a generation runtime.

=== "Local generation"

    ```bash
    python -m pip install "aisketcher[local]==0.3.0"
    ```

    Local presets download their pinned model files only after confirmation.

=== "Studio"

    ```bash
    python -m pip install "aisketcher[demo]==0.3.0"
    aisketcher init
    aisketcher studio
    ```

    `init` writes a versioned per-user settings file and never downloads a
    model. Guided Sample then works from its bundled, hash-verified fixture
    without a network or GPU. Install `aisketcher[local,demo]` instead for live
    local generation. New settings select the T4-validated FLUX.2 Klein Edit
    preset; the weights are still downloaded only after a separate review and
    confirmation.

For the complete model-free first run:

```bash
python -m pip install "aisketcher[demo]==0.3.0" && aisketcher init && aisketcher studio
```

`aisketcher init` protects an existing settings file. Use `--path` for a
project-specific file or `--force` only after reviewing what will be replaced.
See the [configuration reference](reference/configuration.md) for resolution
order and every supported key.

## Run a study

```python
from aisketcher import FakeBackend, Intent, PresetManager, SeedPlan, Studio

preset = "flux2-klein-edit@1"
models = PresetManager()
plan = models.plan_install(preset)
print(plan.license_notice, plan.estimated_bytes, plan.download_bytes, plan.items)

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
model result. Its explicit SDXL preset name keeps this test fixture separate
from the default live FLUX.2 path.

`study.pick(1)` uses the candidate’s stable index, not its visual rank. Inspect
the candidate ID and actual seed in the exported manifest before handing the
study off.

## Choose a first mode

- Start with **Guided Sample** to learn the flow with no model download once the
  canonical fixture status reports Ready.
- Use **FLUX.2 Klein Edit** for new sketch rendering, photo restyling, and
  instruction edits. It is the recommended/default live preset and was
  validated on a 16 GB NVIDIA T4.
- Use **SDXL Canny Lite** or **SDXL Canny Quality** only when an existing
  manifest or an explicit edge-conditioned workflow requires the legacy
  ControlNet path.

The default FLUX.2 profile requires CUDA for the supported interactive path.
CPU execution is not enabled by the packaged Studio. Apple Silicon MPS remains
experimental for the legacy SDXL path and is not the validated FLUX.2 default.

!!! info "Guided fixture"

    The canonical bundle contains its anonymous source, exact prepared input,
    Canny control, four real local SDXL directions, selected result, and a
    hash-verified study manifest. Guided Sample opens that fixture read-only and
    requires no model download. Its SDXL provenance is intentionally preserved;
    it does not redefine the model used for a new live study.

## What the first Studio run does

An uncached model selection opens a preparation step before network access.
Review the pinned repositories and revisions, expected transfer, cache
destination, device guidance, and upstream licenses, then select
**Review & prepare model**. That explicit confirmation prepares both the
selected image model and the pinned 315 MB Korean→English helper when either is
missing. Returning to the sample or leaving setup before pressing that button
starts no download.

**Stop** cancels generation cooperatively. Image-model preparation stops at the
next curated selected-file boundary; a tensor file already in transfer finishes
before the stop is observed. Complete marked image-model entries remain
available, incomplete unmarked image-model destinations are removed, and
**Retry** downloads only what is still missing. Korean-helper setup checks Stop
between tokenizer and model loading and reuses files for its pinned revision;
it does not use the image-model marker-and-cleanup format.
If the page reconnects to the same live browser session, Studio restores the
running or stopping job and its Stop control; refreshing by itself never
cancels GPU work. If the temporary server has ended, the recovery layer tells
you to reload only the latest Studio address.

Generation shows elapsed time and a separate estimate. For example,
`42.3 / 107.6 s` means 42.3 seconds have elapsed against a 107.6-second
estimate; it is not a timeout.

Studio keeps the user’s original creative brief distinct from the prompt sent
to the image model. English needs no translation; the FLUX.2 path can still add
its deterministic reference-edit constraints. Korean is translated with the
pinned local Korean-to-English adapter only after that helper has been
explicitly prepared. Both forms and the translator revision are recorded. If
the helper is unavailable, Studio stops before model/GPU work and preserves the
Korean original instead of silently sending unsupported text.

The result area shows four large direction cards without an inner scrollbar.
Refining a live result opens an additional-instruction field. Refining the
read-only Guided Sample opens a model-preparation layer rather than an error;
**Keep exploring the sample** closes it without changing the selected fixture.

The packaged Gradio app binds to `127.0.0.1` and is intended for one local user.
Browser sessions have isolated run state, while GPU work is serialized across
the process. It is not a ready-made multi-user or public deployment.

## Next

- Understand the [design-lineage model](concepts/design-lineage.md).
- Read each stage in the [complete SDK workflow](sdk/workflow.md).
- Learn what [strict and compatible replay](sdk/export-replay.md) guarantee.
- Check [configuration](reference/configuration.md) and
  [troubleshooting](guides/troubleshooting.md) before a live model install.
