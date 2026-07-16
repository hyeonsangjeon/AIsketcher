# AIsketcher

**Turn one sketch into a traceable family of design directions.**

AIsketcher is a model-agnostic Python SDK for structure-guided visual
exploration. It prepares a sketch, explores several seeded candidates, records
the direction you pick, creates controlled variations, and exports a replayable
manifest. It is designed for product designers, graphic designers, and
sketchers who need more than a one-off image.

> This repository documents AIsketcher 0.2.0. A source checkout is not proof of
> PyPI publication; verify the version on PyPI or install a reviewed wheel or
> pinned revision.

<p align="center">
  <a href="https://hyeonsangjeon.github.io/AIsketcher/canonical-sample/">
    <img src="https://raw.githubusercontent.com/hyeonsangjeon/AIsketcher/main/docs/assets/aisketcher-social-preview-github.jpg" width="1200" alt="Pocket Kingdom paper-art hero concept for AIsketcher">
  </a>
</p>
<p align="center"><sub>Pocket Kingdom hero concept · marketing artwork, not an SDK execution claim · select to inspect the real local source, scout, variations, and replay evidence</sub></p>

[Documentation](https://hyeonsangjeon.github.io/AIsketcher/) ·
[한국어 빠른 시작](https://github.com/hyeonsangjeon/AIsketcher/blob/main/docs/ko/quickstart.md) ·
[Migration from 0.0.x](https://github.com/hyeonsangjeon/AIsketcher/blob/main/docs/guides/migration.md)

## Why AIsketcher

- **Prepare with evidence:** normalize orientation and size, generate a control
  image, and inspect actionable structure diagnostics before spending GPU time.
- **Explore deliberately:** create 1, 4, or 8 candidates with an explicit seed
  plan instead of repeatedly changing an undocumented seed.
- **Pick and vary:** preserve the selected parent and make subtle, balanced, or
  bold variations while keeping chosen constraints locked.
- **Replay the handoff:** export inputs, controls, recipes, seeds, lineage,
  hashes, and runtime information as a portable study.
- **Bring your backend:** use the Diffusers adapter or implement the small
  backend protocol for another local or hosted image model.

## Install

The lightweight SDK does not install Torch, Diffusers, model weights, or the
web app.

```bash
python -m pip install "AIsketcher>=0.2,<0.3"
```

For development from this repository:

```bash
python -m pip install -e ".[dev]"
```

Install optional local generation or the Studio separately:

```bash
python -m pip install "AIsketcher[demo]>=0.2,<0.3"        # Guided Sample Studio
python -m pip install "AIsketcher[local,demo]>=0.2,<0.3"  # Local generation + Studio
```

The complete model-free first run is one line:

```bash
python -m pip install "AIsketcher[demo]>=0.2,<0.3" && aisketcher init && aisketcher studio
```

Model downloads happen only after you explicitly choose a local preset. Guided
Sample mode does not require a model or network connection: this repository
includes a reviewed four-direction fixture with matching hashes and a real
`aisketcher.manifest/v1` manifest.

## Python workflow

```python
from aisketcher import Intent, PresetManager, SeedPlan, Studio

preset = "sdxl-canny-lite@1"
models = PresetManager()
plan = models.plan_install(preset)
print(plan.license_notice, plan.estimated_bytes, plan.items)

# Run this only after reviewing the repositories, revisions, size, and licenses.
if not plan.installed:
    models.install(preset, confirm=True)

studio = Studio.from_preset(preset, device="auto", preset_manager=models)
prepared = studio.prepare("sketch.jpg")

study = studio.explore(
    prepared,
    intent=Intent(
        prompt="A playful paper-cut fantasy kingdom",
        profile="graphic_design",
        structure="balanced",
    ),
    outputs=4,
    seed_plan=SeedPlan.scout(4),
)

choice = study.pick(1)
variants = studio.vary(
    choice,
    outputs=4,
    strength="subtle",
    locks=("structure",),
)

variants.export("pocket-kingdom-run")
report = studio.replay(
    "pocket-kingdom-run/manifest.json",
    mode="strict",
)
```

For a network- and model-free workflow test, use
`Studio(FakeBackend(), preset="sdxl-canny-lite@1")`. Its images are deterministic
fixtures for code and CI, not model-generated creative results.

The exported manifest contains the resolved recipe and actual seeds. It never
contains access tokens, absolute local paths, original upload names, or EXIF
metadata. See the
[complete SDK workflow](https://github.com/hyeonsangjeon/AIsketcher/blob/main/docs/sdk/workflow.md)
and [privacy model](https://github.com/hyeonsangjeon/AIsketcher/blob/main/docs/guides/privacy.md).

## Studio

The included Gradio app has two views backed by the same recipe and selection:

- **Simple** asks for a sketch, a one-sentence brief, a work type, and a
  Loose/Balanced/Faithful structure choice.
- **Advanced** exposes model, Canny, generation, seed, variation, export, and
  replay controls without discarding the Simple session.

Once the optional demo is installed, launch it with:

```bash
aisketcher init
aisketcher studio
```

Start with Guided Sample when no model is installed. It opens the bundled,
verified Pocket Kingdom study in read-only mode. See the
[Studio guide](https://github.com/hyeonsangjeon/AIsketcher/blob/main/docs/studio/simple-advanced.md)
for its current availability and local-only defaults. The versioned YAML,
resolution order, cache settings, and project overrides are documented in the
[configuration reference](https://github.com/hyeonsangjeon/AIsketcher/blob/main/docs/reference/configuration.md).

Repository contributors can still use `python -m examples.studio_app.app` as a
thin compatibility launcher; installed users do not need a source checkout.

## Canonical example

Pocket Kingdom is the canonical `source → control → scout four → pick → vary
four → final → export` example. Its anonymized source, exact prepared input,
Canny control, four real local SDXL directions, four structure-locked
variations, human selections, seeds, lineage, technical scores, and hashes are
checked in with replayable manifests. The manual presents the final result
alongside every alternative instead of substituting private reference art.

Artwork is **not** licensed under MIT. Read the
[artwork notice](https://github.com/hyeonsangjeon/AIsketcher/blob/main/ARTWORK_LICENSE.md)
before using any image from this repository.

## Compatibility

The historical `AIsketcher.img2img` facade remains temporarily available in the
0.2 line and emits a deprecation warning. Cloud translation and credential
arguments from the earliest releases have been removed. Migrate to lowercase
`aisketcher` imports and the study workflow before 0.3.0.

## Development

```bash
python -m pip install -e ".[dev,docs]"
python -m pytest tests/core tests/docs
python -m pytest tests/app tests/test_config.py tests/test_cli.py
mkdocs build --strict
python -m build
python -m twine check dist/*
```

Network and GPU tests are opt-in. Normal CI uses a deterministic fake backend
and never downloads model weights.

## License

Source code and documentation text are licensed under the
[MIT License](https://github.com/hyeonsangjeon/AIsketcher/blob/main/LICENSE).
Images, drawings, generated derivatives, and other artwork are excluded; see
the [artwork notice](https://github.com/hyeonsangjeon/AIsketcher/blob/main/ARTWORK_LICENSE.md).
