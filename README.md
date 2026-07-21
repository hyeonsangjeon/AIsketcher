# AIsketcher

[![PyPI version](https://img.shields.io/pypi/v/AIsketcher.svg)](https://pypi.org/project/AIsketcher/)
[![GitHub tag](https://img.shields.io/github/v/tag/hyeonsangjeon/AIsketcher?sort=semver&label=tag)](https://github.com/hyeonsangjeon/AIsketcher/releases/latest)

**Turn one sketch into a traceable family of design directions.**

AIsketcher is a model-agnostic Python SDK for structure-guided visual
exploration. It prepares a sketch, explores several seeded candidates, records
the direction you pick, creates controlled variations, and exports a replayable
manifest. It is designed for product designers, graphic designers, and
sketchers who need more than a one-off image.

<p align="center">
  <a href="https://hyeonsangjeon.github.io/AIsketcher/canonical-sample/">
    <img src="https://raw.githubusercontent.com/hyeonsangjeon/AIsketcher/main/docs/assets/aisketcher-social-preview-github.jpg" width="1200" alt="Pocket Kingdom paper-art hero concept for AIsketcher">
  </a>
</p>
<p align="center"><sub>Pocket Kingdom hero concept · marketing artwork, not an SDK execution claim · select to inspect the real local source, scout, variations, and replay evidence</sub></p>

[Documentation](https://hyeonsangjeon.github.io/AIsketcher/) ·
[한국어 빠른 시작](https://hyeonsangjeon.github.io/AIsketcher/ko/quickstart/) ·
[PyPI](https://pypi.org/project/AIsketcher/0.2.1/) ·
[Migration from 0.0.x](https://hyeonsangjeon.github.io/AIsketcher/guides/migration/)

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

AIsketcher 0.2.1 is published on
[PyPI](https://pypi.org/project/AIsketcher/0.2.1/). The brand is **AIsketcher**;
the PyPI install identifier is lowercase `aisketcher`, matching the Python
import and CLI:

```bash
pip install aisketcher
```

Pin the public release when you need a reproducible install:

```bash
python -m pip install "aisketcher==0.2.1"
```

The lightweight SDK does not install Torch, Diffusers, model weights, or the
Gradio runtime. Studio code and its Guided Sample are packaged; the `demo`
extra adds the runtime needed to launch the UI.

For development from this repository:

```bash
python -m pip install -e ".[dev]"
```

Install optional local generation or the Studio separately:

```bash
python -m pip install "aisketcher[demo]==0.2.1"
python -m pip install "aisketcher[local,demo]==0.2.1"
```

The complete model-free first run is one line:

```bash
python -m pip install "aisketcher[demo]==0.2.1" && aisketcher init && aisketcher studio
```

Model downloads happen only after you explicitly choose a local preset. Guided
Sample mode does not require a model or network connection: this repository
includes a reviewed four-direction fixture with matching hashes and a real
`aisketcher.manifest/v1` manifest.

Guided Sample works on CPU. Live local generation requires CUDA or experimental
Apple Silicon MPS; CPU generation is disabled by default so users do not fetch
7–9 GB of model data only to discover an unsupported runtime.

## Studio

The packaged Gradio Studio is the fastest way to understand the workflow. This
is the actual English Simple view with a documentation-only heritage study
open:

<p align="center">
  <a href="https://raw.githubusercontent.com/hyeonsangjeon/AIsketcher/main/docs/assets/aisketcher-studio-heritage-fixed-seed-en.jpg">
    <img src="https://raw.githubusercontent.com/hyeonsangjeon/AIsketcher/main/docs/assets/aisketcher-studio-heritage-fixed-seed-en.jpg" width="1200" alt="AIsketcher Studio English Simple view showing a privacy-reviewed family sketch, its selected result, four deterministic directions, and manifest-backed settings">
  </a>
</p>
<p align="center"><sub>Actual local Studio · HPO-selected historical seed 6764547109648557242 · pinned sdxl-canny-lite@1 · select the image to open it full size</sub></p>

This privacy-reviewed family-sketch fixture is used only for documentation and
is separate from the bundled Pocket Kingdom Guided Sample. Its authenticated
manifest fills the visible prompt, profile, and structure controls, and fixes
the HPO-selected direction to seed `6764547109648557242`. Twelve new candidates
were reviewed in four bounded rounds before this direction was selected. Model
weights were already local, so the capture caused no model download or image
upload.

- **Simple** asks for a sketch, a one-sentence brief, a work type, and a
  Loose/Balanced/Faithful structure choice.
- **Advanced** exposes model, Canny, generation, seed, variation, export, and
  replay controls without discarding the Simple session.

Launch it after installing the `demo` extra:

```bash
aisketcher init  # First run only; omit when settings already exist.
aisketcher studio
```

Start with Guided Sample when no model is installed. See the
[Studio guide](https://hyeonsangjeon.github.io/AIsketcher/studio/simple-advanced/)
and [configuration reference](https://hyeonsangjeon.github.io/AIsketcher/reference/configuration/)
for Advanced controls, local-only defaults, versioned YAML, and project
overrides.

## Python workflow

```python
from aisketcher import FakeBackend, Intent, PresetManager, SeedPlan, Studio

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

choice = study.pick(1)  # Stable zero-based index: the second candidate.
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

The exported manifest contains the resolved recipe and actual seeds. Built-in
exports re-encode images without EXIF, omit source filenames, and allowlist
backend metadata. Do not put secrets or private paths in prompts, profiles, or
custom backend identifiers. See the
[complete SDK workflow](https://hyeonsangjeon.github.io/AIsketcher/sdk/workflow/)
and [privacy model](https://hyeonsangjeon.github.io/AIsketcher/guides/privacy/).

## Canonical example

Pocket Kingdom is the canonical `source → control → scout four → pick → vary
four → final → export` example. Its anonymized source, exact prepared input,
Canny control, four real local SDXL directions, four structure-locked
variations, human selections, seeds, lineage, technical scores, and hashes are
checked in with replayable manifests. The manual presents the final result
alongside every alternative instead of substituting private reference art.

Artwork is **not** licensed under MIT. Read the
[artwork notice][artwork-license]
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
[MIT License][mit-license].
Images, drawings, generated derivatives, and other artwork are excluded; see
the [artwork notice][artwork-license].

[mit-license]: https://github.com/hyeonsangjeon/AIsketcher/blob/main/LICENSE
[artwork-license]: https://github.com/hyeonsangjeon/AIsketcher/blob/main/ARTWORK_LICENSE.md
