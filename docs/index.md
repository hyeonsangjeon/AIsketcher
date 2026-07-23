# One sketch. A traceable family of directions.

AIsketcher is a model-agnostic Python SDK for structured visual exploration,
controlled variation, and reproducible creative handoff.

It treats image generation as a design study:

```text
prepare → explore → pick → vary → export → replay
```

The package keeps this workflow separate from any particular checkpoint or
cloud provider. A backend resolves a model-independent intent into capabilities
it can actually support; the resulting recipe, seeds, artifacts, and lineage
are then recorded together.

[Get started](getting-started.md){ .md-button .md-button--primary }
[한국어 빠른 시작](ko/quickstart.md){ .md-button }
[See the Studio](studio/simple-advanced.md){ .md-button }

<a class="hero-art" href="canonical-sample/">
  <img src="assets/aisketcher-social-preview-github.jpg" alt="Pocket Kingdom paper-art hero concept for AIsketcher">
</a>
<p class="hero-caption">
  Pocket Kingdom hero concept — marketing artwork, not an SDK execution claim.
  Open the canonical sample for the real local inputs, candidates, seeds,
  selections, and replay manifests.
</p>

```bash
python -m pip install "aisketcher[demo]==0.3.0" && aisketcher init && aisketcher studio
```

The install identifier, Python import, and CLI are all lowercase
`aisketcher`. This launches Studio. Select **Try the guided sample** to open the
bundled fixture without downloading a model. Its results are read-only:
**Refine this direction** opens the model-preparation layer rather than an
error, and **Keep exploring the sample** closes it without changing the
fixture. Run `aisketcher init` only for the first launch; omit it when settings
already exist. The
[configuration reference](reference/configuration.md) explains the YAML and
project overrides.

[![AIsketcher Studio English Simple view with a privacy-reviewed family sketch, selected result, four deterministic directions, and manifest-backed settings](assets/aisketcher-studio-heritage-fixed-seed-en.jpg)](assets/aisketcher-studio-heritage-fixed-seed-en.jpg)

*Actual local English Studio with the bundled HPO Guided Sample fixed to seed
`6764547109648557242` and pinned historical `sdxl-canny-lite@1` provenance.
This legacy fixture is not the live default. Select the image to inspect the
full-size interface.*

The bundled, privacy-reviewed fixture displays its prompt, profile, and
structure directly from the authenticated manifest. Twelve new candidates were
reviewed in four bounded HPO rounds before the selected direction was captured.
It opens without model weights, network access, or an image upload. Pocket
Kingdom remains a separate documentation-only canonical lineage example.
New live studies use **Auto**, which selects the recommended T4-validated
FLUX.2 Klein Edit path. SDXL Canny remains available for legacy replay or
intentional edge-conditioned work.

## Built for design decisions

| Stage | Design question | Recorded evidence |
| --- | --- | --- |
| Prepare | Is the sketch usable as structure? | normalized source, control, diagnostics |
| Explore | Which directions are worth seeing? | intent, resolved recipe, candidate seeds |
| Pick | Which candidate becomes the parent? | selection and technical badges |
| Vary | What may change and what stays locked? | parent ID, strength, constraint locks |
| Export | Can another person inspect the work? | images, contact sheet, manifest, hashes |
| Replay | Can the run be reconstructed honestly? | model revision, drift report, runtime |

## What it does not promise

AIsketcher does not claim that one seed is universally “best,” that a model is
deterministic across every hardware stack, or that a technical score determines
the most beautiful result. Recommendation badges describe observable structure
and diversity properties; creative selection remains with the designer.

## Try without a model

Studio Guided Sample uses a reviewed, hash-verified local fixture. It does not
download weights or require a GPU. The bundled v0.3 fixture is the
privacy-reviewed HPO hero study with selected seed `6764547109648557242`.
Pocket Kingdom is a separate [documentation-only canonical lineage
example](canonical-sample.md) that shows input, seeds, results, variation, and
evidence together. Preparing a live model is a separate, confirmed action that
also shows the pinned Korean→English helper when it is missing; closing the
layer performs no download.

!!! warning "Artwork has a separate license"

    Code and documentation text are MIT licensed. Drawings and sample images
    are not. Read the project’s
    [artwork notice](https://github.com/hyeonsangjeon/AIsketcher/blob/main/ARTWORK_LICENSE.md)
    before using a visual asset.

## Release status

Version 0.3.0 is available from
[PyPI](https://pypi.org/project/AIsketcher/0.3.0/). Its package description is
the README embedded in the release artifact, so publishing the tagged GitHub
Release updates the PyPI introduction together with the wheel and source
archive. Editing `main` or an existing Release does not rewrite an immutable
PyPI version. The documentation site is deployed separately by the current
manual-only Pages workflow. See the [changelog](changelog.md) and
[0.3.0 release notes](releases/0.3.0.md) for the current boundary.
