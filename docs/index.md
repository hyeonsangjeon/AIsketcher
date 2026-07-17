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
python -m pip install "AIsketcher[demo]==0.2.1" && aisketcher init && aisketcher studio
```

This launches Studio. Select **Try the guided sample** to open the bundled
fixture without downloading a model. Run `aisketcher init` only for the first
launch; omit it when settings already exist. The
[configuration reference](reference/configuration.md) explains the YAML and
project overrides.

[![AIsketcher Studio English Simple view with a privacy-reviewed family sketch, selected result, four deterministic directions, and manifest-backed settings](assets/aisketcher-studio-heritage-fixed-seed-en.jpg)](assets/aisketcher-studio-heritage-fixed-seed-en.jpg)

*Actual local English Studio with the HPO-selected direction fixed to seed
`6764547109648557242` and pinned `sdxl-canny-lite@1`. Select the image to
inspect the full-size interface.*

The documentation-only fixture displays its prompt, profile, and structure
directly from the authenticated manifest. Twelve new candidates were reviewed
in four bounded HPO rounds before the selected direction was captured. It is
separate from the bundled Pocket Kingdom Guided Sample. Model weights were
already local, so this capture required no model download or image upload.

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
download weights or require a GPU. Pocket Kingdom includes its anonymous source,
exact prepared input and Canny control, four real locally generated directions,
human selection, and replay manifest. The [sample page](canonical-sample.md)
shows the input, all seeds, result, and evidence together.

!!! warning "Artwork has a separate license"

    Code and documentation text are MIT licensed. Drawings and sample images
    are not. Read the project’s
    [artwork notice](https://github.com/hyeonsangjeon/AIsketcher/blob/main/ARTWORK_LICENSE.md)
    before using a visual asset.

## Release status

Version 0.2.1 is available from
[PyPI](https://pypi.org/project/AIsketcher/0.2.1/). Its package description is
built from this repository's README, so publishing the tagged release updates
the PyPI introduction together with the installable artifacts. See the
[changelog](changelog.md) and [0.2.1 release notes](releases/0.2.1.md) for the
current boundary.
