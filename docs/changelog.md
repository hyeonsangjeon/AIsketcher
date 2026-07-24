# Changelog

This page tracks user-visible package changes. PyPI is the source of truth for
whether a listed version is publicly available.

## 0.3.0

Version 0.3.0 modernizes the default local model path and makes Studio’s
download, refinement, cancellation, and recovery boundaries explicit.

### Added

- a T4-validated FLUX.2 Klein Edit backend and curated **Auto** route for new
  sketch-to-design, photo-led, and instruction-edit studies;
- an explained model choice in Simple, including cache state, pinned revisions,
  expected transfer, device guidance, and upstream license notices;
- an explicit, pinned local Korean→English helper that preserves the Korean
  original and records the prepared model prompt and translator revision,
  without Transformers fetching an unpinned background conversion PR;
- cooperative Stop behavior for generation and model/helper preparation,
  same-session job reconnection, and a recovery layer for an ended temporary
  Studio server;
- a refinement instruction composer and a model-preparation layer for attempts
  to refine the read-only Guided Sample.

### Changed

- `flux2-klein-edit@1` is the recommended/default live preset; SDXL Canny Lite
  and Quality remain versioned Legacy choices for manifest replay and
  intentional edge-conditioned work;
- the four Simple direction cards are larger and avoid an inner gallery
  scrollbar, while progress distinguishes elapsed time from an estimate;
- model preparation requires an explicit confirmation before downloading the
  selected image model or the Korean helper. Cancelling the layer performs no
  network access, and a retry reuses complete pinned cache entries;
- packaged Studio resolves the Korean helper below the same configured
  AIsketcher cache root as image models, so cache-only reuse does not depend on
  an unrelated global Hugging Face cache;
- FLUX.2 variation levels are applied through explicit deterministic edit
  instructions and recorded as an approximation, while model downloads observe
  Stop between curated file groups;
- release and deployment actions are pinned to reviewed full commit SHAs;
- current install examples use the lowercase `aisketcher` identifier and pin
  version 0.3.0;
- the README embedded in each release artifact remains the PyPI project
  description. Publishing the matching GitHub Release publishes that immutable
  description and package through Trusted Publishing.

### Removed

- the deprecated uppercase `AIsketcher.img2img` facade;
- the deprecated `aisketcher.modelPipe` compatibility module and uppercase
  top-level module packaging.

The historical Pocket Kingdom and heritage manifests keep their recorded SDXL
recipes, seeds, hashes, and AIsketcher 0.2 runtime provenance. See the detailed
[0.3.0 release notes](releases/0.3.0.md).

## 0.2.1

Version 0.2.1 makes the modern v0.2 package the default public PyPI experience
and aligns its presentation with the reviewed Studio workflow.

### Changed

- installation examples now use version-pinned PyPI commands for the core,
  Studio, local-generation, and combined optional dependency profiles;
- the README carried into PyPI metadata describes the current model-agnostic
  SDK rather than the historical AWS workflow;
- the English Studio documentation now leads with an actual local capture of
  the HPO-selected heritage direction: seed `6764547109648557242` with pinned
  preset `sdxl-canny-lite@1`;
- publishing a GitHub Release for a matching `v<version>` tag now starts the
  guarded PyPI workflow automatically. The job verifies the tag and package
  version, runs release tests and distribution scans, smoke-tests the wheel,
  and publishes through OIDC Trusted Publishing without a PyPI API token.

See the detailed [0.2.1 release notes](releases/0.2.1.md).

## 0.2.0

The v0.2 line rebuilds AIsketcher around a model-independent design study:

```text
prepare → explore → pick → vary → export → replay
```

### Added

- normalized sketch preparation with Canny diagnostics;
- explicit seed plans and ordered 1, 4, or 8 candidate studies;
- human selection, parent-child variation lineage, constraint locks, and
  technical recommendation badges;
- portable exports with manifests, file hashes, runtime evidence, strict
  replay, and compatible replay reports;
- curated, immutable SDXL Canny presets with review-before-download plans;
- a Simple/Advanced Gradio Studio and a hash-verified, model-free Guided
  Sample;
- a packaged `aisketcher init` / `aisketcher studio` CLI and versioned YAML
  settings ledger;
- the Pocket Kingdom canonical scout and variation study with complete replay
  evidence.

### Changed

- new code imports the lowercase `aisketcher` package;
- generation is separated from the SDK through a backend capability protocol;
- local model runtimes and Studio are optional installation layers;
- the Studio and its reviewed Guided Sample now ship in the wheel instead of
  requiring a repository checkout;
- seeds are presented as reproducibility coordinates, not universal quality
  scores.

### Removed

- AWS-specific translation, credential, and cloud execution paths;
- arbitrary model URL loading and remote-code execution in the Studio;
- implicit downloads during import or ordinary SDK construction.

### Security and privacy

- model presets pin repository revisions, allow-list fp16 components, require
  SafeTensors, and disable remote code;
- exports omit tokens, original upload names, absolute cache paths, and image
  metadata;
- the Studio binds to loopback, disables public share links, limits uploads,
  and serializes generation by default.

### Compatibility

The `AIsketcher.img2img` facade remains deprecated for the 0.2 line and requires
a caller-provided compatible pipeline. It is scheduled for removal in 0.3.0.
Cloud and credential arguments fail with a migration error instead of being
accepted or logged.

See the detailed [0.2.0 release notes](releases/0.2.0.md) and
[0.0.x migration guide](guides/migration.md).
