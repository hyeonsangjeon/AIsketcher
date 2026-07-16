# Changelog

This page tracks user-visible package changes. PyPI is the source of truth for
whether a listed version is publicly available.

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
