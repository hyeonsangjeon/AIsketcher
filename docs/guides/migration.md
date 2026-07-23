# Migrate from 0.0.x

The original package wrapped one Canny ControlNet call and previously included
cloud-specific prompt translation. Version 0.3.0 completes that migration: the
cloud and credential path, uppercase compatibility module, and `modelPipe`
module are no longer packaged. The replacement is a model-independent study.

## Import and API

| Before | v0.3 |
| --- | --- |
| `import AIsketcher` | `import aisketcher` |
| `img2img(...)` | `prepare()` then `explore()` |
| returned image tuple | `PreparedSketch` and `Study` |
| one seed argument | explicit `SeedPlan` with per-candidate seeds |
| result saved manually | `export()` with manifest and hashes |
| pipeline assumed | backend capabilities are resolved and reported |

Install and import the replacement using the same lowercase identifier:

```bash
python -m pip install "aisketcher==0.3.0"
```

```python
from aisketcher import Intent, SeedPlan, Studio
```

## Compatibility removal in 0.3.0

The v0.2 deprecation window is closed. `import AIsketcher`,
`AIsketcher.img2img`, and `aisketcher.modelPipe` are absent from the v0.3
wheel. Update callers before upgrading; importing those names is now expected
to fail instead of emitting a warning.

Cloud translation and credential arguments are intentionally unsupported. If
an old integration still needs them, remove that configuration rather than
passing it through a compatibility facade. Korean prompt preparation is local,
pinned, and explicit: the Studio shows the helper plan and requires confirmation
before downloading it.

## Suggested migration

1. Remove credentials and cloud translation configuration from application
   settings and deployment secrets.
2. Change every import to lowercase `aisketcher`; remove imports of
   `AIsketcher` and `aisketcher.modelPipe`.
3. Replace `img2img(...)` and tuple handling with `Studio.prepare()`,
   `Studio.explore()`, `PreparedSketch`, `Study`, and `Candidate`.
4. Start new work with Auto/`flux2-klein-edit@1`. Keep an SDXL Canny preset only
   when replaying a recorded legacy manifest or intentionally using edge
   conditioning.
5. Persist exported studies instead of isolated result images.
6. Add strict replay to reviewed examples and compatible replay only where
   substitutions are acceptable.

This v0.3 source migration changes the current tree and new distribution only;
it does not itself rewrite reachable Git commits, remote branches, forks, or
caches, and it does not yank or delete an older PyPI release. Any history
rewrite, force-push, or PyPI lifecycle action is a separate destructive
maintainer decision that requires coordinated backups, branch/tag review, and
explicit approval.

Historical 0.2.x release notes and sample manifests continue to describe the
artifacts that existed at those versions. Do not rewrite their versions, SDXL
recipes, seeds, or hashes while migrating current code.
