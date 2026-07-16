# Migrate from 0.0.x

The original package wrapped one Canny ControlNet call and previously included
cloud-specific prompt translation. v0.2 removes the cloud and credential path
and replaces the one-shot call with a model-independent study.

## Import and API

| Before | v0.2 |
| --- | --- |
| `import AIsketcher` | `import aisketcher` |
| `img2img(...)` | `prepare()` then `explore()` |
| returned image tuple | `PreparedSketch` and `Study` |
| one seed argument | explicit `SeedPlan` with per-candidate seeds |
| result saved manually | `export()` with manifest and hashes |
| pipeline assumed | backend capabilities are resolved and reported |

## Compatibility window

The `AIsketcher.img2img` facade remains for the 0.2 line, emits a deprecation
warning, and requires a caller-provided compatible pipeline. It is scheduled
for removal in 0.3.0.

Cloud translation and credential arguments are intentionally unsupported. If
an old caller supplies one, the facade fails with a migration message instead
of accepting or logging the value.

## Suggested migration

1. Remove credentials and cloud translation configuration from application
   settings and deployment secrets.
2. Change new code to lowercase `aisketcher` imports.
3. Replace tuple handling with `PreparedSketch`, `Study`, and `Candidate`.
4. Choose a backend and versioned preset explicitly.
5. Persist exported studies instead of isolated result images.
6. Add strict replay to reviewed examples and compatible replay only where
   substitutions are acceptable.

This v0.2 source migration changes the current tree and new distribution only;
it does not itself rewrite reachable Git commits, remote branches, forks, or
caches, and it does not yank or delete an older PyPI release. Any history
rewrite, force-push, or PyPI lifecycle action is a separate destructive
maintainer decision that requires coordinated backups, branch/tag review, and
explicit approval.
