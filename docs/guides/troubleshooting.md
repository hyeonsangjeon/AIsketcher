# Troubleshooting

Start with the smallest layer that reproduces the problem. Guided Sample checks
the Studio and reviewed fixture without a model; `FakeBackend` checks the SDK,
lineage, export, and replay without Torch; a live preset adds the local runtime
and model cache.

## The Studio command is missing

Install the optional Studio dependencies in the same Python environment as
AIsketcher. Confirm which executable the shell resolves before reinstalling.

```bash
python -m pip show AIsketcher gradio
command -v aisketcher
```

An editable repository checkout can still run the example directly:

```bash
python -m pip install -e ".[demo]"
python -m examples.studio_app.app
```

## Guided Sample is unavailable

Guided Sample is read-only and hash verified. It refuses to open if its
manifest is absent or if a referenced image no longer matches the recorded
SHA-256 value.

- In a checkout, restore the complete `docs/assets/pocket-kingdom/` bundle from
  the same revision instead of replacing individual files.
- In an installed package, reinstall the same AIsketcher version.
- Do not point the fixture at private source artwork or an arbitrary generation
  result.

## A preset is not installed

`ModelUnavailableError` means at least one pinned model component is missing or
failed validation. Planning is read-only and shows the exact repositories,
revisions, destinations, and expected transfer before any download.

```python
from aisketcher import PresetManager

manager = PresetManager()
plan = manager.plan_install("sdxl-canny-lite@1")
print(plan.cache_dir, plan.download_bytes, plan.items)

# Continue only after reviewing the repositories and upstream licenses.
manager.install("sdxl-canny-lite@1", confirm=True)
```

If installation reports an optional-dependency error, install the local
runtime first:

```bash
python -m pip install "AIsketcher[local]>=0.2,<0.3"
```

Do not copy a partial Hugging Face cache into AIsketcher's managed directory.
The preset manager requires its own marker plus every pinned SafeTensors
component and rejects unsafe checkpoint suffixes.

## The cache is not where expected

The packaged Studio loads YAML first and passes a non-null configured
`cache_dir` to its `PresetManager`. For a direct `PresetManager` integration,
the destination is selected in this order:

1. the `cache_dir` passed to `PresetManager`;
2. `AISKETCHER_CACHE_DIR`;
3. `$XDG_CACHE_HOME/aisketcher` when `XDG_CACHE_HOME` is set;
4. `~/Library/Caches/AIsketcher` on macOS;
5. `~/.cache/aisketcher` on other supported platforms.

Inspect `plan.cache_dir` before approving an installation. Changing the cache
setting does not move or delete an old cache. Cache removal is always an
explicit operation; do not delete a shared directory without reviewing the
reported destination.

## Live generation is disabled on CPU

The v0.2 local SDXL presets require CUDA or Apple Silicon MPS. CPU live
generation is intentionally disabled because the interactive latency and
memory cost do not fit the Studio contract. Use Guided Sample for the
model-free tour, `FakeBackend` for tests, or run a custom backend with its own
deployment policy.

## Apple Silicon runs out of memory

MPS support is experimental and generates candidates sequentially. Start with
the Lite preset, four outputs, and no other large GPU workloads. Restart the
process after an out-of-memory failure so a partially initialized pipeline is
not reused. A seed and recipe remain recorded, but another runtime is not
guaranteed to reproduce identical pixels.

## An upload is rejected

Studio accepts common raster images up to 20 MB and 50 megapixels. Re-export an
oversized input as PNG or JPEG at a practical working resolution. Preparation
corrects EXIF orientation and strips metadata from normalized artifacts; it
does not remove identifying content visible in the pixels.

## Replay stops with drift

Strict replay is expected to stop when an input hash, preset, immutable model
revision, or resolved recipe differs. Treat that as evidence, not as a cache
error. Restore the recorded resources or choose compatible replay and review
every substitution in `report.drift`.

```python
report = studio.replay("design-study/manifest.json", mode="compatible")
print(report.replayed)
print(report.drift)
```

Compatible replay creates a new lineage event. It does not certify that the
new pixels equal the original artifact.

## The interface should not be public

The packaged Studio is a local MVP. Its safety defaults bind to `127.0.0.1`,
disable public share links, and serialize generation. Do not change the bind
address for an internet deployment without authentication, upload retention,
rate limiting, administrator-only model installation, and a separate security
review.

## Report a reproducible problem

Include the AIsketcher version, platform, device, preset name, exported
manifest, and the full exception. Remove private images and tokens first. Do
not paste a model cache path when a repository ID and immutable revision are
enough to reproduce the setup.
