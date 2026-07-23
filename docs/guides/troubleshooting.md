# Troubleshooting

Start with the smallest layer that reproduces the problem. Guided Sample checks
the Studio and reviewed fixture without a model; `FakeBackend` checks the SDK,
lineage, export, and replay without Torch; a live preset adds the local runtime
and model cache.

## The Studio command is missing

Install the optional Studio dependencies in the same Python environment as
AIsketcher. Confirm which executable the shell resolves before reinstalling.

```bash
python -m pip show aisketcher gradio
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

If the sample opens but **Refine this direction** shows a model-preparation
layer, nothing has failed. The bundled sample is intentionally read-only.
Choose the recommended live model to start the confirmed download flow, or
choose **Keep exploring the sample** to close the layer without changing it.

## A preset is not installed

`ModelUnavailableError` means at least one pinned model component is missing or
failed validation. Planning is read-only and shows the exact repositories,
revisions, destinations, and expected transfer before any download.

```python
from aisketcher import PresetManager

manager = PresetManager()
plan = manager.plan_install("flux2-klein-edit@1")
print(plan.cache_dir, plan.download_bytes, plan.items)

# Continue only after reviewing the repositories and upstream licenses.
manager.install("flux2-klein-edit@1", confirm=True)
```

If installation reports an optional-dependency error, install the local
runtime first:

```bash
python -m pip install "aisketcher[local]==0.3.0"
```

Do not copy a partial Hugging Face cache into AIsketcher's managed directory.
The preset manager requires its own marker plus every pinned SafeTensors
component and rejects unsafe checkpoint suffixes.

FLUX.2 Klein Edit is the recommended/default preset for new live studies.
`sdxl-canny-lite@1` and `sdxl-canny@1` are Legacy choices for existing
manifests or explicit Canny ControlNet work; the historical Guided Sample
continues to identify its recorded SDXL recipe.

## A model download was stopped or failed

Image-model preparation checks cancellation at curated selected-file
boundaries. A large file that is already transferring may need to reach that
boundary before the job reports Stopped.

- A complete pinned image-model component with its valid AIsketcher marker
  remains cached.
- An incomplete image-model destination without a valid marker is removed.
- Other installed models are not deleted.
- Select the model again and confirm the rebuilt plan; the retry skips valid marked
  components and downloads only missing files.

The Korean helper uses a pinned Transformers cache rather than those
AIsketcher image-model markers. Stop is checked between tokenizer and model
load boundaries; a retry may reuse files already cached for the same immutable
revision, but does not use the marker-and-cleanup contract above.

Refreshing the browser is not a reliable cancellation mechanism. Use
**Stop** so the active session token reaches the installer or generation
backend. If the page reconnects to the same browser session, Studio restores
the still-running or stopping job and exposes Stop again.

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

## The Korean prompt cannot be prepared

Studio keeps the Korean original separate from the English model-facing
prompt. It does not assume that an image model understands Korean. If the
pinned local Korean-to-English adapter is unavailable, Studio stops before
loading the image model or using the GPU and preserves the original text.

Install the `translate` extra (or the broader `local` extra), then use the model
preparation layer. It shows the pinned helper revision, roughly 315 MB transfer,
cache reuse, and license alongside the selected image model.
**Review & prepare model** explicitly confirms both preparations; leaving setup
before pressing it performs no download. As an alternative, enter a reviewed
English prompt. Do not work around the message by placing credentials or an
unreviewed remote translation URL in configuration.

## Live generation is disabled on CPU

The supported interactive FLUX.2 Klein path requires CUDA and was validated on
a 16 GB NVIDIA T4. CPU live generation is intentionally disabled because the
latency and memory cost do not fit the Studio contract. Use Guided Sample for
the model-free tour, `FakeBackend` for tests, or run a custom backend with its
own deployment policy.

## FLUX.2 is selected on Apple Silicon

The packaged FLUX.2 Klein backend is not validated for MPS. Select a supported
CUDA device for the recommended profile. The Legacy SDXL backend retains
experimental MPS support and generates candidates sequentially; use it only
when that compatibility path is intentional.

After an accelerator out-of-memory failure, Studio evicts the failed runtime
and clears available accelerator caches so the next request can construct a
clean pipeline. Reduce output count or image size before retrying. A seed and
recipe remain recorded, but another runtime is not guaranteed to reproduce
identical pixels.

## Generation looks stuck or shows `42.3 / 107.6 s`

The first number is elapsed time and the second is an estimate based on the
current phase; the estimate is not a timeout. First use can include model
loading, and four outputs are generated sequentially to bound GPU memory.

While work is active, **Stop** appears. Queued work is removed immediately.
Active generation stops cooperatively at a denoising-step or output boundary,
so a provider callback may take a moment to unwind. Completed candidates remain
visible. A duplicate request from the same browser session is rejected until
the first backend callback has fully stopped. Reconnecting that browser session
restores the active or stopping status; refreshing alone does not cancel the
backend.

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

The packaged Studio is a local, single-user MVP. Its safety defaults bind to
`127.0.0.1`, disable public share links, keep run state in session-isolated
workspaces, and serialize GPU work process-wide. A browser refresh does not
guarantee that an already-running server job is cancelled.

Do not change the bind address for a multi-user or internet deployment without
authentication, durable job ownership, upload retention, rate limiting,
administrator-only model installation, and a separate security review.

If an old temporary Gradio share page reports an HTML/JSON parse error or
`Unexpected token '<'`, treat the link as expired or restarted. Use the latest
Studio address; the session-ended layer is connection guidance, not a model
failure.

## Report a reproducible problem

Include the AIsketcher version, platform, device, preset name, exported
manifest, and the full exception. Remove private images and tokens first. Do
not paste a model cache path when a repository ID and immutable revision are
enough to reproduce the setup.
