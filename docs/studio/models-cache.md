# Models and cache

AIsketcher does not bundle model weights. Guided Sample requires none; local
generation uses pinned Hugging Face repositories after the user confirms a
preset’s license, files, cache destination, and estimated download size.

## Preset cards

| Studio option | Versioned preset | Intended use | Approximate first download |
| --- | --- | --- | ---: |
| Guided Sample | bundled fixture | learn the workflow | none |
| FLUX.2 Klein Edit — recommended | `flux2-klein-edit@1` | fast sketch rendering, photo restyling, and instruction edits on a 16 GB NVIDIA T4 | about 16.2 GB |
| SDXL Canny Lite — legacy | `sdxl-canny-lite@1` | replay existing SDXL manifests and lower-memory edge conditioning | about 7.3 GB |
| SDXL Canny Quality — legacy | `sdxl-canny@1` | replay existing full-ControlNet SDXL recipes | about 9.4 GB |
| Korean→English helper | pinned `facebook/m2m100_418M` adapter | protect recognized design terms, then prepare model-facing English while preserving Korean input | about 1.9 GB when absent |

FLUX.2 Klein Edit is the default for new live studies. It uses the uploaded
image as a reference and does not use Canny ControlNet. The two SDXL presets
remain available for explicit installation and replay when needed, but are
labeled Legacy rather than presented as the modern default.

Sizes cover only the pinned, allow-listed files at the revisions below. Studio
shows the model’s intended use, expected transfer, cache policy, device
expectation, and license confirmation; `plan_install()` exposes the exact
destinations for programmatic review before downloading.

The Studio preparation layer also shows the pinned Korean→English helper when
it is missing. **Review & prepare model** explicitly confirms the selected
image model and helper together. Leaving setup before pressing that button
performs no network access.

## Pinned repositories

The curated image presets and local translation helper use immutable revisions
of:

| Preset | Role | Repository | Revision |
| --- | --- | --- | --- |
| FLUX.2 Klein Edit | base/edit | `black-forest-labs/FLUX.2-klein-4B` | `e7b7dc27f91deacad38e78976d1f2b499d76a294` |
| FLUX.2 Klein Edit | decoder | `black-forest-labs/FLUX.2-small-decoder` | `a3efc24f613ef42d9428af62fdbd6f5fd8856c4a` |
| SDXL Canny legacy | base | `stabilityai/stable-diffusion-xl-base-1.0` | `462165984030d82259a11f4367a4eed129e94a7b` |
| SDXL Canny legacy | lite control | `diffusers/controlnet-canny-sdxl-1.0-small` | `edd85f64c5f87dfb6d73762949d9daca16389518` |
| SDXL Canny legacy | quality control | `diffusers/controlnet-canny-sdxl-1.0` | `eb115a19a10d14909256db740ed109532ab1483c` |
| Korean→English helper | translation | `facebook/m2m100_418M` | `55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636` |

Image-model loading is restricted to SafeTensors with remote code disabled. A
mutable branch name is not sufficient for a replayable preset. FLUX.2 Klein’s
pinned weights are Apache-2.0; the legacy SDXL components retain their pinned
upstream OpenRAIL terms. The Korean→English helper is MIT-licensed and uses its
pinned PyTorch state dictionary through the weights-only loader; its license is
shown separately from the selected image model's license during preparation.

FLUX.2 Klein Edit resolves to at most 1024 × 1024, four steps, guidance 1.0,
the Flow Match Euler scheduler, and reference-image control. The two SDXL
presets resolve to 1024 × 1024, 30 steps, guidance 5.0, UniPC, and Canny
ControlNet before an Advanced override. Structure strength resolves to 0.55
for Loose, 0.75 for Balanced, and 0.95 for Faithful. These values belong to
versioned presets and may change only under a new preset version.

## Cache behavior

Weights stay in AIsketcher’s platform user cache, outside the repository,
wheel, and exported study. A manifest records repository and immutable
revision, never an access token or the local cache’s absolute path.
The packaged Studio keeps image-model directories below `models/` and its
pinned Korean helper below `translation/` under the same resolved cache root.
It does not depend on an unrelated process-wide Hugging Face cache, so a helper
prepared through Studio remains available to that Studio in cache-only mode.

The cache root resolves in this order:

1. an explicit `PresetManager(cache_dir=...)` argument;
2. `cache_dir` loaded by the packaged Studio from its YAML configuration;
3. `AISKETCHER_CACHE_DIR`;
4. `$XDG_CACHE_HOME/aisketcher` when set;
5. `~/Library/Caches/AIsketcher` on macOS;
6. `~/.cache/aisketcher` on other supported platforms.

To keep Studio model files on another volume, set `cache_dir` in the per-user
configuration created by `aisketcher init`:

```yaml
schema_version: 1
cache_dir: "/Volumes/Models/AIsketcher"
```

For a direct Python integration or temporary shell override:

```bash
export AISKETCHER_CACHE_DIR=/Volumes/Models/AIsketcher
```

See [Configuration](../reference/configuration.md) for precedence and the
complete schema.

Removing a cache is an explicit user operation. The Studio reports what will be
removed and does not delete unrelated model caches.

The pinned M2M100 Korean→English revision publishes seven runtime files:
weights, model and generation configuration, and tokenizer vocabulary/config.
AIsketcher verifies the reviewed size and SHA-256 of every one before loading
the local snapshot, and disables Transformers’ optional background SafeTensors
conversion helper during that load. This prevents an independently moving
`refs/pr/*` conversion artifact from being fetched and keeps the reviewed
revision authoritative. The exact allow-listed first transfer is about 1.9 GB.
Any pre-existing process policy for that Transformers setting is restored
immediately afterward.

Before tokenization, a deterministic glossary replaces recognized Korean
visual-design terms with their reviewed English production vocabulary. Studio
then runs Korean→English translation locally and records the exact source text,
prepared English, helper ID, and immutable revision as separate prompt
provenance. The glossary is intentionally bounded and does not imply that every
Korean phrase will translate perfectly.

## First download, Stop, and retry

Selecting an uncached model does not immediately start network access. Review
the repositories, immutable revisions, expected transfer, destination,
license, and device guidance, then confirm the separate download action.
`allow_downloads: true` permits this flow but is never consent by itself.

During image-model preparation, **Stop** is checked between selected download
groups and between streamed SHA-256 chunks. It never turns a partial or
unverified tensor into a usable cache. Stopping a read-only integrity pass
leaves the existing files in place for a later retry. A failed fresh download
removes only that incomplete managed destination; it never removes another
installed preset or follows a symlink outside the managed cache.

The Korean helper uses the pinned Transformers cache below `translation/`
rather than AIsketcher image-model markers. Its cooperative Stop check occurs
at every reviewed file boundary, during streamed hashing, and between the
tokenizer and model loads. A later retry can reuse files already present for
the same immutable revision, but the image-model marker-and-cleanup guarantee
above does not apply to that cache format.

The completion marker uses the versioned `aisketcher-model-cache` schema and
records the repository, immutable revision, allowlist, hash policy, reviewed
artifact sizes and SHA-256 values, and the verified file-stat fingerprint. The
marker is provenance, not an authentication token: a fresh Studio process does
not trust it by itself. Policy v2 streams every exact curated runtime file —
configuration, scheduler, tokenizer, index, and LFS weight payloads — through
the reviewed SHA-256 policy before the backend may load it. A process-local
receipt then permits fast reuse only while the marker and every verified
file's size, modification/change time, device and inode remain unchanged.
Legacy markers are fully verified and upgraded; a forged marker, configuration
edit, or same-size tamper is rejected.

Opening Studio uses a non-verifying display plan so a multi-gigabyte hash pass
cannot hide the page behind a connection delay. Until **Review & prepare
model** finishes, an existing cache may therefore appear as **Not yet verified
· download if absent** and the displayed transfer is a conservative maximum,
not proof that it will be downloaded. On the validated T4, checking the
roughly 16.2 GB FLUX cache took up to about one minute with zero network
transfer. The same process reuses its receipt immediately afterward.

After a successful first download, later Studio sessions load the same pinned
files from the configured local cache; they do not redownload them merely
because the browser was refreshed.

Refreshing is not cancellation. If the same browser session reconnects while a
job is running or stopping, Studio restores that status and the Stop control.
If the temporary server itself ended, the connection-recovery layer directs the
user to the latest Studio address.

## Install from Python

Planning is read-only. Installation requires a separate confirmation after the
application has displayed the plan.

```python
from aisketcher import PresetManager

manager = PresetManager()
plan = manager.plan_install("flux2-klein-edit@1")

print(plan.license_notice)
print(plan.estimated_bytes, plan.download_bytes)
for item in plan.items:
    print(item.repo_id, item.revision, item.destination, item.installed)

# Call only after the user reviews the information above.
result = manager.install("flux2-klein-edit@1", confirm=True)
```

`PresetManager(allow_downloads=False)` is the required setting for a hosted or
read-only environment where an administrator, rather than a web user, manages
the model cache.

## Custom repositories

Arbitrary repository URLs, single checkpoint files, custom Python pipelines,
pickle weights, and `trust_remote_code` are outside the curated Studio path.
Implement a backend in Python when a controlled deployment needs another
model; do not expose a public URL field that turns the Studio into a
remote-code loader.
