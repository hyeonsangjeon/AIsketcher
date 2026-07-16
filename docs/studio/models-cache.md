# Models and cache

AIsketcher does not bundle model weights. Guided Sample requires none; local
generation uses pinned Hugging Face repositories after the user confirms a
preset’s license, files, cache destination, and estimated download size.

## Preset cards

| Preset | Intended use | Approximate first download |
| --- | --- | ---: |
| Guided Sample | learn the workflow | none |
| Lite | lower-memory SDXL Canny exploration | about 7.3 GB |
| Quality | full SDXL Canny exploration | about 9.4 GB |

Sizes are exact for the pinned fp16 component allow lists at the revisions
below. Studio shows the preset, approximate display size, cache policy, and
license confirmation; `plan_install()` exposes the complete file destinations
for programmatic review before downloading.

## Pinned repositories

The v0.2 preset catalog uses immutable revisions of:

| Role | Repository | Revision |
| --- | --- | --- |
| Base | `stabilityai/stable-diffusion-xl-base-1.0` | `462165984030d82259a11f4367a4eed129e94a7b` |
| Lite control | `diffusers/controlnet-canny-sdxl-1.0-small` | `edd85f64c5f87dfb6d73762949d9daca16389518` |
| Quality control | `diffusers/controlnet-canny-sdxl-1.0` | `eb115a19a10d14909256db740ed109532ab1483c` |

Loading is restricted to SafeTensors with remote code disabled. A mutable branch
name is not sufficient for a replayable preset.

Both v1 SDXL presets resolve to 1024 × 1024, 30 steps, guidance 5.0, and the
UniPC scheduler before an Advanced override. Structure strength resolves to
0.55 for Loose, 0.75 for Balanced, and 0.95 for Faithful. These values belong
to the versioned preset and may change only under a new preset version.

## Cache behavior

Weights stay in AIsketcher’s platform user cache, outside the repository,
wheel, and exported study. A manifest records repository and immutable
revision, never an access token or the local cache’s absolute path.

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

## Install from Python

Planning is read-only. Installation requires a separate confirmation after the
application has displayed the plan.

```python
from aisketcher import PresetManager

manager = PresetManager()
plan = manager.plan_install("sdxl-canny-lite@1")

print(plan.license_notice)
print(plan.estimated_bytes)
for item in plan.items:
    print(item.repo_id, item.revision, item.destination, item.installed)

# Call only after the user reviews the information above.
result = manager.install("sdxl-canny-lite@1", confirm=True)
```

`PresetManager(allow_downloads=False)` is the required setting for a hosted or
read-only environment where an administrator, rather than a web user, manages
the model cache.

## Custom repositories

Arbitrary repository URLs, single checkpoint files, custom Python pipelines,
pickle weights, and `trust_remote_code` are outside v0.2. Implement a backend in
Python when a controlled deployment needs another model; do not expose a public
URL field that turns the Studio into a remote-code loader.
