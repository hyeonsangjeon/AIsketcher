# Configuration reference

AIsketcher uses a versioned YAML settings file for repeatable Studio defaults.
The first-run command creates a private per-user file; an optional
`aisketcher.yaml` in a project directory can override only the values that team
members need to share.

```bash
aisketcher init
```

The command does not download a model. It refuses to replace an existing file
unless overwrite is explicit. Print the resolved location after creation and
review it before starting Studio.

To create and then use an explicit project file:

```bash
aisketcher init --path ./aisketcher.yaml
aisketcher studio --config ./aisketcher.yaml
```

## Example

```yaml
schema_version: 1
preset: "sdxl-canny-lite@1"
device: "auto"
output_count: 4
seed_mode: "scout"
seed: null
language: "en"
cache_dir: null
allow_downloads: true
```

Every file must declare `schema_version`. Files with a future schema, unknown
keys, duplicate keys, or unsupported values fail closed with a validation
error.

## Settings

| Key | Default | Accepted values | Meaning |
| --- | --- | --- | --- |
| `schema_version` | `1` | `1` | Settings contract version; required in every file. |
| `preset` | `sdxl-canny-lite@1` | A registered preset or alias | Initial local model preset. It is normalized to its canonical versioned name. |
| `device` | `auto` | `auto`, `cuda`, `mps`, `cpu` | Preferred backend device. The built-in live SDXL Studio does not support CPU generation. |
| `output_count` | `4` | `1`, `4`, `8` | Initial number of scout or variation outputs. |
| `seed_mode` | `scout` | `scout`, `locked`, `explicit` | Initial seed-plan mode. |
| `seed` | `null` | non-negative 63-bit integer or `null` | Starting value for `locked` mode only. A seed only identifies a run when paired with its complete recipe and runtime. |
| `language` | `en` | `en`, `ko` | Initial Studio interface language. |
| `cache_dir` | `null` | path string or `null` | Managed model-cache root. `~` is expanded when the path is used. |
| `allow_downloads` | `true` | `true`, `false` | Whether this configured runtime may perform an explicitly confirmed preset installation. |

`allow_downloads: true` is not download consent. A local preset still requires
the application to display `plan_install()` and call `install(...,
confirm=True)`. Set it to `false` in hosted or read-only environments where an
administrator prepares the cache.

## Resolution order

Values are merged from least to most specific:

1. package defaults;
2. the platform user file;
3. `aisketcher.yaml` in the current working directory.

Set `AISKETCHER_CONFIG` to use one explicit project override instead of the
current-directory file:

```bash
export AISKETCHER_CONFIG=/path/to/team-settings.yaml
aisketcher studio
```

An explicit path must exist. A missing automatically discovered user or
current-directory file is normal; a missing `AISKETCHER_CONFIG` target is an
error.

The per-user location follows the platform configuration convention. Ask the
package for the exact path instead of hard-coding it:

```python
from aisketcher.config import default_user_config_path

print(default_user_config_path())
```

## User and project files

Keep machine-specific values such as `device` and `cache_dir` in the user file.
A project file is appropriate for a shared preset, output count, or interface
language:

```yaml
# ./aisketcher.yaml
schema_version: 1
preset: "sdxl-canny-lite@1"
output_count: 4
seed_mode: "scout"
seed: null
```

To prefill a recorded seed for a controlled comparison, create a user file with
both fields explicitly:

```bash
aisketcher init --outputs 1 --seed-mode locked --seed 6764547109648557242
```

Locked mode intentionally requires one output; repeating one unchanged request
would only create duplicate candidates. That value is within the supported
63-bit range. It does not reproduce pixels
from a different checkpoint, scheduler, prompt, control image, or hardware
runtime; export the new study manifest as the actual evidence.

Do not put tokens, credentials, private image paths, or Hugging Face cache
contents in either file. AIsketcher’s curated v0.2 presets use public pinned
repositories and do not define a credential field.

## Environment variables

| Variable | Scope | Behavior |
| --- | --- | --- |
| `AISKETCHER_CONFIG` | configuration | Selects an explicit project override file. |
| `AISKETCHER_CACHE_DIR` | model cache | Overrides the platform cache for a `PresetManager` that was not given `cache_dir`. |
| `XDG_CACHE_HOME` | model cache | Supplies the parent of the default `aisketcher` cache on platforms that use the XDG convention. |

For the direct Python API, an explicit `PresetManager(cache_dir=...)` argument
takes precedence over cache environment variables. The YAML loader returns
`config.cache_path`; pass that path and `config.allow_downloads` when building
your own application:

```python
from aisketcher import PresetManager
from aisketcher.config import load_config

config = load_config()
models = PresetManager(
    cache_dir=config.cache_path,
    allow_downloads=config.allow_downloads,
)
```

## YAML boundary

The parser intentionally accepts a small, dependency-free YAML scalar subset:
top-level keys with strings, integers, booleans, or `null`. Nested mappings,
lists, tags, anchors, objects, and block values are rejected. This keeps the
settings contract auditable and prevents a configuration file from becoming a
code-loading surface.

Use the Python API to create a settings file programmatically:

```python
from aisketcher.config import AIsketcherConfig, save_config

path = save_config(AIsketcherConfig(language="ko"))
print(path)
```

New files and atomic replacement files are owner-readable and owner-writable.
An existing destination is protected unless `overwrite=True` is passed.

## Schema changes

Do not silently change `schema_version: 1`. A future package that changes the
meaning or shape of a setting must publish an explicit migration. If a newer
file is opened by an older package, preserve it and upgrade the package rather
than deleting unknown values until the migration notes are reviewed.
