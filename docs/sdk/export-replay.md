# Export and replay

An export is the handoff boundary. It combines viewable assets with a
machine-readable `aisketcher.manifest/v1` record.

## Manifest contents

- normalized source and control references;
- file SHA-256 values;
- versioned preset and immutable model revisions;
- original intent and resolved recipe;
- actual seed for every candidate;
- technical signal values and display badges;
- pick and variation parent-child relationships;
- AIsketcher, backend, and runtime versions.

It excludes access tokens, absolute paths, original upload filenames, and EXIF
metadata.

## Strict replay

Strict mode rejects a replay when a required file hash, preset, model revision,
or resolved recipe has drifted. Use it for reviewed handoffs and canonical
examples.

```python
report = studio.replay("design-study/manifest.json", mode="strict")
```

## Compatible replay

Compatible mode may use a supported replacement. Every changed component is
listed in the report; no substitution is silent.

```python
report = studio.replay("design-study/manifest.json", mode="compatible")
assert report.replayed
print(report.drift)
```

Compatible replay is useful when old hardware or a checkpoint is unavailable,
but its output is a new lineage event rather than an assertion that the original
run was reproduced exactly.

## Reproducibility boundary

Pinned seeds and revisions are necessary but may not make pixels identical
across CUDA, MPS, CPU kernels, precision modes, and library versions. Compare
the replay report and artifact hashes instead of assuming seed equality alone
is proof.
