# Privacy and asset handling

Visual exploration often begins with personal sketches or photographs.
AIsketcher treats minimization and traceable export as part of the workflow.

## What an export keeps

- anonymized derived filenames;
- normalized source and control artifacts selected for the study;
- model-independent intent and resolved generation settings;
- seeds, lineage, hashes, and runtime versions.

## What an export removes

- EXIF, GPS, camera, and capture-time metadata;
- the original upload filename;
- absolute filesystem and model-cache paths;
- API tokens, access keys, cookies, and authorization headers.

An export cannot erase identifying content visibly drawn or photographed in the
pixels. Review every public asset after metadata stripping.

## Studio boundaries

The local Studio binds to loopback by default, uses separate session workspaces,
limits upload size, and does not create a public share URL. Public hosting needs
an explicit deployment review, authentication and retention policy, and model
installation must be restricted to administrators.

Studio writes normalized uploads and generated candidates to an OS temporary
workspace. An expired run is removed when the registry next prunes, and the
auto-created workspace is removed when the controller closes or the Python
process exits normally. A caller-supplied workspace root itself is never
deleted, although registered run directories inside it are cleaned. A crash,
forced termination, or machine power loss can bypass normal cleanup, so shared
or hosted systems must also apply an operating-system retention job.

## Canonical artwork

The Pocket Kingdom source is private family artwork. Only the reviewed,
anonymized derivative may enter `docs/assets/pocket-kingdom/`. Private
art-direction references and screenshots must stay outside the repository.

Before accepting an image in that directory, validation checks that:

- metadata is absent;
- dimensions are appropriate for documentation;
- filenames reveal no person or device information;
- a result has an accompanying AIsketcher manifest and file hashes;
- the separate artwork notice is linked nearby.

See [ARTWORK_LICENSE.md](https://github.com/hyeonsangjeon/AIsketcher/blob/main/ARTWORK_LICENSE.md)
for reuse restrictions.
