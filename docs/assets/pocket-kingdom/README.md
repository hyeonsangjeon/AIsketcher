# Pocket Kingdom asset contract

Pocket Kingdom is the reviewed canonical AIsketcher example. The public source
is an anonymized drawing; private originals and art-direction references stay
outside this repository. Every published result is an actual output of the
pinned AIsketcher v0.2 pipeline.

```text
source.png                  public high-resolution source
control.png                 1024px preparation evidence
prepared/source.png         normalized source referenced by the manifest
prepared/control.png        exact Canny condition referenced by the manifest
scout/scout-01..04.png      ordered seeded directions
scout/contact-sheet.png     ordered overview
variation/variation-01..04  subtle children of the selected scout
variation/manifest.json     replayable lineage and MPS runtime evidence
result.png                  human-selected variation
manifest.json               replayable documentation scout record
provenance.json             publication and artwork provenance
```

Publication requirements:

- derived, anonymous filenames only;
- no EXIF, XMP, GPS, device, name, or capture-time metadata;
- immutable model revisions and the fully resolved recipe;
- matching SHA-256 hashes for every manifest artifact;
- exact seeds, selection, scores, and finite-output runtime evidence;
- no secret, access token, absolute path, or original upload name;
- artwork governed by `ARTWORK_LICENSE.md`, not the MIT source-code license.

`manifest.json` is the replayable scout contract for this documentation-only
canonical lineage example. The separate variation manifest points back to its
scout selection. `result.png` is a documentation alias of the human-selected
variation and is independently hashed in `provenance.json`. The packaged v0.3
Studio Guided Sample instead uses the privacy-reviewed HPO hero study.
