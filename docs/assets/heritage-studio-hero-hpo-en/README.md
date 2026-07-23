# Studio Hero HPO fixture

This directory is the real, model-generated four-seed study shown in the
English Studio Hero screenshot and bundled as the v0.3 Guided Sample. Pocket
Kingdom remains a separate documentation-only canonical lineage example.

The selected first direction uses the recorded 2023 seed
`6764547109648557242` with the pinned `sdxl-canny-lite@1` preset. The other
three values are the two alternatives preserved in the same notebook cell and
one explicit adjacent value. No random seed appears in this fixture.

## Search and selection

The July 2026 Hero search generated 12 new images in four bounded rounds:

1. four fixed-seed material directions;
2. three fixed-seed prompt repairs;
3. two fixed-seed ControlNet strengths with a minimal prompt; and
4. three additional preserved-seed runs around the best material direction.

Every round used the same reviewed source, pinned local model revisions, Canny
`140/160`, UniPC, and no downloads. The selected material direction was then
compared with the previous Hero incumbent and the verified 2023 output.

Technical scores were used only to remove obvious failures. The visual gates
were: a strong central read at thumbnail size; survival of the main left,
center, and right landmarks; dimensional material depth; a clean crop; and no
empty center, pseudo-text, malformed face, or dominant background pattern.
The historical seed won the final seed sweep. Alternative 1 left the center
unfinished, alternative 2 introduced pseudo-text, and the adjacent seed was
overexposed.

The selected result is intentionally a modern SDXL reinterpretation, not a
claim that the old DreamShaper/SD 1.5 pixels can be reproduced by the current
default model. A seed is meaningful only with its complete recipe and model
revisions.

## Evidence

`manifest.json` records the prompt, negative prompt, exact model revisions,
seed order, selection, Canny settings, scheduler, steps, CFG, ControlNet
strength, runtime versions, and every artifact hash. `provenance.json` records
the HPO scope, offline generation boundary, privacy review, and selection
rationale.

The source and generated artwork are not covered by the repository's MIT
license. They remain subject to the root `ARTWORK_LICENSE.md`.
