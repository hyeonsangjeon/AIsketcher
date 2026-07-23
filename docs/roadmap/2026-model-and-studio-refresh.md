# 2026 model and Studio refresh

Status: implementation checkpoint, 2026-07-24

This document records the product and technical decisions agreed for the next
AIsketcher update. It is a working checkpoint, not a promise that every model
listed below has passed the hardware benchmark.

## Why the current default must change

The current `SDXL Base + Canny ControlNet` path is useful for strict edge
conditioning, but it is not an instruction-editing model. A dense Canny map
from a photograph can overpower semantic requests such as changing clothing,
materials, or art direction. Translating the same request to English does not
fix that architectural mismatch.

The refresh therefore does not replace one checkpoint with another universal
checkpoint. It routes sketch structure and photographic editing to different
model families.

## Candidate model profiles

All zero-click defaults must be public, commercially usable, revision-pinned,
and independently licensed at the base-model, adapter, and preprocessor level.

| Product profile | Candidate | Intended use | Current decision |
| --- | --- | --- | --- |
| Auto / Fast Edit | `black-forest-labs/FLUX.2-klein-4B` | Photo restyling, instruction edits, multi-reference work | Benchmark on T4; leading photo/default-edit candidate |
| Auto / Structure | `Tongyi-MAI/Z-Image-Turbo` + Alibaba-PAI Union 2.1 `2602-8steps` lite | Pencil sketch, line art, Canny/HED/Scribble-guided rendering | Benchmark on T4; leading sketch/default-structure candidate |
| Pro Quality | `Qwen/Qwen-Image-Edit-2511` with 2509 structural path where needed | Identity-sensitive portrait/product editing and geometric edits | A100/H100 option, not a T4 default |
| Legacy | SDXL Base + SDXL Canny | Replay and compatibility with existing manifests | Keep available, remove from the new Auto default |
| Experimental | `microsoft/Mage-Flow-Edit-Turbo` | Newly released instruction editing | Observe and benchmark; insufficient adoption evidence for default |

`FLUX.2-klein-4B`, Z-Image-Turbo, the Alibaba-PAI adapter, and the Qwen
checkpoints above are Apache-2.0 candidates. Non-commercial, gated, or
territory-incompatible weights are excluded from zero-click defaults. This
excludes FLUX dev/klein 9B, HunyuanImage variants whose license excludes South
Korea, and Stability checkpoints whose commercial terms are unsuitable for an
unconditional package default.

Korean input still needs a prompt-normalization/translation layer because none
of the shortlisted official model cards guarantees Korean instruction support.

## Auto routing

`Auto (recommended)` is the Simple-mode default.

- Sparse monochrome line work, drawings, and explicit structure controls route
  to the Structure profile.
- Photographs and requests that change clothing, materials, lighting, identity,
  or scene semantics route to the Fast Edit profile.
- Ambiguous inputs show the detected route and let the user change it before
  downloading weights.
- Advanced mode exposes the exact profile, revision, controls, and memory
  requirements.

The router decision and selected model profile are written to the exported
manifest so a result can be replayed.

## Simple-mode model selector

Simple mode also exposes a compact model selector rather than hiding all model
choice. Each curated option shows:

- a one-line “best for” description;
- expected first-download size and warm-generation time for the active device;
- minimum/recommended GPU memory;
- installed, downloading, ready, unavailable, or experimental state;
- license name and a link to the model card.

Selecting an uninstalled public default starts an explicit first-use flow:

1. Explain that the model will be downloaded from the internet once and reused
   from the local cache afterward.
2. Show real byte/file progress, not an indeterminate animation.
3. Keep generation disabled until the complete pinned snapshot is verified.
4. Provide Cancel and Retry.
5. Preserve already installed models when another download fails.

Arbitrary model URLs remain an Advanced-only feature. They are marked
`unverified/custom`, require a rights acknowledgement, and never silently
replace the curated default.

## Explore and refine interaction

Exploration produces four large, readable directions without an inner
horizontal scrollbar. Selecting a direction preserves its image, structure,
seed, model profile, and parent manifest.

`Refine this direction` opens an additional short instruction field. Examples:

- “Make the armor more modern.”
- “Keep the background and turn only the character into pen art.”
- “Use warmer paper and fewer decorative details.”

The refinement request is composed from the original brief plus the additional
instruction. If the field is empty, the product applies a documented automatic
refinement preset. The extra instruction is stored separately in the child
manifest instead of destructively overwriting the original brief.

## Progress, cancellation, and refresh behavior

Refreshing or closing a browser tab must not be presented as cancellation. A
server job otherwise continues consuming the GPU.

While a job is queued or running, the primary action changes to `Stop`:

- queued jobs are removed immediately;
- active Diffusers jobs receive a per-session cancellation token;
- the pipeline checks the token at step callbacks and between output seeds;
- already completed candidates remain visible;
- cancellation releases temporary tensors, runs accelerator cache cleanup, and
  reports `Stopped by user`;
- the same mechanism cancels first-time model downloads at a safe file
  boundary.

The UI reports phase and honest timing:

- waiting in queue;
- downloading model;
- loading model;
- generating candidate N of M;
- refining;
- saving/exporting.

Elapsed time and an estimate are labeled separately. A value such as
`42.3 / 107.6 s` must be explained as elapsed/estimated, never implied to be a
hard timeout.

## Canonical first-run experience

`Try the guided sample` must use the exact canonical README hero input, prompt,
model profile, revision, seed set, structure settings, and selected result.
Opening it must clear any stale prompt from a previously uploaded image. The
English Studio uses the English screenshot and copy; the Korean manual uses the
Korean screenshot and copy.

The bundled sample remains available without downloading a model. Live
generation is a separate, clearly labeled action.

The Guided Sample is read-only until a live model is installed, but that is a
normal product transition rather than an error. If the user selects
`Refine this direction` from the bundled sample, do not raise a red error toast
or replace every component with an error badge. Open a centered, accessible
layer instead:

- title: `Ready to make this direction yours?`;
- explain that the preview is bundled and live refinement needs one model;
- primary action: `Choose a model and refine`;
- secondary action: `Keep exploring the sample`;
- show the Auto-recommended model, “best for” copy, first-download bytes,
  estimated setup time, device compatibility, and license before confirmation;
- continue into the same download/progress/cancel flow used by the Simple-mode
  model selector;
- preserve the selected sample and refinement instruction while the model is
  being prepared.

Only genuine failures after the user starts setup should use error styling, and
those errors must remain localized to the layer rather than replacing the
entire Studio surface.

## Model registry and download policy

Each registry entry records at least:

- `model_id`, immutable `revision`, and verified file hashes;
- `license_id`, `license_url`, redistribution notice, and commercial-use flag;
- gated/click-through status and territory exclusions;
- runtime family and required optional dependency;
- supported input modes and control types;
- minimum VRAM profile, download bytes, and tested device results.

Weights are never shipped inside the PyPI wheel. Public curated weights are
downloaded on first use into the configured cache. Model, adapter, and
preprocessor licenses are evaluated independently.

Preprocessor constraints:

- OpenCV Canny is the safe default.
- Pose should use an Apache-compatible DWPose path, not bundled CMU OpenPose
  weights.
- Only the Apache-2.0 Depth Anything V2 Small weights may be a curated default;
  larger non-commercial variants are excluded.
- InsightFace-dependent InstantID/IP-Adapter FaceID paths are not commercial
  defaults.

## Benchmark gate

No new profile becomes `Auto` until it passes the same fixed benchmark corpus:

- canonical README sketch;
- additional sparse and dense sketches;
- a portrait/photo semantic-edit case;
- a product/photo restyle case.

Record cold start, warm latency, four-output latency, peak VRAM/RAM, download
bytes, failure rate, prompt adherence, structure/identity preservation,
aesthetic preference, and repeatability. T4 is the minimum live-GPU gate for
Fast and Structure; Pro is benchmarked on A100/H100.
