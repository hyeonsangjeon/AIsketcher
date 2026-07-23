# 2026 model and Studio refresh

Status: v0.3.0 implementation record with explicitly marked future work,
2026-07-24

This document records the product and technical decisions implemented for
AIsketcher v0.3.0. A model listed as a candidate is not implied to have passed
the hardware benchmark.

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
| Auto / Fast Edit | `black-forest-labs/FLUX.2-klein-4B` | Sketch rendering, photo restyling, instruction edits | T4 validated; v0.3.0 default |
| Auto / Structure | `black-forest-labs/FLUX.2-klein-4B` | Sketch-to-design with the source image used as the structural reference | T4 validated; v0.3.0 default |
| Structure candidate | `Tongyi-MAI/Z-Image-Turbo` + Alibaba-PAI Union 2.1 `2602-8steps` lite | Canny/HED/Scribble-guided rendering | Benchmark required; not in Auto |
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

## Korean prompts

The Studio accepts Korean as a first-class input language even when the selected
image model does not guarantee Korean prompt following.

- Preserve and display the user's Korean original.
- Detect Korean locally and prepare an English model prompt before generation.
- Show the prepared prompt in a collapsible `Prompt sent to the model` field so
  the user can inspect or edit it.
- Store the original, prepared prompt, translation provider/model revision, and
  whether the user edited the translation in the manifest.
- Never translate image pixels or upload an image merely to translate text.
- If no translator is installed/configured, explain the limitation and let the
  user choose between installing the optional local translator or editing the
  English prompt manually.

The zero-credential local path is the public Apache-2.0
`Helsinki-NLP/opus-mt-ko-en` Marian checkpoint, loaded lazily and pinned like
other optional weights. The explicit model-preparation action discloses and
prepares this 315 MB helper alongside the selected image model; normal prompt
submission never starts an undisclosed network download. Azure Translator or
another user-configured provider may be an alternative, but is never silently
required. Translation and design prompt enhancement are separate steps:
translation preserves meaning, while a small deterministic template expresses
the requested edit and preservation constraints without inventing new creative
content.

## Auto routing

`Auto (recommended)` is the Simple-mode default. In v0.3.0 both sketches and
photos resolve to the same T4-validated FLUX.2 Klein profile. The registry
records future route metadata, but v0.3.0 does not claim to classify the
uploaded image or persist an inferred route.

- Sparse monochrome line work and drawings use the FLUX.2 source image as the
  structural reference.
- Photographs and requests that change clothing, materials, lighting, identity,
  or scene semantics use the same validated FLUX.2 edit runtime.
- Advanced mode exposes the exact profile, revision, controls, and memory
  requirements.

The resolved preset and its pinned model revisions are written to the exported
manifest so a result can be replayed. Input classification, a visible route
override, and route-specific manifest fields remain future work and require
their own benchmark gate before they can change `Auto`.

## Simple-mode model selector

Simple mode also exposes a compact model selector rather than hiding all model
choice. Each curated option shows:

- a one-line “best for” description;
- expected first-download size and the tested T4 warm-generation range where
  one has been recorded;
- minimum/recommended GPU memory;
- whether each pinned component is already cached or still missing;
- license name and a link to the model card.

Selecting an uninstalled public default starts an explicit first-use flow:

1. Explain that the model will be downloaded from the internet once and reused
   from the local cache afterward.
2. Show the current installation phase and preserve the reviewed byte estimate;
   v0.3.0 does not claim live byte-level Hub telemetry.
3. Keep generation disabled until every required pinned file group is present
   and the local completion marker is valid.
4. Provide Stop and allow retry. An active transfer stops at the next safe
   selected-file boundary rather than pretending a partial tensor file is
   usable.
5. Preserve already installed models when another download fails.

Arbitrary model URLs are intentionally not accepted in v0.3.0 Simple or
Advanced mode. A future custom-model flow would need a rights acknowledgement,
safe-file policy, isolated cache namespace, and an explicit
`unverified/custom` label before it could be exposed.

## Explore and refine interaction

Exploration produces four large, readable directions without an inner
horizontal scrollbar. Selecting a direction preserves its image, structure,
seed, model profile, and parent manifest.

The same sizing rule applies to refinement results. Four intermediate results
must be useful at normal page zoom without opening the fullscreen viewer: each
card keeps the source aspect ratio, has a practical minimum height, and is not
cropped into a narrow strip by `object-fit: cover`.

Opening and closing Gallery preview/fullscreen must be reversible. Closing with
`X` restores the original gallery in place, the previous page scroll position,
and document overflow. It must not leave a detached thumbnail strip at the top
of the page, an empty gallery shell, duplicate preview nodes, or a fullscreen
z-index/overflow class on the document. This interaction is a required browser
regression test on both exploration and refinement galleries.

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

GPU work is guarded process-wide, not only by one browser queue. Explore,
refine, and replay calls from different sessions wait for the same accelerator
slot, while a duplicate request from one session is rejected until its current
backend callback has fully unwound. If CUDA/MPS reports out-of-memory, the
failed runtime is evicted, temporary accelerator memory is released, and the
next request constructs a clean runtime instead of reusing a poisoned
pipeline.

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

Temporary Gradio share links may return an HTML error page after the remote
process restarts. A client-side JSON parse failure in that situation must not
be presented as model failure or collapse the whole Studio into error badges.
The page shows a localized session-ended layer with reload/latest-link
guidance; a refreshed deployment remains the source of truth.

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

## Session recovery

A stale browser page can outlive a restarted local server or temporary Gradio
share tunnel. In that case the old frontend may receive an HTML reconnect/404
page where it expected a JSON event response and report `Unexpected token '<'`.
Treat this as an expired session, not a generation failure:

- keep the last rendered images and controls intact;
- do not replace every component with an error badge;
- show one recovery layer with **Reload this address** and **Keep this screen**;
- after a reload of the same live server, restore language, prompt, selected
  model, and non-sensitive controls from browser/session state;
- clearly state when an in-flight server job cannot be recovered.

The browser regression suite includes a server restart while an old page
remains open, followed by a primary-action click.

v0.3.0 uses Gradio `BrowserState` for a stable, non-sensitive browser session
identifier. Refreshing the same running server can therefore rediscover and
stop the active job, and the newest retained result can be restored after the
backend finishes. A process restart cannot recover GPU memory or an in-flight
Python call that no longer exists; the reconnect layer explains that boundary.

This is session recovery, not a user-account system. The local Studio does not
store email addresses, passwords, OAuth tokens, or cross-device projects.
Persistent multi-user accounts would require an explicit authentication,
authorization, retention, and storage design and remain outside the packaged
local MVP.

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
