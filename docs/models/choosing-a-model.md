# Choose a model by the constraint you need

AIsketcher is not a checkpoint leaderboard. Its job is to keep the same
`prepare → explore → pick → vary → export → replay` contract while a backend
changes. The model choice should therefore describe a capability, not imply
that one model is best for every input.

## Current product roles

| Role | Current path | Use it when | Important limit |
| --- | --- | --- | --- |
| Guided tour | bundled, hash-verified study | you want to understand the workflow immediately | read-only; no new image is generated |
| Fast Edit | FLUX.2 Klein 4B | speed, photo restyling, flexible sketch interpretation, and instruction edits matter | reference-image editing is not Canny and does not guarantee exact line locking |
| Structure Lock | SDXL + Canny ControlNet | preserving a line drawing or replaying an existing study matters more than current-generation edit quality | stable legacy fallback, not the quality showcase |
| Structure candidate | Z-Image Turbo + Union 2.1 Lite | modern Canny, HED, Scribble, or related structure control | not a default until the benchmark below passes; large first download and a separate VideoX-Fun runtime |
| Pro candidates | Qwen Image Edit 2509/2511 | high-end structure, product, character, or identity-sensitive work | 20B-class models; not appropriate for a T4-first experience |

[FLUX.2 Klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
is the current concrete Fast Edit model. Its official interface accepts a
reference image and editing instruction. AIsketcher no longer labels that path
as an intelligent Auto router or describes it as strict structure control.

[Z-Image Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) and the
[Alibaba-PAI Union 2.1 adapter](https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1)
are the leading Structure candidate because their public Apache-2.0 path
supports Canny, HED, Scribble, depth, pose, and other controls. The adapter is
not yet a native Diffusers preset, and the reviewed runtime payload is too
large to silently become a Simple-mode default.

[Qwen Image Edit 2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) is the
Pro Structure candidate; [2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
is the Pro Consistency candidate. They stay out of the local default until a
larger-GPU backend and the same replay contract are validated.

## The default-model gate

A candidate can replace the current Fast Edit or Structure role only after a
reproducible study covers all of the following:

1. At least 12 inputs across sparse pencil lines, dense child drawings,
   product outlines, graphic compositions, people, and ordinary photos.
2. Four recorded seeds for every model/input/prompt combination.
3. The same reviewed English model prompt, with the Korean source retained
   separately when translation is involved.
4. Blind designer preference, structure similarity, prompt adherence,
   visible distortion/failure rate, first-download bytes, warm latency,
   peak VRAM, and cancellation behavior.
5. Separate results for T4 FP16 and a BF16-capable L4, A10, A100, or H100.

The winning model must improve both human preference and the constraint it is
meant to serve. A higher edge score alone is not enough, and a beautiful image
that discards the sketch is not a Structure winner.

## Why the package still matters when models change

The portable value is the study around the model: explicit seeds, a chosen
parent, controlled variations, model revisions, source and output hashes,
prompt provenance, and strict or compatible replay. Custom local or hosted
backends implement the small `Backend` protocol and keep those records without
forking the workflow.

If you can run a candidate model, use the
[model benchmark report form](https://github.com/hyeonsangjeon/AIsketcher/issues/new?template=model-benchmark.yml)
to contribute a comparable result rather than a screenshot without settings.
