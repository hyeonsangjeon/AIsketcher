# Seeds and output counts

A seed identifies a random stream. It is useful for replay and nearby
exploration; it is not a portable quality score.

## Seed plans

- `SeedPlan.scout(count)` derives a documented set of independent seeds for a
  broad first pass.
- `SeedPlan.locked(seed)` preserves one seed for a controlled run. The Python
  API can repeat it, but identical requests can produce duplicate candidates;
  Studio therefore pairs Locked mode with one output.
- `SeedPlan.explicit([...])` uses caller-provided seeds exactly.

Every candidate records the actual integer used. A backend must not share one
mutable generator across a batch and then describe the batch as independently
seeded.

## Choose an output count

| Count | Use it when | Trade-off |
| ---: | --- | --- |
| 1 | checking a recipe or device | fastest, no meaningful comparison |
| 4 | normal scouting and variation | default balance of breadth and cost |
| 8 | a direction is still ambiguous | broader but slower and harder to review |

The Simple view always uses four. Advanced exposes 1, 4, and 8 so the contact
sheet and review flow remain predictable.

## A useful designer loop

1. Scout four independent seeds with Balanced structure.
2. Pick on concept, not on a technical badge alone.
3. Vary the selected candidate with a structure lock and subtle strength.
4. Change one recipe dimension at a time when comparing runs.
5. Export before starting a materially different direction.

Do not publish a “recommended seeds” list as if the numbers generalize across
models, prompts, schedulers, or inputs. Publish the complete recipe and result
lineage instead.

## A real heritage-seed check

The
[AWS Korea 2023 Coding School notebook at `main@4b3a220`](https://github.com/hyeonsangjeon/aws-korea-2023-coding-school/blob/4b3a22098f239dda18134eb5128094aa9892ed5e/aws-korea-coding-school-2023.ipynb)
recorded `6764547109648557242` as its active first-exercise seed. Its same code
cell preserved `6854547109648557242` and `6634547109688557242` in comments,
alongside 40 steps, CFG 7, and Canny thresholds 140/160. AIsketcher 0.2 reran
all three numbers against the current pinned SDXL Canny Lite recipe. It used
the already installed cache in offline mode, so the comparison downloaded zero
bytes.

The notebook contains two prompt records that should not be conflated. Its
opening lesson text proposes `알파벳 A, 글자, 추상화, 멋진 펜아트`; the saved
stdout for the active-seed execution records `특이한 풍경, 환상적인 그림, 복잡한
나라, 귀여운 케릭터`. The provenance file preserves both as the documented brief
and recorded runtime prompt, respectively.

Only the three numeric seeds were carried into the v2 compatibility run. The
input, prompt, model stack, and generation recipe deliberately changed:

| Dimension | 2023 notebook evidence | AIsketcher 0.2 compatibility run |
| --- | --- | --- |
| Source | `present_image1.jpeg` | reviewed metadata-free Pocket Kingdom source |
| Prompt | saved Korean runtime prompt above | expanded English graphic-design prompt |
| Base / ControlNet | DreamShaper / SD Canny, revisions unpinned | SDXL Base / SDXL Canny Small, revisions pinned |
| Scheduler | PNDM | UniPC |
| Steps / CFG | 40 / 7 | 24 / 5.5 |
| Canny low / high | 140 / 160 | 90 / 224 |

This is a seed-compatibility study, not a claim that v2 reproduces the complete
2023 recipe or its pixels.

<figure class="canonical-source">
  <img
    src="../../assets/heritage-seed-study/legacy-2023-cell-output.png"
    alt="Source sketch and vivid generated result embedded in the 2023 notebook execution cell"
    data-lightbox
    data-caption="Verified legacy execution evidence: source and generated result embedded in the exact active-seed notebook cell"
  >
  <figcaption>
    Verified 2023 execution evidence · active seed
    <code>6764547109648557242</code> · 40 steps · CFG 7 · Canny 140/160.
    This is the image embedded in that execution cell, not the separate showcase
    JPEG.
  </figcaption>
</figure>

The legacy stack did not pin model revisions, so this evidence is preserved as
a historical result rather than promised as a pixel-identical v2 replay.
AIsketcher 0.2 separately reran the three notebook values with its current
pinned recipe. The current images are retained as compatibility evidence, not
as a visual recommendation:

<details>
  <summary>Open the current v2 compatibility contact sheet</summary>
  <figure class="canonical-source">
    <img
      src="../../assets/heritage-seed-study/contact-sheet.png"
      alt="Three current AIsketcher outputs from the recorded 2023 seed and two neighboring notebook seeds"
      data-lightbox
      data-caption="Current v2 comparison: exact active 2023 seed and two values preserved in the notebook comments"
    >
    <figcaption>
      Left: exact active 2023 seed and canonical reference. Center and right:
      the two preserved comment values. Select the sheet to inspect it at full
      size.
    </figcaption>
  </figure>
</details>

| Seed | Structure | Edge cleanliness | Distinctiveness | Review |
| ---: | ---: | ---: | ---: | --- |
| `6764547109648557242` | 0.271662 | 0.616346 | 0.194801 | exact recorded active seed |
| `6854547109648557242` | 0.322524 | 0.811936 | 0.198649 | three technical badges; not an aesthetic selection |
| `6634547109688557242` | 0.181476 | 0.373457 | 0.187767 | third comparison direction |

The center seed scored higher on all three automated diagnostics for this run,
but those diagnostics do not decide what is beautiful. The exact active 2023
seed remains the canonical reference. The important result is that the old
number was accepted exactly as a 63-bit seed and the new execution produced a
fresh [manifest](../assets/heritage-seed-study/manifest.json) with pinned model
revisions and artifact hashes.

The legacy input JPEG was not copied because it contains camera and GPS EXIF.
The current study uses the reviewed metadata-free source. The separate legacy
showcase JPEG is not claimed as the recorded seed's output; the embedded cell
image above is the verified execution evidence. See the bundle's
[provenance contract](../assets/heritage-seed-study/README.md).
