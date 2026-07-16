# Heritage seed comparison asset contract

This bundle is a new AIsketcher 0.2 local run made from the reviewed,
metadata-free Pocket Kingdom source. It tests the active seed recorded in the
[AWS Korea 2023 Coding School notebook at `main@4b3a220`](https://github.com/hyeonsangjeon/aws-korea-2023-coding-school/blob/4b3a22098f239dda18134eb5128094aa9892ed5e/aws-korea-coding-school-2023.ipynb)
alongside two values preserved in that notebook's comments.

The notebook's opening lesson text proposes the brief `알파벳 A, 글자, 추상화,
멋진 펜아트`. The saved stdout in the active-seed code cell records a different
runtime prompt, `특이한 풍경, 환상적인 그림, 복잡한 나라, 귀여운 케릭터`.
`provenance.json` retains both labels rather than presenting either as the
other.

The old JPEG input is intentionally not copied here because it contains camera
and GPS EXIF. Its content hash and Git blob ID are retained in
`provenance.json`. The opening notebook cell displays the separate
`present_result_sample.jpeg` showcase, but neither that cell nor the file ties
the showcase to the active seed. It also differs from the result embedded in
the active execution cell, so the provenance records the file while leaving
its active-seed attribution unverified. The notebook does embed a 1600×600
source-and-result PNG directly in the active execution cell. That metadata-free
image is retained as `legacy-2023-cell-output.png`, with its notebook commit,
cell/output indexes, and hashes recorded in `provenance.json`.

The manifest records the pinned SDXL Canny Lite recipe and all three exact
63-bit seeds. Only the numeric seeds carry over: the v2 study uses a replacement
source, expanded prompt, 24 steps, CFG 5.5, and Canny 90/224 rather than the
legacy recipe. Candidate 1, the notebook's active seed, remains the canonical
selection. Candidate 2 holds the three technical badges for this set, but those
badges only report observable structure, edge, and set-distance properties;
they are not an aesthetic choice or a universal quality score.

All model files were already present in the managed cache. This comparison ran
offline and downloaded no weights. Visual assets remain subject to the root
`ARTWORK_LICENSE.md`.
