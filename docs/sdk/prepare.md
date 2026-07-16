# Prepare

`prepare()` turns an uploaded or local image into a normalized,
backend-independent structure artifact.

## Processing contract

- Apply EXIF orientation once, then remove EXIF from the derived asset.
- Convert grayscale, palette, and alpha images to RGB.
- Preserve aspect ratio and use dimensions compatible with the selected preset.
- Generate a three-channel Canny image from explicit or recommended thresholds.
- Keep the original source file unchanged.
- Hash exported derivatives rather than exposing the upload’s original name or
  absolute path.

## Diagnostics

| Signal | What it helps detect | Typical response |
| --- | --- | --- |
| Contrast | faint pencil marks | increase contrast or lower thresholds |
| Edge density | too little or too much structure | adjust thresholds or simplify input |
| Fragmentation | many short, disconnected edges | clean the sketch or soften Canny |
| Crop risk | important marks near the boundary | add safe margin before generating |

Diagnostics are descriptive. They do not rewrite an input without the caller
requesting the corresponding preparation option.

## Canny thresholds

Recommended thresholds are an inspection result, not a hidden setting. When
accepted, they become part of the resolved recipe and manifest. Advanced users
can override them; the exported study preserves both the recommendation and the
values actually used.

## Input limits in Studio

The local Studio accepts images up to 20 MB and 50 megapixels. Oversized or
invalid files are rejected before decoding or generation. Exports receive new,
non-identifying names.
