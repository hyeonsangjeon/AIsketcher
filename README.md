# AIsketcher

[![CI](https://github.com/hyeonsangjeon/AIsketcher/actions/workflows/ci.yml/badge.svg)](https://github.com/hyeonsangjeon/AIsketcher/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/AIsketcher?style=flat-square)](https://pypi.org/project/AIsketcher/)
[![Python](https://img.shields.io/pypi/pyversions/AIsketcher?style=flat-square)](https://pypi.org/project/AIsketcher/)
[![Downloads](https://static.pepy.tech/badge/AIsketcher)](https://pepy.tech/project/AIsketcher)
[![License: MIT](https://img.shields.io/badge/license-MIT-2f855a?style=flat-square)](https://github.com/hyeonsangjeon/AIsketcher/blob/main/LICENSE)

Turn an image's structure and a natural-language prompt into a new image with
Stable Diffusion, ControlNet Canny conditioning, and optional Amazon Translate.

AIsketcher handles the repetitive parts around the model call: EXIF orientation,
aspect-ratio-preserving resize, Canny edge extraction, deterministic seeds, prompt
translation, and resizing the generated result back to the source dimensions.

## Results

![AIsketcher temple generation example](https://raw.githubusercontent.com/hyeonsangjeon/AIsketcher/main/pic/yahunjeon.png)

![AIsketcher architectural generation example](https://raw.githubusercontent.com/hyeonsangjeon/AIsketcher/main/pic/seowonjeon.png)

## Why AIsketcher

- **Structure-aware generation**: preserves the composition of the source image with
  ControlNet Canny edges.
- **Multilingual prompts**: translates prompts through Amazon Translate before inference.
- **Reproducible output**: uses an explicit seed and configurable inference settings.
- **Image-safe preprocessing**: normalizes EXIF orientation, RGB mode, aspect ratio, and
  ControlNet-compatible dimensions.
- **Pipeline-agnostic helper**: accepts an already configured Diffusers ControlNet pipeline,
  so model loading and device placement stay under your control.

## Installation

```bash
python -m pip install AIsketcher
```

To try the current development version:

```bash
python -m pip install "AIsketcher @ git+https://github.com/hyeonsangjeon/AIsketcher.git"
```

Python 3.9 or newer is required. A CUDA-capable GPU is strongly recommended for generation;
CPU inference works but is substantially slower.

## Quick Start

The following example loads the original DreamShaper and Canny ControlNet models used by
this project, generates one image, and saves the source, control map, and result.

```python
import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

import AIsketcher

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=dtype,
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "Lykon/DreamShaper",
    controlnet=controlnet,
    torch_dtype=dtype,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

original, edges, generated = AIsketcher.img2img(
    "input.jpg",
    "a quiet seaside library, warm morning light, detailed pencil texture",
    num_steps=30,
    guidance_scale=8.0,
    seed=42,
    low=100,
    high=200,
    pipe=pipe,
)

original.save("original.png")
edges.save("control-edges.png")
generated.save("generated.png")
```

The first model download is several gigabytes. Hugging Face caches downloaded weights, so
later runs reuse the local copy.

## Translate Prompts

AIsketcher can translate a prompt before passing it to the diffusion pipeline. Prefer the
standard AWS credential provider chain: environment variables, `~/.aws/credentials`, IAM
roles, or workload identity. Do not commit access keys to source code.

```python
translation = {
    "region_name": "ap-northeast-2",
    "SourceLanguageCode": "ko",
    "TargetLanguageCode": "en",
}

original, edges, generated = AIsketcher.img2img(
    "input.jpg",
    "비 오는 날의 조용한 한옥 도서관, 연필 스케치",
    pipe=pipe,
    trans_info=translation,
    seed=42,
)
```

Set `SourceLanguageCode` to `auto` or omit it to let Amazon Translate detect the source
language. Explicit `aws_access_key_id`, `aws_secret_access_key`, and `aws_session_token`
values are supported for backward compatibility, but the provider chain is safer for
production use.

Amazon Translate requires an AWS account and may incur usage charges.

## API

### `img2img(...)`

```python
AIsketcher.img2img(
    img_path,
    prompt,
    num_steps=20,
    guidance_scale=7,
    seed=0,
    low=100,
    high=200,
    pipe=None,
    trans_info=None,
    *,
    image_size=800,
    default_prompt=AIsketcher.DEFAULT_PROMPT,
    negative_prompt=AIsketcher.DEFAULT_NEGATIVE_PROMPT,
)
```

| Parameter | Purpose |
| --- | --- |
| `img_path` | Local path, HTTP(S) URL, or `PIL.Image.Image` source. |
| `prompt` | Prompt sent to the pipeline, optionally after translation. |
| `num_steps` | Number of diffusion inference steps. |
| `guidance_scale` | Strength of prompt guidance. |
| `seed` | Integer seed used for reproducible generation. |
| `low`, `high` | Canny edge thresholds satisfying `0 <= low < high <= 255`. |
| `pipe` | Loaded, callable Diffusers ControlNet pipeline. |
| `trans_info` | Optional Amazon Translate settings. |
| `image_size` | Longest side used for the ControlNet input. |
| `default_prompt` | Quality prefix prepended to the user's prompt. |
| `negative_prompt` | Negative prompt supplied to the diffusion pipeline. |

Returns `(original_image, canny_image, generated_image)` as Pillow images. The generated
image is resized to the original image dimensions.

### Image utilities

```python
resized = AIsketcher.resize_image("input.jpg", 800)
oriented = AIsketcher.correct_image_orientation(resized)
```

### Translation utility

```python
english = AIsketcher.translate_language(
    "안녕하세요",
    {"SourceLanguageCode": "ko", "TargetLanguageCode": "en"},
)
```

## Model and Safety Notes

- The MIT license in this repository applies to AIsketcher code. DreamShaper, ControlNet,
  Stable Diffusion, and generated outputs may have separate licenses and usage terms.
- Keep the Diffusers safety checker enabled unless you have an appropriate replacement.
- A negative prompt reduces some unwanted output but is not a safety guarantee.
- Generated images can reflect limitations and biases in the selected model.

Review the model cards before production use:

- [Lykon/DreamShaper](https://huggingface.co/Lykon/DreamShaper)
- [lllyasviel/sd-controlnet-canny](https://huggingface.co/lllyasviel/sd-controlnet-canny)

## Development

The test suite does not download model weights, require a GPU, or call AWS.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "Pillow>=9.5,<13" "pytest>=8,<9" "ruff>=0.12,<1" "build>=1.2,<2" "twine>=6,<7"
python -m pip install --no-deps -e .

ruff check .
pytest
python -m build
twine check dist/*
```

## References

- [Hugging Face Diffusers ControlNet guide](https://huggingface.co/docs/diffusers/using-diffusers/controlnet)
- [Stable Diffusion ControlNet pipeline API](https://huggingface.co/docs/diffusers/api/pipelines/controlnet)
- [Amazon Translate `translate_text`](https://docs.aws.amazon.com/translate/latest/dg/API_TranslateText.html)

## License

[MIT](https://github.com/hyeonsangjeon/AIsketcher/blob/main/LICENSE)
