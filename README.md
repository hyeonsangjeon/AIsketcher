[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](https://raw.githubusercontent.com/hyeonsangjeon/youtube-dl-nas/master/LICENSE)
[![Downloads](https://pepy.tech/badge/ai-sketcher)](https://pepy.tech/project/AIsketcher)

# AIsketcher

- Stable Diffusion model : Lykon/DreamShaper[1] 
- Text-to-Image Generation with ControlNet Conditioning : used Canny edge detection [2][3]
- prompt translator english to korean : Amazon Translate [4]

Text-to-image generation using Huggingface stable diffusion ControlNet conditioning and AWS Translate's prompt translation function

![screenshot1](https://github.com/hyeonsangjeon/AIsketcher/blob/main/pic/yahunjeon.png?raw=true)
![screenshot2](https://github.com/hyeonsangjeon/AIsketcher/blob/main/pic/seowonjeon.png?raw=true)

## Project Description
This function takes two inputs: an image and a prompt text, utilizing the power of multi-modal models.
In this project, I used Stable Diffusion, where prompts were written in English. However, for users who predominantly use other languages, it can be challenging to express the details of their input sentences. Therefore, we utilize user's language for the input prompt, and the corresponding text is machine-translated to English using Amazon Translate before being fed into the model.

Prerequisite: Load the ControlNetModel and StableDiffusionModel into the StableDiffusionControlNet Pipeline and prepare the PNDMScheduler.
```python
controlnet_model = "lllyasviel/sd-controlnet-canny"
sd_model = "Lykon/DreamShaper"

controlnet = ControlNetModel.from_pretrained(
    controlnet_model,
    torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    sd_model,
    controlnet=controlnet,
    torch_dtype=torch.float16
)

pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
```

## Function Workflow

1. Resize the input image to 800x800.
2. Extract the edges, which are the key features of the input image, using the Canny function.
3. If the input sentence contains the Amazon Translate dictionary (trans_info) variable, translate the sentence to English.
4. Feed the translated prompt and the extracted edge image into the StableDiffusionControlNet Pipeline to generate a new image.
5. Resize the output image back to the original size of the input image and display it.

This workflow allows for the generation of new images based on input images and prompts, with the option of translating the prompts to English for non-English input sentences.

### Usage

```bash
pip install AIsketcher
```


#### case1. English Prompt 

```python
import AIsketcher
from PIL import Image
import numpy as np
file_name = 'hello.jpg'

input_text = 'Cute, (hungry), plump, sitting at a table by the beach, warm feeling, beautiful shining eyes, seascape'

num_steps = 50
guidance_scale = 17
seed =6764547109648557242 
low = 140
high = 160

image, canny_image, out_image = AIsketcher.img2img(file_name,  input_text,  num_steps, guidance_scale, seed, low, high, pipe)
Image.fromarray(np.concatenate([image.resize(out_image.size), out_image], axis=1))
```

#### case2. Korean Prompt without IAM AccessRole

```python
import AIsketcher
from PIL import Image
import numpy as np
file_name = 'hello.jpg'
input_text = '귀여운, (배가고픈), 포동포동한, 해변가 식탁에 앉은, 따뜻한 느낌, 아름답고 빛나는 눈, 바다풍경'

trans_info = {
            'region_name' : 'us-east-1', #user region
            'aws_access_key_id' : '{{YOUR_ACCESS_KEY}}',
            'aws_secret_access_key' : '{{YOUR_SECRET_KEY}}',
            'SourceLanguageCode' : 'ko',
            'TargetLanguageCode' : 'en'
        }

num_steps = 50
guidance_scale = 17
seed =6764547109648557242 
low = 140
high = 160

image, canny_image, out_image = AIsketcher.img2img(file_name,  input_text,  num_steps, guidance_scale, seed, low, high, pipe, trans_info)
```

#### case3. Korean Prompt with IAM AccessRole between SageMaker and Translate
```python
import AIsketcher
from PIL import Image
import numpy as np
file_name = 'hello.jpg'
input_text = '귀여운, (배가고픈), 포동포동한, 해변가 식탁에 앉은, 따뜻한 느낌, 아름답고 빛나는 눈, 바다풍경'

trans_info = {
            'region_name' : 'us-east-1', #user region
            'SourceLanguageCode' : 'ko',
            'TargetLanguageCode' : 'en'
        }

num_steps = 50
guidance_scale = 17
seed =6764547109648557242 
low = 140
high = 160

image, canny_image, out_image = AIsketcher.img2img(file_name,  input_text,  num_steps, guidance_scale, seed, low, high, pipe, trans_info)
```


### Default Parameters Used
default_prompt
```text
(8k, best quality, masterpiece:1.2), (realistic, photo-realistic:1.37), ultra-detailed,
```
negative_prompt
```text
NSFW, lowres, ((bad anatomy)), ((bad hands)), text, missing finger, extra digits, fewer digits, blurry, ((mutated hands and fingers)), (poorly drawn face), ((mutation)), ((deformed face)), (ugly), ((bad proportions)), ((extra limbs)), extra face, (double head), (extra head), ((extra feet)), monster, logo, cropped, worst quality, low quality, normal quality, jpeg, humpbacked, long body, long neck, ((jpeg artifacts))
```

| Variables      | Description                                                                                                     |
|----------------|-----------------------------------------------------------------------------------------------------------------|
| num_steps      | Number of steps to run the diffusion process for                                                                |  
| guidance_scale | Creativity value adjustment, a parameter that controls how much the image generation process follows the text prompt | 
| seed           | a number used to initialize the generation in the stable diffusion model                                        |
| low            | Canny Edge Detection lowpass filter threshold                                                                   |
| high           | Canny Edge Detection highpass filter threshold                                                                  |
| pipe           | PNDMScheduler                                                                                |
| trans_info     | Amazon Translate parameters,                                                                                       |



### References 
- `[1]`. Lykon/DreamShaper, Stable Diffusion model, https://huggingface.co/Lykon/DreamShaper
- `[2]`. Text-to-Image Generation with ControlNet Conditioning, https://huggingface.co/docs/diffusers/v0.14.0/en/api/pipelines/stable_diffusion/controlnet
- `[3]`. Controlnet - Canny Version ,https://huggingface.co/lllyasviel/sd-controlnet-canny
- `[4]`. Amazon Translate, https://aws.amazon.com/ko/translate/
- `[5]`. Amazon Translate, source language code, https://docs.aws.amazon.com/translate/latest/dg/what-is-languages.html