from diffusers.utils import load_image
import numpy as np
import boto3
import cv2
from PIL import Image, ExifTags
from diffusers import DDPMScheduler
import torch
import random, sys
import boto3


def img2img(img_path, prompt, num_steps=20, guidance_scale=7, seed=0, low=100, high=200, pipe=None, trans_info=None):
    image = load_image(img_path)
    default_prompt = "(8k, best quality, masterpiece:1.2), (realistic, photo-realistic:1.37), ultra-detailed,"
    negative_prompt = "NSFW, lowres, ((bad anatomy)), ((bad hands)), text, missing finger, extra digits, fewer digits, blurry, ((mutated hands and fingers)), (poorly drawn face), ((mutation)), ((deformed face)), (ugly), ((bad proportions)), ((extra limbs)), extra face, (double head), (extra head), ((extra feet)), monster, logo, cropped, worst quality, low quality, normal quality, jpeg, humpbacked, long body, long neck, ((jpeg artifacts))"

    # resize image for fitting stable diffusion 800 x 800
    res_img = resize_image(img_path,800)
    np_image = np.array(res_img)

    #prompt Translate
    if trans_info is not None :
        input_prompt = default_prompt + translate_language(prompt, trans_info)
    else:
        input_prompt = default_prompt + prompt


    canny_image = cv2.Canny(np_image, low, high)
    canny_image = canny_image[:, :, None]
    canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
    canny_image = Image.fromarray(canny_image)

    out_image = pipe(
        input_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=torch.manual_seed(seed),
        image=canny_image
    ).images[0]
    # resizing to original image size
    resized_out_image = out_image.resize(image.size)
    return image, canny_image, resized_out_image

def correct_image_orientation(image):
    try:
        # Get the EXIF data of an image.
        exif = image._getexif()
        if exif is not None:
            # Find the tag that determines the direction of rotation.
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            # Rotate the image according to the rotation direction.
            if exif[orientation] == 2:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            elif exif[orientation] == 3:
                image = image.rotate(180)
            elif exif[orientation] == 4:
                image = image.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
            elif exif[orientation] == 5:
                image = image.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
            elif exif[orientation] == 6:
                image = image.rotate(-90, expand=True)
            elif exif[orientation] == 7:
                image = image.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
            elif exif[orientation] == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # If there is no EXIF information, the image is returned as is.
        pass
    return image


def resize_image(image_path, pixels):
    image = Image.open(image_path)
    image = correct_image_orientation(image)
    # Determine the aspect ratio
    aspect_ratio = image.width / image.height

    # Calculate new dimensions based on the longer side being 800 pixels
    if image.width > image.height:
        new_width = pixels
        new_height = int(pixels / aspect_ratio)
    else:
        new_height = pixels
        new_width = int(pixels * aspect_ratio)

    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.LANCZOS) # update  DeprecationWarning: ANTIALIAS is deprecated and will be removed in Pillow 10 (2023-07-01). Use LANCZOS or Resampling.LANCZOS instead.
    return resized_image

def translate_language(input, trans_info):
    if 'iam_access' in trans_info and trans_info['iam_access']:
        translate = boto3.client(service_name='translate', region_name=trans_info['region_name'], use_ssl=True)
    elif 'iam_access' in trans_info and not trans_info['iam_access']:
        translate = boto3.client(service_name='translate', region_name=trans_info['region_name'], use_ssl=True,
                                 aws_access_key_id=trans_info['aws_access_key_id'], aws_secret_access_key=trans_info['aws_secret_access_key'])
    else:
        translate = boto3.client(service_name='translate', region_name=trans_info['region_name'], use_ssl=True,
                                 aws_access_key_id=trans_info['aws_access_key_id'], aws_secret_access_key=trans_info['aws_secret_access_key'])

    prompt = translate.translate_text(Text=input, SourceLanguageCode=trans_info['SourceLanguageCode'], TargetLanguageCode=trans_info['TargetLanguageCode'])['TranslatedText']
    return prompt