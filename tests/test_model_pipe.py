import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from PIL import Image

import AIsketcher
from AIsketcher import modelPipe


def test_resize_image_preserves_aspect_ratio() -> None:
    source = Image.new("RGB", (400, 200), "white")

    resized = AIsketcher.resize_image(source, 800)

    assert resized.mode == "RGB"
    assert resized.size == (800, 400)


@pytest.mark.parametrize("pixels", [0, -1, 1.5, True])
def test_resize_image_rejects_invalid_size(pixels: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        AIsketcher.resize_image(Image.new("RGB", (10, 10)), pixels)  # type: ignore[arg-type]


def test_img2img_calls_pipeline_and_restores_original_size() -> None:
    source = Image.new("RGB", (100, 50), "white")
    generated = Image.new("RGB", (800, 400), "blue")
    canny = Image.new("RGB", (800, 400), "black")
    pipeline = Mock(return_value=SimpleNamespace(images=[generated]))
    generator = object()

    with (
        patch.object(modelPipe, "_create_canny_image", return_value=canny),
        patch.object(modelPipe, "_seeded_generator", return_value=generator),
    ):
        original, control, output = AIsketcher.img2img(
            source,
            "architectural pencil sketch",
            num_steps=24,
            guidance_scale=8.5,
            seed=42,
            low=80,
            high=160,
            pipe=pipeline,
        )

    assert original.size == (100, 50)
    assert control is canny
    assert output.size == original.size
    prompt = pipeline.call_args.args[0]
    assert prompt.endswith(", architectural pencil sketch")
    assert pipeline.call_args.kwargs == {
        "negative_prompt": AIsketcher.DEFAULT_NEGATIVE_PROMPT,
        "num_inference_steps": 24,
        "guidance_scale": 8.5,
        "generator": generator,
        "image": canny,
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"pipe": None}, "pipe"),
        ({"prompt": "  "}, "prompt"),
        ({"num_steps": 0}, "num_steps"),
        ({"seed": -1}, "seed"),
        ({"low": 200, "high": 100}, "Canny thresholds"),
        ({"image_size": 7}, "image_size"),
    ],
)
def test_img2img_validates_inputs(overrides: dict[str, object], message: str) -> None:
    arguments: dict[str, object] = {
        "img_path": Image.new("RGB", (10, 10)),
        "prompt": "sketch",
        "pipe": Mock(),
    }
    arguments.update(overrides)

    with pytest.raises(ValueError, match=message):
        AIsketcher.img2img(**arguments)  # type: ignore[arg-type]


def test_translate_language_uses_injected_client() -> None:
    client = Mock()
    client.translate_text.return_value = {"TranslatedText": "a quiet mountain village"}

    result = AIsketcher.translate_language(
        "조용한 산골 마을",
        {"SourceLanguageCode": "ko", "TargetLanguageCode": "en"},
        client=client,
    )

    assert result == "a quiet mountain village"
    client.translate_text.assert_called_once_with(
        Text="조용한 산골 마을",
        SourceLanguageCode="ko",
        TargetLanguageCode="en",
    )


def test_translate_language_defaults_to_source_auto() -> None:
    client = Mock()
    client.translate_text.return_value = {"TranslatedText": "hello"}

    AIsketcher.translate_language(
        "bonjour",
        {"TargetLanguageCode": "en"},
        client=client,
    )

    assert client.translate_text.call_args.kwargs["SourceLanguageCode"] == "auto"


def test_translate_language_requires_target_language() -> None:
    with pytest.raises(ValueError, match="TargetLanguageCode"):
        AIsketcher.translate_language("hello", {}, client=Mock())


def test_translate_client_uses_iam_provider_chain_without_static_keys() -> None:
    boto3 = SimpleNamespace(client=Mock(return_value=object()))
    config = {
        "region_name": "ap-northeast-2",
        "iam_access": True,
        "aws_access_key_id": "ignored-access-key",
        "aws_secret_access_key": "ignored-secret-key",
    }

    with patch.dict(sys.modules, {"boto3": boto3}):
        modelPipe._create_translate_client(config)

    boto3.client.assert_called_once_with(
        service_name="translate",
        use_ssl=True,
        region_name="ap-northeast-2",
    )


def test_translate_client_passes_complete_explicit_credentials() -> None:
    boto3 = SimpleNamespace(client=Mock(return_value=object()))
    config = {
        "region_name": "us-east-1",
        "aws_access_key_id": "access-key",
        "aws_secret_access_key": "secret-key",
        "aws_session_token": "session-token",
    }

    with patch.dict(sys.modules, {"boto3": boto3}):
        modelPipe._create_translate_client(config)

    boto3.client.assert_called_once_with(
        service_name="translate",
        use_ssl=True,
        region_name="us-east-1",
        aws_access_key_id="access-key",
        aws_secret_access_key="secret-key",
        aws_session_token="session-token",
    )


def test_package_exports_version_and_public_api() -> None:
    assert AIsketcher.__version__ == "0.1.0"
    assert set(AIsketcher.__all__) >= {"img2img", "resize_image", "translate_language"}
