"""Curated 2026 model metadata and first-download policy.

This module deliberately contains metadata only. It does not import model
runtimes or download weights. Production-ready entries mirror the versioned
presets used by the installer; benchmark-required and experimental entries stay
visible for evaluation without becoming downloadable defaults.

The registry separates two kinds of integrity:

* every repository is pinned to an immutable Hub commit;
* every selected runtime file is pinned to its reviewed SHA-256 digest.

For installable presets this includes Git-backed configuration, scheduler, and
tokenizer files as well as LFS-backed weights. ``download_bytes`` is the sum of
the selected runtime files, rather than every example image or alternate
precision contained in a model repository. The v0.3 installer uses pinned
revisions, an exact file allowlist, and these digests to verify every freshly
downloaded or newly encountered cache. A process-local stat receipt avoids
repeatedly hashing an unchanged multi-GB cache; the persistent marker alone is
never trusted as an authentication token.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

from .errors import ValidationError

_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_APACHE_2 = "apache-2.0"
_APACHE_2_URL = "https://www.apache.org/licenses/LICENSE-2.0"
_MIT_URL = "https://opensource.org/license/mit"


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return str(self.value)


class HashPolicy(_StringEnum):
    """How a future installer must verify a selected model snapshot."""

    PINNED_COMMIT_AND_LFS_SHA256 = "pinned-commit-and-lfs-sha256-v1"
    PINNED_COMMIT_AND_RUNTIME_SHA256 = "pinned-commit-and-runtime-sha256-v2"


class RuntimeFamily(_StringEnum):
    """Runtime adapter family required by a profile."""

    AUTO_ROUTER = "auto-router"
    DIFFUSERS_FLUX2_KLEIN = "diffusers-flux2-klein"
    DIFFUSERS_Z_IMAGE_FUN_CONTROL = "diffusers-z-image-fun-control"
    DIFFUSERS_QWEN_IMAGE_EDIT = "diffusers-qwen-image-edit"
    DIFFUSERS_SDXL_CONTROLNET = "diffusers-sdxl-controlnet"
    MAGE_FLOW_REFERENCE = "mage-flow-reference"


class ModelStatus(_StringEnum):
    """Product-readiness state, independent of license/download eligibility."""

    READY = "ready"
    BENCHMARK_REQUIRED = "benchmark-required"
    PRO_QUALITY = "pro-quality"
    LEGACY = "legacy"
    EXPERIMENTAL = "experimental"


class InputMode(_StringEnum):
    TEXT = "text"
    SKETCH = "sketch"
    PHOTO = "photo"
    IMAGE_EDIT = "image-edit"
    MULTI_REFERENCE = "multi-reference"


class ControlType(_StringEnum):
    IMAGE_EDIT = "image-edit"
    MULTI_REFERENCE = "multi-reference"
    CANNY = "canny"
    HED = "hed"
    SCRIBBLE = "scribble"
    DEPTH = "depth"
    POSE = "pose"
    TILE = "tile"
    INPAINT = "inpaint"


@dataclass(frozen=True, slots=True)
class VerifiedFile:
    """One selected Hub runtime file with its expected digest and size."""

    path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if (
            not self.path
            or "\\" in self.path
            or self.path.startswith("/")
            or ".." in self.path.split("/")
        ):
            raise ValidationError("VerifiedFile.path must be a safe repository-relative path")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes <= 0
        ):
            raise ValidationError("VerifiedFile.size_bytes must be a positive integer")
        if not _SHA256_PATTERN.fullmatch(self.sha256):
            raise ValidationError("VerifiedFile.sha256 must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    """Pinned repository component used by one or more curated profiles."""

    artifact_id: str
    model_id: str
    revision: str
    role: str
    files: tuple[VerifiedFile, ...]
    license_id: str
    license_url: str
    redistribution_notice: str
    commercial_use: bool
    public: bool
    gated: bool
    territory_exclusions: tuple[str, ...] = ()
    hash_policy: HashPolicy = HashPolicy.PINNED_COMMIT_AND_LFS_SHA256
    metadata_verified_on: str = "2026-07-24"

    def __post_init__(self) -> None:
        if not self.artifact_id or not self.model_id or "/" not in self.model_id:
            raise ValidationError("ModelArtifact requires an id and a namespaced model_id")
        if not _COMMIT_PATTERN.fullmatch(self.revision):
            raise ValidationError("ModelArtifact.revision must be an immutable commit SHA")
        if not self.role or not self.files:
            raise ValidationError("ModelArtifact requires a role and verified files")
        paths = tuple(item.path for item in self.files)
        if len(paths) != len(set(paths)):
            raise ValidationError("ModelArtifact file paths must be unique")
        if not self.license_id or not self.license_url.startswith("https://"):
            raise ValidationError("ModelArtifact requires a license id and HTTPS license URL")
        if not self.redistribution_notice.strip():
            raise ValidationError("ModelArtifact requires a redistribution notice")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", self.metadata_verified_on):
            raise ValidationError("metadata_verified_on must use YYYY-MM-DD")

    @property
    def download_bytes(self) -> int:
        return sum(item.size_bytes for item in self.files)

    @property
    def model_card_url(self) -> str:
        return f"https://huggingface.co/{self.model_id}/tree/{self.revision}"


@dataclass(frozen=True, slots=True)
class AutoRoute:
    """A declared Auto route; detection itself is implemented elsewhere."""

    input_kind: str
    profile_id: str

    def __post_init__(self) -> None:
        if not self.input_kind.strip() or not self.profile_id.strip():
            raise ValidationError("AutoRoute fields cannot be empty")


@dataclass(frozen=True, slots=True)
class CuratedModelProfile:
    """Designer-facing model option composed from pinned Hub artifacts."""

    profile_id: str
    label: str
    artifacts: tuple[ModelArtifact, ...]
    runtime_family: RuntimeFamily
    optional_dependency: str
    input_modes: tuple[InputMode, ...]
    control_types: tuple[ControlType, ...]
    minimum_vram_gb: int
    recommended_vram_gb: int
    status: ModelStatus
    best_for: str
    zero_click_enabled: bool
    auto_routes: tuple[AutoRoute, ...] = ()
    tested_devices: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.profile_id or not self.label or not self.best_for.strip():
            raise ValidationError("CuratedModelProfile requires id, label, and best_for")
        if not self.optional_dependency:
            raise ValidationError("CuratedModelProfile.optional_dependency cannot be empty")
        if (
            isinstance(self.minimum_vram_gb, bool)
            or isinstance(self.recommended_vram_gb, bool)
            or not isinstance(self.minimum_vram_gb, int)
            or not isinstance(self.recommended_vram_gb, int)
            or self.minimum_vram_gb <= 0
            or self.recommended_vram_gb < self.minimum_vram_gb
        ):
            raise ValidationError("CuratedModelProfile VRAM values are invalid")
        if not self.input_modes:
            raise ValidationError("CuratedModelProfile requires at least one input mode")
        if len(self.input_modes) != len(set(self.input_modes)):
            raise ValidationError("CuratedModelProfile input modes must be unique")
        if len(self.control_types) != len(set(self.control_types)):
            raise ValidationError("CuratedModelProfile control types must be unique")
        artifact_ids = tuple(item.artifact_id for item in self.artifacts)
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValidationError("CuratedModelProfile artifacts must be unique")
        if self.runtime_family is RuntimeFamily.AUTO_ROUTER:
            if self.artifacts or not self.auto_routes or self.zero_click_enabled:
                raise ValidationError(
                    "Auto router must contain routes, no artifacts, and no direct download"
                )
        elif self.auto_routes:
            raise ValidationError("Only the Auto router may declare auto_routes")
        elif not self.artifacts:
            raise ValidationError("A concrete model profile requires artifacts")

        if self.zero_click_enabled:
            if self.status is not ModelStatus.READY:
                raise ValidationError(
                    "Zero-click profiles must pass the hardware benchmark and be READY"
                )
            for artifact in self.artifacts:
                if (
                    artifact.license_id != _APACHE_2
                    or not artifact.commercial_use
                    or not artifact.public
                    or artifact.gated
                    or artifact.territory_exclusions
                    or not artifact.files
                ):
                    raise ValidationError(
                        "Zero-click profiles require verified, public, ungated, "
                        "commercial Apache-2.0 artifacts with no territory exclusions"
                    )

    @property
    def download_bytes(self) -> int:
        return sum(item.download_bytes for item in self.artifacts)

    @property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(item.model_id for item in self.artifacts)

    @property
    def revisions(self) -> tuple[str, ...]:
        return tuple(item.revision for item in self.artifacts)

    @property
    def license_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.license_id for item in self.artifacts))

    @property
    def gated(self) -> bool:
        return any(item.gated for item in self.artifacts)

    @property
    def territory_exclusions(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                territory
                for artifact in self.artifacts
                for territory in artifact.territory_exclusions
            )
        )


def _file(path: str, size_bytes: int, sha256: str) -> VerifiedFile:
    return VerifiedFile(path=path, size_bytes=size_bytes, sha256=sha256)


_APACHE_NOTICE = (
    "Weights are not bundled with AIsketcher. Preserve the upstream Apache-2.0 "
    "license and notices when redistributing a downloaded snapshot."
)

_ARTIFACTS = {
    "flux2-klein-4b": ModelArtifact(
        artifact_id="flux2-klein-4b",
        model_id="black-forest-labs/FLUX.2-klein-4B",
        revision="e7b7dc27f91deacad38e78976d1f2b499d76a294",
        role="base-edit",
        files=(
            _file(
                "model_index.json",
                446,
                "51a76cb1cf3ed37423a1128c79c22faee8e6fbe7f5aaeb737f0a258930dbaac0",
            ),
            _file(
                "scheduler/scheduler_config.json",
                486,
                "067afb012cef64553a763447d1efd93daeffcc0123ca7e25b09f8de20b90762e",
            ),
            _file(
                "text_encoder/config.json",
                1_536,
                "214b4c29a0d975e9fddf9994a5673f22cb2c4c5750352f9227c2c3251ebeab40",
            ),
            _file(
                "text_encoder/model-00001-of-00002.safetensors",
                4_967_215_360,
                "8c0506e7f4936fa7e26183a4fd8da4e2bdbc5990ba64ae441f965d51228f36ea",
            ),
            _file(
                "text_encoder/model-00002-of-00002.safetensors",
                3_077_766_632,
                "82f2bd839378541b0557bfabaf37c7d3d637071fdcb73302dedd7cf61162ce07",
            ),
            _file(
                "text_encoder/model.safetensors.index.json",
                32_855,
                "06b3d5319b6d76d1a4a2433419180016cfd54ed62d086a5e6567a809f8c82634",
            ),
            _file(
                "tokenizer/added_tokens.json",
                707,
                "c0284b582e14987fbd3d5a2cb2bd139084371ed9acbae488829a1c900833c680",
            ),
            _file(
                "tokenizer/chat_template.jinja",
                4_168,
                "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
            ),
            _file(
                "tokenizer/merges.txt",
                1_671_853,
                "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
            ),
            _file(
                "tokenizer/special_tokens_map.json",
                613,
                "76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd",
            ),
            _file(
                "tokenizer/tokenizer.json",
                11_422_654,
                "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
            ),
            _file(
                "tokenizer/tokenizer_config.json",
                5_404,
                "443bfa629eb16387a12edbf92a76f6a6f10b2af3b53d87ba1550adfcf45f7fa0",
            ),
            _file(
                "tokenizer/vocab.json",
                2_776_833,
                "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
            ),
            _file(
                "transformer/config.json",
                541,
                "09733c74a3da6d17dd0a0472a091a8950c7c6935889c32c16cc800ede05029de",
            ),
            _file(
                "transformer/diffusion_pytorch_model.safetensors",
                7_751_109_744,
                "9f29f9edcfdae452a653ffb51a534ca4decd389952c225724ff3b94042612a6e",
            ),
            _file(
                "vae/config.json",
                821,
                "0d6dfb69ae95a5e2ac9836284bbb63d8b38ce67b25ba2dff380752b2a10ab948",
            ),
            _file(
                "vae/diffusion_pytorch_model.safetensors",
                168_120_878,
                "ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04",
            ),
        ),
        license_id=_APACHE_2,
        license_url=_APACHE_2_URL,
        redistribution_notice=_APACHE_NOTICE,
        commercial_use=True,
        public=True,
        gated=False,
        hash_policy=HashPolicy.PINNED_COMMIT_AND_RUNTIME_SHA256,
    ),
    "flux2-small-decoder": ModelArtifact(
        artifact_id="flux2-small-decoder",
        model_id="black-forest-labs/FLUX.2-small-decoder",
        revision="a3efc24f613ef42d9428af62fdbd6f5fd8856c4a",
        role="decoder",
        files=(
            _file(
                "config.json",
                842,
                "88641eb0ebe46877b32822fb91aff828574765ef6b2b4bbc73d379051d006267",
            ),
            _file(
                "diffusion_pytorch_model.safetensors",
                249_521_340,
                "d8d52ba036475f5fb07c8b435e176d3d97ebfa82f0d1a1c317f9cc1e25bd013b",
            ),
        ),
        license_id=_APACHE_2,
        license_url=_APACHE_2_URL,
        redistribution_notice=_APACHE_NOTICE,
        commercial_use=True,
        public=True,
        gated=False,
        hash_policy=HashPolicy.PINNED_COMMIT_AND_RUNTIME_SHA256,
    ),
    "z-image-turbo": ModelArtifact(
        artifact_id="z-image-turbo",
        model_id="Tongyi-MAI/Z-Image-Turbo",
        revision="f332072aa78be7aecdf3ee76d5c247082da564a6",
        role="base-generation",
        files=(
            _file(
                "text_encoder/model-00001-of-00003.safetensors",
                3_957_900_840,
                "328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223",
            ),
            _file(
                "text_encoder/model-00002-of-00003.safetensors",
                3_987_450_520,
                "6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5",
            ),
            _file(
                "text_encoder/model-00003-of-00003.safetensors",
                99_630_640,
                "7ca841ee75b9c61267c0c6148fd8d096d3d21b6d3e161256a9b878154f91fc52",
            ),
            _file(
                "tokenizer/tokenizer.json",
                11_422_654,
                "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
            ),
            _file(
                "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
                9_973_693_184,
                "95facd593e2549e8252acb571c653d57f7ddb7f1060d4e81712f152555a88804",
            ),
            _file(
                "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
                9_973_714_824,
                "a4bbe43ee184a1fb5af4b412d27555f532893bdc3165b1149e304ed82b5d7015",
            ),
            _file(
                "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
                4_672_282_880,
                "aba4e37a590e63210878160a718d916d80398f4e1f78ab6c9b2b2a00d92769fa",
            ),
            _file(
                "vae/diffusion_pytorch_model.safetensors",
                167_666_902,
                "f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3",
            ),
        ),
        license_id=_APACHE_2,
        license_url=_APACHE_2_URL,
        redistribution_notice=_APACHE_NOTICE,
        commercial_use=True,
        public=True,
        gated=False,
    ),
    "z-image-union-lite-2602": ModelArtifact(
        artifact_id="z-image-union-lite-2602",
        model_id="alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1",
        revision="5155fc56d17821007d6f62ac192c09e0f0e72016",
        role="controlnet-union",
        files=(
            _file(
                "Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2602-8steps.safetensors",
                2_016_627_488,
                "3ea098db9bd145be525c7e2366920b6d76c5ffd46b3d7aa8169bbc943fdaee35",
            ),
        ),
        license_id=_APACHE_2,
        license_url=_APACHE_2_URL,
        redistribution_notice=_APACHE_NOTICE,
        commercial_use=True,
        public=True,
        gated=False,
    ),
    "qwen-image-edit-2511": ModelArtifact(
        artifact_id="qwen-image-edit-2511",
        model_id="Qwen/Qwen-Image-Edit-2511",
        revision="6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9",
        role="base-edit",
        files=(
            _file(
                "processor/tokenizer.json",
                11_421_896,
                "9c5ae00e602b8860cbd784ba82a8aa14e8feecec692e7076590d014d7b7fdafa",
            ),
            _file(
                "text_encoder/model-00001-of-00004.safetensors",
                4_968_243_304,
                "d725335e4ea2399be706469e4b8807716a8fa64bd03468252e9f7acf2415fee4",
            ),
            _file(
                "text_encoder/model-00002-of-00004.safetensors",
                4_991_495_816,
                "b1830db6908dcc76df3a71492acbcf2b8cac130114cf1f3c2d9edae8de8c6de3",
            ),
            _file(
                "text_encoder/model-00003-of-00004.safetensors",
                4_932_751_040,
                "09c1807c6d00d7cab94f7db39d4c02ebb8537225ccde383861ac48db97945aa6",
            ),
            _file(
                "text_encoder/model-00004-of-00004.safetensors",
                1_691_924_384,
                "5dd068336d14d45ffb43cef374d286cc6ba9d8741b028f90a7d040d847961f4a",
            ),
            _file(
                "transformer/diffusion_pytorch_model-00001-of-00005.safetensors",
                9_973_578_592,
                "2a0c30c9ba44a5f11c21ca139e37951430bbde814ff4e0b5b1a68b80530e7a1a",
            ),
            _file(
                "transformer/diffusion_pytorch_model-00002-of-00005.safetensors",
                9_987_326_072,
                "54ec249b07b4376e19cf16b764054f03ca03ae2cfbd9939453e2085f4e9bd259",
            ),
            _file(
                "transformer/diffusion_pytorch_model-00003-of-00005.safetensors",
                9_987_307_440,
                "c55157843525653161e8f6af5acc670ba3aceff04284f7cf657199d24d065e16",
            ),
            _file(
                "transformer/diffusion_pytorch_model-00004-of-00005.safetensors",
                9_930_685_712,
                "ffcfb5a4895702635890a67bad183591e0ae515d794bdcb26e217b27a7f6d12d",
            ),
            _file(
                "transformer/diffusion_pytorch_model-00005-of-00005.safetensors",
                982_130_472,
                "2b2556b736629e10a5a0dfa14606f2057f4f81c2ba53f94103682c7ac42d4940",
            ),
            _file(
                "vae/diffusion_pytorch_model.safetensors",
                253_806_966,
                "0c8bc8b758c649abef9ea407b95408389a3b2f610d0d10fcb054fe171d0a8344",
            ),
        ),
        license_id=_APACHE_2,
        license_url=_APACHE_2_URL,
        redistribution_notice=_APACHE_NOTICE,
        commercial_use=True,
        public=True,
        gated=False,
    ),
    "sdxl-base-1.0": ModelArtifact(
        artifact_id="sdxl-base-1.0",
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        revision="462165984030d82259a11f4367a4eed129e94a7b",
        role="base-generation",
        files=(
            _file(
                "model_index.json",
                609,
                "6d7b93508390ab91ac5bfbe4aeb4dc2d83f7bb1b05fb069d714b5b0c75f70d44",
            ),
            _file(
                "scheduler/scheduler_config.json",
                479,
                "af3e45a949aff8b8341ab8b811429ec03fee857a700a1d9477363e4fff9666e2",
            ),
            _file(
                "text_encoder/config.json",
                565,
                "39b8b2e4b1949e36969caa425b6c81c68bace99198dd9078ce05d16ad401fe7f",
            ),
            _file(
                "text_encoder/model.fp16.safetensors",
                246_144_152,
                "660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd",
            ),
            _file(
                "text_encoder_2/config.json",
                575,
                "a892d1c3a69a7e9247a24de2bc1d5891e3109a54696e53be20093af671072c34",
            ),
            _file(
                "text_encoder_2/model.fp16.safetensors",
                1_389_382_176,
                "ec310df2af79c318e24d20511b601a591ca8cd4f1fce1d8dff822a356bcdb1f4",
            ),
            _file(
                "tokenizer/merges.txt",
                524_619,
                "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a",
            ),
            _file(
                "tokenizer/special_tokens_map.json",
                472,
                "c4864a9376a8401918425bed71fc14fc0e81f9b59ec45c1cf96cccb2df508eac",
            ),
            _file(
                "tokenizer/tokenizer_config.json",
                737,
                "19d7b034cb0cc3ce9766c2231373ab8aa8991fc72e2c8f76558bfaae3de0d563",
            ),
            _file(
                "tokenizer/vocab.json",
                1_059_962,
                "e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349",
            ),
            _file(
                "tokenizer_2/merges.txt",
                524_619,
                "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a",
            ),
            _file(
                "tokenizer_2/special_tokens_map.json",
                460,
                "f118ab3a983206e4f32583448de6bd6aae4ee21869135cef1f5848a753cdaab6",
            ),
            _file(
                "tokenizer_2/tokenizer_config.json",
                725,
                "c9d23941f76a41cbd50eda9290f57be7828f0a7a677939e9ef181f7e12bd1bdf",
            ),
            _file(
                "tokenizer_2/vocab.json",
                1_059_962,
                "e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349",
            ),
            _file(
                "unet/config.json",
                1_680,
                "30ebc70750223e59006f7f2b4e1e6c102570aa19a9c4ae3e1fbe7591332dbae6",
            ),
            _file(
                "unet/diffusion_pytorch_model.fp16.safetensors",
                5_135_149_760,
                "83e012a805b84c7ca28e5646747c90a243c65c8ba4f070e2d7ddc9d74661e139",
            ),
            _file(
                "vae/config.json",
                642,
                "0b331c8ac22ded5f9997a144a575c1d113d6169aff262c353f39015bd24a6264",
            ),
            _file(
                "vae/diffusion_pytorch_model.fp16.safetensors",
                167_335_342,
                "bcb60880a46b63dea58e9bc591abe15f8350bde47b405f9c38f4be70c6161e68",
            ),
        ),
        license_id="openrail++",
        license_url=(
            "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/"
            "blob/462165984030d82259a11f4367a4eed129e94a7b/LICENSE.md"
        ),
        redistribution_notice=(
            "Weights are not bundled. Redistribution and use remain subject to "
            "the pinned upstream OpenRAIL++ license and its use restrictions."
        ),
        commercial_use=True,
        public=True,
        gated=False,
        hash_policy=HashPolicy.PINNED_COMMIT_AND_RUNTIME_SHA256,
    ),
    "sdxl-canny-small": ModelArtifact(
        artifact_id="sdxl-canny-small",
        model_id="diffusers/controlnet-canny-sdxl-1.0-small",
        revision="edd85f64c5f87dfb6d73762949d9daca16389518",
        role="controlnet-canny",
        files=(
            _file(
                "config.json",
                1_259,
                "3ef4f560b0d21eadee5da2ac4fa047603ed85aa61d24ef32017a7b7d37e97a2d",
            ),
            _file(
                "diffusion_pytorch_model.fp16.safetensors",
                320_237_179,
                "fde4888a5f0a5648118991cc50e0ac4d60a2356dbaddf5e0649dd69c1119a2f9",
            ),
        ),
        license_id="openrail++",
        license_url=(
            "https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0-small/"
            "tree/edd85f64c5f87dfb6d73762949d9daca16389518"
        ),
        redistribution_notice=(
            "Weights are not bundled. Review the pinned model card and preserve "
            "the upstream OpenRAIL++ terms when redistributing this adapter."
        ),
        commercial_use=True,
        public=True,
        gated=False,
        hash_policy=HashPolicy.PINNED_COMMIT_AND_RUNTIME_SHA256,
    ),
    "sdxl-canny-quality": ModelArtifact(
        artifact_id="sdxl-canny-quality",
        model_id="diffusers/controlnet-canny-sdxl-1.0",
        revision="eb115a19a10d14909256db740ed109532ab1483c",
        role="controlnet-canny",
        files=(
            _file(
                "config.json",
                1_309,
                "81f9bfed178f666c332892cd568a78ed8918fe08aaee994d05428a75ab44f2ca",
            ),
            _file(
                "diffusion_pytorch_model.fp16.safetensors",
                2_502_139_136,
                "b2e7d3921058a442cc80430d1ec8847f42599c705e2451c95e77cf4dcf8d6c25",
            ),
        ),
        license_id="openrail++",
        license_url=(
            "https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0/"
            "tree/eb115a19a10d14909256db740ed109532ab1483c"
        ),
        redistribution_notice=(
            "Weights are not bundled. Review the pinned model card and preserve "
            "the upstream OpenRAIL++ terms when redistributing this adapter."
        ),
        commercial_use=True,
        public=True,
        gated=False,
        hash_policy=HashPolicy.PINNED_COMMIT_AND_RUNTIME_SHA256,
    ),
    "mage-flow-edit-turbo": ModelArtifact(
        artifact_id="mage-flow-edit-turbo",
        model_id="microsoft/Mage-Flow-Edit-Turbo",
        revision="14427bd7627d3a25436497a5939e1096f6a0d523",
        role="base-edit",
        files=(
            _file(
                "text_encoder/model-00001-of-00002.safetensors",
                4_967_229_296,
                "30a01a0556622645a3cce87b655bbbbbc1f170c196099f1b666c93202c3339a9",
            ),
            _file(
                "text_encoder/model-00002-of-00002.safetensors",
                3_908_490_048,
                "046296a2a387efb43b0c997d5833c789604d168834f6e0d3064bf7bb13d002a6",
            ),
            _file(
                "transformer/diffusion_pytorch_model.safetensors",
                8_231_536_760,
                "29c3726ecd64afe149eef28af3e27b6b40de52646bfd16757a37da4b6fbcf288",
            ),
            _file(
                "vae/diffusion_pytorch_model.safetensors",
                345_053_056,
                "34e076dc1e8a15321e1e07be5111d59cf16dd10b804b7c7e20b4de29013427e0",
            ),
        ),
        license_id="mit",
        license_url=_MIT_URL,
        redistribution_notice=(
            "Weights are not bundled. Preserve the upstream MIT copyright and "
            "license notice if a snapshot is redistributed."
        ),
        commercial_use=True,
        public=True,
        gated=False,
    ),
}

MODEL_ARTIFACTS: Mapping[str, ModelArtifact] = MappingProxyType(_ARTIFACTS)

_PROFILES = {
    "auto": CuratedModelProfile(
        profile_id="auto",
        label="Auto — T4 validated",
        artifacts=(),
        runtime_family=RuntimeFamily.AUTO_ROUTER,
        optional_dependency="aisketcher[local]",
        input_modes=(InputMode.SKETCH, InputMode.PHOTO),
        control_types=(),
        minimum_vram_gb=16,
        recommended_vram_gb=24,
        status=ModelStatus.READY,
        best_for="A safe default for both sketches and photos on a 16 GB NVIDIA T4.",
        zero_click_enabled=False,
        auto_routes=(
            AutoRoute("sparse-sketch-or-line-art", "flux2-klein-4b"),
            AutoRoute("photo-or-semantic-edit", "flux2-klein-4b"),
        ),
        tested_devices=("NVIDIA Tesla T4 16 GB / Azure Standard_NC4as_T4_v3",),
    ),
    "flux2-klein-4b": CuratedModelProfile(
        profile_id="flux2-klein-4b",
        label="FLUX.2 Klein 4B — Fast Edit",
        artifacts=(
            MODEL_ARTIFACTS["flux2-klein-4b"],
            MODEL_ARTIFACTS["flux2-small-decoder"],
        ),
        runtime_family=RuntimeFamily.DIFFUSERS_FLUX2_KLEIN,
        optional_dependency="aisketcher[local]",
        input_modes=(
            InputMode.TEXT,
            InputMode.SKETCH,
            InputMode.PHOTO,
            InputMode.IMAGE_EDIT,
        ),
        control_types=(ControlType.IMAGE_EDIT,),
        minimum_vram_gb=13,
        recommended_vram_gb=16,
        status=ModelStatus.READY,
        best_for=(
            "Fast sketch rendering, photo restyling, and instruction edits "
            "on a 16 GB NVIDIA T4."
        ),
        zero_click_enabled=True,
        tested_devices=("NVIDIA Tesla T4 16 GB / Azure Standard_NC4as_T4_v3",),
    ),
    "z-image-turbo-union-lite": CuratedModelProfile(
        profile_id="z-image-turbo-union-lite",
        label="Z-Image Turbo + PAI Union Lite — Structure",
        artifacts=(
            MODEL_ARTIFACTS["z-image-turbo"],
            MODEL_ARTIFACTS["z-image-union-lite-2602"],
        ),
        runtime_family=RuntimeFamily.DIFFUSERS_Z_IMAGE_FUN_CONTROL,
        optional_dependency="aisketcher[local]",
        input_modes=(InputMode.TEXT, InputMode.SKETCH),
        control_types=(
            ControlType.CANNY,
            ControlType.HED,
            ControlType.SCRIBBLE,
            ControlType.DEPTH,
            ControlType.POSE,
            ControlType.TILE,
            ControlType.INPAINT,
        ),
        minimum_vram_gb=16,
        recommended_vram_gb=24,
        status=ModelStatus.BENCHMARK_REQUIRED,
        best_for="Pencil sketches, line art, and controlled structure rendering.",
        zero_click_enabled=False,
    ),
    "qwen-image-edit-quality": CuratedModelProfile(
        profile_id="qwen-image-edit-quality",
        label="Qwen Image Edit 2511 — Pro Quality",
        artifacts=(MODEL_ARTIFACTS["qwen-image-edit-2511"],),
        runtime_family=RuntimeFamily.DIFFUSERS_QWEN_IMAGE_EDIT,
        optional_dependency="aisketcher[local]",
        input_modes=(
            InputMode.PHOTO,
            InputMode.IMAGE_EDIT,
            InputMode.MULTI_REFERENCE,
        ),
        control_types=(ControlType.IMAGE_EDIT, ControlType.MULTI_REFERENCE),
        minimum_vram_gb=40,
        recommended_vram_gb=80,
        status=ModelStatus.PRO_QUALITY,
        best_for="Identity-sensitive portrait, product, and geometric edits on A100/H100.",
        zero_click_enabled=False,
    ),
    "sdxl-canny-legacy": CuratedModelProfile(
        profile_id="sdxl-canny-legacy",
        label="SDXL Canny — Legacy Replay",
        artifacts=(
            MODEL_ARTIFACTS["sdxl-base-1.0"],
            MODEL_ARTIFACTS["sdxl-canny-small"],
        ),
        runtime_family=RuntimeFamily.DIFFUSERS_SDXL_CONTROLNET,
        optional_dependency="aisketcher[local]",
        input_modes=(InputMode.TEXT, InputMode.SKETCH),
        control_types=(ControlType.CANNY,),
        minimum_vram_gb=12,
        recommended_vram_gb=16,
        status=ModelStatus.LEGACY,
        best_for="Replaying existing SDXL Canny manifests and strict edge conditioning.",
        zero_click_enabled=False,
    ),
    "mage-flow-edit-turbo-experimental": CuratedModelProfile(
        profile_id="mage-flow-edit-turbo-experimental",
        label="Mage Flow Edit Turbo — Experimental",
        artifacts=(MODEL_ARTIFACTS["mage-flow-edit-turbo"],),
        runtime_family=RuntimeFamily.MAGE_FLOW_REFERENCE,
        optional_dependency="aisketcher[local]",
        input_modes=(
            InputMode.TEXT,
            InputMode.PHOTO,
            InputMode.IMAGE_EDIT,
            InputMode.MULTI_REFERENCE,
        ),
        control_types=(ControlType.IMAGE_EDIT, ControlType.MULTI_REFERENCE),
        minimum_vram_gb=24,
        recommended_vram_gb=40,
        status=ModelStatus.EXPERIMENTAL,
        best_for="Evaluation of a newly released instruction-editing model.",
        zero_click_enabled=False,
    ),
}


def _validate_registry(profiles: Mapping[str, CuratedModelProfile]) -> None:
    for key, profile in profiles.items():
        if key != profile.profile_id:
            raise RuntimeError(f"Profile key {key!r} does not match its profile_id")
        for route in profile.auto_routes:
            if route.profile_id not in profiles or route.profile_id == "auto":
                raise RuntimeError(f"Auto route points to unknown profile {route.profile_id!r}")


_validate_registry(_PROFILES)
CURATED_MODEL_PROFILES: Mapping[str, CuratedModelProfile] = MappingProxyType(_PROFILES)


def get_model_profile(profile_id: str) -> CuratedModelProfile:
    """Return one curated profile without performing downloads or imports."""

    try:
        return CURATED_MODEL_PROFILES[profile_id]
    except KeyError as exc:
        available = ", ".join(sorted(CURATED_MODEL_PROFILES))
        raise ValidationError(
            f"Unknown model profile {profile_id!r}. Available profiles: {available}"
        ) from exc


def zero_click_profiles() -> tuple[CuratedModelProfile, ...]:
    """Return benchmarked profiles allowed in explicit first-use setup."""

    return tuple(
        profile
        for profile in CURATED_MODEL_PROFILES.values()
        if profile.zero_click_enabled
    )


__all__ = [
    "CURATED_MODEL_PROFILES",
    "MODEL_ARTIFACTS",
    "AutoRoute",
    "ControlType",
    "CuratedModelProfile",
    "HashPolicy",
    "InputMode",
    "ModelArtifact",
    "ModelStatus",
    "RuntimeFamily",
    "VerifiedFile",
    "get_model_profile",
    "zero_click_profiles",
]
