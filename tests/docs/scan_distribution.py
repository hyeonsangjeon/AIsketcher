"""Reject secrets, legacy cloud dependencies, and model weights in distributions."""

from __future__ import annotations

import argparse
import io
import re
import tarfile
import zipfile
from collections.abc import Iterable, Iterator
from pathlib import Path

WEIGHT_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".gguf",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
SECRET_PATTERNS = {
    "AWS access key": re.compile(rb"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "OpenAI API key": re.compile(rb"\bsk-(?:proj-)?[A-Za-z0-9_-]{16,}\b"),
    "legacy OpenAI API key": re.compile(rb"\bsk_[A-Za-z0-9_-]{16,}\b"),
    "Hugging Face token": re.compile(rb"\bhf_[A-Za-z0-9]{16,}\b"),
    "GitHub token": re.compile(
        rb"\b(?:gh[pousr]_[A-Za-z0-9_]{16,}|github_pat_[A-Za-z0-9_]{16,})\b"
    ),
    "credential assignment": re.compile(
        rb"(?i)\b(?:access[_ -]?key|secret[_ -]?key|token|password)\s*[:=]\s*['\"]?[A-Za-z0-9._~+/=-]{16,}"
    ),
    "bearer token": re.compile(rb"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{16,}"),
    "private key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}
LEGACY_IMPORT = re.compile(rb"(?m)^\s*(?:from|import)\s+boto3\b")
LEGACY_DEPENDENCY = re.compile(rb"(?im)^Requires-Dist:\s*boto3(?:\s|;|\(|$)")
REQUIRED_WHEEL_FILES = {
    "aisketcher/cli.py",
    "aisketcher/config.py",
    "aisketcher/flux2_backend.py",
    "aisketcher/model_registry.py",
    "aisketcher/prompt_normalization.py",
    "aisketcher/studio_app/__init__.py",
    "aisketcher/studio_app/app.py",
    "aisketcher/studio_app/i18n.py",
    "aisketcher/studio_app/runtime.py",
    "aisketcher/studio_app/styles.css",
    "aisketcher/studio_app/assets/pocket-kingdom/ARTWORK_NOTICE.md",
    "aisketcher/studio_app/assets/pocket-kingdom/manifest.json",
    "aisketcher/studio_app/assets/pocket-kingdom/prepared/control.png",
    "aisketcher/studio_app/assets/pocket-kingdom/prepared/source.png",
    "aisketcher/studio_app/assets/pocket-kingdom/scout/contact-sheet.png",
    "aisketcher/studio_app/assets/pocket-kingdom/scout/scout-01.png",
    "aisketcher/studio_app/assets/pocket-kingdom/scout/scout-02.png",
    "aisketcher/studio_app/assets/pocket-kingdom/scout/scout-03.png",
    "aisketcher/studio_app/assets/pocket-kingdom/scout/scout-04.png",
}

REPOSITORY_IGNORED_PARTS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "site",
        "tmp",
    }
)
MAX_DISTRIBUTION_FILE_BYTES = 20 * 1024 * 1024


def archive_entries(path: Path) -> Iterator[tuple[str, bytes]]:
    if path.suffix == ".whl" or zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if not info.is_dir():
                    yield info.filename, archive.read(info)
        return

    if tarfile.is_tarfile(path):
        with tarfile.open(path, "r:*") as archive:
            for member in archive.getmembers():
                if member.isfile():
                    fileobj = archive.extractfile(member)
                    if fileobj is not None:
                        yield member.name, fileobj.read()
        return

    raise ValueError(f"unsupported distribution archive: {path}")


def oversized_archive_entries(path: Path) -> list[tuple[str, int]]:
    if path.suffix == ".whl" or zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            return [
                (info.filename, info.file_size)
                for info in archive.infolist()
                if not info.is_dir()
                and info.file_size > MAX_DISTRIBUTION_FILE_BYTES
            ]

    if tarfile.is_tarfile(path):
        with tarfile.open(path, "r:*") as archive:
            return [
                (member.name, member.size)
                for member in archive.getmembers()
                if member.isfile()
                and member.size > MAX_DISTRIBUTION_FILE_BYTES
            ]

    raise ValueError(f"unsupported distribution archive: {path}")


def is_package_python(name: str) -> bool:
    normalized = "/" + name.replace("\\", "/")
    return name.endswith(".py") and any(
        marker in normalized
        for marker in ("/src/aisketcher/", "/aisketcher/")
    )


def image_metadata(name: str, data: bytes) -> set[str]:
    try:
        from PIL import Image
    except ImportError:
        return {"Pillow unavailable"}

    try:
        with Image.open(io.BytesIO(data)) as image:
            metadata = {
                key.lower()
                for key, value in image.info.items()
                if value not in (None, b"", "")
            }
            metadata.discard("dpi")
            if image.getexif():
                metadata.add("exif")
            return metadata & {"exif", "xmp", "xml", "photoshop", "icc_profile"}
    except Exception as exc:  # pragma: no cover - archive diagnostics
        return {f"unreadable image: {exc}"}


def scan_archive(path: Path) -> list[str]:
    oversized = oversized_archive_entries(path)
    failures = [
        (
            f"{name}: file exceeds the {MAX_DISTRIBUTION_FILE_BYTES:,}-byte "
            f"distribution limit ({size:,} bytes)"
        )
        for name, size in oversized
    ]
    if oversized:
        return failures
    entries = list(archive_entries(path))
    for name, data in entries:
        normalized = "/" + name.replace("\\", "/")
        suffix = Path(name).suffix.lower()
        if "/test/" in normalized or "/pic/" in normalized:
            failures.append(f"{name}: legacy test/personal asset trees must not be distributed")
        if suffix in WEIGHT_SUFFIXES:
            failures.append(f"{name}: model weight files must not be distributed")

        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(data):
                failures.append(f"{name}: contains a possible {label}")

        if name.endswith(("METADATA", "PKG-INFO")) and LEGACY_DEPENDENCY.search(data):
            failures.append(f"{name}: declares the removed boto3 dependency")
        if is_package_python(name) and LEGACY_IMPORT.search(data):
            failures.append(f"{name}: imports the removed boto3 dependency")

        if suffix in IMAGE_SUFFIXES and any(
            marker in normalized
            for marker in (
                "/docs/assets/pocket-kingdom/",
                "/studio_app/assets/pocket-kingdom/",
            )
        ):
            metadata = image_metadata(name, data)
            if metadata:
                failures.append(f"{name}: embedded metadata present ({', '.join(sorted(metadata))})")
    if path.suffix == ".whl":
        contents = {name for name, _ in entries}
        missing = sorted(REQUIRED_WHEEL_FILES - contents)
        if missing:
            failures.append(f"wheel is missing packaged Studio files: {', '.join(missing)}")
        entry_points = [
            data
            for name, data in entries
            if name.endswith(".dist-info/entry_points.txt")
        ]
        if len(entry_points) != 1 or b"aisketcher = aisketcher.cli:main" not in entry_points[0]:
            failures.append("wheel does not declare the aisketcher console entry point")
        forbidden_roots = ("docs/", "examples/", "pic/", "test/", "tests/")
        leaked = sorted(name for name in contents if name.startswith(forbidden_roots))
        if leaked:
            failures.append(f"wheel contains repository-only trees: {', '.join(leaked)}")
    return failures


def scan(paths: Iterable[Path]) -> list[str]:
    failures: list[str] = []
    for path in paths:
        if not path.is_file():
            failures.append(f"{path}: distribution does not exist")
            continue
        failures.extend(f"{path.name}: {failure}" for failure in scan_archive(path))
    return failures


def scan_repository(root: Path) -> list[str]:
    """Scan current source files for credential-shaped values before packaging."""

    failures: list[str] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part in REPOSITORY_IGNORED_PARTS for part in relative.parts):
            continue
        if not path.is_file() or path.is_symlink():
            continue
        try:
            size = path.stat().st_size
            if size > MAX_DISTRIBUTION_FILE_BYTES:
                failures.append(
                    f"repository/{relative}: file exceeds the "
                    f"{MAX_DISTRIBUTION_FILE_BYTES:,}-byte distribution "
                    f"limit ({size:,} bytes)"
                )
                continue
            data = path.read_bytes()
        except OSError as exc:
            failures.append(f"repository/{relative}: cannot be read ({exc})")
            continue
        if path.suffix.lower() in WEIGHT_SUFFIXES:
            failures.append(
                f"repository/{relative}: model weight files must not be distributed"
            )
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(data):
                failures.append(f"repository/{relative}: contains a possible {label}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path)
    parser.add_argument("archives", nargs="*", type=Path)
    args = parser.parse_args()
    if not args.archives and args.repository is None:
        parser.error("provide at least one archive or --repository")
    failures = scan(args.archives)
    if args.repository is not None:
        failures.extend(scan_repository(args.repository.resolve()))
    if failures:
        print("distribution policy scan failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    scope = f"{len(args.archives)} archive(s)"
    if args.repository is not None:
        scope += " and the repository"
    print(f"distribution policy scan passed for {scope}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
