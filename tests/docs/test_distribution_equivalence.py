from __future__ import annotations

import gzip
import io
import stat
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest
from tests.docs.verify_distribution_equivalence import (
    DistributionEquivalenceError,
    verify_equivalent,
)

SCRIPT = Path(__file__).with_name("verify_distribution_equivalence.py")


def _write_wheel(
    path: Path,
    entries: list[tuple[str, bytes, int]],
    *,
    timestamp: tuple[int, int, int, int, int, int] = (2024, 1, 1, 0, 0, 0),
    compression: int = zipfile.ZIP_DEFLATED,
) -> None:
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        archive.comment = f"container-{timestamp}".encode()
        for name, data, mode in entries:
            info = zipfile.ZipInfo(name, date_time=timestamp)
            info.create_system = 3
            info.external_attr = mode << 16
            info.compress_type = compression
            archive.writestr(info, data)


def _tar_bytes(
    entries: list[tuple[str, bytes, int, bytes, str]],
    *,
    member_mtime: int,
    owner: tuple[int, int, str, str],
) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        for name, data, mode, entry_type, linkname in entries:
            info = tarfile.TarInfo(name)
            info.mode = mode
            info.mtime = member_mtime
            info.uid, info.gid, info.uname, info.gname = owner
            info.type = entry_type
            info.linkname = linkname
            if entry_type in {tarfile.REGTYPE, tarfile.AREGTYPE}:
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
            else:
                archive.addfile(info)
    return output.getvalue()


def _write_sdist(
    path: Path,
    entries: list[tuple[str, bytes, int, bytes, str]],
    *,
    member_mtime: int,
    gzip_mtime: int,
    gzip_name: str,
    owner: tuple[int, int, str, str] = (1000, 1000, "builder", "builder"),
) -> None:
    tar_data = _tar_bytes(entries, member_mtime=member_mtime, owner=owner)
    with path.open("wb") as output, gzip.GzipFile(
        filename=gzip_name,
        mode="wb",
        fileobj=output,
        mtime=gzip_mtime,
    ) as compressed:
        compressed.write(tar_data)


def test_wheels_ignore_container_metadata_order_and_implicit_directories(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.whl"
    published = tmp_path / "published.whl"
    _write_wheel(
        candidate,
        [
            ("pkg/", b"", stat.S_IFDIR | 0o755),
            ("pkg/data.txt", b"same data", stat.S_IFREG | 0o644),
            ("pkg/tool", b"#!/bin/sh\n", stat.S_IFREG | 0o755),
        ],
        timestamp=(2024, 1, 1, 0, 0, 0),
        compression=zipfile.ZIP_STORED,
    )
    _write_wheel(
        published,
        [
            ("pkg/tool", b"#!/bin/sh\n", stat.S_IFREG | 0o755),
            ("pkg/data.txt", b"same data", stat.S_IFREG | 0o600),
        ],
        timestamp=(2026, 7, 24, 2, 0, 0),
        compression=zipfile.ZIP_DEFLATED,
    )

    verify_equivalent(candidate, published)


def test_sdists_ignore_tar_and_gzip_container_metadata(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.tar.gz"
    published = tmp_path / "published.tar.gz"
    candidate_entries = [
        ("project-1.0/", b"", 0o755, tarfile.DIRTYPE, ""),
        ("project-1.0/PKG-INFO", b"Version: 1.0\n", 0o644, tarfile.REGTYPE, ""),
        ("project-1.0/bin/run", b"#!/bin/sh\n", 0o755, tarfile.REGTYPE, ""),
    ]
    published_entries = [
        ("project-1.0/bin/run", b"#!/bin/sh\n", 0o755, tarfile.REGTYPE, ""),
        ("project-1.0/PKG-INFO", b"Version: 1.0\n", 0o600, tarfile.REGTYPE, ""),
    ]
    _write_sdist(
        candidate,
        candidate_entries,
        member_mtime=1,
        gzip_mtime=2,
        gzip_name="candidate-source.tar",
    )
    _write_sdist(
        published,
        published_entries,
        member_mtime=9_999_999,
        gzip_mtime=8_888_888,
        gzip_name="published-source.tar",
        owner=(42, 43, "other", "group"),
    )

    verify_equivalent(candidate, published)


@pytest.mark.parametrize(
    ("candidate_entries", "published_entries", "message"),
    [
        (
            [("pkg/data", b"candidate", stat.S_IFREG | 0o644)],
            [("pkg/data", b"published", stat.S_IFREG | 0o644)],
            "content differs",
        ),
        (
            [("pkg/candidate", b"same", stat.S_IFREG | 0o644)],
            [("pkg/published", b"same", stat.S_IFREG | 0o644)],
            "candidate-only member",
        ),
        (
            [
                ("pkg/data", b"same", stat.S_IFREG | 0o755),
            ],
            [
                ("pkg/data", b"same", stat.S_IFREG | 0o644),
            ],
            "executable bit differs",
        ),
        (
            [("pkg/", b"", stat.S_IFDIR | 0o755)],
            [("pkg/", b"", stat.S_IFDIR | 0o644)],
            "executable bit differs",
        ),
        (
            [
                ("pkg/item", b"same", stat.S_IFREG | 0o644),
                ("pkg/target", b"target", stat.S_IFREG | 0o644),
            ],
            [
                ("pkg/item", b"target", stat.S_IFLNK | 0o777),
                ("pkg/target", b"target", stat.S_IFREG | 0o644),
            ],
            "entry type differs",
        ),
    ],
)
def test_wheel_semantic_differences_fail(
    tmp_path: Path,
    candidate_entries: list[tuple[str, bytes, int]],
    published_entries: list[tuple[str, bytes, int]],
    message: str,
) -> None:
    candidate = tmp_path / "candidate.whl"
    published = tmp_path / "published.whl"
    _write_wheel(candidate, candidate_entries)
    _write_wheel(published, published_entries)

    with pytest.raises(DistributionEquivalenceError, match=message):
        verify_equivalent(candidate, published)


def test_sdist_executable_bit_difference_fails(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.tar.gz"
    published = tmp_path / "published.tar.gz"
    _write_sdist(
        candidate,
        [("project/bin/run", b"same", 0o755, tarfile.REGTYPE, "")],
        member_mtime=1,
        gzip_mtime=1,
        gzip_name="candidate",
    )
    _write_sdist(
        published,
        [("project/bin/run", b"same", 0o644, tarfile.REGTYPE, "")],
        member_mtime=2,
        gzip_mtime=2,
        gzip_name="published",
    )

    with pytest.raises(DistributionEquivalenceError, match="executable bit differs"):
        verify_equivalent(candidate, published)


def test_unsafe_and_duplicate_wheel_paths_are_rejected(tmp_path: Path) -> None:
    safe = tmp_path / "safe.whl"
    unsafe = tmp_path / "unsafe.whl"
    duplicate = tmp_path / "duplicate.whl"
    _write_wheel(safe, [("pkg/data", b"data", stat.S_IFREG | 0o644)])
    _write_wheel(unsafe, [("../escape", b"data", stat.S_IFREG | 0o644)])
    with pytest.warns(UserWarning, match="Duplicate name"):
        _write_wheel(
            duplicate,
            [
                ("pkg/data", b"first", stat.S_IFREG | 0o644),
                ("pkg/data", b"second", stat.S_IFREG | 0o644),
            ],
        )

    with pytest.raises(DistributionEquivalenceError, match="unsafe member path"):
        verify_equivalent(unsafe, safe)
    with pytest.raises(DistributionEquivalenceError, match="duplicate normalized"):
        verify_equivalent(duplicate, safe)


def test_duplicate_sdist_paths_and_escaping_links_are_rejected(tmp_path: Path) -> None:
    safe = tmp_path / "safe.tar.gz"
    duplicate = tmp_path / "duplicate.tar.gz"
    escaping_link = tmp_path / "escaping.tar.gz"
    _write_sdist(
        safe,
        [("project/data", b"data", 0o644, tarfile.REGTYPE, "")],
        member_mtime=1,
        gzip_mtime=1,
        gzip_name="safe",
    )
    _write_sdist(
        duplicate,
        [
            ("project/data", b"first", 0o644, tarfile.REGTYPE, ""),
            ("project/data", b"second", 0o644, tarfile.REGTYPE, ""),
        ],
        member_mtime=1,
        gzip_mtime=1,
        gzip_name="duplicate",
    )
    _write_sdist(
        escaping_link,
        [("link", b"", 0o777, tarfile.SYMTYPE, "../../outside")],
        member_mtime=1,
        gzip_mtime=1,
        gzip_name="escaping",
    )

    with pytest.raises(DistributionEquivalenceError, match="duplicate normalized"):
        verify_equivalent(duplicate, safe)
    with pytest.raises(DistributionEquivalenceError, match="escaping link target"):
        verify_equivalent(escaping_link, safe)


def test_distribution_types_must_match(tmp_path: Path) -> None:
    wheel = tmp_path / "candidate.whl"
    sdist = tmp_path / "published.tar.gz"
    _write_wheel(wheel, [("pkg/data", b"same", stat.S_IFREG | 0o644)])
    _write_sdist(
        sdist,
        [("pkg/data", b"same", 0o644, tarfile.REGTYPE, "")],
        member_mtime=1,
        gzip_mtime=1,
        gzip_name="published",
    )

    with pytest.raises(DistributionEquivalenceError, match="distribution types differ"):
        verify_equivalent(wheel, sdist)


def test_cli_reports_success_and_a_clear_mismatch(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.whl"
    matching = tmp_path / "matching.whl"
    different = tmp_path / "different.whl"
    _write_wheel(candidate, [("pkg/data", b"same", stat.S_IFREG | 0o644)])
    _write_wheel(matching, [("pkg/data", b"same", stat.S_IFREG | 0o600)])
    _write_wheel(different, [("pkg/data", b"different", stat.S_IFREG | 0o644)])

    success = subprocess.run(
        [sys.executable, str(SCRIPT), str(candidate), str(matching)],
        check=False,
        capture_output=True,
        text=True,
    )
    mismatch = subprocess.run(
        [sys.executable, str(SCRIPT), str(candidate), str(different)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert success.returncode == 0
    assert "normalized extracted contents match" in success.stdout
    assert mismatch.returncode == 1
    assert "distribution equivalence check failed" in mismatch.stderr
    assert "content differs for 'pkg/data'" in mismatch.stderr
