"""Compare two distribution archives by their safely normalized extracted content."""

from __future__ import annotations

import argparse
import hashlib
import stat
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Literal

ArchiveKind = Literal["wheel", "sdist"]
EntryKind = Literal["file", "directory", "symlink", "hardlink"]

MAX_ARCHIVE_ENTRIES = 100_000
MAX_ENTRY_BYTES = 256 * 1024 * 1024
MAX_TOTAL_BYTES = 1024 * 1024 * 1024
MAX_PATH_CHARACTERS = 4096
MAX_LINK_TARGET_BYTES = 4096
READ_CHUNK_BYTES = 1024 * 1024
MAX_REPORTED_DIFFERENCES = 20


class DistributionEquivalenceError(ValueError):
    """Raised when an archive is unsafe, invalid, or not equivalent."""


@dataclass(frozen=True)
class NormalizedEntry:
    """Metadata that affects the content produced by extracting an entry."""

    kind: EntryKind
    size: int = 0
    digest: str | None = None
    executable: bool = False
    link_target: str | None = None


@dataclass(frozen=True)
class DistributionSnapshot:
    kind: ArchiveKind
    entries: dict[str, NormalizedEntry]


def _normalized_member_name(raw_name: str, *, is_directory: bool) -> str:
    if not raw_name:
        raise DistributionEquivalenceError("archive contains an empty member path")
    if len(raw_name) > MAX_PATH_CHARACTERS:
        raise DistributionEquivalenceError(
            f"archive member path exceeds {MAX_PATH_CHARACTERS} characters"
        )
    if "\x00" in raw_name:
        raise DistributionEquivalenceError(
            f"archive contains a NUL byte in member path {raw_name!r}"
        )
    if "\\" in raw_name:
        raise DistributionEquivalenceError(
            f"archive contains a non-POSIX member path {raw_name!r}"
        )
    if raw_name.startswith("/"):
        raise DistributionEquivalenceError(
            f"archive contains an absolute member path {raw_name!r}"
        )

    name = raw_name[:-1] if is_directory and raw_name.endswith("/") else raw_name
    if not name or (not is_directory and name.endswith("/")):
        raise DistributionEquivalenceError(
            f"archive contains an invalid member path {raw_name!r}"
        )

    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise DistributionEquivalenceError(
            f"archive contains an unsafe member path {raw_name!r}"
        )
    if path.as_posix() != name:
        raise DistributionEquivalenceError(
            f"archive contains a non-canonical member path {raw_name!r}"
        )
    if ":" in path.parts[0]:
        raise DistributionEquivalenceError(
            f"archive contains a drive-like member path {raw_name!r}"
        )
    return name


def _validate_link_target(
    raw_target: str,
    *,
    member_name: str,
    relative_to_parent: bool,
) -> str:
    if not raw_target or "\x00" in raw_target or "\\" in raw_target:
        raise DistributionEquivalenceError(
            f"{member_name!r} has an unsafe link target {raw_target!r}"
        )
    if len(raw_target.encode("utf-8")) > MAX_LINK_TARGET_BYTES:
        raise DistributionEquivalenceError(
            f"{member_name!r} has a link target larger than "
            f"{MAX_LINK_TARGET_BYTES} bytes"
        )
    target = PurePosixPath(raw_target)
    if target.is_absolute() or (target.parts and ":" in target.parts[0]):
        raise DistributionEquivalenceError(
            f"{member_name!r} has an unsafe link target {raw_target!r}"
        )

    resolved_parts = (
        list(PurePosixPath(member_name).parent.parts) if relative_to_parent else []
    )
    for part in target.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if not resolved_parts:
                raise DistributionEquivalenceError(
                    f"{member_name!r} has an escaping link target {raw_target!r}"
                )
            resolved_parts.pop()
        else:
            resolved_parts.append(part)
    return PurePosixPath(*resolved_parts).as_posix()


def _hash_stream(stream: BinaryIO, *, expected_size: int, member_name: str) -> str:
    if expected_size < 0 or expected_size > MAX_ENTRY_BYTES:
        raise DistributionEquivalenceError(
            f"{member_name!r} has an unsafe declared size of {expected_size} bytes"
        )

    digest = hashlib.sha256()
    size = 0
    while chunk := stream.read(READ_CHUNK_BYTES):
        size += len(chunk)
        if size > expected_size or size > MAX_ENTRY_BYTES:
            raise DistributionEquivalenceError(
                f"{member_name!r} expands beyond its safe declared size"
            )
        digest.update(chunk)
    if size != expected_size:
        raise DistributionEquivalenceError(
            f"{member_name!r} declared {expected_size} bytes but yielded {size}"
        )
    return digest.hexdigest()


class _SnapshotBuilder:
    def __init__(self, kind: ArchiveKind) -> None:
        self.kind = kind
        self.entries: dict[str, NormalizedEntry] = {}
        self.total_bytes = 0
        self.member_count = 0

    def add(
        self,
        raw_name: str,
        entry: NormalizedEntry,
        *,
        is_directory: bool = False,
    ) -> str:
        self.member_count += 1
        if self.member_count > MAX_ARCHIVE_ENTRIES:
            raise DistributionEquivalenceError(
                f"archive exceeds the {MAX_ARCHIVE_ENTRIES} entry safety limit"
            )

        name = _normalized_member_name(raw_name, is_directory=is_directory)
        if name in self.entries:
            raise DistributionEquivalenceError(
                f"archive contains duplicate normalized member path {name!r}"
            )
        self.total_bytes += entry.size
        if self.total_bytes > MAX_TOTAL_BYTES:
            raise DistributionEquivalenceError(
                f"archive exceeds the {MAX_TOTAL_BYTES} byte expansion safety limit"
            )
        self.entries[name] = entry
        return name

    def finish(self) -> DistributionSnapshot:
        entries = dict(self.entries)
        for name in tuple(entries):
            parent = PurePosixPath(name).parent
            while parent != PurePosixPath("."):
                parent_name = parent.as_posix()
                existing = entries.get(parent_name)
                if existing is not None and existing.kind != "directory":
                    raise DistributionEquivalenceError(
                        f"archive member {name!r} has non-directory parent {parent_name!r}"
                    )
                entries.setdefault(
                    parent_name,
                    NormalizedEntry(kind="directory", executable=True),
                )
                parent = parent.parent
        for name, entry in entries.items():
            if entry.kind != "hardlink":
                continue
            target_name = _validate_link_target(
                entry.link_target or "",
                member_name=name,
                relative_to_parent=False,
            )
            seen = {name}
            while True:
                target = entries.get(target_name)
                if target is None:
                    raise DistributionEquivalenceError(
                        f"hardlink {name!r} targets missing member {target_name!r}"
                    )
                if target.kind == "file":
                    break
                if target.kind != "hardlink" or target_name in seen:
                    raise DistributionEquivalenceError(
                        f"hardlink {name!r} does not resolve to a regular file"
                    )
                seen.add(target_name)
                target_name = _validate_link_target(
                    target.link_target or "",
                    member_name=target_name,
                    relative_to_parent=False,
                )
        return DistributionSnapshot(kind=self.kind, entries=entries)


def _zip_entry_kind(info: zipfile.ZipInfo) -> EntryKind:
    unix_mode = (info.external_attr >> 16) & 0xFFFF if info.create_system == 3 else 0
    unix_type = stat.S_IFMT(unix_mode)
    if info.is_dir() or unix_type == stat.S_IFDIR:
        if info.is_dir() and unix_type not in {0, stat.S_IFDIR}:
            raise DistributionEquivalenceError(
                f"{info.filename!r} has conflicting ZIP directory metadata"
            )
        return "directory"
    if unix_type in {0, stat.S_IFREG}:
        return "file"
    if unix_type == stat.S_IFLNK:
        return "symlink"
    raise DistributionEquivalenceError(
        f"{info.filename!r} uses unsupported ZIP entry type {oct(unix_type)}"
    )


def _snapshot_wheel(path: Path) -> DistributionSnapshot:
    builder = _SnapshotBuilder("wheel")
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise DistributionEquivalenceError(f"{path} is not a readable wheel: {exc}") from exc

    with archive:
        for info in archive.infolist():
            if info.flag_bits & 0x1:
                raise DistributionEquivalenceError(
                    f"{info.filename!r} is encrypted and cannot be verified"
                )
            kind = _zip_entry_kind(info)
            executable = False
            if kind == "file" and info.create_system == 3:
                executable = bool(((info.external_attr >> 16) & 0xFFFF) & 0o111)

            if kind == "directory":
                unix_mode = (
                    (info.external_attr >> 16) & 0xFFFF
                    if info.create_system == 3
                    else 0
                )
                builder.add(
                    info.filename,
                    NormalizedEntry(
                        kind="directory",
                        executable=bool(unix_mode & 0o111) if unix_mode else True,
                    ),
                    is_directory=True,
                )
                continue

            if info.file_size > MAX_ENTRY_BYTES:
                raise DistributionEquivalenceError(
                    f"{info.filename!r} has an unsafe declared size of "
                    f"{info.file_size} bytes"
                )
            with archive.open(info) as stream:
                digest = _hash_stream(
                    stream,
                    expected_size=info.file_size,
                    member_name=info.filename,
                )

            if kind == "symlink":
                if info.file_size > MAX_LINK_TARGET_BYTES:
                    raise DistributionEquivalenceError(
                        f"{info.filename!r} has a link target larger than "
                        f"{MAX_LINK_TARGET_BYTES} bytes"
                    )
                with archive.open(info) as stream:
                    target_bytes = stream.read(MAX_LINK_TARGET_BYTES + 1)
                try:
                    target = target_bytes.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise DistributionEquivalenceError(
                        f"{info.filename!r} has a non-UTF-8 symlink target"
                    ) from exc
                name = _normalized_member_name(info.filename, is_directory=False)
                _validate_link_target(
                    target,
                    member_name=name,
                    relative_to_parent=True,
                )
                builder.add(
                    info.filename,
                    NormalizedEntry(
                        kind="symlink",
                        size=info.file_size,
                        digest=digest,
                        link_target=target,
                    ),
                )
            else:
                builder.add(
                    info.filename,
                    NormalizedEntry(
                        kind="file",
                        size=info.file_size,
                        digest=digest,
                        executable=executable,
                    ),
                )
    return builder.finish()


def _snapshot_open_sdist(archive: tarfile.TarFile) -> DistributionSnapshot:
    builder = _SnapshotBuilder("sdist")
    for member in archive.getmembers():
        if member.isdir():
            builder.add(
                member.name,
                NormalizedEntry(
                    kind="directory",
                    executable=bool(member.mode & 0o111),
                ),
                is_directory=True,
            )
            continue
        if member.isfile():
            stream = archive.extractfile(member)
            if stream is None:
                raise DistributionEquivalenceError(
                    f"could not read regular file {member.name!r}"
                )
            with stream:
                digest = _hash_stream(
                    stream,
                    expected_size=member.size,
                    member_name=member.name,
                )
            builder.add(
                member.name,
                NormalizedEntry(
                    kind="file",
                    size=member.size,
                    digest=digest,
                    executable=bool(member.mode & 0o111),
                ),
            )
            continue
        if member.issym() or member.islnk():
            kind: EntryKind = "symlink" if member.issym() else "hardlink"
            name = _normalized_member_name(member.name, is_directory=False)
            _validate_link_target(
                member.linkname,
                member_name=name,
                relative_to_parent=member.issym(),
            )
            target_bytes = member.linkname.encode("utf-8")
            builder.add(
                member.name,
                NormalizedEntry(
                    kind=kind,
                    size=len(target_bytes),
                    digest=hashlib.sha256(target_bytes).hexdigest(),
                    link_target=member.linkname,
                ),
            )
            continue
        raise DistributionEquivalenceError(
            f"{member.name!r} uses unsupported TAR entry type {member.type!r}"
        )
    return builder.finish()


def _snapshot_sdist(path: Path) -> DistributionSnapshot:
    try:
        with tarfile.open(path, mode="r:*") as archive:
            return _snapshot_open_sdist(archive)
    except (OSError, tarfile.TarError) as exc:
        raise DistributionEquivalenceError(f"{path} is not a readable sdist: {exc}") from exc


def snapshot_distribution(path: Path) -> DistributionSnapshot:
    """Read a wheel or tar-based sdist into a safe, normalized snapshot."""

    path = path.resolve()
    if not path.is_file():
        raise DistributionEquivalenceError(f"distribution does not exist: {path}")
    try:
        if path.suffix.lower() == ".whl":
            return _snapshot_wheel(path)
        if tarfile.is_tarfile(path):
            return _snapshot_sdist(path)
        if zipfile.is_zipfile(path):
            raise DistributionEquivalenceError(
                f"{path} is a ZIP archive but does not have a .whl filename"
            )
    except DistributionEquivalenceError:
        raise
    except (
        EOFError,
        NotImplementedError,
        OSError,
        UnicodeError,
        tarfile.TarError,
        zipfile.BadZipFile,
    ) as exc:
        raise DistributionEquivalenceError(
            f"could not safely read distribution {path}: {exc}"
        ) from exc
    else:
        raise DistributionEquivalenceError(
            f"unsupported distribution archive {path}; "
            "expected a wheel or tar-based sdist"
        )


def _entry_differences(
    candidate: DistributionSnapshot,
    published: DistributionSnapshot,
) -> list[str]:
    differences: list[str] = []
    candidate_names = set(candidate.entries)
    published_names = set(published.entries)
    for name in sorted(candidate_names - published_names):
        differences.append(f"candidate-only member: {name}")
    for name in sorted(published_names - candidate_names):
        differences.append(f"published-only member: {name}")

    for name in sorted(candidate_names & published_names):
        candidate_entry = candidate.entries[name]
        published_entry = published.entries[name]
        if candidate_entry.kind != published_entry.kind:
            differences.append(
                f"entry type differs for {name!r}: "
                f"candidate={candidate_entry.kind}, published={published_entry.kind}"
            )
            continue
        if candidate_entry.executable != published_entry.executable:
            differences.append(
                f"executable bit differs for {name!r}: "
                f"candidate={candidate_entry.executable}, "
                f"published={published_entry.executable}"
            )
        if (
            candidate_entry.size != published_entry.size
            or candidate_entry.digest != published_entry.digest
            or candidate_entry.link_target != published_entry.link_target
        ):
            differences.append(f"content differs for {name!r}")
    return differences


def verify_equivalent(candidate_path: Path, published_path: Path) -> None:
    """Raise when two archives differ after safe extraction normalization."""

    candidate = snapshot_distribution(candidate_path)
    published = snapshot_distribution(published_path)
    if candidate.kind != published.kind:
        raise DistributionEquivalenceError(
            f"distribution types differ: candidate={candidate.kind}, "
            f"published={published.kind}"
        )

    differences = _entry_differences(candidate, published)
    if differences:
        shown = differences[:MAX_REPORTED_DIFFERENCES]
        suffix = ""
        if len(differences) > len(shown):
            suffix = f"\n- ... and {len(differences) - len(shown)} more difference(s)"
        raise DistributionEquivalenceError(
            "normalized distribution contents differ:\n- "
            + "\n- ".join(shown)
            + suffix
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path, metavar="CANDIDATE")
    parser.add_argument("published", type=Path, metavar="PUBLISHED")
    args = parser.parse_args(argv)

    try:
        verify_equivalent(args.candidate, args.published)
    except DistributionEquivalenceError as exc:
        print(f"distribution equivalence check failed: {exc}", file=sys.stderr)
        return 1

    print(
        "distribution equivalence check passed: normalized extracted contents match"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
