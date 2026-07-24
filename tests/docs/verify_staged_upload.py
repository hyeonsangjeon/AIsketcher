"""Validate the files left in a Trusted Publishing staging directory."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

ATTESTATION_SUFFIX = ".publish.attestation"


class StagedUploadError(ValueError):
    """Raised when the upload directory crosses the reviewed file boundary."""


def staged_distribution_names(
    directory: Path,
    expected_names: Iterable[str],
) -> frozenset[str]:
    """Return staged distributions after validating action-created sidecars."""

    expected = frozenset(expected_names)
    if not expected:
        raise StagedUploadError("at least one expected distribution is required")

    entries = tuple(directory.iterdir())
    invalid_entries = sorted(
        entry.name for entry in entries if not entry.is_file() or entry.is_symlink()
    )
    if invalid_entries:
        raise StagedUploadError(
            f"staged upload directory contains invalid entries: {invalid_entries}"
        )

    names = {entry.name for entry in entries}
    distributions = names & expected
    allowed_attestations = {f"{name}{ATTESTATION_SUFFIX}" for name in distributions}
    unexpected = sorted(names - distributions - allowed_attestations)
    if unexpected:
        raise StagedUploadError(f"staged upload directory contains unexpected files: {unexpected}")
    return frozenset(distributions)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--expected", action="append", required=True)
    args = parser.parse_args()

    try:
        distributions = staged_distribution_names(
            args.directory,
            args.expected,
        )
    except (OSError, StagedUploadError) as error:
        raise SystemExit(str(error)) from error

    print(
        "verified staged distributions and exact Trusted Publishing "
        f"attestation sidecars: {sorted(distributions)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
