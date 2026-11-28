"""Contract tests for `make help` across the root Makefile and its includes.

`help` greps `$(MAKEFILE_LIST)`, which is one file until a target moves into an
included fragment. `grep` prefixes every match with its filename once more than
one file is named, and the two-column output then reads `Makefile build`
instead of `build … Build the debug binary`: the target becomes the
description and the description disappears. The Kani wiring is now an included
fragment, so that bug is live rather than hypothetical.

These tests run the real target and assert the columns for a root target and
for both fragment targets, so a change to the pipeline that loses either
column fails here rather than shipping a help screen that names no purpose.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MAKEFILE_PATH = REPO_ROOT / "Makefile"
KANI_FRAGMENT_PATH = REPO_ROOT / "tools" / "kani" / "kani.mk"


@pytest.fixture(scope="module")
def help_lines() -> list[str]:
    """Return `make help`'s output lines."""
    completed = subprocess.run(
        ["make", "--no-print-directory", "help"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    return [line for line in completed.stdout.splitlines() if line.strip()]


def _description_of(lines: list[str], target: str) -> str:
    """Return the description `help` prints for `target`, or fail."""
    for line in lines:
        columns = line.split()
        if columns and columns[0] == target:
            return line.strip()
    raise AssertionError(f"help lists no {target} target; got {lines!r}")


@pytest.mark.parametrize(
    ("target", "description"),
    (
        ("build", "Build debug binary"),
        ("kani", "Run Kani practical harnesses"),
        ("kani-full", "Run all Kani formal verification harnesses"),
    ),
)
def test_help_names_each_target_with_its_description(
    help_lines: list[str], target: str, description: str
) -> None:
    """A root target and both fragment targets keep their own columns.

    Reading the line rather than matching the description alone is what
    distinguishes `build  Build debug binary` from `Makefile  build`, where
    the target has slid into the description column.
    """
    line = _description_of(help_lines, target)
    assert description in line, (
        f"help must describe {target} as {description!r}, got {line!r}"
    )
    assert "Makefile" not in line and "kani.mk" not in line, (
        f"help must not print the source file as the target, got {line!r}"
    )


def test_help_keeps_its_heading(help_lines: list[str]) -> None:
    """The heading stays on the first line whatever the include list holds."""
    assert help_lines[0] == "Available targets:", (
        f"help must open with its heading, got {help_lines[:2]!r}"
    )


def test_help_covers_the_included_fragment() -> None:
    """The fragment's targets are listed at all, not only the root file's."""
    assert KANI_FRAGMENT_PATH.is_file(), "the Kani fragment is missing"
    text = MAKEFILE_PATH.read_text(encoding="utf-8")
    assert "include tools/kani/kani.mk" in text, (
        "the root Makefile must include the Kani fragment"
    )
    assert "kani:" in KANI_FRAGMENT_PATH.read_text(encoding="utf-8"), (
        "the fragment must define the kani target"
    )