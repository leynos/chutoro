"""The Whitaker recipe runs under the toolchain its installer manages.

The Concordat ``rust-build-defaults`` rule (BD-008) wants a direct Whitaker
route isolated from the development flags: its ``RUSTFLAGS`` is the literal
boundary, so it carries neither the parallel frontend, the ``mold`` linker, nor
a Cranelift selection, and it inherits nothing from the caller. This contract
reads what ``make -n lint-whitaker`` prints on a Linux host and holds that
boundary, so an edit that restores the linker flag or an inherited
``RUSTFLAGS`` fails here instead of in the audit.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import typing as typ
from pathlib import Path

import pytest

ROOT: typ.Final[Path] = Path(__file__).resolve().parents[2]
DEVELOPMENT_FLAGS: typ.Final[tuple[str, ...]] = (
    "-Zthreads",
    "-fuse-ld=mold",
    "codegen-backend",
    "cranelift",
)


def _dry_run() -> str:
    """Return the commands ``make -n lint-whitaker`` prints on a Linux host."""
    make = shutil.which("make")
    if make is None:
        pytest.skip("make is not installed")
    result = subprocess.run(  # noqa: S603 - a fixed command
        [make, "-n", "-B", "BUILD_HOST_OS=Linux", "lint-whitaker"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"make -n lint-whitaker failed:\n{result.stderr}"
    return result.stdout


def test_the_whitaker_recipe_sets_a_literal_warning_only_rustflags() -> None:
    """The command assigns ``RUSTFLAGS="-D warnings"`` and nothing else."""
    commands = [line for line in _dry_run().splitlines() if "whitaker" in line]
    assert len(commands) == 1, f"expected one Whitaker command, found {commands}"
    assert re.match(r'^RUSTFLAGS="-D warnings" (?:\S+/)?whitaker --all ', commands[0]), (
        f"Whitaker must run under a literal RUSTFLAGS boundary: {commands[0]}"
    )


@pytest.mark.parametrize("flag", DEVELOPMENT_FLAGS)
def test_the_whitaker_recipe_carries_no_development_flag(flag: str) -> None:
    """Neither the frontend, the linker, nor a backend reaches Whitaker."""
    assert flag not in _dry_run(), f"{flag} must not reach the Whitaker route"


def test_the_whitaker_recipe_inherits_no_caller_rustflags() -> None:
    """A caller's ``RUSTFLAGS`` must not be spliced into the Whitaker command."""
    assert "${RUSTFLAGS" not in _dry_run(), "Whitaker must not inherit RUSTFLAGS"
