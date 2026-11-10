"""Contract tests for the development route's toolchain-pin guard.

``DEV_FAST_TOOLCHAIN`` is read from ``tools/dev-fast/TOOLCHAIN`` with
``$(file <…)``. That function yields nothing for a missing file and strips to
nothing for an empty one, and ``cargo +`` with nothing after it falls back to
the default toolchain. Without a guard the development route would compile
with a toolchain the pin never names and report success, which is the one
outcome the pin exists to prevent.

The guard is per recipe, not a file-wide ``$(error)``, because the standard
gates parse the same variables and a broken development pin must not fail
them. Both halves are contracted here.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from dev_fast_make_support import (
    DEV_FAST_TOOLCHAIN_PATH,
    MAKEFILE_PATH,
    REPO_ROOT,
    _mutated_makefile,
)

DEVELOPMENT_TARGETS = ("dev-build", "dev-test", "install-dev-fast")

#: The guard's definition. Mutating it is how the mutation case removes the
#: check from every recipe at once, without touching what the recipes call.
GUARD_ANCHOR = (
    "DEV_FAST_TOOLCHAIN_CHECK = $(if $(DEV_FAST_TOOLCHAIN),,"
)

#: The same assignment with the check dropped, modelling the code before the
#: guard existed. Every development recipe keeps calling `$(DEV_FAST_CHECK)`,
#: so the mutation has to empty the guard rather than remove its call.
MUTATED_GUARD = "DEV_FAST_TOOLCHAIN_CHECK ="


def _guard_definition() -> str:
    """Return the guard's whole assignment line, to use as a mutation anchor.

    ``GUARD_ANCHOR`` is only the line's prefix: the ``$(error …)`` it guards
    sits in the tail. Replacing the prefix alone would leave that error firing
    unconditionally, which models a stricter guard rather than an absent one.
    """
    source = MAKEFILE_PATH.read_text(encoding="utf-8")
    definitions = [
        line for line in source.splitlines() if line.startswith(GUARD_ANCHOR)
    ]
    assert len(definitions) == 1, (
        f"the guard must be defined on exactly one line, found {definitions!r}"
    )
    return definitions[0]


#: What each recipe emits once the guard is gone and the pin reads empty. The
#: two Cargo routes lose the channel name after `cargo +`; the installer loses
#: it between `toolchain install` and its first flag. Naming them per target
#: keeps the mutation honest for the one route that never calls Cargo.
UNPINNED_ROUTE_MARKERS = {
    "dev-build": "probe-cargo + --config",
    "dev-test": "probe-cargo + --config",
    "install-dev-fast": "rustup toolchain install  --profile",
}

#: The message the guard must produce, and the marker the refusal cases read.
MISSING_PIN_MESSAGE = "tools/dev-fast/TOOLCHAIN is missing or empty"

#: Targets that parse the development pin without running it. A broken pin
#: must leave every one of them working.
STANDARD_TARGETS = ("build", "test", "check-fmt", "help")


def _run_in_pin_directory(
    directory: Path, target: str, *overrides: str
) -> subprocess.CompletedProcess[str]:
    """Dry-run one target where a relative pin path resolves to ``directory``.

    ``$(file <tools/dev-fast/TOOLCHAIN)`` resolves against the working
    directory, so a private copy of the Makefile is not enough: the probe has
    to run from a directory whose own pin is the one under test.
    """
    return subprocess.run(
        [
            "make",
            "--no-print-directory",
            "--dry-run",
            "--always-make",
            *overrides,
            target,
            "CARGO=probe-cargo",
        ],
        cwd=directory,
        check=False,
        capture_output=True,
        text=True,
    )


def _pin_directory(tmp_path: Path) -> Path:
    """Copy the Makefile into a directory whose pin the caller then controls."""
    directory = tmp_path / "repo"
    (directory / "tools" / "dev-fast").mkdir(parents=True)
    (directory / "tools" / "kani").mkdir(parents=True)
    (directory / "Makefile").write_text(
        MAKEFILE_PATH.read_text(encoding="utf-8"), encoding="utf-8"
    )
    for relative in (Path("tools/dev-fast/config.toml"), Path("tools/kani/VERSION")):
        (directory / relative).write_text(
            (REPO_ROOT / relative).read_text(encoding="utf-8"), encoding="utf-8"
        )
    return directory


@pytest.mark.parametrize("target", DEVELOPMENT_TARGETS)
def test_development_targets_refuse_an_empty_toolchain_pin(
    target: str, tmp_path: Path
) -> None:
    """An empty pin must stop the target rather than select a default toolchain."""
    directory = _pin_directory(tmp_path)
    (directory / "tools" / "dev-fast" / "TOOLCHAIN").write_text("   \n", encoding="utf-8")

    completed = _run_in_pin_directory(directory, target)

    assert completed.returncode != 0, (
        f"{target} must fail when the toolchain pin is empty: {completed.stdout!r}"
    )
    assert MISSING_PIN_MESSAGE in completed.stderr, (
        f"{target} must report the empty pin: {completed.stderr!r}"
    )


@pytest.mark.parametrize("target", DEVELOPMENT_TARGETS)
def test_development_targets_refuse_an_absent_toolchain_pin(
    target: str, tmp_path: Path
) -> None:
    """A missing pin file reads as empty, so it must stop the target too."""
    directory = _pin_directory(tmp_path)

    completed = _run_in_pin_directory(directory, target)

    assert completed.returncode != 0, (
        f"{target} must fail when the toolchain pin is absent: {completed.stdout!r}"
    )
    assert MISSING_PIN_MESSAGE in completed.stderr, (
        f"{target} must report the absent pin: {completed.stderr!r}"
    )


def test_a_present_pin_leaves_the_route_intact(tmp_path: Path) -> None:
    """The guard must not fire for the pin the repository actually ships.

    Without this control the two refusal cases above would also pass against a
    route that had stopped working for an unrelated reason.
    """
    directory = _pin_directory(tmp_path)
    (directory / "tools" / "dev-fast" / "TOOLCHAIN").write_text(
        DEV_FAST_TOOLCHAIN_PATH.read_text(encoding="utf-8"), encoding="utf-8"
    )

    completed = _run_in_pin_directory(
        directory, "dev-build", "DEV_FAST_HOST_OS=Linux"
    )

    assert completed.returncode == 0, (
        f"a present pin must leave the route usable: {completed.stderr!r}"
    )
    assert "probe-cargo +" in completed.stdout, (
        f"the route must still name the pinned toolchain: {completed.stdout!r}"
    )


@pytest.mark.parametrize("target", STANDARD_TARGETS)
def test_standard_targets_ignore_a_broken_toolchain_pin(
    target: str, tmp_path: Path
) -> None:
    """A broken development pin must not fail a standard target.

    This is why the guard is per recipe. A file-wide ``$(error)`` would stop
    ``build``, ``test`` and the formatting gates as well, turning a
    development-only pin into a repository-wide outage.
    """
    directory = _pin_directory(tmp_path)

    completed = _run_in_pin_directory(directory, target)

    assert MISSING_PIN_MESSAGE not in completed.stderr, (
        f"{target} must not report the development pin: {completed.stderr!r}"
    )


@pytest.mark.parametrize("target", DEVELOPMENT_TARGETS)
def test_dropping_the_pin_guard_reaches_cargo_with_no_toolchain(
    target: str, tmp_path: Path
) -> None:
    """Removing the guard must let an empty pin reach the command, as it once did.

    The guard's whole purpose is to stand between an empty read and
    ``cargo +``. The pin is left present here, and blanked through
    ``DEV_FAST_TOOLCHAIN`` instead, so the guard's absence is the only thing
    that decides the outcome: a present pin that still reads empty is exactly
    the state the recipes must not pass through. Mutating the guard rather
    than one recipe is what makes this a check on every development target.
    """
    makefile = _mutated_makefile(tmp_path, _guard_definition(), MUTATED_GUARD)

    directory = _pin_directory(tmp_path / "mutated")
    (directory / "Makefile").write_text(
        makefile.read_text(encoding="utf-8"), encoding="utf-8"
    )

    completed = _run_in_pin_directory(
        directory,
        target,
        "DEV_FAST_HOST_OS=Linux",
        "DEV_FAST_TOOLCHAIN=",
    )

    assert MISSING_PIN_MESSAGE not in completed.stderr, (
        f"{target} must stop checking the pin once the guard is removed: "
        f"{completed.stderr!r}"
    )
    marker = UNPINNED_ROUTE_MARKERS[target]
    assert marker in completed.stdout, (
        f"{target} must reach the command with no toolchain named: "
        f"{completed.stdout!r}"
    )
