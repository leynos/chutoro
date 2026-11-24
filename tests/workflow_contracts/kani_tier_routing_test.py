"""Behavioural tests for how the Makefile routes its Kani tiers.

A recipe that merely mentions the checker proves nothing: it could echo the
command, or leave it in a comment and run ``true``. These tests run each tier
against a recording checker, so the argv that reaches it, the order the calls
run in, and the status the tier propagates are observed rather than asserted as
recipe text.

The recipe reading is :mod:`kani_recipe_reading`'s; this module is the part
that executes.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

from kani_recipe_reading import (
    CHECKER_WRAPPER_RE,
    KANI_COMMAND_RE,
    PACKAGE_ARGUMENT_RE,
    executable_commands,
    logical_lines,
    names,
    selections,
    wrappers,
)
from workflow_support import ROOT

# ---------------------------------------------------------------------------
# The checker wrapper
# ---------------------------------------------------------------------------

def test_every_fast_call_is_wrapped_by_the_checker(fast_recipe: str) -> None:
    """Every pull-request Kani call runs behind the manifest checker.

    A recipe that invokes `cargo kani` directly would report Kani's exit status,
    which does not count harnesses, so a filter that silently matched less
    would pass. The checker is what makes the tier mean its manifest scope.
    """
    commands = executable_commands(fast_recipe)
    found = wrappers(fast_recipe)
    assert len(found) == len(commands), (
        "every kani: command must be wrapped by $(KANI_SUITE); "
        f"{len(commands)} commands and {len(found)} wrappers found"
    )
    assert {wrapper["scope"] for wrapper in found} == {"fast"}, (
        "every kani: wrapper must name the fast scope, got "
        f"{[wrapper['scope'] for wrapper in found]!r}"
    )


def test_every_full_call_is_wrapped_by_the_checker(full_recipe: str) -> None:
    """Every nightly call runs behind the manifest checker, for the full scope."""
    commands = executable_commands(full_recipe)
    found = wrappers(full_recipe)
    assert len(found) == len(commands), (
        "every kani-full: command must be wrapped by $(KANI_SUITE); "
        f"{len(commands)} commands and {len(found)} wrappers found"
    )
    assert {wrapper["scope"] for wrapper in found} == {"full"}, (
        "every kani-full: wrapper must name the full scope, got "
        f"{[wrapper['scope'] for wrapper in found]!r}"
    )


@pytest.mark.parametrize(
    ("recipe_fixture", "scope"),
    (("fast_recipe", "fast"), ("full_recipe", "full")),
)
def test_the_wrapper_names_the_package_the_command_selects(
    recipe_fixture: str, scope: str, request: pytest.FixtureRequest
) -> None:
    """The checker's package and the delegated `-p` must agree.

    The checker derives its expected set from `--package`; the command runs
    inside `-p`. A mismatch would compare one package's results with another
    package's manifest scope and fail for the wrong reason.
    """
    recipe = request.getfixturevalue(recipe_fixture)
    for line in logical_lines(recipe):
        command = KANI_COMMAND_RE.match(line)
        if command is None:
            continue
        wrapper = CHECKER_WRAPPER_RE.match(line)
        assert wrapper is not None, f"unwrapped Kani command: {line!r}"
        packages = names(command["arguments"], PACKAGE_ARGUMENT_RE)
        assert packages == [wrapper["package"]], (
            "the wrapper's --package must match the delegated command's -p: "
            f"{wrapper['package']!r} against {packages!r} in {line!r}"
        )
        assert wrapper["scope"] == scope, (
            f"the wrapper must name the {scope} scope, got {wrapper['scope']!r}"
        )




# ---------------------------------------------------------------------------
# The Makefile's own routing
# ---------------------------------------------------------------------------

RECORDER = r"""#!/bin/sh
{
  for arg in "$@"; do
    printf '%s\t' "$arg"
  done
  printf '\n'
} >> "$FAKE_SUITE_LOG"
count=$(wc -l < "$FAKE_SUITE_LOG")
if [ "$count" -eq "${FAKE_SUITE_FAIL_AT:-0}" ]; then
  exit 9
fi
exit 0
"""


def _recorder(directory: Path) -> tuple[Path, Path]:
    """Write the recorder and return it with the log it appends to."""
    tool = directory / "kani_suite_recorder"
    tool.write_text(RECORDER, encoding="utf-8")
    tool.chmod(0o755)
    return tool, directory / "invocations.log"


def _run_tier(tool: Path, log: Path, fail_at: int | None, target: str) -> subprocess.CompletedProcess[str]:
    """Run one Make tier against the recorder."""
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"MAKEFLAGS", "MFLAGS", "KANI_SUITE"}
    }
    environment |= {"FAKE_SUITE_LOG": str(log)}
    if fail_at is not None:
        environment["FAKE_SUITE_FAIL_AT"] = str(fail_at)
    return subprocess.run(
        [
            "make",
            "--no-print-directory",
            f"KANI_SUITE={tool}",
            target,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )


def _recorded(log: Path) -> list[list[str]]:
    """Return each recorder invocation's argv, in call order."""
    if not log.exists():
        return []
    return [
        line.rstrip("\t").split("\t")
        for line in log.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _selectors(argv: list[str]) -> list[str]:
    """Return the `--harness` values in one recorded invocation."""
    return [
        argv[index + 1]
        for index, argument in enumerate(argv[:-1])
        if argument == "--harness"
    ]


@pytest.mark.parametrize(
    ("target", "recipe_fixture", "scope"),
    (("kani", "fast_recipe", "fast"), ("kani-full", "full_recipe", "full")),
)
def test_make_tier_runs_the_checker_for_each_command_in_order(
    tmp_path: Path,
    target: str,
    recipe_fixture: str,
    scope: str,
    request: pytest.FixtureRequest,
) -> None:
    """Each tier's checked calls run in recipe order, one per command.

    The expected count comes from the recipe rather than a literal, so a tier
    that gains or loses a command moves this contract with it instead of
    failing on a number nobody meant to pin.
    """
    recipe = request.getfixturevalue(recipe_fixture)
    tool, log = _recorder(tmp_path)
    completed = _run_tier(tool, log, None, target)
    assert completed.returncode == 0, (
        f"make {target} failed: stdout={completed.stdout!r} "
        f"stderr={completed.stderr!r}"
    )
    commands = selections(recipe)
    recorded = _recorded(log)
    assert len(recorded) == len(commands), (
        f"expected {len(commands)} checked calls for {target}, got {recorded!r}"
    )
    assert all(f"--scope\t{scope}" in "\t".join(argv) for argv in recorded), (
        f"every {target} call must name the {scope} scope: {recorded!r}"
    )
    expected = [name for _, names in commands for name in names]
    actual = [selector for argv in recorded for selector in _selectors(argv)]
    assert actual == expected, (
        "the checked selectors must run in recipe order: "
        f"expected {expected!r}, got {actual!r}"
    )


@pytest.mark.parametrize("fail_at", (1, 3, 6), ids=("first", "middle", "last"))
def test_a_failing_checker_stops_the_fast_tier(
    tmp_path: Path, fast_recipe: str, fail_at: int
) -> None:
    """A failure at the first, middle or last call stops the tier there.

    The checker propagates a delegated failure, so the tier must stop rather
    than continue to the next proof and report a later success as the run's
    outcome.
    """
    tool, log = _recorder(tmp_path)
    completed = _run_tier(tool, log, fail_at, "kani")
    assert completed.returncode != 0, (
        f"make kani must fail when checked call {fail_at} fails: "
        f"stdout={completed.stdout!r}"
    )
    recorded = _recorded(log)
    assert len(recorded) == fail_at, (
        f"the tier must stop at call {fail_at}, recorded {len(recorded)} calls"
    )
    expected = [name for _, names in selections(fast_recipe) for name in names]
    assert _selectors(recorded[-1]) == [expected[fail_at - 1]], (
        "the failing call must be the selector at its position: "
        f"expected {expected[fail_at - 1]!r}, recorded {recorded[-1]!r}"
    )


def _joined_dry_run(stdout: str) -> list[str]:
    """Join a dry run's continued shell lines into whole commands."""
    return logical_lines(stdout)


def test_make_dry_run_shows_each_wrapped_call(fast_recipe: str) -> None:
    """`make -n kani` prints the six checker invocations and their commands."""
    completed = subprocess.run(
        ["make", "-n", "--no-print-directory", "kani"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    commands = [
        line for line in _joined_dry_run(completed.stdout) if "cargo kani" in line
    ]
    assert len(commands) == 6, (
        f"expected six wrapped commands in the dry run, got {commands!r}"
    )
    for line in commands:
        assert "target/debug/kani_suite --manifest tools/kani/harnesses.toml" in line, (
            f"the dry run must show the checker: {line!r}"
        )
        assert "--manifest tools/kani/harnesses.toml" in line, line
        assert " --scope fast " in line, line
        assert "--exact" in line, line
        assert "--harness " in line, line
    expected = [name for _, names in selections(fast_recipe) for name in names]
    printed = [
        found.group(1)
        for line in commands
        if (found := re.search(r"--harness\s+(\S+)", line)) is not None
    ]
    assert printed == expected, (
        f"the dry run must show the selectors in recipe order: {printed!r}"
    )
