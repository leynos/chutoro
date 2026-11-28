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
import shlex
import subprocess
from pathlib import Path

import pytest

from kani_recipe_reading import (
    CHECKER_WRAPPER_RE,
    KANI_COMMAND_RE,
    KANI_MAKEFILE_PATH,
    executable_commands,
    logical_lines,
    package_names,
    selections,
    wrappers,
)
from workflow_support import ROOT

#: Each Makefile tier, the conftest fixture holding its recipe, and the scope
#: the tier's wrapper must name. The three travel together, so a test takes the
#: tier and reads the rest rather than passing the same fact twice.
TIERS: dict[str, tuple[str, str]] = {
    "kani": ("fast_recipe", "fast"),
    "kani-full": ("full_recipe", "full"),
}

# ---------------------------------------------------------------------------
# The checker wrapper
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("recipe_fixture", "scope"),
    (("fast_recipe", "fast"), ("full_recipe", "full")),
)
def test_every_call_is_wrapped_by_the_checker(
    recipe_fixture: str, scope: str, request: pytest.FixtureRequest
) -> None:
    """Every Kani call in a tier runs behind the manifest checker.

    A recipe that invokes `cargo kani` directly would report Kani's exit status,
    which does not count harnesses, so a filter that silently matched less
    would pass. The checker is what makes the tier mean its manifest scope.
    """
    recipe = request.getfixturevalue(recipe_fixture)
    commands = executable_commands(recipe)
    found = wrappers(recipe)
    assert len(found) == len(commands), (
        f"every {scope} command must be wrapped by $(KANI_SUITE); "
        f"{len(commands)} commands and {len(found)} wrappers found"
    )
    assert {wrapper["scope"] for wrapper in found} == {scope}, (
        f"every {scope} wrapper must name its own scope, got "
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
        packages = package_names(command["arguments"])
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
            f"KANI_SUITE={shlex.quote(str(tool))}",
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


#: A `--harness` selector in either spelling: `--harness NAME` or
#: `--harness=NAME`. Both reach the checker as the same option.
_SELECTOR_RE = re.compile(r"--harness(?:=|\s+)(?P<name>\S+)")


def _selectors(argv: list[str]) -> list[str]:
    """Return the `--harness` values in one recorded invocation."""
    found: list[str] = []
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == "--harness" and index + 1 < len(argv):
            found.append(argv[index + 1])
            index += 2
            continue
        attached = re.fullmatch(r"--harness=(?P<name>.+)", argument)
        if attached is not None:
            found.append(attached["name"])
        index += 1
    return found


@pytest.mark.parametrize("target", TIERS)
def test_make_tier_runs_the_checker_for_each_command_in_order(
    tmp_path: Path,
    target: str,
    request: pytest.FixtureRequest,
) -> None:
    """Each tier's checked calls run in recipe order, one per command.

    The expected count comes from the recipe rather than a literal, so a tier
    that gains or loses a command moves this contract with it instead of
    failing on a number nobody meant to pin.
    """
    recipe_fixture, scope = TIERS[target]
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
    expected = [names for _, names in commands]
    actual = [_selectors(argv) for argv in recorded]
    assert actual == expected, (
        "the checked selectors must run in recipe order, one list per call: "
        f"expected {expected!r}, got {actual!r}"
    )


@pytest.mark.parametrize("target", TIERS)
@pytest.mark.parametrize("position", ("first", "middle", "last"))
def test_a_failing_checker_stops_the_tier(
    tmp_path: Path,
    target: str,
    position: str,
    request: pytest.FixtureRequest,
) -> None:
    """A failure at the first, middle or last call stops the tier there.

    The checker propagates a delegated failure, so the tier must stop rather
    than continue to the next command and report a later success as the run's
    outcome. Both tiers are exercised, because the nightly one is the only
    place several proofs run at all.
    """
    recipe_fixture, _scope = TIERS[target]
    recipe = request.getfixturevalue(recipe_fixture)
    commands = selections(recipe)
    fail_at = {"first": 1, "middle": (len(commands) + 1) // 2, "last": len(commands)}[
        position
    ]
    tool, log = _recorder(tmp_path)
    completed = _run_tier(tool, log, fail_at, target)
    assert completed.returncode != 0, (
        f"make {target} must fail when checked call {fail_at} fails: "
        f"stdout={completed.stdout!r}"
    )
    recorded = _recorded(log)
    assert len(recorded) == fail_at, (
        f"the {target} tier must stop at call {fail_at}, recorded "
        f"{len(recorded)} calls"
    )
    expected = commands[fail_at - 1][1]
    assert _selectors(recorded[-1]) == expected, (
        "the failing call must be the one at that position: "
        f"expected {expected!r}, recorded {recorded[-1]!r}"
    )


def test_the_checker_build_pins_the_host_triple() -> None:
    """The build names its target, so the output path is the documented one.

    A `build.target` in a user-level Cargo configuration, or a
    `CARGO_BUILD_TARGET` in the environment, moves the executable under
    `<target-dir>/<triple>/debug/`. Passing `--target` with the host triple
    outranks both, so `<target-dir>/<host>/debug/kani_suite` is the path the
    build just wrote rather than one it might not have.
    """
    makefile = KANI_MAKEFILE_PATH.read_text(encoding="utf-8")
    assert '--target "$$host"' in makefile, (
        "the checker build must name its target triple"
    )
    assert '$(KANI_SUITE_TARGET_DIR)/$$host/debug/kani_suite' in makefile, (
        "the copy must read the path that triple's build writes"
    )


def test_the_checker_is_published_by_rename() -> None:
    """A rename replaces the entry atomically; a direct copy would not.

    Copying onto a checker another `make` is running fails with ETXTBSY, and a
    reader could observe a partial file. The copy therefore lands beside the
    destination and is renamed over it.
    """
    makefile = KANI_MAKEFILE_PATH.read_text(encoding="utf-8")
    assert 'mv -f "$(KANI_SUITE_DIR)/kani_suite.$$$$"' in makefile, (
        "the checker must be published by a rename within its own directory"
    )


def _joined_dry_run(stdout: str) -> list[str]:
    """Join a dry run's continued shell lines into whole commands."""
    return logical_lines(stdout)


def test_make_dry_run_shows_each_wrapped_call(fast_recipe: str) -> None:
    """`make -n kani` prints one checker invocation per fast-tier command."""
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
    expected_count = len(selections(fast_recipe))
    assert len(commands) == expected_count, (
        f"expected {expected_count} wrapped commands in the dry run, got "
        f"{commands!r}"
    )
    for line in commands:
        # The binary path is quoted because it is derived from CARGO_TARGET_DIR
        # and the host triple, and --manifest's value is quoted so a path with
        # a space reaches the checker as one argument.
        assert 'kani_suite" --manifest "tools/kani/harnesses.toml"' in line, (
            f"the dry run must show the checker: {line!r}"
        )
        assert " --scope fast " in line, line
        assert "--exact" in line, line
        assert "--harness " in line, line
    expected = [name for _, names in selections(fast_recipe) for name in names]
    printed = [
        found.group(1)
        for line in commands
        if (found := _SELECTOR_RE.search(line)) is not None
    ]
    assert printed == expected, (
        f"the dry run must show the selectors in recipe order: {printed!r}"
    )
