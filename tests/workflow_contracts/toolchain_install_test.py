"""Contract-test how workflows invoke rustup.

`rustup --component` takes one value. Written as `--component clippy
rustfmt`, the second name is parsed as a toolchain and rustup exits 1, so
the lane dies before it does any work. `nightly-portable-simd` shipped
that for as long as it has existed and never once ran its tests: on a day
its gate fired it failed here, and on every other day the gate skipped the
work and the run reported success in seconds. Nothing in the repository
could see the difference, because the flag is only wrong at the point
rustup parses it.

These tests read the invocation rather than the step's name, so deleting
the fix while keeping the step fails them.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import itertools as it
import re
import typing as typ

import pytest
from workflow_support import (
    jobs,
    load_workflow,
    run_script,
    steps,
    workflow_names,
)

#: A `rustup` invocation, however its line continuations are written. The
#: script is joined first, so a command split across lines is one match
#: rather than none.
RUSTUP_INVOCATION = re.compile(r"\brustup\s+toolchain\s+install\b[^\n;&|]*")

#: The flag whose repeated form is the whole point of this module.
COMPONENT_FLAG = "--component"

#: A shell line continuation, and only that.
#:
#: The shell joins lines on a backslash immediately followed by a newline. A
#: backslash followed by a space or a tab escapes that character instead, and
#: the command ends at the newline. Matching the looser `\\\s*\n` would join
#: lines the shell does not, so a truncated install would read here as a
#: well-formed one, which is the opposite of what a contract is for.
CONTINUATION = re.compile(r"\\\n[ \t]*")


class ComponentGroup(typ.NamedTuple):
    """One `--component` flag, with enough context to name it in a failure."""

    job_name: str
    invocation: str
    values: list[str]


def _rustup_installs(script: str) -> list[str]:
    """Return each rustup toolchain install in a step's script.

    Genuine continuations are joined first, so a command split across lines
    is one invocation rather than none. A backslash the shell would not treat
    as a continuation is left alone, and the invocation then ends on it, which
    is what `test_no_rustup_install_ends_on_a_dangling_backslash` looks for.
    """
    joined = CONTINUATION.sub(" ", script)
    return [match.group(0) for match in RUSTUP_INVOCATION.finditer(joined)]


def _component_values(invocation: str) -> list[list[str]]:
    """Return the tokens following each `--component` flag.

    A value run ends at the next flag, so a `--component` immediately
    followed by one yields an empty list rather than swallowing it.
    """
    tokens = invocation.split()
    return [
        list(it.takewhile(lambda token: not token.startswith("-"), tokens[index + 1 :]))
        for index, token in enumerate(tokens)
        if token == COMPONENT_FLAG
    ]


def _rustup_invocations(workflow_name: str) -> typ.Iterator[tuple[str, str]]:
    """Yield every rustup toolchain install in a workflow, with its job name."""
    for job_name, definition in jobs(load_workflow(workflow_name)).items():
        for step in steps(definition):
            for invocation in _rustup_installs(run_script(step)):
                yield job_name, invocation


def _component_groups_in_step(step: dict[str, typ.Any]) -> typ.Iterator[tuple[str, list[str]]]:
    """Yield each `--component` flag found in one step, with its invocation."""
    for invocation in _rustup_installs(run_script(step)):
        for values in _component_values(invocation):
            yield invocation, values


def _component_groups(workflow_name: str) -> typ.Iterator[ComponentGroup]:
    """Yield every `--component` flag in a workflow, wherever it appears."""
    for job_name, definition in jobs(load_workflow(workflow_name)).items():
        for invocation, values in _steps_of(definition):
            yield ComponentGroup(job_name, invocation, values)


def _steps_of(definition: dict[str, typ.Any]) -> typ.Iterator[tuple[str, list[str]]]:
    """Yield each `--component` flag across one job's steps."""
    for step in steps(definition):
        yield from _component_groups_in_step(step)


@pytest.mark.parametrize("workflow_name", workflow_names())
def test_each_rustup_component_carries_its_own_flag(workflow_name: str) -> None:
    """Fail when one `--component` flag is given more than one value.

    rustup reads the second value as a toolchain name and exits before
    installing anything, so the step fails and every step after it is
    skipped.
    """
    for group in _component_groups(workflow_name):
        assert len(group.values) == 1, (
            f"{workflow_name}:{group.job_name} installs {group.values} under "
            f"one {COMPONENT_FLAG} flag; rustup reads {group.values[1:]} as "
            "toolchain names and exits 1"
        )


@pytest.mark.parametrize("workflow_name", workflow_names())
def test_no_rustup_install_asks_for_a_component_it_does_not_name(
    workflow_name: str,
) -> None:
    """Every `--component` flag has a value, rather than swallowing the next flag."""
    for group in _component_groups(workflow_name):
        assert group.values, (
            f"{workflow_name}:{group.job_name} passes {COMPONENT_FLAG} with no "
            f"value in {group.invocation!r}"
        )


@pytest.mark.parametrize("workflow_name", workflow_names())
def test_no_rustup_install_ends_on_a_dangling_backslash(workflow_name: str) -> None:
    """A rustup install must not be truncated by a broken continuation.

    `\\` followed by a space and a newline is not a continuation: the
    backslash escapes the space and the shell ends the command at the newline,
    dropping every flag on the lines below. The run then installs a toolchain
    with none of the components asked for, and succeeds, so nothing downstream
    reports it. The reading above leaves such a backslash in place precisely
    so it can be seen here.
    """
    for job_name, invocation in _rustup_invocations(workflow_name):
        assert not invocation.rstrip().endswith("\\"), (
            f"{workflow_name}:{job_name} ends a rustup install on a backslash "
            f"the shell does not read as a continuation, so everything after "
            f"it is dropped: {invocation!r}"
        )


@pytest.mark.parametrize(
    ("script", "expected"),
    [
        pytest.param(
            "rustup toolchain install nightly \\\n  --component clippy",
            ["rustup toolchain install nightly  --component clippy"],
            id="continuation-is-joined",
        ),
        pytest.param(
            "rustup toolchain install nightly \\ \n  --component clippy",
            ["rustup toolchain install nightly \\ "],
            id="backslash-space-newline-is-not-a-continuation",
        ),
        pytest.param(
            "rustup toolchain install nightly \\\t\n  --component clippy",
            ["rustup toolchain install nightly \\\t"],
            id="backslash-tab-newline-is-not-a-continuation",
        ),
    ],
)
def test_the_reading_joins_only_what_the_shell_joins(
    script: str, expected: list[str]
) -> None:
    """Whitespace between a backslash and a newline ends the command.

    Joining it anyway would let a truncated install read as a well-formed one,
    which is the defect this module exists to catch rather than to hide.
    """
    assert _rustup_installs(script) == expected


#: The trybuild test whose expected output is toolchain-specific.
#:
#: Its fixture pins stable's refusal to compile `std::simd` without the
#: feature. On nightly, rustc adds `help: add #![feature(portable_simd)]` to
#: each of those errors because there it can be enabled, so the fixture
#: mismatches for a reason about the toolchain rather than about the gating.
TOOLCHAIN_SPECIFIC_TRYBUILD_TEST = "portable_simd_without_feature_is_rejected"

#: The lane that runs the dense provider's tests on nightly.
NIGHTLY_SIMD_WORKFLOW = "nightly-portable-simd.yml"


def _nightly_cargo_tests() -> list[str]:
    """Return each `cargo +nightly test` command in the nightly SIMD lane."""
    return [
        " ".join(invocation.split())
        for _, invocation in _run_scripts(NIGHTLY_SIMD_WORKFLOW)
        for invocation in CONTINUATION.sub(" ", invocation).split("\n")
        if "cargo +nightly test" in invocation
    ]


def _run_scripts(workflow_name: str) -> typ.Iterator[tuple[str, str]]:
    """Yield every step's `run` script in a workflow, with its job name."""
    for job_name, definition in jobs(load_workflow(workflow_name)).items():
        for step in steps(definition):
            script = run_script(step)
            if script:
                yield job_name, script


def test_the_nightly_lane_skips_the_toolchain_specific_trybuild_test() -> None:
    """The nightly SIMD lane excludes the trybuild test by name.

    Without the exclusion the lane fails on a fixture mismatch rather than on
    anything it exists to check, which is how it spent its whole existence
    reporting nothing useful (#262). The assertion is on the command's own
    `--skip` argument, so deleting the exclusion while leaving the step in
    place fails here.
    """
    commands = _nightly_cargo_tests()
    assert commands, (
        f"{NIGHTLY_SIMD_WORKFLOW} must run the dense provider's tests on "
        "nightly; no `cargo +nightly test` command was found"
    )
    for command in commands:
        assert f"--skip {TOOLCHAIN_SPECIFIC_TRYBUILD_TEST}" in command, (
            f"{NIGHTLY_SIMD_WORKFLOW} must skip "
            f"{TOOLCHAIN_SPECIFIC_TRYBUILD_TEST} on nightly, because its "
            f"expected output pins stable's diagnostics; got {command!r}"
        )


def test_only_the_nightly_lane_skips_that_trybuild_test() -> None:
    """No other workflow skips it, so stable keeps running it.

    The exclusion is narrow on purpose: the check earns its place on stable,
    where the refusal it asserts is the real behaviour. A second lane adopting
    the same `--skip` would quietly drop that coverage everywhere.
    """
    for workflow_name in workflow_names():
        if workflow_name == NIGHTLY_SIMD_WORKFLOW:
            continue
        for job_name, script in _run_scripts(workflow_name):
            assert TOOLCHAIN_SPECIFIC_TRYBUILD_TEST not in script, (
                f"{workflow_name}:{job_name} excludes "
                f"{TOOLCHAIN_SPECIFIC_TRYBUILD_TEST}; only the nightly lane "
                "has a toolchain reason to, and stable must keep running it"
            )
