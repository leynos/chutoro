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


class ComponentGroup(typ.NamedTuple):
    """One `--component` flag, with enough context to name it in a failure."""

    job_name: str
    invocation: str
    values: list[str]


def _rustup_installs(script: str) -> list[str]:
    """Return each rustup toolchain install in a step's script."""
    joined = re.sub(r"\\\s*\n\s*", " ", script)
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
