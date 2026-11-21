"""Read `rustup toolchain install` invocations out of workflow steps.

`rustup --component` takes one value. Written as `--component clippy
rustfmt`, the second name is parsed as a toolchain and rustup exits 1, so a
lane dies before it does any work. This module owns the one reading of those
invocations: it joins the line continuations the shell joins and groups each
`--component` flag with the values that follow it, so the contracts in
`toolchain_install_test.py` do not each parse the text again.

It also holds the nightly portable SIMD lane's presence contract, because the
reading that contract needs is the same one. Keeping the parser and that
contract together leaves the test module below the 400-line limit.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import itertools as it
import re
import typing as typ

from workflow_support import jobs, load_workflow, run_script, steps

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

#: The lane whose nightly installer this module holds to its purpose.
NIGHTLY_WORKFLOW = "nightly-portable-simd.yml"
NIGHTLY_JOB = "nightly-portable-simd"

#: The toolchain, profile, and components the nightly lane exists to install.
NIGHTLY_TOOLCHAIN = "nightly"
NIGHTLY_PROFILE = "minimal"
NIGHTLY_COMPONENTS = frozenset({"clippy", "rustfmt"})


class ComponentGroup(typ.NamedTuple):
    """One `--component` flag, with enough context to name it in a failure."""

    job_name: str
    invocation: str
    values: list[str]


def rustup_installs(script: str) -> list[str]:
    """Return each rustup toolchain install in a step's script.

    Genuine continuations are joined first, so a command split across lines
    is one invocation rather than none. A backslash the shell would not treat
    as a continuation is left alone, and the invocation then ends on it, which
    is what `test_no_rustup_install_ends_on_a_dangling_backslash` looks for.
    """
    joined = CONTINUATION.sub(" ", script)
    return [match.group(0) for match in RUSTUP_INVOCATION.finditer(joined)]


def component_values(invocation: str) -> list[list[str]]:
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


def rustup_invocations(workflow_name: str) -> typ.Iterator[tuple[str, str]]:
    """Yield every rustup toolchain install in a workflow, with its job name."""
    for job_name, definition in jobs(load_workflow(workflow_name)).items():
        for step in steps(definition):
            for invocation in rustup_installs(run_script(step)):
                yield job_name, invocation


def run_scripts(workflow_name: str) -> typ.Iterator[tuple[str, str]]:
    """Yield every workflow `run` script with its job name."""
    for job_name, definition in jobs(load_workflow(workflow_name)).items():
        for step in steps(definition):
            script = run_script(step)
            if script:
                yield job_name, script


def component_groups_in_step(
    step: dict[str, typ.Any],
) -> typ.Iterator[tuple[str, list[str]]]:
    """Yield each `--component` flag found in one step, with its invocation."""
    for invocation in rustup_installs(run_script(step)):
        for values in component_values(invocation):
            yield invocation, values


def component_groups(workflow_name: str) -> typ.Iterator[ComponentGroup]:
    """Yield every `--component` flag in a workflow, wherever it appears."""
    for job_name, definition in jobs(load_workflow(workflow_name)).items():
        for invocation, values in steps_of(definition):
            yield ComponentGroup(job_name, invocation, values)


def steps_of(
    definition: dict[str, typ.Any],
) -> typ.Iterator[tuple[str, list[str]]]:
    """Yield each `--component` flag across one job's steps."""
    for step in steps(definition):
        yield from component_groups_in_step(step)


def installer_invocations(job_definition: dict[str, typ.Any]) -> list[str]:
    """Return each rustup install in one job, in step order."""
    return [
        invocation
        for step in steps(job_definition)
        for invocation in rustup_installs(run_script(step))
    ]


def _toolchain_of(invocation: str) -> str | None:
    """Return the toolchain named after `rustup toolchain install`."""
    tokens = invocation.split()
    return tokens[3] if len(tokens) > 3 else None


def _profile_of(invocation: str) -> str | None:
    """Return the value given to `--profile`, if any."""
    tokens = invocation.split()
    return next(
        (
            tokens[index + 1]
            for index, token in enumerate(tokens[:-1])
            if token == "--profile"
        ),
        None,
    )


def components_of(invocation: str) -> set[str]:
    """Return every component one invocation names, splitting comma lists."""
    return {
        component
        for values in component_values(invocation)
        for value in values
        for component in value.split(",")
        if component
    }


def assert_nightly_installer(job_definition: dict[str, typ.Any]) -> None:
    """Require the nightly lane's one installer to name both components.

    The lane exists to run portable-SIMD tests under a nightly compiler. An
    installer that is absent, names another toolchain, or omits a component
    leaves the lane either unable to start or silently testing the stable
    toolchain, which is the vacuous pass this contract exists to catch.
    Exactly one invocation is required so a second install cannot supply the
    missing component under a different profile or toolchain.

    Parameters
    ----------
    job_definition : dict[str, typing.Any]
        The parsed `nightly-portable-simd` job.

    Examples
    --------
    A comma-separated component list names both components:

    >>> assert_nightly_installer({"steps": [{"run": (
    ...     "rustup toolchain install nightly --profile minimal "
    ...     "--component clippy,rustfmt"
    ... )}]})
    """
    installers = installer_invocations(job_definition)
    assert len(installers) == 1, (
        f"{NIGHTLY_WORKFLOW}:{NIGHTLY_JOB} must install a toolchain exactly "
        f"once, got {installers!r}"
    )
    invocation = installers[0]
    assert _toolchain_of(invocation) == NIGHTLY_TOOLCHAIN, (
        f"{NIGHTLY_WORKFLOW}:{NIGHTLY_JOB} must install {NIGHTLY_TOOLCHAIN}, "
        f"got {_toolchain_of(invocation)!r} in {invocation!r}"
    )
    assert _profile_of(invocation) == NIGHTLY_PROFILE, (
        f"{NIGHTLY_WORKFLOW}:{NIGHTLY_JOB} must install the "
        f"{NIGHTLY_PROFILE} profile, got {_profile_of(invocation)!r} in "
        f"{invocation!r}"
    )
    missing = NIGHTLY_COMPONENTS - components_of(invocation)
    assert not missing, (
        f"{NIGHTLY_WORKFLOW}:{NIGHTLY_JOB} must install "
        f"{sorted(NIGHTLY_COMPONENTS)}, missing {sorted(missing)} in "
        f"{invocation!r}"
    )