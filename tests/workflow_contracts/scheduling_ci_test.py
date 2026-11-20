"""Contract tests for where the scheduling gate runs in CI.

A Make target nobody runs protects nothing. These tests read the
``build-test`` job in ``.github/workflows/ci.yml`` and require the scheduling
target to run there, beside ``make test-workflow-contracts``, after
``cargo-nextest`` is installed, and in a step that fails the job when it fails.

The recipe's own contents are pinned by ``scheduling_contract_test``; this
module covers only the wiring that decides whether the recipe ever executes.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import shlex

import pytest
from scheduling_reading import SHELL_SEPARATORS, TARGET
from workflow_support import job, run_script, steps

#: The script that installs ``cargo-nextest`` for the job.
NEXTEST_INSTALLER = "scripts/install-nextest.sh"

#: Words that run the command after them. ``bash scripts/install-nextest.sh``
#: installs; ``echo scripts/install-nextest.sh`` only prints the path.
INTERPRETERS = frozenset({"bash", "sh", "dash"})


def _joined_lines(script: str) -> list[str]:
    """Return `script`'s lines with backslash continuations joined.

    A workflow script wraps long commands with a trailing backslash. Passing
    one of those lines to ``shlex`` on its own raises ``No escaped
    character``, because the escape has nothing left to escape, so the
    continuations are resolved first.

    Examples
    --------
    The continuation is built with ``chr(92)`` and ``chr(10)``, because a
    docstring holding those escapes would end the example's line:

    >>> _joined_lines(
    ...     "cargo build " + chr(92) + chr(10) + "  --release"
    ... )
    ['cargo build    --release']
    """
    lines: list[str] = []
    pending = ""
    for raw in script.splitlines():
        line = raw.rstrip()
        if line.endswith("\\"):
            pending = f"{pending}{line[:-1]} "
            continue
        lines.append(f"{pending}{line}")
        pending = ""
    if pending.strip():
        lines.append(pending)
    return lines


def _simple_commands(script: str) -> list[list[str]]:
    """Return the simple commands `script` runs, one token list each.

    A step's script can hold several commands, separated by newlines or by
    shell operators, and a comment line names a script without running it.
    Splitting first is what makes "is the installer executed?" a question
    about a command's first word rather than about a substring.

    Examples
    --------
    >>> _simple_commands("echo scripts/install-nextest.sh")
    [['echo', 'scripts/install-nextest.sh']]
    >>> _simple_commands("# scripts/install-nextest.sh")
    []
    """
    commands: list[list[str]] = []
    for line in _joined_lines(script):
        lexer = shlex.shlex(line, posix=True, punctuation_chars=";&|")
        lexer.whitespace_split = True
        current: list[str] = []
        for token in lexer:
            if token in SHELL_SEPARATORS:
                if current:
                    commands.append(current)
                current = []
                continue
            current.append(token)
        if current:
            commands.append(current)
    return commands


def installs_nextest(script: str) -> bool:
    """Return whether `script` executes the nextest installer.

    The path must be a command's first word, or the argument of a shell
    interpreter. A substring check accepted ``echo scripts/install-nextest.sh``
    and a commented-out line, so the ordering contract could pass without
    anything being installed.

    Examples
    --------
    >>> installs_nextest("scripts/install-nextest.sh")
    True
    >>> installs_nextest("echo scripts/install-nextest.sh")
    False
    """
    return any(
        command[0] == NEXTEST_INSTALLER
        or (command[0] in INTERPRETERS and NEXTEST_INSTALLER in command[1:])
        for command in _simple_commands(script)
    )


def _build_test_steps() -> list[dict[str, object]]:
    """Return the ``build-test`` steps that run a script, in order.

    The step mapping is kept rather than only its command because a step can
    carry a condition that skips it. Reading the command alone would accept a
    step guarded by ``if: false`` as evidence that the gate runs.
    """
    return [
        step
        for step in steps(job("ci.yml", "build-test"))
        if run_script(step).strip()
    ]


def _build_test_runs() -> list[str]:
    """Return the run scripts of the ``build-test`` job, in order."""
    return [run_script(step).strip() for step in _build_test_steps()]


def _step_running(command: str) -> dict[str, object] | None:
    """Return the ``build-test`` step whose script is exactly `command`."""
    return next(
        (
            step
            for step in _build_test_steps()
            if run_script(step).strip() == command
        ),
        None,
    )


def _assert_unconditional(step: dict[str, object] | None, command: str) -> None:
    """Fail unless `step` exists and its failure would fail the job.

    These steps have no side effect worth guarding: they either run the
    evidence or the job has not done its work. A condition, including
    ``if: false``, would let the step disappear while the contract still
    found its command in the file, and ``continue-on-error`` would let a
    failing regression leave the job green.
    """
    assert step is not None, f"build-test must run {command}"
    condition = step.get("if")
    assert condition is None, (
        f"{command} must run unconditionally; it is guarded by {condition!r}"
    )
    assert "continue-on-error" not in step, (
        f"{command} must fail the job when it fails; it is non-blocking"
    )


def test_ci_runs_the_scheduling_target_beside_the_workflow_contracts() -> None:
    """The gate must run in the build job, not only in a local recipe."""
    runs = _build_test_runs()
    assert f"make {TARGET}" in runs, (
        f"build-test must run make {TARGET}; it runs {runs}"
    )
    assert "make test-workflow-contracts" in runs, (
        "the scheduling gate belongs with the workflow contracts"
    )
    _assert_unconditional(_step_running(f"make {TARGET}"), f"make {TARGET}")


def test_the_build_test_job_is_not_conditional() -> None:
    """A job-level condition skips every step the contract relies on.

    The step check reads only the scheduling step's own guard. A condition
    on ``build-test`` itself would skip the whole job, including that step,
    and the evidence would never run.
    """
    definition = job("ci.yml", "build-test")
    condition = definition.get("if")
    assert condition is None, (
        f"build-test must run unconditionally; it is guarded by {condition!r}"
    )
    assert "continue-on-error" not in definition, (
        "build-test must fail when a gate fails; the job tolerates failure"
    )


def test_ci_installs_nextest_before_the_scheduling_target() -> None:
    """The target cannot select tests if the nextest binary is absent.

    ``scripts/install-nextest.sh`` is the job's only nextest installation. A
    scheduling step placed before it would fail for a reason unrelated to the
    regressions it exists to run, and either step being skipped would break
    the chain just as surely as running them in the wrong order.
    """
    runs = _build_test_runs()
    install = next(
        (index for index, run in enumerate(runs) if installs_nextest(run)),
        None,
    )
    target = next(
        (index for index, run in enumerate(runs) if run == f"make {TARGET}"),
        None,
    )
    assert install is not None, "build-test must install cargo-nextest"
    assert target is not None, f"build-test must run make {TARGET}"
    assert install < target, (
        "cargo-nextest must be installed before the scheduling target runs"
    )
    installer = next(
        (
            step
            for step in _build_test_steps()
            if installs_nextest(run_script(step))
        ),
        None,
    )
    _assert_unconditional(installer, NEXTEST_INSTALLER)


@pytest.mark.parametrize(
    ("script", "expected"),
    [
        pytest.param(NEXTEST_INSTALLER, True, id="the-installer"),
        pytest.param(f"bash {NEXTEST_INSTALLER}", True, id="through-a-shell"),
        pytest.param(
            f"make test && {NEXTEST_INSTALLER}", True, id="after-another-command"
        ),
        pytest.param(f"echo {NEXTEST_INSTALLER}", False, id="printed"),
        pytest.param(f"# {NEXTEST_INSTALLER}", False, id="commented"),
        pytest.param(
            f"echo hi  # {NEXTEST_INSTALLER}", False, id="trailing-comment"
        ),
    ],
)
def test_the_installer_selector_reads_executable_use(
    script: str, expected: bool
) -> None:
    """Printing or commenting the path must not read as installing it.

    The ordering contract places the installer before the scheduling target,
    so a selector that accepted a mention would let a step that installs
    nothing satisfy it.
    """
    assert installs_nextest(script) is expected, (
        f"installs_nextest must return {expected} for {script!r}"
    )