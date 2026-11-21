"""Contract tests for where the scheduling gate runs in CI.

A Make target nobody runs protects nothing. These tests read the
``build-test`` job in ``.github/workflows/ci.yml`` and require the scheduling
target to run there, beside ``make test-workflow-contracts``, after
``cargo-nextest`` is installed, and in a step that fails the job when it fails.

The recipe's own contents are pinned by ``scheduling_contract_test``. This
module covers only the wiring that decides whether the recipe ever executes,
reading each step's script through ``scheduling_script_reading`` and the shell
that runs it through ``scheduling_shell_reading``.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import pytest
from scheduling_reading import TARGET
from scheduling_script_reading import NEXTEST_INSTALLER, installs_nextest
from scheduling_selector_cases import SELECTOR_CASES
from scheduling_shell_reading import effective_shell, runs_the_script
from workflow_support import job, load_workflow, run_script, steps


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
    failing regression leave the job green. The shell matters as well: a
    custom template such as ``bash -n {0}`` passes every check above while
    only parsing the script, so the effective shell must execute it.
    """
    assert step is not None, f"build-test must run {command}"
    condition = step.get("if")
    assert condition is None, (
        f"{command} must run unconditionally; it is guarded by {condition!r}"
    )
    assert "continue-on-error" not in step, (
        f"{command} must fail the job when it fails; it is non-blocking"
    )
    shell = effective_shell(
        step, job("ci.yml", "build-test"), load_workflow("ci.yml")
    )
    assert runs_the_script(shell), (
        f"{command} must execute; its shell {shell!r} would only parse it"
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
        pytest.param(script, expected, id=name)
        for script, expected, name in SELECTOR_CASES
    ],
)
def test_the_installer_selector_reads_executable_use(
    script: str, expected: bool
) -> None:
    """Printing, commenting, parsing or masking the path installs nothing.

    The ordering contract places the installer before the scheduling target,
    so a selector that accepted a mention would let a step that installs
    nothing satisfy it. The same is true of a command whose failure is
    swallowed: ``installer || true`` runs the script and reports success
    whether or not it worked, ``bash -n`` only parses it, and a ``||`` further
    along an ``&&`` chain masks the installer's failure just as well as one
    placed right after it. A preceding ``||`` is the mirror image: ``true ||
    installer`` gives the selector a command that runs to completion and never
    installs, and so does a preceding ``&&`` whose left side may fail, as
    ``false && installer`` never runs. Only ``;`` starts a command list that
    runs whatever came before it. A ``;`` after the installer ends the
    installer's own list, so its failure is not exempt from ``errexit`` and
    the step stops there: ``installer; true`` and ``installer ; make test``
    both count, and so does a trailing ``installer ;``. A trailing ``&`` lets
    the rest of the step start first whether it follows the installer or a
    later command in the chain.

    Text decides the reading as well as syntax: a quoted ``&&`` is an
    argument, not an operator, and an escaped operator is data for the same
    reason, so a step that echoes two escaped ampersands runs nothing. A ``#``
    inside a redirection's word does not start a comment, so the mask after
    it stays visible. ``errexit`` completes the picture: the installer's
    failure in an ``&&`` chain is exempt unless it is the line's last command,
    so a later success absorbs it. The chain therefore has to end the script,
    and a comment after it changes nothing — no command follows to replace
    the status. A chain that ends in a trailing operator continues onto the
    next line, so the two lines are read as the one command list the shell
    runs, and a backslash continuation inside the path itself joins the two
    halves into the word the shell runs. A here-document's body is input for
    its command rather than a command line, so a path written there installs
    nothing; and ``errexit`` itself can be turned off, so ``set +e`` before
    the installer makes its failure ignorable until ``set -e`` restores it.
    A container decides where a command runs, and this reading tracks none:
    an installer inside an ``if`` body is refused whether the branch is taken
    or not, because a branch is not an unconditional installation. A ``set``
    inside a pipeline element or a backgrounded command runs in a subshell of
    its own, so it changes nothing the enclosing script sees and leaves
    ``errexit`` as it was. A here-document's terminator is compared as the
    shell compares it: a quoted delimiter makes the body literal, so a
    backslash there is a character, while an unquoted body deletes
    backslash-newline pairs and can close on a line the literal body would
    keep.
    """
    assert installs_nextest(script) is expected, (
        f"installs_nextest must return {expected} for {script!r}"
    )
