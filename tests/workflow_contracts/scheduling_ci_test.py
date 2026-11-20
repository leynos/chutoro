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
        pytest.param(NEXTEST_INSTALLER, True, id="the-installer"),
        pytest.param(f"bash {NEXTEST_INSTALLER}", True, id="through-a-shell"),
        pytest.param(
            f"make test && {NEXTEST_INSTALLER}",
            False,
            id="after-an-unproven-command",
        ),
        pytest.param(
            f"false && {NEXTEST_INSTALLER}",
            False,
            id="skipped-by-a-preceding-failure",
        ),
        pytest.param(
            f"echo done ; {NEXTEST_INSTALLER}",
            True,
            id="after-a-semicolon",
        ),
        pytest.param(
            f"false && make test ; {NEXTEST_INSTALLER}",
            True,
            id="after-a-semicolon-that-resets-the-chain",
        ),
        pytest.param(
            f"{NEXTEST_INSTALLER} && make test-workflow-contracts",
            True,
            id="before-another-command",
        ),
        pytest.param(f"echo {NEXTEST_INSTALLER}", False, id="printed"),
        pytest.param(f"# {NEXTEST_INSTALLER}", False, id="commented"),
        pytest.param(
            f"echo hi  # {NEXTEST_INSTALLER}", False, id="trailing-comment"
        ),
        pytest.param(
            f"{NEXTEST_INSTALLER} || true", False, id="failure-masked"
        ),
        pytest.param(
            f"{NEXTEST_INSTALLER}; true", True, id="a-semicolon-does-not-mask"
        ),
        pytest.param(
            f"{NEXTEST_INSTALLER} ; make test",
            True,
            id="a-semicolon-ends-the-installers-own-list",
        ),
        pytest.param(f"{NEXTEST_INSTALLER} ;", True, id="a-trailing-semicolon"),
        pytest.param(
            f"bash -n {NEXTEST_INSTALLER}", False, id="syntax-check-only"
        ),
        pytest.param(
            f"bash {NEXTEST_INSTALLER} || true",
            False,
            id="a-shell-that-masks",
        ),
        pytest.param(
            f"{NEXTEST_INSTALLER} && echo installed || true",
            False,
            id="masked-later-on-the-line",
        ),
        pytest.param(
            f"{NEXTEST_INSTALLER} && make test && echo done",
            True,
            id="a-chain-that-carries-the-status",
        ),
        pytest.param(
            f"true || {NEXTEST_INSTALLER}",
            False,
            id="skipped-by-a-preceding-success",
        ),
        pytest.param(
            f"{NEXTEST_INSTALLER} &", False, id="sent-to-the-background"
        ),
        pytest.param(
            f"{NEXTEST_INSTALLER} && echo done &",
            False,
            id="backgrounded-at-the-end",
        ),
        pytest.param(
            f"echo '&&' {NEXTEST_INSTALLER}",
            False,
            id="a-quoted-operator",
        ),
        pytest.param(
            "echo " + chr(92) + "&" + chr(92) + "& " + NEXTEST_INSTALLER,
            False,
            id="an-escaped-operator",
        ),
        pytest.param(
            "scripts/install-" + chr(92) + "\nnextest.sh",
            True,
            id="a-path-split-by-a-continuation",
        ),
        pytest.param(
            f"{NEXTEST_INSTALLER} >install#log || true",
            False,
            id="a-mask-after-a-word-internal-hash",
        ),
        pytest.param(
            f"{NEXTEST_INSTALLER} && echo installed\ntrue",
            False,
            id="a-chain-swallowed-by-a-later-line",
        ),
        pytest.param(
            f"{NEXTEST_INSTALLER} && echo installed\n# installed",
            True,
            id="a-chain-that-ends-the-script",
        ),
        pytest.param(
            f"{NEXTEST_INSTALLER} # don't skip this",
            True,
            id="a-comment-with-an-apostrophe",
        ),
        pytest.param(
            f"{NEXTEST_INSTALLER} &&\ntrue",
            True,
            id="a-chain-continued-on-the-next-line",
        ),
        pytest.param(
            f"true ||\n{NEXTEST_INSTALLER}",
            False,
            id="skipped-across-a-continued-line",
        ),
        pytest.param(
            f"{NEXTEST_INSTALLER} &&\n# installed\ntrue",
            True,
            id="a-continuation-past-a-comment-line",
        ),
        pytest.param(
            f"set +e\n{NEXTEST_INSTALLER}\ntrue",
            False,
            id="errexit-disabled-before-the-installer",
        ),
        pytest.param(
            f"set -e\n{NEXTEST_INSTALLER}",
            True,
            id="errexit-enabled-before-the-installer",
        ),
        pytest.param(
            f"set +e\nset -e\n{NEXTEST_INSTALLER}",
            True,
            id="errexit-restored-before-the-installer",
        ),
        pytest.param(
            f"set +e\nset -- -e\n{NEXTEST_INSTALLER}",
            False,
            id="a-positional-parameter-named-e",
        ),
        pytest.param(
            f"set +e\nset value -e\n{NEXTEST_INSTALLER}",
            False,
            id="a-positional-parameter-before-errexit",
        ),
        pytest.param(
            f"set +e\nset -o pipefail -e\n{NEXTEST_INSTALLER}",
            True,
            id="an-option-operand-before-errexit",
        ),
        pytest.param(
            f"set +e\nfalse && set -e\n{NEXTEST_INSTALLER}\ntrue",
            False,
            id="an-errexit-change-on-a-skipped-branch",
        ),
        pytest.param(
            f"exit 0\n{NEXTEST_INSTALLER}",
            False,
            id="an-installer-after-an-exit",
        ),
        pytest.param(
            f"{NEXTEST_INSTALLER}\nexit 0",
            True,
            id="an-installer-before-an-exit",
        ),
        pytest.param(
            f"cat <<'EOF'\n{NEXTEST_INSTALLER}\nEOF",
            False,
            id="a-here-document-body",
        ),
        pytest.param(
            f"cat <<'EOF'\n{NEXTEST_INSTALLER}\nEOF\n{NEXTEST_INSTALLER}",
            True,
            id="after-a-here-document-body",
        ),
        pytest.param(
            f"cat <<'EOF'\n\tEOF\n{NEXTEST_INSTALLER}\nEOF",
            False,
            id="a-tab-indented-terminator-of-an-exact-document",
        ),
        pytest.param(
            f"cat <<-EOF\n\t{NEXTEST_INSTALLER}\n\tEOF\n{NEXTEST_INSTALLER}",
            True,
            id="a-tab-indented-terminator-of-a-stripping-document",
        ),
        pytest.param(
            f"cat <<-EOF\n  EOF\n{NEXTEST_INSTALLER}\nEOF",
            False,
            id="a-space-indented-terminator-of-a-stripping-document",
        ),
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
    """
    assert installs_nextest(script) is expected, (
        f"installs_nextest must return {expected} for {script!r}"
    )
