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

from scheduling_reading import TARGET
from workflow_support import job, run_script, steps


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
        (index for index, run in enumerate(runs) if "install-nextest.sh" in run),
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
            if "install-nextest.sh" in run_script(step)
        ),
        None,
    )
    _assert_unconditional(installer, "scripts/install-nextest.sh")