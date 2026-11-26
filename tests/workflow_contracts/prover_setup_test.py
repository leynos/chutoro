"""Contract tests for how the workflows provision the shared prover tooling.

The prover targets shell out to ``uv tool run``, so ``uv`` must be on ``PATH``
before a target starts, and the install itself must remain one delegation to a
prover Make target rather than a repository downloader. Each workflow that
installs a prover therefore carries the same three contracts: a commit-pinned
``setup-uv`` step, caching left to ``setup-rust``, and an install that follows
the setup.

Every one of those steps is also unconditional. The nightly job used to decide
in an earlier step whether to install and run at all, which made a step's
condition part of the contract; that gate is gone, because a skipped nightly
run is not evidence that its commit was verified. The setup, install and
execution steps now run whole, in order, on every run.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import collections.abc as cabc
import re
from pathlib import Path

import pytest
from workflow_support import installs_prover, job, run_script, steps, uses_reference

#: The uv action, which puts `uv tool run` on PATH for the prover target.
SETUP_UV_RE = re.compile(r"^astral-sh/setup-uv@[0-9a-f]{40}$")

#: The Python the prover tool needs, and the input that pins it.
PYTHON_VERSION = "3.14"

#: Each workflow, its job, the tool its install step asks the shared
#: install-prover-tools action for, and the execution command it runs.
PROVER_JOBS = (
    ("kani-pr.yml", "kani", "kani", "make kani"),
    ("ci.yml", "verus-proofs", "verus", "make verus"),
    ("nightly-kani.yml", "kani-full", "kani", "make kani-full"),
)

JOB_IDS = tuple(
    f"{Path(workflow_name).stem}-{job_name}"
    for workflow_name, job_name, _, _ in PROVER_JOBS
)


def _step_index(
    job_steps: list[dict[str, object]],
    predicate: cabc.Callable[[dict[str, object]], bool],
) -> int | None:
    """Return the index of the first step satisfying the predicate."""
    return next(
        (index for index, step in enumerate(job_steps) if predicate(step)), None
    )


def _uv_setup_index(job_steps: list[dict[str, object]]) -> int:
    """Return the index of the commit-pinned uv setup step."""
    index = _step_index(
        job_steps, lambda step: bool(SETUP_UV_RE.match(uses_reference(step)))
    )
    assert index is not None, (
        "a step must install uv pinned to a 40-hex commit, got "
        f"{[uses_reference(step) for step in job_steps]!r}"
    )
    return index


@pytest.mark.parametrize(
    ("workflow_name", "job_name", "install", "run"),
    PROVER_JOBS,
    ids=JOB_IDS,
)
def test_uv_setup_is_pinned_and_precedes_the_install(
    workflow_name: str, job_name: str, install: str, run: str
) -> None:
    """The provisioned `uv` is pinned, uncached, and installed before use.

    `make install-*` shells out to `uv tool run`, so a setup step that runs
    afterwards leaves the target with no `uv` at all. Caching is disabled
    because `setup-rust` already gave `~/.cache/uv` an owner; a second one
    would compete for the same key.
    """
    job_steps = steps(job(workflow_name, job_name))
    setup_at = _uv_setup_index(job_steps)

    with_block = job_steps[setup_at].get("with")
    assert isinstance(with_block, dict), (
        f"{workflow_name}:{job_name} must pass inputs to the uv setup step, "
        f"got {with_block!r}"
    )
    assert with_block.get("python-version") == PYTHON_VERSION, (
        f"{workflow_name}:{job_name} must run the prover tool on Python "
        f"{PYTHON_VERSION}, got {with_block.get('python-version')!r}"
    )
    assert with_block.get("enable-cache") is False, (
        f"{workflow_name}:{job_name} must leave the uv cache to setup-rust, "
        f"got enable-cache: {with_block.get('enable-cache')!r}"
    )

    install_at = _step_index(job_steps, lambda step: installs_prover(step, install))
    assert install_at is not None, (
        f"{workflow_name}:{job_name} has no SHA-pinned install-prover-tools step "
        f"for {install!r}, got {[uses_reference(step) for step in job_steps]!r}"
    )
    assert setup_at < install_at, (
        f"{workflow_name}:{job_name} sets up uv at step {setup_at} but installs "
        f"at step {install_at}, so `uv tool run` is not on PATH when it starts"
    )


@pytest.mark.parametrize(
    ("workflow_name", "job_name", "install", "run"),
    PROVER_JOBS,
    ids=JOB_IDS,
)
def test_setup_install_and_run_are_unconditional_and_ordered(
    workflow_name: str, job_name: str, install: str, run: str
) -> None:
    """Setup, install and execution run whole, in that order, every time.

    A condition on any of the three would let a run finish without
    verification while the job still reported a conclusion. The nightly job
    used to guard all three with one gate; that gate is deleted, and the
    deleted behaviour must not reappear as a step condition.
    """
    job_steps = steps(job(workflow_name, job_name))
    positions = {
        "uv setup": _uv_setup_index(job_steps),
        "install": _step_index(job_steps, lambda step: installs_prover(step, install)),
        "execution": _step_index(job_steps, lambda step: run_script(step).strip() == run),
    }
    missing = [name for name, index in positions.items() if index is None]
    assert not missing, (
        f"{workflow_name}:{job_name} has no step for {missing!r}, got "
        f"{[run_script(step) for step in job_steps]!r}"
    )
    ordered = sorted(positions.values())
    assert ordered == [positions["uv setup"], positions["install"], positions["execution"]], (
        f"{workflow_name}:{job_name} must set up uv, install, then run, got "
        f"{positions!r}"
    )
    conditional = {
        name: job_steps[index].get("if")
        for name, index in positions.items()
        if "if" in job_steps[index]
    }
    assert not conditional, (
        f"{workflow_name}:{job_name} must run these steps unconditionally, so "
        f"a green job means the proof was executed; found conditions "
        f"{conditional!r}"
    )
    # A tolerant proof step reports success over a failed run, which is the
    # same defect as a condition: the job concludes without the guarantee.
    tolerant = {
        name: job_steps[index].get("continue-on-error")
        for name, index in positions.items()
        if "continue-on-error" in job_steps[index]
    }
    assert not tolerant, (
        f"{workflow_name}:{job_name} must not tolerate a failing proof step; "
        f"found continue-on-error on {tolerant!r}"
    )
    # The job itself must be unconditional too. A job-level `if` skips the
    # whole proof run while the job still reports a conclusion, and a
    # job-level `continue-on-error` turns a failed proof into a green run.
    # The required-context contract checks this for the required jobs; the
    # nightly one is not required, so nothing else would.
    document = job(workflow_name, job_name)
    gated = [key for key in ("if", "continue-on-error") if key in document]
    assert not gated, (
        f"{workflow_name}:{job_name} must run its proof unconditionally; "
        f"found job-level {gated!r}"
    )