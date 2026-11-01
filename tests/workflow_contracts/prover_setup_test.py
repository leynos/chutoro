"""Contract tests for how the workflows provision the shared prover tooling.

The prover targets shell out to ``uv tool run``, so ``uv`` must be on ``PATH``
before a target starts, and the install itself must remain one delegation to a
prover Make target rather than a repository downloader. Each workflow that
installs a prover therefore carries the same three contracts: a commit-pinned
``setup-uv`` step, caching left to ``setup-rust``, and an install that follows
the setup.

The nightly gate runs only when its own gate step agrees, which makes a step's
condition part of the contract: the setup, install, and execution steps must
share one condition, or the job fails on exactly the night it was selected to
run.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import collections.abc as cabc
import re

import pytest
from workflow_support import job, run_script, steps, uses_reference

#: The uv action, which puts `uv tool run` on PATH for the prover target.
SETUP_UV_RE = re.compile(r"^astral-sh/setup-uv@[0-9a-f]{40}$")

#: The Python the prover tool needs, and the input that pins it.
PYTHON_VERSION = "3.14"

#: Each workflow, its job, and the install and execution commands it runs.
PROVER_JOBS = (
    ("kani-pr.yml", "kani", "make install-kani", "make kani"),
    ("ci.yml", "verus-proofs", "make install-verus", "make verus"),
    ("nightly-kani.yml", "kani-full", "make install-kani", "make kani-full"),
)

JOB_IDS = tuple(f"{name[1]}-{job_name}" for name, job_name, _, _ in PROVER_JOBS)


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


def _condition(step: dict[str, object]) -> str | None:
    """Return a step's `if` condition, normalised for whitespace."""
    condition = step.get("if")
    return condition.strip() if isinstance(condition, str) else None


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

    install_at = _step_index(
        job_steps, lambda step: run_script(step).strip() == install
    )
    assert install_at is not None, (
        f"{workflow_name}:{job_name} runs no {install!r} step, got "
        f"{[run_script(step) for step in job_steps]!r}"
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
def test_setup_install_and_run_share_one_condition(
    workflow_name: str, job_name: str, install: str, run: str
) -> None:
    """A conditional install and run must share the setup's condition.

    The nightly job decides in an earlier step whether the toolchain is worth
    installing. If the gate guards the install but not the run, the run fails
    for want of a verifier; if it guards the setup but not the install, the
    install fails for want of `uv`. The three conditions must therefore agree,
    whichever condition that is.
    """
    job_steps = steps(job(workflow_name, job_name))
    positions = {
        "uv setup": _uv_setup_index(job_steps),
        "install": _step_index(
            job_steps, lambda step: run_script(step).strip() == install
        ),
        "execution": _step_index(job_steps, lambda step: run_script(step).strip() == run),
    }
    missing = [name for name, index in positions.items() if index is None]
    assert not missing, (
        f"{workflow_name}:{job_name} has no step for {missing!r}, got "
        f"{[run_script(step) for step in job_steps]!r}"
    )

    conditions = {
        name: _condition(job_steps[index]) for name, index in positions.items()
    }
    assert len(set(conditions.values())) == 1, (
        f"{workflow_name}:{job_name} must guard both sides of the prover "
        f"install with one condition, got {conditions!r}"
    )