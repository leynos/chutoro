"""Contract tests for how a workflow provisions the shared prover tooling.

The prover targets shell out to ``uv tool run``, so ``uv`` must be on ``PATH``
before the target starts, and the install itself must remain one delegation to
``make install-kani`` rather than a repository downloader. These contracts live
beside ``kani_pr_test`` rather than inside it so each module stays within the
repository's module-size limit while the pull-request gate's provisioning
order is still pinned.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import re

import pytest
from workflow_support import job, run_script, steps, uses_reference

#: The pull-request Kani workflow, the only gate that installs a prover.
PR_WORKFLOW = "kani-pr.yml"
PR_JOB = "kani"

#: The uv action, which puts `uv tool run` on PATH for the prover target.
SETUP_UV_RE = re.compile(r"^astral-sh/setup-uv@[0-9a-f]{40}$")

#: The one command that installs the verifier for this gate.
INSTALL_COMMAND = "make install-kani"


@pytest.fixture(scope="module")
def pr_steps() -> list[dict[str, object]]:
    """Return the gating job's steps, in execution order."""
    return steps(job(PR_WORKFLOW, PR_JOB))


def _run_commands(pr_steps: list[dict[str, object]]) -> list[str]:
    """Return every step's `run` script, in the order the job executes them."""
    return [run_script(step).strip() for step in pr_steps if run_script(step)]


def test_kani_install_delegates_to_the_pinned_prover_tool(
    pr_steps: list[dict[str, object]],
) -> None:
    """The verifier installs through the shared, commit-pinned prover tool.

    The download, checksum, and extraction mechanics belong to
    ``rust-prover-tools``; the repository keeps only the target. Asserting the
    command rather than the step's name leaves renaming free while deleting the
    delegation fails.
    """
    commands = _run_commands(pr_steps)
    assert commands.count(INSTALL_COMMAND) == 1, (
        f"expected exactly one {INSTALL_COMMAND!r} step, found {commands!r}"
    )


def test_prover_tool_setup_precedes_the_install(
    pr_steps: list[dict[str, object]],
) -> None:
    """`uv` must be on PATH before the target that shells out to it runs."""
    setup_at = next(
        (
            index
            for index, step in enumerate(pr_steps)
            if SETUP_UV_RE.match(uses_reference(step))
        ),
        None,
    )
    assert setup_at is not None, (
        "a step must install uv pinned to a 40-hex commit, got "
        f"{[uses_reference(step) for step in pr_steps]!r}"
    )
    install_at = next(
        (
            index
            for index, step in enumerate(pr_steps)
            if run_script(step).strip() == INSTALL_COMMAND
        ),
        None,
    )
    assert install_at is not None, (
        f"no step runs {INSTALL_COMMAND!r}, got {_run_commands(pr_steps)!r}"
    )
    assert setup_at < install_at, (
        f"uv setup at step {setup_at} must precede the prover install at "
        f"step {install_at}, or `uv tool run` is not on PATH when it starts"
    )
