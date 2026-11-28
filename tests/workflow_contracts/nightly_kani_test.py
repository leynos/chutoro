"""Contract tests for the nightly Kani workflow's freshness guarantee.

The nightly run exists to verify one exact commit, unconditionally. A
commit-age gate used to decide whether to run, so a missed or
infrastructure-failed night could age out and be followed by a skipped green
workflow: the head was never verified and nothing said so. These tests pin the
replacement behaviour, because none of it is observable from the job's exit
status alone:

- the checkout resolves ``${{ github.sha }}``, not the branch tip;
- a step records the checkout's own ``git rev-parse HEAD`` in the job summary
  before the suite runs, so a failed run still names what it was verifying;
- neither the job nor any step carries a condition, so no age gate, ``needs``
  or ``continue-on-error`` can suppress or tolerate the run; and
- the suite is the full tier, not the practical one.

Reintroducing an ``if:`` on the job, replacing the checkout ref with the branch
tip, or dropping the recorded SHA would otherwise pass every test in the tree.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "nightly-kani.yml"
CHECKOUT_RE = re.compile(r"^actions/checkout@[0-9a-f]{40}$")


@pytest.fixture(scope="module")
def workflow() -> dict[str, object]:
    """Parse the nightly workflow once for every contract test."""
    return yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))


def _job(workflow: dict[str, object]) -> dict[str, object]:
    """Return the workflow's single verification job."""
    jobs = workflow.get("jobs")
    assert isinstance(jobs, dict), "the workflow must declare a jobs mapping"
    assert list(jobs) == ["kani-full"], (
        f"expected a single job named 'kani-full', found {sorted(jobs)}"
    )
    return jobs["kani-full"]


def _steps(workflow: dict[str, object]) -> list[dict[str, object]]:
    """Return the verification job's step list."""
    steps = _job(workflow).get("steps")
    assert isinstance(steps, list) and steps, "jobs.kani-full.steps is missing"
    return steps


def _runs(steps: list[dict[str, object]]) -> list[tuple[int, str]]:
    """Return each step's `run` script with its position."""
    return [
        (index, step["run"])
        for index, step in enumerate(steps)
        if isinstance(step.get("run"), str)
    ]


def test_schedule_and_dispatch_are_both_declared(
    workflow: dict[str, object],
) -> None:
    """The nightly runs on its schedule and can be dispatched by hand."""
    triggers = workflow.get("on", workflow.get(True))
    assert isinstance(triggers, dict), "the workflow must declare an on: mapping"
    assert "schedule" in triggers, "on.schedule is missing"
    assert "workflow_dispatch" in triggers, "on.workflow_dispatch is missing"


def test_the_checkout_verifies_the_events_commit(
    workflow: dict[str, object],
) -> None:
    """Checkout resolves the event SHA and drops the workflow token.

    Defaulting to the branch tip would verify whatever the branch pointed at
    when the runner started, which for a scheduled run is not the commit the
    event named. Persisting credentials would also leave the token in the
    checkout's Git configuration.
    """
    steps = _steps(workflow)
    checkout = steps[0]
    uses = checkout.get("uses")
    assert isinstance(uses, str) and CHECKOUT_RE.match(uses), (
        f"the first step must be actions/checkout pinned to a 40-hex commit "
        f"SHA, got {uses!r}"
    )
    with_block = checkout.get("with")
    assert isinstance(with_block, dict), "the checkout step must carry with:"
    assert with_block.get("ref") == "${{ github.sha }}", (
        "the checkout must resolve the event's commit, got "
        f"{with_block.get('ref')!r}"
    )
    assert with_block.get("persist-credentials") is False, (
        "the checkout must set persist-credentials: false so the workflow "
        f"token is not retained, got {with_block!r}"
    )


def test_the_checked_out_commit_is_recorded_before_verification(
    workflow: dict[str, object],
) -> None:
    """A step writes the checkout's own HEAD to the job summary first.

    The reading is ``git rev-parse HEAD`` rather than the event payload, so a
    checkout that resolved somewhere else is visible. It must precede the
    suite, because a failed run is exactly the one whose commit is worth
    knowing.
    """
    steps = _steps(workflow)
    recorded = [
        index
        for index, run in _runs(steps)
        if "git rev-parse HEAD" in run and "GITHUB_STEP_SUMMARY" in run
    ]
    assert len(recorded) == 1, (
        "exactly one step must record the checked-out commit in the job "
        f"summary; found {len(recorded)}"
    )
    suite = [
        index
        for index, run in _runs(steps)
        if any(word.strip() == "make kani-full" for word in run.splitlines())
    ]
    assert len(suite) == 1, (
        f"exactly one step must run the full suite, found {len(suite)}"
    )
    assert recorded[0] < suite[0], (
        f"the commit must be recorded at {recorded[0]} before the suite at "
        f"{suite[0]}"
    )


def test_neither_the_job_nor_a_step_can_be_skipped(
    workflow: dict[str, object],
) -> None:
    """No condition, dependency or tolerance exists anywhere in the job.

    This is the freshness rule in its strongest form. An ``if:`` reintroduces
    an age gate; ``needs`` lets an upstream decision suppress the run; and
    ``continue-on-error`` reports success over a failed suite. Each one turns
    a missing verification into a green workflow.
    """
    job = _job(workflow)
    gated = [key for key in ("if", "needs", "continue-on-error") if key in job]
    assert not gated, (
        "jobs.kani-full must carry no condition, dependency or tolerance, got "
        f"{ {key: job.get(key) for key in gated}!r}"
    )
    conditional = [
        (step.get("name"), key)
        for step in _steps(workflow)
        for key in ("if", "needs", "continue-on-error")
        if key in step
    ]
    assert not conditional, (
        "no step of the nightly job may be conditional or tolerant, got "
        f"{conditional!r}"
    )


def test_the_age_gate_is_gone(workflow: dict[str, object]) -> None:
    """Nothing invokes the deleted commit-age gate binary."""
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "kani_nightly_gate" not in text, (
        "the commit-age gate binary was deleted; the workflow must not call it"
    )


def test_permissions_are_read_only(workflow: dict[str, object]) -> None:
    """The verification job's token grants contents: read and nothing broader."""
    permissions = _job(workflow).get("permissions")
    assert permissions == {"contents": "read"}, (
        "jobs.kani-full.permissions must be exactly {'contents': 'read'}, got "
        f"{permissions!r}"
    )