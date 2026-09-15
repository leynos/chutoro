"""Contract-test where each job runs.

Chutoro pays for the jobs on the developer feedback path, where GitHub's
queue can stretch to hours, and for those only. Weekly, nightly, mutation,
and administrative jobs are off that path, so a paid queue buys them
nothing while their long runtimes would dominate the bill; they stay on
GitHub-hosted runners. These tests fail on the pull request that moves a
job across that line, rather than after an invoice arrives.

The line is drawn at the event, not at the workflow file. A job that can
run for a scheduled event stays GitHub-hosted even when it shares a file
with paid work, and it may earn a paid runner for its pull-request runs
only by selecting the label from ``github.event_name``, which is what
``event_selected_runners`` reads. Push and dispatch runs of a lane that
already runs on pull requests are the same work on the same cache, so
they are not what this contract is guarding against.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import typing as typ

import pytest
import yaml
from workflow_support import (
    ACTIONLINT_CONFIG,
    GITHUB_HOSTED_LABELS,
    NON_FORK_GUARD,
    PULL_REQUEST_EVENT_GUARD,
    conditional_runner,
    is_pull_request_only,
    is_reusable_call,
    job,
    jobs,
    load_workflow,
    runner_labels,
    triggers,
    workflow_names,
)

#: Every job authorized to use a paid runner, and the label it must carry.
#: The shape is asserted by value rather than left free, because a paid
#: runner is the one thing here that silently costs more when someone
#: reaches for a bigger one. Two cores is the estate default and the
#: measurement supports it on each lane; the before-and-after wall times per
#: lane are in "Runner shapes" in docs/developers-guide.md. Raising a shape
#: needs a wall-time or disk measurement in the pull request that raises it.
PAID_JOBS = {
    ("benchmark-regressions.yml", "benchmark-policy"): "ubicloud-standard-2",
    ("benchmark-regressions.yml", "benchmark-smoke"): "ubicloud-standard-2",
    ("ci.yml", "build-test"): "ubicloud-standard-2",
    ("ci.yml", "verus-proofs"): "ubicloud-standard-2",
    ("coverage-main.yml", "coverage-upload"): "ubicloud-standard-2",
    ("kani-pr.yml", "kani"): "ubicloud-standard-2",
    ("property-tests.yml", "property-tests-pr"): "ubicloud-standard-2",
}

#: Event names that never represent a pull-request feedback loop. ``push``
#: and ``workflow_dispatch`` were here until the Linux lanes moved: the
#: trunk coverage job runs on ``push`` and writes the compiler-cache key
#: every pull request reads, and that key carries ``runner.environment``, so
#: a writer left on GitHub's runners would fill a store its readers cannot
#: see. A dispatch is a manual re-run of a lane that already runs paid.
#: ``schedule`` is the line that stays: nothing behind it blocks a
#: developer, and GitHub's Linux minutes are free on a public repository.
OFF_PATH_EVENTS = frozenset({"schedule"})


def _self_hosted(labels: list[str]) -> bool:
    """Report whether any label falls outside GitHub's hosted pool."""
    return any(label not in GITHUB_HOSTED_LABELS for label in labels)


# A job that picks its runner from `github.event_name` is billed only for
# the arm it takes, so the two arms are read separately. Reading the raw
# `runs-on` instead would see one unrecognized expression string and treat
# every such job as if it were always paid, which would put the two
# benchmark lanes below in breach of a rule they keep.
def _paid_labels(job_definition: dict[str, typ.Any]) -> list[str]:
    """Return the labels a job can bill for, fixed or conditional.

    Examples
    --------
    >>> _paid_labels({"runs-on": "ubuntu-latest"})
    []
    >>> _paid_labels(
    ...     {
    ...         "runs-on": "${{ github.event.pull_request.head.repo.fork"
    ...         " && 'ubuntu-latest' || 'ubicloud-standard-2' }}"
    ...     }
    ... )
    ['ubicloud-standard-2']
    """
    selected = conditional_runner(job_definition)
    labels = (
        [selected.paid, selected.otherwise]
        if selected
        else runner_labels(job_definition)
    )
    return [label for label in labels if label not in GITHUB_HOSTED_LABELS]


@pytest.mark.parametrize("workflow_name", workflow_names())
def test_only_approved_jobs_use_a_paid_runner(workflow_name: str) -> None:
    """Fail when any job other than an approved one leaves GitHub hosting."""
    for job_name, definition in jobs(load_workflow(workflow_name)).items():
        labels = _paid_labels(definition)
        if not labels:
            continue
        expected = PAID_JOBS.get((workflow_name, job_name))
        assert expected is not None, (
            f"{workflow_name}:{job_name} uses {labels}, but only "
            f"{sorted(name for _, name in PAID_JOBS)} may leave GitHub hosting"
        )
        assert labels == [expected], (
            f"{workflow_name}:{job_name} must use {expected}, not {labels}"
        )


@pytest.mark.parametrize("coordinate", sorted(PAID_JOBS))
def test_each_paid_job_keeps_its_current_label(coordinate: tuple[str, str]) -> None:
    """Pin each paid runner's shape, the setting that costs more silently.

    Parametrizing over the approved coordinates rather than over the
    workflows is what makes a deleted or renamed lane fail here: iterating
    the workflows would simply stop looking at a job that had gone.
    """
    workflow_name, job_name = coordinate
    label = PAID_JOBS[coordinate]
    labels = _paid_labels(job(workflow_name, job_name))
    assert labels == [label], (
        f"{workflow_name}:{job_name} must keep {label}, found {labels}"
    )


@pytest.mark.parametrize("workflow_name", workflow_names())
def test_jobs_reachable_off_the_feedback_path_stay_github_hosted(
    workflow_name: str,
) -> None:
    """Keep scheduled work on GitHub's runners, arm by arm."""
    workflow = load_workflow(workflow_name)
    if not OFF_PATH_EVENTS & set(triggers(workflow)):
        return
    for job_name, definition in jobs(workflow).items():
        if is_reusable_call(definition) or is_pull_request_only(definition):
            continue
        selected = conditional_runner(definition)
        if selected is not None:
            assert PULL_REQUEST_EVENT_GUARD in selected.guards, (
                f"{workflow_name}:{job_name} can run for a scheduled event "
                "and chooses a runner, so its condition must include the "
                f"{PULL_REQUEST_EVENT_GUARD} guard; found "
                f"{sorted(selected.guards)}"
            )
            assert selected.otherwise in GITHUB_HOSTED_LABELS, (
                f"{workflow_name}:{job_name} falls back to "
                f"{selected.otherwise}, which a scheduled run would bill"
            )
            continue
        labels = runner_labels(definition)
        assert not _self_hosted(labels), (
            f"{workflow_name}:{job_name} can run for a non-pull-request event, "
            f"so it must stay GitHub-hosted; found {labels}"
        )


@pytest.mark.parametrize("workflow_name", workflow_names())
def test_paid_pull_request_lanes_fall_back_for_forks(workflow_name: str) -> None:
    """A fork's pull request cannot obtain an Ubicloud runner at all.

    Without the fallback the lane does not run slowly, it does not run: the
    job sits unassignable and the pull request never reports. So every paid
    lane reachable from a ``pull_request`` event has to name the fork test
    and fall back to GitHub's pool, and the fallback label is checked too,
    because falling back to another paid label would fix nothing.
    """
    workflow = load_workflow(workflow_name)
    if "pull_request" not in triggers(workflow):
        return
    for job_name, definition in jobs(workflow).items():
        if is_reusable_call(definition) or not _paid_labels(definition):
            continue
        selected = conditional_runner(definition)
        assert selected is not None, (
            f"{workflow_name}:{job_name} takes a paid runner on a pull "
            "request with no fork fallback, so a fork's pull request would "
            "never be assigned a runner"
        )
        assert NON_FORK_GUARD in selected.guards, (
            f"{workflow_name}:{job_name} chooses a runner without the "
            f"{NON_FORK_GUARD} guard; found {sorted(selected.guards)}"
        )
        assert selected.otherwise in GITHUB_HOSTED_LABELS, (
            f"{workflow_name}:{job_name} falls back to {selected.otherwise}, "
            "which a fork cannot obtain either"
        )


@pytest.mark.parametrize("workflow_name", workflow_names())
def test_every_paid_job_bounds_its_runtime(workflow_name: str) -> None:
    """A runaway paid job must hit a timeout rather than burn the budget."""
    for job_name, definition in jobs(load_workflow(workflow_name)).items():
        if not _paid_labels(definition):
            continue
        assert isinstance(definition.get("timeout-minutes"), int), (
            f"{workflow_name}:{job_name} runs on a paid runner and must set "
            "timeout-minutes"
        )


def test_actionlint_registers_exactly_the_labels_in_use() -> None:
    """Keep the lint allow-list and the workflows in step.

    An unregistered label fails actionlint; a registered but unused label
    hides a runner assignment that has already been retired.
    """
    config = yaml.safe_load(ACTIONLINT_CONFIG.read_text(encoding="utf-8"))
    registered = set(config["self-hosted-runner"]["labels"])
    in_use = {
        label
        for workflow_name in workflow_names()
        for definition in jobs(load_workflow(workflow_name)).values()
        for label in _paid_labels(definition)
    }
    assert registered == in_use, (
        f"{ACTIONLINT_CONFIG} registers {sorted(registered)} but the "
        f"workflows use {sorted(in_use)}"
    )
