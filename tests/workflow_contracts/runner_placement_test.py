"""Contract-test where each job runs.

Chutoro pays for its pull-request property suite on an Ubicloud runner
because that job sits on the developer feedback path, where GitHub's queue
can stretch to hours. Nothing else qualifies. Weekly, nightly, mutation,
and administrative jobs are off that path, so a paid queue buys them
nothing while their long runtimes would dominate the bill; they stay on
GitHub-hosted runners. These tests fail on the pull request that moves a
job across that line, rather than after an invoice arrives.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import pytest
import yaml
from workflow_support import (
    ACTIONLINT_CONFIG,
    GITHUB_HOSTED_LABELS,
    is_pull_request_only,
    is_reusable_call,
    job,
    jobs,
    load_workflow,
    runner_labels,
    triggers,
    workflow_names,
)

#: The single job authorized to use a paid runner, and its label.
PAID_JOBS = {("property-tests.yml", "property-tests-pr"): "ubicloud-standard-8"}

#: Event names that never represent a pull-request feedback loop.
OFF_PATH_EVENTS = frozenset({"schedule", "push", "workflow_dispatch"})


def _self_hosted(labels: list[str]) -> bool:
    """Report whether any label falls outside GitHub's hosted pool."""
    return any(label not in GITHUB_HOSTED_LABELS for label in labels)


@pytest.mark.parametrize("workflow_name", workflow_names())
def test_only_the_pull_request_property_suite_uses_a_paid_runner(
    workflow_name: str,
) -> None:
    """Fail when any job other than the approved one leaves GitHub hosting."""
    for job_name, definition in jobs(load_workflow(workflow_name)).items():
        labels = runner_labels(definition)
        if not _self_hosted(labels):
            continue
        expected = PAID_JOBS.get((workflow_name, job_name))
        assert expected is not None, (
            f"{workflow_name}:{job_name} uses {labels}, but only "
            f"{sorted(name for _, name in PAID_JOBS)} may leave GitHub hosting"
        )
        assert labels == [expected], (
            f"{workflow_name}:{job_name} must use {expected}, not {labels}"
        )


def test_the_paid_property_job_keeps_its_current_label() -> None:
    """Preserve the pull-request suite's eight-core capacity."""
    for (workflow_name, job_name), label in PAID_JOBS.items():
        assert runner_labels(job(workflow_name, job_name)) == [label]


@pytest.mark.parametrize("workflow_name", workflow_names())
def test_jobs_reachable_off_the_feedback_path_stay_github_hosted(
    workflow_name: str,
) -> None:
    """Keep scheduled, push, and dispatched work on GitHub's runners."""
    workflow = load_workflow(workflow_name)
    if not OFF_PATH_EVENTS & set(triggers(workflow)):
        return
    for job_name, definition in jobs(workflow).items():
        if is_reusable_call(definition) or is_pull_request_only(definition):
            continue
        labels = runner_labels(definition)
        assert not _self_hosted(labels), (
            f"{workflow_name}:{job_name} can run for a non-pull-request event, "
            f"so it must stay GitHub-hosted; found {labels}"
        )


@pytest.mark.parametrize("workflow_name", workflow_names())
def test_every_paid_job_bounds_its_runtime(workflow_name: str) -> None:
    """A runaway paid job must hit a timeout rather than burn the budget."""
    for job_name, definition in jobs(load_workflow(workflow_name)).items():
        if not _self_hosted(runner_labels(definition)):
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
        for label in runner_labels(definition)
        if label not in GITHUB_HOSTED_LABELS
    }
    assert registered == in_use, (
        f"{ACTIONLINT_CONFIG} registers {sorted(registered)} but the "
        f"workflows use {sorted(in_use)}"
    )
