"""Contract for the status checks that must block a merge to ``main``.

The live ``main-required-checks`` ruleset requires three pull-request
contexts: ``build-test``, ``verus-proofs`` and ``kani``. A required context is
matched against a job's reported name, which is the job identifier unless a
``name`` override renames it. For the required set to mean anything, each
context must therefore correspond to exactly one pull-request job whose
identifier is the context, and that job must be unable to disappear or report
a conclusion without doing its work.

This module holds the repository side of that contract: exactly one job per
context, no display-name override, no matrix, no job-level condition, no
reusable-workflow call, and no path filter on the owning workflow's
pull-request trigger. The live ruleset is an administrator action recorded in
the developer's guide; the static list here is compared against that guide, so
the documented enforcement and the configured enforcement cannot drift apart.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import re
import typing as typ
from pathlib import Path

import pytest
from workflow_support import ROOT, all_workflow_documents, triggers

GUIDE_PATH = ROOT / "docs" / "developers-guide.md"

#: The contexts the active ruleset must require. These are job identifiers,
#: not display names, because a required check matches the reported name.
REQUIRED_CONTEXTS = ("build-test", "verus-proofs", "kani")

#: The sentence in the developer's guide that states the same list. The
#: contract reads the backticked names from it, so the prose is the source the
#: test compares against rather than a second hand-maintained list.
DOCUMENTED_RE = re.compile(
    r"Required pull-request contexts on `main`: (?P<contexts>[^.\n]+)\."
)

#: Keys that would stop a job being a plain, always-present check. ``name``
#: changes the reported context; the rest change whether or how it reports.
FORBIDDEN_JOB_KEYS = ("if", "strategy", "uses")


class CheckError(ValueError):
    """Raised when a required context cannot be attributed to one job."""


def _pull_request_workflows() -> dict[str, dict[str, typ.Any]]:
    """Return each workflow document declaring a pull-request trigger."""
    return {
        name: document
        for name, document in all_workflow_documents().items()
        if "pull_request" in triggers(document)
    }


def _jobs(documents: dict[str, dict[str, typ.Any]]) -> dict[str, list[tuple[str, dict]]]:
    """Return each job identifier mapped to the workflows that declare it."""
    found: dict[str, list[tuple[str, dict[str, typ.Any]]]] = {}
    for name, document in documents.items():
        jobs = document.get("jobs")
        if not isinstance(jobs, dict):
            continue
        for job_id, job in jobs.items():
            if isinstance(job, dict):
                found.setdefault(job_id, []).append((name, job))
    return found


@pytest.fixture(scope="module")
def documents() -> dict[str, dict[str, typ.Any]]:
    """Return every workflow declaring a pull-request trigger."""
    return _pull_request_workflows()


@pytest.fixture(scope="module")
def jobs(documents: dict[str, dict[str, typ.Any]]) -> dict[str, list[tuple[str, dict]]]:
    """Return every pull-request job identifier and its workflows."""
    return _jobs(documents)


def test_no_job_renames_itself_to_a_required_context(
    documents: dict[str, dict[str, typ.Any]],
) -> None:
    """A display name may collide with a required context.

    GitHub matches a required check on the *reported* name, and a job's `name`
    may differ from its identifier. A second job called `kani` through `name:`
    would therefore report the same required context, and whichever run
    finished first would be the one the ruleset saw. Grouping by identifier
    alone cannot see that collision, so every pull-request job's effective
    reported name is checked here.
    """
    collisions: list[tuple[str, str, str]] = []
    for workflow, document in documents.items():
        jobs = document.get("jobs")
        if not isinstance(jobs, dict):
            continue
        for job_id, job in jobs.items():
            if not isinstance(job, dict):
                continue
            reported = job.get("name", job_id)
            if reported in REQUIRED_CONTEXTS and reported != job_id:
                collisions.append((workflow, job_id, str(reported)))
    assert not collisions, (
        "these jobs rename themselves to a required context, so the ruleset "
        f"could match the wrong run: {collisions!r}"
    )


@pytest.mark.parametrize("context", REQUIRED_CONTEXTS)
def test_each_required_context_is_exactly_one_job(
    context: str, jobs: dict[str, list[tuple[str, dict]]]
) -> None:
    """A context must name one pull-request job, not zero and not two.

    Zero would mean the ruleset waits for a check nothing reports, which
    blocks every merge; two would mean the ruleset matches whichever run
    reports first, so a passing duplicate could hide a failing one.
    """
    owners = jobs.get(context, [])
    assert len(owners) == 1, (
        f"required context {context!r} must be exactly one pull-request job; "
        f"found {[(workflow, job) for workflow, job in owners]!r}"
    )


@pytest.mark.parametrize("context", REQUIRED_CONTEXTS)
def test_a_required_job_has_no_display_name_override(
    context: str, jobs: dict[str, list[tuple[str, dict]]]
) -> None:
    """The reported name must be the job identifier the ruleset requires."""
    for workflow, job in jobs.get(context, []):
        assert "name" not in job, (
            f"{workflow}:{context} must not override its reported name; a "
            f"`name` turns the required context into a name the ruleset does "
            f"not require. Found {job.get('name')!r}"
        )


@pytest.mark.parametrize("context", REQUIRED_CONTEXTS)
def test_a_required_job_cannot_be_skipped_or_renamed_per_run(
    context: str, jobs: dict[str, list[tuple[str, dict]]]
) -> None:
    """No condition, matrix or reusable call may vary a required job.

    A job-level ``if`` can skip the check while the run reports success, a
    matrix multiplies the reported name (``build-test (ubuntu)``), and a
    reusable call is one indirection between the required identifier and the
    work. None of the three can belong to a required check.
    """
    for workflow, job in jobs.get(context, []):
        forbidden = [key for key in FORBIDDEN_JOB_KEYS if key in job]
        assert not forbidden, (
            f"{workflow}:{context} must be a plain job; found {forbidden!r} in "
            f"{sorted(job)!r}"
        )


@pytest.mark.parametrize("context", REQUIRED_CONTEXTS)
def test_the_owning_trigger_declares_no_path_filter(
    context: str,
    jobs: dict[str, list[tuple[str, dict]]],
    documents: dict[str, dict[str, typ.Any]],
) -> None:
    """A path filter would leave the required context absent.

    An absent required check blocks a merge, which is fail-closed, but it also
    means a change that alters what the job verifies without matching a listed
    path never receives a result. The job must therefore run on every pull
    request to the branch the ruleset protects.
    """
    for workflow, _ in jobs.get(context, []):
        pull_request = triggers(documents[workflow])["pull_request"]
        # A bare `pull_request:` key parses as None, which GitHub reads as an
        # unconfigured trigger: every pull request, with no filter. Treating
        # the value as a mapping before normalising it would raise TypeError
        # instead of accepting the unfiltered trigger.
        configured = pull_request if isinstance(pull_request, dict) else {}
        filters = [
            key for key in ("paths", "paths-ignore") if key in configured
        ]
        assert not filters, (
            f"{workflow}:{context} is a required check, so its pull_request "
            f"trigger must declare no path filter; found {filters!r}"
        )


def test_the_documented_required_contexts_match_the_contract() -> None:
    """The developer's guide must state the same required list.

    The live ruleset is changed by an administrator, so the guide is where the
    repository records what that change must be. Comparing the two makes a
    ruleset change without a documentation change fail here, and a
    documentation change without a ruleset change visible in review.
    """
    text = GUIDE_PATH.read_text(encoding="utf-8")
    found = DOCUMENTED_RE.search(text)
    assert found is not None, (
        "the developer's guide must state the required pull-request contexts "
        f"as {DOCUMENTED_RE.pattern!r}"
    )
    documented = tuple(
        re.findall(r"`([^`]+)`", found["contexts"])
    )
    assert documented == REQUIRED_CONTEXTS, (
        "the developer's guide and this contract must name the same required "
        f"contexts: guide {documented!r}, contract {REQUIRED_CONTEXTS!r}"
    )