"""Contract for the status checks that must block a merge to ``main``.

Three pull-request contexts are intended to gate ``main``: ``build-test``,
``verus-proofs`` and ``kani``. ``build-test`` and ``verus-proofs`` are
enforced by the live ``main-required-checks`` ruleset; adding ``kani`` is the
administrator action recorded in the developer's guide, and until it happens
that job reports on every pull request without blocking a merge.

A required context is matched against a job's reported name, which is the job
identifier unless a ``name`` override renames it. For the set to mean
anything, each context must therefore correspond to exactly one
pull-request job whose identifier is the context, and that job must be unable
to disappear or report a conclusion without doing its work.

This module is the policy check behind that: exactly one job per context, no
display-name override, no matrix, no job-level condition, no
reusable-workflow call, and no path filter on the owning workflow's
pull-request trigger. It compares that list with the one the developer's
guide states, so the documented policy and the local contract cannot drift
apart. The guide, not this test, records which of the three is already live.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import re
import typing as typ

import pytest
from github_branch_patterns import branch_pattern_regex, selected_by
from workflow_support import ROOT, all_workflow_documents

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
#: changes the reported context; ``strategy`` multiplies it; ``uses`` makes the
#: identifier an indirection; ``if`` skips it outright. ``needs`` is here
#: because a job whose dependency fails or is skipped is itself reported
#: skipped, and a skipped required check is neither a pass nor a run.
FORBIDDEN_JOB_KEYS = ("if", "strategy", "uses", "needs", "continue-on-error")


class CheckError(ValueError):
    """Raised when a required context cannot be attributed to one job."""


def _events(document: dict[str, typ.Any], name: str) -> dict[str, typ.Any]:
    """Return a workflow's events as a mapping, whatever form it was written in.

    GitHub accepts three spellings of ``on``: a mapping of event to its
    configuration, a bare scalar (``on: push``), and a sequence
    (``on: [pull_request, push]``). PyYAML turns the last two into a string
    and a list, and a mapping-only reader would raise on either -- during
    fixture setup, so every test in the module would error over a workflow
    that need not concern it.

    Returns
    -------
    dict[str, typing.Any]
        Each event mapped to its configuration, or None when it has none.

    Raises
    ------
    AssertionError
        When ``on`` is absent or is some other type, and the message names
        ``name`` so the offending file is findable.
    """
    found = document.get("on", document.get(True))
    if isinstance(found, str):
        return {found: None}
    if isinstance(found, list):
        return {event: None for event in found if isinstance(event, str)}
    if isinstance(found, dict):
        return found
    message = f"{name} must declare an on: mapping, sequence or scalar"
    raise AssertionError(message)


def _pull_request_workflows() -> dict[str, dict[str, typ.Any]]:
    """Return each workflow document declaring a pull-request trigger."""
    return {
        name: document
        for name, document in all_workflow_documents().items()
        if "pull_request" in _events(document, name)
    }


def _jobs(documents: dict[str, dict[str, typ.Any]]) -> dict[str, list[tuple[str, dict]]]:
    """Return each job identifier mapped to the workflows that declare it."""
    found: dict[str, list[tuple[str, dict[str, typ.Any]]]] = {}
    for name, document in documents.items():
        for job_id, job in _document_jobs(document):
            found.setdefault(job_id, []).append((name, job))
    return found


def _document_jobs(
    document: dict[str, typ.Any],
) -> typ.Iterator[tuple[str, dict[str, typ.Any]]]:
    """Yield each job mapping one workflow document declares."""
    jobs = document.get("jobs")
    if not isinstance(jobs, dict):
        return
    for job_id, job in jobs.items():
        if isinstance(job, dict):
            yield job_id, job


def _name_collisions(
    documents: dict[str, dict[str, typ.Any]],
) -> list[tuple[str, str, str]]:
    """Return each job reported under a required context's name."""
    return [
        (workflow, job_id, str(job["name"]))
        for workflow, document in documents.items()
        for job_id, job in _document_jobs(document)
        if _renames_to_required(job, job_id)
    ]


def _renames_to_required(job: dict[str, typ.Any], job_id: str) -> bool:
    """Return whether `job` reports a required context under another name."""
    reported = job.get("name", job_id)
    return reported in REQUIRED_CONTEXTS and reported != job_id


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
    collisions = _name_collisions(documents)
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
def test_the_owning_trigger_covers_every_pull_request(
    context: str,
    jobs: dict[str, list[tuple[str, dict]]],
    documents: dict[str, dict[str, typ.Any]],
) -> None:
    """The trigger must reach ``main`` for every pull request, unfiltered.

    A path filter would leave the required context absent on a change that
    alters what the job verifies without matching a listed path. A branch
    filter does the same for every pull request it excludes: GitHub never
    starts the workflow, so the check never reports. Either way the merge is
    blocked, and the job's verification is skipped without a result.
    """
    for workflow, _ in jobs.get(context, []):
        pull_request = _events(documents[workflow], workflow)["pull_request"]
        # A bare `pull_request:` key parses as None, which GitHub reads as an
        # unconfigured trigger: every pull request, with no filter. Treating
        # the value as a mapping before normalising it would raise TypeError
        # instead of accepting the unfiltered trigger.
        configured = pull_request if isinstance(pull_request, dict) else {}
        _require_unfiltered(workflow, context, configured)


#: The activity types a required job must run on. A pull request is checked
#: when it opens, when it gains a commit, and when it reopens; a `types` list
#: that omits one of these leaves the required check absent for that change.
REQUIRED_TYPES = ("opened", "synchronize", "reopened")


def _require_unfiltered(
    workflow: str, context: str, configured: dict[str, typ.Any]
) -> None:
    """Require a trigger that reaches main for every relevant activity."""
    _require_no_path_filter(workflow, context, configured)
    _require_main_selected(workflow, context, configured)
    _require_required_types(workflow, context, configured)


def _require_no_path_filter(
    workflow: str, context: str, configured: dict[str, typ.Any]
) -> None:
    """Require a trigger with no path filter, which would leave it absent."""
    filters = [key for key in ("paths", "paths-ignore") if key in configured]
    assert not filters, (
        f"{workflow}:{context} is a required check, so its pull_request "
        f"trigger must declare no path filter; found {filters!r}"
    )


def _require_main_selected(
    workflow: str, context: str, configured: dict[str, typ.Any]
) -> None:
    """Require the branch filters to select main and not exclude it."""
    branches = configured.get("branches")
    if branches is not None:
        assert selected_by(branches, "main"), (
            f"{workflow}:{context} is a required check, so its pull_request "
            f"trigger must select main; found branches {branches!r}"
        )
    ignored = configured.get("branches-ignore")
    if ignored is None:
        return
    excluded = [
        pattern
        for pattern in ignored
        if branch_pattern_regex(pattern).fullmatch("main")
    ]
    assert not excluded, (
        f"{workflow}:{context} is a required check, so its pull_request "
        f"trigger must not exclude main; found branches-ignore {excluded!r}"
    )


def _require_required_types(
    workflow: str, context: str, configured: dict[str, typ.Any]
) -> None:
    """Require an explicit activity list to include every relevant activity."""
    types = configured.get("types")
    if types is None:
        return
    missing = [kind for kind in REQUIRED_TYPES if kind not in types]
    assert not missing, (
        f"{workflow}:{context} is a required check, so its pull_request "
        f"trigger must run on {missing!r} as well; found types {types!r}"
    )


@pytest.mark.parametrize(
    ("patterns", "expected"),
    (
        (("main",), True),
        (("**",), True),
        (("m*",), True),
        (("ma?n",), False),
        (("[m]ain",), True),
        (("mai+n",), True),
        (("mai+",), False),
        (("develop",), False),
        (("main", "!main"), False),
        (("!main", "main"), True),
    ),
)
def test_branch_patterns_follow_github_filtering(
    patterns: tuple[str, ...], expected: bool
) -> None:
    """A branch filter is a glob with ordered exclusions, not literal names.

    ``m*`` matches ``main``, and a later positive pattern can undo a ``!``
    exclusion, so a literal membership test would accept configurations that
    exclude the protected branch.
    """
    assert selected_by(patterns, "main") is expected, (
        f"{patterns!r} selecting main must be {expected}"
    )


@pytest.mark.parametrize(
    ("configured", "rejected"),
    (
        ({"branches-ignore": ["m*"]}, True),
        ({"branches": ["main", "!main"]}, True),
        ({"branches": ["develop"]}, True),
        ({"branches": ["ma?n"]}, True),
        ({"branches-ignore": ["[m]ain"]}, True),
        ({"types": ["closed"]}, True),
        ({"branches": ["**"]}, False),
        ({"types": ["opened", "synchronize", "reopened"]}, False),
        ({}, False),
    ),
)
def test_a_trigger_that_misses_an_activity_is_rejected(
    configured: dict[str, typ.Any], rejected: bool
) -> None:
    """Every spelling that leaves main unchecked must fail the trigger check."""
    if not rejected:
        _require_unfiltered("fixture.yml", "build-test", configured)
        return
    with pytest.raises(AssertionError):
        _require_unfiltered("fixture.yml", "build-test", configured)


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