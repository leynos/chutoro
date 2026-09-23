"""CV-005: only `main` writes persistent coverage state.

A pull-request lane measures coverage and compares it with the ratcheted
baseline `main` produced. It does not publish the report, call the CodeScene
action, run a `cs-coverage` command, name the CodeScene host, or hold the
credential any of those needs. The check step that used to sit in `ci.yml` is
why a CodeScene outage or a token change could redden a pull request that had
touched nothing to do with coverage.

This module reads the repository's own workflows. Over those alone a reader
that answered nothing agrees with a correct one exactly, so every reader here
is also driven over synthetic documents in `coverage_boundary_readers_test.py`
and `coverage_publisher_test.py`.

Run via ``make test-workflow-contracts``.
"""

import typing as typ

import pytest
from coverage_boundary import (
    GENERATE_COVERAGE_ACTION,
    PUBLICATION_OPT_OUT_INPUT,
    PUBLICATION_OPT_OUT_VALUE,
    UPLOAD_COVERAGE_ACTION,
    action_of,
    declines_the_generated_report_archive,
    pull_request_offenders,
)
from coverage_publisher import (
    TRUNK_BRANCH,
    cancelling_scopes,
    push_branches,
    upload_condition_offences,
)
from workflow_reach import is_reachable_by_a_pull_request
from workflow_support import WORKFLOW_DIR, job, parse_workflow_text, steps, workflow_paths

#: The lane that owns the upload, and is therefore the one exemption.
PUBLISHER_WORKFLOW: typ.Final[str] = "coverage-main.yml"

#: The pull-request lane that measures coverage.
MEASURING_LANE: typ.Final[tuple[str, str]] = ("ci.yml", "build-test")

#: The mode that publishes, and the action's default when `mode` is absent.
#: The other, `check`, reads a report and uploads nothing.
UPLOAD_MODE: typ.Final[str] = "upload"

#: The checkout action, whose history depth the measuring lane leaves shallow.
CHECKOUT_ACTION: typ.Final[str] = "actions/checkout"


def _raw_texts() -> dict[str, str]:
    """Return every workflow's raw text, by file name."""
    return {path.name: path.read_text(encoding="utf-8") for path in workflow_paths()}


def _documents(raw_texts: dict[str, str]) -> dict[str, dict[str, typ.Any]]:
    """Return every workflow that parses to a mapping, by file name."""
    parsed = {name: parse_workflow_text(text) for name, text in raw_texts.items()}
    return {name: document for name, document in parsed.items() if isinstance(document, dict)}


def _publisher() -> dict[str, typ.Any]:
    """Return the publisher's parsed document."""
    document = parse_workflow_text((WORKFLOW_DIR / PUBLISHER_WORKFLOW).read_text(encoding="utf-8"))
    assert isinstance(document, dict), f"{PUBLISHER_WORKFLOW} must parse to a mapping"
    return document


def _publisher_uploads(document: dict[str, typ.Any]) -> list[dict[str, typ.Any]]:
    """Return the publisher's calls to the CodeScene action."""
    return [
        step
        for definition in (document.get("jobs") or {}).values()
        for step in steps(definition if isinstance(definition, dict) else {})
        if action_of(step) == UPLOAD_COVERAGE_ACTION
    ]


#: The binding the upload step must carry, and the input that must read it.
#: Asserted positively: a guard on `env.CS_ACCESS_TOKEN != ''` passes with the
#: binding deleted, and the upload then skips on every run without failing.
TOKEN_BINDING: typ.Final[str] = "${{ secrets.CS_ACCESS_TOKEN }}"
TOKEN_INPUT: typ.Final[str] = "${{ env.CS_ACCESS_TOKEN }}"


def _assert_the_token_is_bound(step: dict[str, typ.Any]) -> None:
    """Require the upload step to bind the credential and pass it on."""
    env = step.get("env") if isinstance(step.get("env"), dict) else {}
    with_ = step.get("with") if isinstance(step.get("with"), dict) else {}
    assert env.get("CS_ACCESS_TOKEN") == TOKEN_BINDING, (
        f"the upload step must bind CS_ACCESS_TOKEN to {TOKEN_BINDING}; it "
        f"binds {env.get('CS_ACCESS_TOKEN')!r}, so its guard would skip it"
    )
    assert with_.get("access-token") == TOKEN_INPUT, (
        f"the upload must pass {TOKEN_INPUT} as access-token; it passes "
        f"{with_.get('access-token')!r}"
    )


@pytest.mark.parametrize("name", [path.name for path in workflow_paths()])
def test_no_pull_request_workflow_touches_the_publication_surface(name: str) -> None:
    """The boundary, over every workflow a pull request can reach.

    `coverage-main.yml` is the exemption and the only one. A lane that a pull
    request can start, and every local workflow it calls, must not publish the
    report, invoke the CodeScene action, run its command, name its host,
    forward every secret, or carry its credential, because each needs a secret
    or a service that a pull request's own run cannot depend on, and none of
    them tells the author anything the ratchet does not.
    """
    raw_texts = _raw_texts()
    documents = _documents(raw_texts)
    document = documents.get(name)
    if document is None or name == PUBLISHER_WORKFLOW:
        return
    if not is_reachable_by_a_pull_request(document):
        return
    offenders = pull_request_offenders(name, documents, raw_texts)
    assert not offenders, (
        f"{name} can be reached by a pull request, so it and every local "
        f"workflow it calls must leave the coverage publication surface to "
        f"{PUBLISHER_WORKFLOW}: {offenders}"
    )


def test_the_measuring_lane_compares_against_the_ratchet_and_keeps_the_report() -> None:
    """Removing the upload must not leave the lane measuring nothing.

    Deleting a CodeScene step satisfies "no CodeScene on a pull request" and
    would also satisfy it with the coverage build deleted. The lane still has
    to generate a report and compare it with the baseline `main` wrote, and it
    has to decline the action's own archive, which is the only place the
    boundary is observable from this file: the archive is a step inside the
    action, and no scan of this workflow's steps can see it.
    """
    workflow_name, job_name = MEASURING_LANE
    found = [
        step
        for step in steps(job(workflow_name, job_name))
        if action_of(step) == GENERATE_COVERAGE_ACTION
    ]
    assert len(found) == 1, (
        f"{workflow_name}:{job_name} must generate coverage exactly once; it "
        f"has {len(found)} such steps"
    )
    with_ = found[0].get("with")
    assert isinstance(with_, dict), "the coverage step must declare inputs"
    assert with_.get("with-ratchet") == "true", (
        f"the pull-request lane must compare against the baseline "
        f"{PUBLISHER_WORKFLOW} wrote; it passes {with_.get('with-ratchet')!r}"
    )
    assert declines_the_generated_report_archive(found[0]), (
        f"the pull-request lane must pass {PUBLICATION_OPT_OUT_INPUT}: "
        f"{PUBLICATION_OPT_OUT_VALUE}, or the action publishes the report this "
        f"boundary exists to keep local"
    )


def test_the_measuring_lane_checks_out_shallow() -> None:
    """Full history was for the CodeScene check, and that check is gone.

    `cs-coverage check` diffed the pull request against its merge base, which
    is why the lane fetched every commit. The ratchet compares a percentage
    with a cached baseline and reads no history, so a deep checkout would be
    minutes of cloning that nothing consumes. Bringing it back needs a step
    that uses the merge base, and that step has to argue its way past this.
    """
    workflow_name, job_name = MEASURING_LANE
    checkouts = [
        step
        for step in steps(job(workflow_name, job_name))
        if action_of(step) == CHECKOUT_ACTION
    ]
    assert checkouts, f"{workflow_name}:{job_name} must check the repository out"
    depths = [
        step["with"].get("fetch-depth", 1) if isinstance(step.get("with"), dict) else 1
        for step in checkouts
    ]
    assert all(str(depth) == "1" for depth in depths), (
        f"{workflow_name}:{job_name} must keep the action's shallow default; "
        f"its checkouts ask for fetch-depth {depths}"
    )


def test_the_publisher_keeps_the_upload_this_boundary_moved_to_it() -> None:
    """A rule that only forbids the upload elsewhere is satisfied by deleting it.

    The upload is not abolished; it is owned. If `coverage-main.yml` stopped
    making it, CodeScene would have no coverage at all and every contract above
    would still pass.
    """
    document = _publisher()
    assert not is_reachable_by_a_pull_request(document), (
        f"{PUBLISHER_WORKFLOW} must not be reachable by a pull request, or "
        f"moving the upload into it moves nothing"
    )
    calls = _publisher_uploads(document)
    assert calls, (
        f"{PUBLISHER_WORKFLOW} must keep the CodeScene upload; without it "
        f"nothing publishes coverage and every rule above is vacuous"
    )
    # The action has more than one mode and only one of them publishes.
    # Asserting the reference alone would be satisfied by switching this call
    # to `check`, which uploads nothing while every rule above still passes.
    # An absent `mode` is the action's own default, `upload`, so it counts.
    modes = [
        str(step["with"].get("mode", UPLOAD_MODE))
        if isinstance(step.get("with"), dict)
        else UPLOAD_MODE
        for step in calls
    ]
    for step in calls:
        _assert_the_token_is_bound(step)
    assert UPLOAD_MODE in modes, (
        f"{PUBLISHER_WORKFLOW} must call the action in `{UPLOAD_MODE}` mode, "
        f"which is also its default when `mode` is absent; the calls pass "
        f"{modes}"
    )


def test_the_publisher_uploads_from_the_trunk_alone() -> None:
    """The dispatch trigger can be started from any branch.

    So the push filter is not enough on its own: the upload step's condition
    must carry the trunk guard as a conjunct, and the push filter must name
    the trunk and nothing else.
    """
    document = _publisher()
    assert push_branches(document) == [TRUNK_BRANCH], (
        f"{PUBLISHER_WORKFLOW} must publish on a push to {TRUNK_BRANCH} alone; "
        f"its push filter is {push_branches(document)}"
    )
    offences = [
        offence
        for step in _publisher_uploads(document)
        for offence in upload_condition_offences(step.get("if"))
    ]
    assert not offences, f"{PUBLISHER_WORKFLOW} can upload off the trunk: {offences}"


def test_the_publisher_queues_rather_than_cancels() -> None:
    """A cancelled publisher leaves the next pull request a stale baseline.

    It abandons the upload and the ratchet write together, and nothing reports
    it. The pull-request lanes may cancel superseded runs; this one may not.
    """
    scopes = cancelling_scopes(_publisher())
    assert not scopes, (
        f"{PUBLISHER_WORKFLOW} must queue its runs; cancel-in-progress is set "
        f"on {scopes}"
    )
