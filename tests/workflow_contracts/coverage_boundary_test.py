"""CV-005: only `main` writes persistent coverage state.

A pull-request lane measures coverage and compares it with the ratcheted
baseline `main` produced. It does not publish the report, call the CodeScene
action, run a `cs-coverage` command, or hold the credential either of those
needs. The check step that used to sit in `ci.yml` is why a CodeScene outage or
a token change could redden a pull request that had touched nothing to do with
coverage.

Every assertion here that reads this repository's files is paired with one that
drives the reader over a synthetic document. Over the repository's own
workflows a reader that answered nothing agrees with a correct one exactly, so
the rule would pass with every detector deleted.

Run via ``make test-workflow-contracts``.
"""

import collections.abc as cabc
import pathlib
import typing as typ

import pytest
import yaml
from coverage_boundary import (
    COVERAGE_COMMAND,
    CREDENTIAL_ENVIRONMENT_KEY,
    GENERATE_COVERAGE_ACTION,
    PUBLICATION_OPT_OUT_INPUT,
    PUBLICATION_OPT_OUT_VALUE,
    UPLOAD_COVERAGE_ACTION,
    action_of,
    coverage_surface_offenders,
    declares_trigger,
    declines_the_generated_report_archive,
    is_reachable_by_a_pull_request,
    local_workflows_called_by,
    publishes_the_coverage_report,
)
from workflow_support import WORKFLOW_DIR, job, steps, workflow_names

#: The lane that owns the upload, and is therefore the one exemption.
PUBLISHER_WORKFLOW: typ.Final[str] = "coverage-main.yml"

#: The pull-request lane that measures coverage.
MEASURING_LANE: typ.Final[tuple[str, str]] = ("ci.yml", "build-test")

#: The mode that publishes, and the action's default when `mode` is absent.
#: The other, `check`, reads a report and uploads nothing.
UPLOAD_MODE: typ.Final[str] = "upload"


def _raw(name: str) -> str:
    """Return one workflow file's raw text."""
    return (WORKFLOW_DIR / name).read_text(encoding="utf-8")


def _parsed(name: str) -> dict[str, typ.Any] | None:
    """Return one workflow's parsed document, or None when it is not a mapping."""
    document = yaml.safe_load(_raw(name))
    return document if isinstance(document, dict) else None


def _document(name: str) -> dict[str, typ.Any]:
    """Return one workflow's parsed document, or an empty one."""
    return _parsed(name) or {}


def _reachable_from(name: str) -> list[str]:
    """Return a workflow and every local reusable workflow it can reach.

    Parameters
    ----------
    name : str
        The entry workflow's file name.

    Returns
    -------
    list[str]
        The entry workflow first, then each local `jobs.<id>.uses` target it
        reaches, transitively, each once. A trigger-based reading alone would
        exempt a child: the child declares `workflow_call`, not
        `pull_request`, so the boundary would stop at the parent while the
        child held the credential. The walk tracks what it has seen, so a
        cycle, which GitHub rejects but a half-finished edit can produce,
        terminates here rather than recursing.
    """
    seen: list[str] = []
    pending = [name]
    while pending:
        current = pending.pop(0)
        if current in seen or not (WORKFLOW_DIR / current).is_file():
            continue
        seen.append(current)
        pending.extend(local_workflows_called_by(_document(current)))
    return seen


@pytest.fixture(name="synthetic")
def synthetic_fixture() -> cabc.Callable[[str], dict[str, typ.Any]]:
    """Return a factory for one-job pull-request workflows.

    A factory rather than a document, because every case here wants a
    different step body and a shared document would have to be mutated.
    """
    return _synthetic


def _synthetic(step_body: str) -> dict[str, typ.Any]:
    """Return a one-job pull-request workflow declaring the given steps."""
    text = (
        "on:\n  pull_request:\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
        f"    steps:\n{step_body}"
    )
    parsed = yaml.safe_load(text)
    assert isinstance(parsed, dict), "the synthetic workflow must parse to a mapping"
    return parsed


@pytest.mark.parametrize("name", workflow_names())
def test_no_pull_request_workflow_touches_the_publication_surface(name: str) -> None:
    """The boundary, over every workflow a pull request can reach.

    `coverage-main.yml` is the exemption and the only one. A lane that a pull
    request can start must not publish the report, invoke the CodeScene action,
    run its command, or carry its credential, because all four need a secret
    that a pull request's own run cannot be trusted with and none of them
    tells the author anything the ratchet does not.
    """
    document = _parsed(name)
    if document is None or name == PUBLISHER_WORKFLOW:
        return
    if not is_reachable_by_a_pull_request(document):
        return
    offenders = [
        offence
        for reached in _reachable_from(name)
        for offence in coverage_surface_offenders(reached, _document(reached), _raw(reached))
    ]
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


def test_the_publisher_keeps_the_upload_this_boundary_moved_to_it() -> None:
    """A rule that only forbids the upload elsewhere is satisfied by deleting it.

    The upload is not abolished; it is owned. If `coverage-main.yml` stopped
    making it, CodeScene would have no coverage at all and every contract above
    would still pass.
    """
    raw = (WORKFLOW_DIR / PUBLISHER_WORKFLOW).read_text(encoding="utf-8")
    document = yaml.safe_load(raw)
    assert isinstance(document, dict), f"{PUBLISHER_WORKFLOW} must parse"
    assert not is_reachable_by_a_pull_request(document), (
        f"{PUBLISHER_WORKFLOW} must not be reachable by a pull request, or "
        f"moving the upload into it moves nothing"
    )
    calls = [
        step
        for definition in (document.get("jobs") or {}).values()
        for step in steps(definition if isinstance(definition, dict) else {})
        if action_of(step) == UPLOAD_COVERAGE_ACTION
    ]
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
    assert UPLOAD_MODE in modes, (
        f"{PUBLISHER_WORKFLOW} must call the action in `{UPLOAD_MODE}` mode, "
        f"which is also its default when `mode` is absent; the calls pass "
        f"{modes}"
    )


@pytest.mark.parametrize(
    ("step_body", "expected"),
    [
        pytest.param(
            f"      - uses: {UPLOAD_COVERAGE_ACTION}@abc\n",
            "invokes the CodeScene coverage action",
            id="the-codescene-action",
        ),
        pytest.param(
            f"      - run: {COVERAGE_COMMAND} check --format lcov\n",
            f"runs a {COVERAGE_COMMAND} command",
            id="the-command-form",
        ),
        pytest.param(
            "      - uses: actions/upload-artifact@abc\n"
            "        with:\n          path: lcov.info\n",
            "publishes the coverage report as an artefact",
            id="an-artefact-upload-of-the-report",
        ),
        pytest.param(
            "      - uses: actions/upload-artifact@abc\n",
            "publishes the coverage report as an artefact",
            id="an-artefact-upload-of-the-workspace",
        ),
        pytest.param(
            f"      - uses: {GENERATE_COVERAGE_ACTION}@abc\n",
            "without declining its own archive",
            id="a-coverage-call-that-keeps-its-archive",
        ),
    ],
)
def test_each_forbidden_element_is_reported(
    synthetic: cabc.Callable[[str], dict[str, typ.Any]], step_body: str, expected: str
) -> None:
    """Each offence, added back one at a time.

    This repository's workflows carry none of these, so the contract above
    cannot separate a working detector from a deleted one. Each case here is
    the only evidence that one of them fires.
    """
    offenders = coverage_surface_offenders("scratch.yml", synthetic(step_body), "")
    assert any(expected in offence for offence in offenders), (
        f"a lane declaring this step must be reported as {expected!r}; "
        f"the reading gave {offenders}"
    )


@pytest.mark.parametrize(
    ("path", "publishes"),
    [
        pytest.param("lcov.info", True, id="the-report-by-name"),
        pytest.param(".", True, id="the-workspace-as-a-dot"),
        pytest.param("./", True, id="the-workspace-with-a-separator"),
        pytest.param("../workspace", True, id="a-path-reaching-upward"),
        pytest.param("**/*.info", True, id="a-glob-that-matches-the-report"),
        pytest.param("${{ env.COVERAGE_OUTPUT }}", True, id="an-unresolved-expression"),
        pytest.param("${{ github.workspace }}", True, id="the-workspace-by-expression"),
        pytest.param("", True, id="an-empty-path"),
        pytest.param("dist/\nlcov.info", True, id="one-safe-entry-and-one-not"),
        pytest.param("dist/", False, id="a-named-directory"),
        pytest.param("/tmp/bench.log", False, id="an-absolute-path-elsewhere"),
        pytest.param(
            "/tmp/bench-${{ matrix.bench }}.log", False, id="an-absolute-path-with-an-expression"
        ),
        pytest.param(
            "${{ matrix.x == 'y' && '/tmp/a.log' || '' }}",
            False,
            id="an-expression-choosing-between-absolute-paths",
        ),
        pytest.param(
            "**/proptest-regressions/**", False, id="a-glob-that-cannot-match-the-report"
        ),
    ],
)
def test_an_artefact_path_is_judged_by_what_it_can_carry(
    synthetic: cabc.Callable[[str], dict[str, typ.Any]], path: str, *, publishes: bool
) -> None:
    """Fail closed, but only where failing closed says something.

    A substring test for `lcov.info` clears the first nine of these while each
    uploads the report. Refusing every pattern and every expression, which was
    the first draft, condemned the last five, and four of those are real steps
    in this repository's own workflows: log uploads under `/tmp` and the
    proptest regression directory. So a pattern is tested against the places
    the report could sit, and an expression is cleared only when every
    path-shaped literal in it is absolute.

    `path` is newline-separated, and one unsafe entry condemns the step
    whatever the others name.
    """
    if "\n" in path:
        # A multi-line `path` has to go in as a block scalar. A quoted scalar
        # with `\n` inside puts a literal backslash-n in the value, and the
        # reader then sees one entry rather than two.
        rendered = "|\n" + "".join(f"            {line}\n" for line in path.split("\n"))
    else:
        rendered = f"{path!r}\n"
    step_body = (
        "      - uses: actions/upload-artifact@abc\n"
        "        with:\n"
        f"          path: {rendered}"
    )
    offenders = coverage_surface_offenders("scratch.yml", synthetic(step_body), "")
    reported = any("publishes the coverage report" in offence for offence in offenders)
    assert reported is publishes, (
        f"a path of {path!r} must read as publishes={publishes}; the reading "
        f"gave {offenders}"
    )


def test_a_local_reusable_workflow_is_reached_through_its_caller(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A child declares `workflow_call`, not `pull_request`.

    So a trigger-based reading stops at the parent, and a credential in the
    child is never asked about. This drives the walk over a tree built for it,
    because this repository has no local reusable workflow and the contract
    above would pass with the traversal deleted.

    The cycle case is here because a half-finished edit can produce one, and a
    walk that recursed on it would hang the suite rather than fail it.
    """
    monkeypatch.setattr("coverage_boundary_test.WORKFLOW_DIR", tmp_path)
    (tmp_path / "parent.yml").write_text(
        "on:\n  pull_request:\njobs:\n  call:\n    uses: ./.github/workflows/child.yml\n",
        encoding="utf-8",
    )
    (tmp_path / "child.yml").write_text(
        "on:\n  workflow_call:\njobs:\n  call:\n    uses: ./.github/workflows/parent.yml\n",
        encoding="utf-8",
    )
    assert _reachable_from("parent.yml") == ["parent.yml", "child.yml"], (
        "the walk must reach the child through its caller and stop on the cycle"
    )
    assert _reachable_from("child.yml") == ["child.yml", "parent.yml"], (
        "the walk is the same from either end of the cycle"
    )


def test_an_ordinary_lane_is_not_accused(
    synthetic: cabc.Callable[[str], dict[str, typ.Any]],
) -> None:
    """The other direction, so the detectors discriminate rather than accuse."""
    offenders = coverage_surface_offenders(
        "scratch.yml", synthetic("      - run: make test\n"), ""
    )
    assert not offenders, (
        f"an ordinary lane must produce no offence; it produced {offenders}"
    )


def test_the_compliant_coverage_call_passes_its_own_rule(
    synthetic: cabc.Callable[[str], dict[str, typ.Any]],
) -> None:
    """The shape the contract asks for must itself be accepted."""
    offenders = coverage_surface_offenders(
        "scratch.yml",
        synthetic(
            f"      - uses: {GENERATE_COVERAGE_ACTION}@abc\n"
            f"        with:\n"
            f"          with-ratchet: 'true'\n"
            f"          {PUBLICATION_OPT_OUT_INPUT}: '{PUBLICATION_OPT_OUT_VALUE}'\n"
        ),
        "",
    )
    assert not offenders, f"the prescribed shape must pass; it gave {offenders}"


@pytest.mark.parametrize("suffix", ["-legacy", "-v2"])
def test_a_lookalike_action_is_a_different_action(
    synthetic: cabc.Callable[[str], dict[str, typ.Any]], suffix: str
) -> None:
    """Splitting on the version separator is what tells them apart.

    A prefix match would report `upload-codescene-coverage-legacy` as the real
    action, which is an accusation rather than a finding.
    """
    offenders = coverage_surface_offenders(
        "scratch.yml",
        synthetic(f"      - uses: {UPLOAD_COVERAGE_ACTION}{suffix}@abc\n"),
        "",
    )
    assert not offenders, (
        f"{UPLOAD_COVERAGE_ACTION}{suffix} is a different action; the reading "
        f"gave {offenders}"
    )


def test_an_artefact_step_naming_another_path_is_not_an_offence(
    synthetic: cabc.Callable[[str], dict[str, typ.Any]],
) -> None:
    """Uploading something other than the report is allowed."""
    step_body = (
        "      - uses: actions/upload-artifact@abc\n        with:\n          path: dist/\n"
    )
    document = synthetic(step_body)
    assert not coverage_surface_offenders("scratch.yml", document, ""), (
        "an artefact step naming a path that is not the report must pass"
    )
    assert not publishes_the_coverage_report(steps(document["jobs"]["a"])[0])


def test_the_credential_is_found_in_text_the_parser_would_drop(
    synthetic: cabc.Callable[[str], dict[str, typ.Any]],
) -> None:
    """A comment is not a hiding place.

    The parsed scan alone would miss a credential named in a comment or in a
    shape the parser flattened, and a workflow that mentions it is a workflow
    somebody is about to wire it into.
    """
    clean = synthetic("      - run: make test\n")
    offenders = coverage_surface_offenders(
        "scratch.yml", clean, f"# see {CREDENTIAL_ENVIRONMENT_KEY} in main\n"
    )
    assert any("raw text" in offence for offence in offenders), (
        f"the credential must be reported from the raw text; got {offenders}"
    )


@pytest.mark.parametrize(
    ("declaration", "reachable"),
    [
        pytest.param("on: pull_request\n", True, id="a-scalar-trigger"),
        pytest.param("on: [push, pull_request]\n", True, id="a-sequence-trigger"),
        pytest.param("on:\n  pull_request:\n", True, id="a-mapping-trigger"),
        pytest.param("on:\n  pull_request_target:\n", True, id="the-privileged-variant"),
        pytest.param("on:\n  workflow_run:\n", True, id="a-resumed-run"),
        pytest.param("on:\n  push:\n", False, id="a-push-lane"),
        pytest.param("on:\n  schedule:\n", False, id="a-scheduled-lane"),
    ],
)
def test_the_trigger_reading_accepts_every_shape_on_takes(
    declaration: str, reachable: bool
) -> None:
    """`on:` has four spellings and PyYAML reads a bare `on` as True.

    A reading that knew only the mapping form would call `on: pull_request`
    unreachable and exempt it from the whole rule, which is the failure mode
    that would make this contract quietly cover less than it claims.
    """
    parsed = yaml.safe_load(f"{declaration}jobs:\n  a:\n    steps: []\n")
    assert is_reachable_by_a_pull_request(parsed) is reachable, (
        f"{declaration!r} must read as reachable={reachable}"
    )
    assert declares_trigger(parsed, "pull_request") is (
        "pull_request" in declaration and "pull_request_target" not in declaration
    )
