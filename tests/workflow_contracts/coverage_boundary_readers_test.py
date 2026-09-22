"""The CV-005 boundary readers, driven over documents built to test them.

This repository's workflows carry none of the offences the boundary forbids and
call no local reusable workflow, so the repository-level contract cannot tell
a working detector from a deleted one. Each case here is the only evidence
that one of them fires, and the accepted cases are what stop the detectors
from accusing honest steps.

Run via ``make test-workflow-contracts``.
"""

import collections.abc as cabc
import typing as typ

import pytest
import yaml
from coverage_boundary import (
    CODESCENE_HOST,
    COVERAGE_COMMAND,
    CREDENTIAL_ENVIRONMENT_KEY,
    GENERATE_COVERAGE_ACTION,
    PUBLICATION_OPT_OUT_INPUT,
    PUBLICATION_OPT_OUT_VALUE,
    UPLOAD_COVERAGE_ACTION,
    coverage_surface_offenders,
    publishes_the_coverage_report,
    pull_request_offenders,
)
from workflow_reach import (
    declares_trigger,
    is_reachable_by_a_pull_request,
    local_workflow_target,
    reachable_workflows,
)
from workflow_support import parse_workflow_text, steps

#: What a pull-request job would write to forward the credential by name.
_FORWARDED_TOKEN: typ.Final[str] = "${{ secrets.CS_ACCESS_TOKEN }}"


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
    parsed = parse_workflow_text(text)
    assert isinstance(parsed, dict), "the synthetic workflow must parse to a mapping"
    return parsed


def _tree(texts: dict[str, str]) -> tuple[dict[str, dict[str, typ.Any]], dict[str, str]]:
    """Return a synthetic repository's parsed documents and raw texts."""
    documents = {name: parse_workflow_text(text) for name, text in texts.items()}
    return {name: doc for name, doc in documents.items() if isinstance(doc, dict)}, texts


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
        pytest.param(
            "      - run: curl -fsS https://api.CodeScene.io/v2/projects/1\n",
            f"references {CODESCENE_HOST}",
            id="the-host-by-another-route",
        ),
        pytest.param(
            f"      - env:\n          T: {_FORWARDED_TOKEN}\n        run: make test\n",
            f"parsed value references {CREDENTIAL_ENVIRONMENT_KEY}",
            id="the-credential-in-a-parsed-value",
        ),
        pytest.param(
            "      - run: echo ${{ secrets.cs_access_token }}\n",
            f"parsed value references {CREDENTIAL_ENVIRONMENT_KEY}",
            id="the-credential-in-another-case",
        ),
    ],
)
def test_each_forbidden_element_is_reported(
    synthetic: cabc.Callable[[str], dict[str, typ.Any]], step_body: str, expected: str
) -> None:
    """Each offence, added back one at a time, with no raw text to lean on.

    The raw text is passed empty, so the credential cases are answered by the
    parsed reading alone; the raw reading has its own case below.
    """
    offenders = coverage_surface_offenders("scratch.yml", synthetic(step_body), "")
    assert any(expected in offence for offence in offenders), (
        f"a lane declaring this step must be reported as {expected!r}; "
        f"the reading gave {offenders}"
    )


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("false", id="an-unquoted-boolean"),
        pytest.param("'False'", id="a-capitalised-string"),
        pytest.param("'0'", id="a-zero"),
        pytest.param("''", id="an-empty-string"),
        pytest.param("'no'", id="a-yaml-1-1-no"),
        pytest.param("${{ false }}", id="an-expression"),
    ],
)
def test_only_the_literal_opt_out_declines_the_archive(
    synthetic: cabc.Callable[[str], dict[str, typ.Any]], value: str
) -> None:
    """The opt-out is the string the action compares against, and nothing else.

    Each of these is falsy to somebody. The contract asks for the one spelling
    whose meaning does not depend on which reader coerces it, so a boolean or
    an expression that happens to evaluate false stays an offence and is
    written the prescribed way instead.
    """
    offenders = coverage_surface_offenders(
        "scratch.yml",
        synthetic(
            f"      - uses: {GENERATE_COVERAGE_ACTION}@abc\n"
            f"        with:\n          {PUBLICATION_OPT_OUT_INPUT}: {value}\n"
        ),
        "",
    )
    assert any("without declining its own archive" in offence for offence in offenders), (
        f"{PUBLICATION_OPT_OUT_INPUT}: {value} is not the literal "
        f"{PUBLICATION_OPT_OUT_VALUE!r}; the reading gave {offenders}"
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
    uploads the report. Refusing every pattern and every expression condemned
    the last five, and four of those are real steps in this repository's own
    workflows: log uploads under `/tmp` and the proptest regression directory.
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


def test_a_reusable_child_is_judged_with_its_caller() -> None:
    """The probe: a child that declares `workflow_call` and nothing else.

    A pull-request job calls it with `secrets: inherit`, and it curls the
    CodeScene API with the inherited credential. Its triggers make it look
    unreachable, and its caller names neither the host nor the credential, so
    a reading of the entry alone passes the pair. Every offence has to come
    back through the closure.
    """
    documents, raw_texts = _tree(
        {
            "parent.yml": (
                "on:\n  pull_request:\njobs:\n  call:\n"
                "    uses: ./.github/workflows/child.yml\n    secrets: inherit\n"
            ),
            "child.yml": (
                "on:\n  workflow_call:\njobs:\n  probe:\n    runs-on: ubuntu-latest\n"
                "    steps:\n      - run: |\n          curl -H \"Authorization: Bearer "
                f"{_FORWARDED_TOKEN}\" https://api.codescene.io/v2/projects\n"
            ),
        }
    )
    offenders = pull_request_offenders("parent.yml", documents, raw_texts)
    expected = [
        "parent.yml:call forwards every secret",
        f"child.yml: raw text references {CREDENTIAL_ENVIRONMENT_KEY}",
        f"child.yml: parsed value references {CODESCENE_HOST}",
    ]
    for fragment in expected:
        assert any(fragment in offence for offence in offenders), (
            f"the closure must report {fragment!r}; it gave {offenders}"
        )


def test_a_local_call_to_nothing_is_reported_rather_than_skipped() -> None:
    """A target that is not there is a workflow no rule was asked about."""
    documents, raw_texts = _tree(
        {"parent.yml": "on: pull_request\njobs:\n  a:\n    uses: .github/workflows/gone.yml\n"}
    )
    offenders = pull_request_offenders("parent.yml", documents, raw_texts)
    assert any("gone.yml" in offence for offence in offenders), (
        f"a call to a missing local workflow must be reported; it gave {offenders}"
    )


def test_the_walk_stops_on_a_cycle_and_reads_each_workflow_once() -> None:
    """A half-finished edit can produce a cycle; a recursing walk would hang."""
    documents, _ = _tree(
        {
            "parent.yml": "on: pull_request\njobs:\n  a:\n    uses: ./.github/workflows/child.yml\n",
            "child.yml": "on: workflow_call\njobs:\n  b:\n    uses: ./.github/workflows/parent.yml\n",
        }
    )
    assert reachable_workflows("parent.yml", documents).reached == ["parent.yml", "child.yml"]
    assert reachable_workflows("child.yml", documents).reached == ["child.yml", "parent.yml"]


@pytest.mark.parametrize(
    ("uses", "target"),
    [
        pytest.param("./.github/workflows/child.yml", "child.yml", id="with-a-leading-dot"),
        pytest.param(".github/workflows/child.yml", "child.yml", id="without-one"),
        pytest.param("leynos/shared-actions/.github/workflows/x.yml@v1", None, id="foreign"),
        pytest.param("./.github/actions/setup", None, id="an-action-directory"),
    ],
)
def test_a_local_call_is_recognised_by_its_shape(uses: str, target: str | None) -> None:
    """Local by shape, not by an enumerated list of prefixes.

    A prefix list is a list somebody has to remember to extend; a path under
    the workflow directory is local however it is introduced.
    """
    assert local_workflow_target(uses) == target


def test_an_ordinary_lane_is_not_accused(
    synthetic: cabc.Callable[[str], dict[str, typ.Any]],
) -> None:
    """The other direction, so the detectors discriminate rather than accuse."""
    assert not coverage_surface_offenders("scratch.yml", synthetic("      - run: make test\n"), "")


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
    """A prefix match would report `upload-codescene-coverage-legacy` as real."""
    offenders = coverage_surface_offenders(
        "scratch.yml", synthetic(f"      - uses: {UPLOAD_COVERAGE_ACTION}{suffix}@abc\n"), ""
    )
    assert not offenders, f"{UPLOAD_COVERAGE_ACTION}{suffix} is a different action: {offenders}"


def test_an_artefact_step_naming_another_path_is_not_an_offence(
    synthetic: cabc.Callable[[str], dict[str, typ.Any]],
) -> None:
    """Uploading something other than the report is allowed."""
    document = synthetic(
        "      - uses: actions/upload-artifact@abc\n        with:\n          path: dist/\n"
    )
    assert not coverage_surface_offenders("scratch.yml", document, "")
    assert not publishes_the_coverage_report(steps(document["jobs"]["a"])[0])


def test_the_credential_is_found_in_text_the_parser_would_drop(
    synthetic: cabc.Callable[[str], dict[str, typ.Any]],
) -> None:
    """A comment is not a hiding place, and the parsed reading cannot see one."""
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
        pytest.param("'on': pull_request\n", True, id="a-quoted-string-key"),
        pytest.param("'on': push\non: pull_request\n", True, id="both-keys-at-once"),
        pytest.param("on:\n  pull_request_target:\n", True, id="the-privileged-variant"),
        pytest.param("on:\n  workflow_run:\n", True, id="a-resumed-run"),
        pytest.param("on:\n  push:\n", False, id="a-push-lane"),
        pytest.param("on:\n  schedule:\n", False, id="a-scheduled-lane"),
    ],
)
def test_the_trigger_reading_accepts_every_shape_on_takes(
    declaration: str, *, reachable: bool
) -> None:
    """`on:` has three shapes and two keys, and a resolving loader reads both.

    A bare `on` resolves to the boolean True and a quoted one stays a string,
    so a reader narrowed to either key exempts the other's workflows from the
    whole rule. The cases are parsed by the resolving loader the contract
    uses, which is the only way the two keys can differ.
    """
    parsed = parse_workflow_text(f"{declaration}jobs:\n  a:\n    steps: []\n")
    assert isinstance(parsed, dict)
    assert is_reachable_by_a_pull_request(parsed) is reachable, (
        f"{declaration!r} must read as reachable={reachable}"
    )
    assert declares_trigger(parsed, "pull_request") is (
        "pull_request" in declaration and "pull_request_target" not in declaration
    )


def test_a_key_declared_twice_is_refused_rather_than_resolved() -> None:
    """PyYAML keeps the second `runs-on` and says nothing.

    A lane could then carry one label where a reviewer reads it and another
    where the runner does. The loader every reader here goes through refuses
    the document instead.
    """
    with pytest.raises(yaml.constructor.ConstructorError, match="duplicate key 'runs-on'"):
        parse_workflow_text(
            "on: pull_request\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
            "    runs-on: ubicloud-standard-2\n    steps: []\n"
        )
