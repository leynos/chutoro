"""The publisher readers, driven over conditions and documents built for them.

The repository's own publisher is one condition and one trigger block, so a
reader that accepted everything would agree with it. These cases include the
exact spellings that defeated the substring readings elsewhere in the estate.

Run via ``make test-workflow-contracts``.
"""

import pytest
from coverage_publisher import (
    TRUNK_REF_GUARD,
    cancelling_scopes,
    push_branches,
    upload_condition_offences,
)
from workflow_support import parse_workflow_text

#: The non-empty credential guard the publisher carries beside the trunk one.
_TOKEN_GUARD = "env.CS_ACCESS_TOKEN != ''"


@pytest.mark.parametrize(
    "condition",
    [
        pytest.param(f"{_TOKEN_GUARD} && {TRUNK_REF_GUARD}", id="the-prescribed-pair"),
        pytest.param(f"${{{{ {TRUNK_REF_GUARD} && {_TOKEN_GUARD} }}}}", id="wrapped-and-reordered"),
        pytest.param(
            f"{_TOKEN_GUARD}  &&  github.ref  ==  'refs/heads/main'", id="spaced-differently"
        ),
        pytest.param(
            f"{TRUNK_REF_GUARD} && github.actor != 'a||b'", id="an-or-inside-a-quoted-literal"
        ),
    ],
)
def test_a_condition_requiring_the_trunk_is_accepted(condition: str) -> None:
    """The guard is a conjunct wherever it sits and however it is spaced."""
    assert upload_condition_offences(condition) == []


@pytest.mark.parametrize(
    "condition",
    [
        pytest.param(
            f"{_TOKEN_GUARD} && {TRUNK_REF_GUARD} || github.event_name == 'workflow_dispatch'",
            id="the-guard-made-optional-by-an-or",
        ),
        pytest.param(_TOKEN_GUARD, id="no-guard-at-all"),
        pytest.param(f"{_TOKEN_GUARD} && github.ref != 'refs/heads/main'", id="the-negation"),
        pytest.param(
            f"{_TOKEN_GUARD} && !({TRUNK_REF_GUARD})", id="the-guard-inside-a-negation"
        ),
        pytest.param(
            f"{_TOKEN_GUARD} && github.base_ref == 'refs/heads/main'", id="a-sibling-field"
        ),
        pytest.param(None, id="an-absent-condition"),
        pytest.param("", id="an-empty-condition"),
    ],
)
def test_a_condition_that_lets_a_branch_upload_is_refused(condition: str | None) -> None:
    """Each of these contains the guard's text or its neighbour, and none holds.

    The first is the sweep's own mutation: a substring test for the guard
    passes it while a dispatch from any branch uploads.
    """
    assert upload_condition_offences(condition), (
        f"{condition!r} does not confine the upload to the trunk"
    )


@pytest.mark.parametrize(
    ("declaration", "branches"),
    [
        pytest.param("on:\n  push:\n    branches: [main]\n", ["main"], id="the-trunk"),
        pytest.param("'on':\n  push:\n    branches: [main]\n", ["main"], id="a-quoted-key"),
        pytest.param(
            "on:\n  push:\n    branches: [main, 'release/*']\n",
            ["main", "release/*"],
            id="a-second-branch",
        ),
        pytest.param("on:\n  push:\n    tags: ['v*']\n", None, id="a-tag-workflow"),
        pytest.param("on: push\n", None, id="an-unfiltered-push"),
    ],
)
def test_the_push_filter_is_read_as_declared(declaration: str, branches: list[str] | None) -> None:
    """Read, not judged: the contract compares the answer with the trunk alone.

    A tag workflow and an unfiltered push both answer "no filter", which the
    contract then refuses, rather than being mistaken for a trunk publisher.
    """
    document = parse_workflow_text(f"{declaration}jobs: {{}}\n")
    assert isinstance(document, dict)
    assert push_branches(document) == branches


@pytest.mark.parametrize(
    ("concurrency", "cancels"),
    [
        pytest.param("  group: g\n  cancel-in-progress: true\n", True, id="a-literal-true"),
        pytest.param("  group: g\n  cancel-in-progress: 'true'\n", True, id="a-quoted-true"),
        pytest.param(
            "  group: g\n  cancel-in-progress: ${{ github.event_name == 'push' }}\n",
            True,
            id="an-expression",
        ),
        pytest.param("  group: g\n  cancel-in-progress: false\n", False, id="a-literal-false"),
        pytest.param("  group: g\n", False, id="no-setting"),
    ],
)
@pytest.mark.parametrize("scope", ["workflow", "job"])
def test_a_cancelling_publisher_is_refused_at_either_scope(
    concurrency: str, scope: str, *, cancels: bool
) -> None:
    """A job-level setting cancels as surely as a workflow-level one.

    An expression is refused even where it would evaluate false today, since
    it is one edit from cancelling a publisher halfway through its write.
    """
    if scope == "workflow":
        text = f"concurrency:\n{concurrency}jobs:\n  a:\n    steps: []\n"
    else:
        indented = "".join(f"    {line}\n" for line in concurrency.splitlines())
        text = f"jobs:\n  a:\n    concurrency:\n{indented}    steps: []\n"
    document = parse_workflow_text(text)
    assert isinstance(document, dict)
    assert bool(cancelling_scopes(document)) is cancels
