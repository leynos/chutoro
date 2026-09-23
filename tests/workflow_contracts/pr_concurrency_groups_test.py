"""Drive the concurrency-group renderer directly.

The repository's own groups are all acceptable, so the file-driven contract
alone cannot show that the rules refuse anything. These cases feed the
renderer the shapes the rules exist to refuse, and the one they accept.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import pytest
from pr_concurrency_groups import (
    FIRST_PUSH,
    UnmodelledGroupError,
    keeps_runs_together_and_apart,
    render_group,
)

#: The group every workflow here uses.
ESTATE_GROUP = (
    "${{ github.workflow }}-${{ github.event.pull_request.number || github.run_id }}"
)


@pytest.mark.parametrize(
    ("template", "verdict"),
    [
        (ESTATE_GROUP, "accept"),
        ("pr-${{ github.event.pull_request.number || github.ref }}", "refuse"),
        ("pr-${{ github.run_id || github.event.pull_request.number }}", "refuse"),
        ("kani-pr-${{ github.ref }}", "refuse"),
        ("${{ github.workflow }}-${{ github.run_id }}", "refuse"),
        ("${{ github.workflow }}-${{ github.sha }}", "refuse"),
        ("${{ github.workflow }}-${{ github.head_ref }}", "refuse"),
        ("pr-${{ github.event.pull_request.number || github.base_ref }}", "refuse"),
        ("one-group-for-everything", "refuse"),
    ],
    ids=[
        "estate",
        "ref-fallback",
        "run-id-first",
        "ref-only",
        "run-id-only",
        "sha",
        "head-ref",
        "base-ref-fallback",
        "constant",
    ],
)
def test_the_group_rules_accept_and_refuse_the_known_shapes(
    template: str, verdict: str
) -> None:
    """Only the estate group keeps one pull request together and the rest apart."""
    assert keeps_runs_together_and_apart(template) is (verdict == "accept")


@pytest.mark.parametrize(
    "template",
    [
        "${{ format('{0}', github.ref) }}",
        "${{ github.ref || format('{0}', github.sha) }}",
        "pr-${{ github.ref",
    ],
    ids=["function", "unmodelled-right-operand", "unclosed"],
)
def test_an_unmodelled_group_expression_is_refused(template: str) -> None:
    """A function call, a comparison or an unclosed opener fails loudly."""
    with pytest.raises(UnmodelledGroupError):
        render_group(template, FIRST_PUSH)
