"""The reach readers, driven over graphs, calls and triggers built for them.

This repository calls no local reusable workflow and declares none of the
rarer triggers, so over its own files a reader that answered nothing would
agree with a correct one. These cases are the evidence each reading fires.

Run via ``make test-workflow-contracts``.
"""

import typing as typ

import pytest
from workflow_reach import (
    declares_trigger,
    is_reachable_by_a_pull_request,
    local_workflow_target,
    reachable_workflows,
)
from workflow_support import parse_workflow_text


def _documents(texts: dict[str, str]) -> dict[str, dict[str, typ.Any]]:
    """Return a synthetic repository's parsed documents."""
    parsed = {name: parse_workflow_text(text) for name, text in texts.items()}
    return {name: doc for name, doc in parsed.items() if isinstance(doc, dict)}


def test_the_walk_stops_on_a_cycle_and_reads_each_workflow_once() -> None:
    """A half-finished edit can produce a cycle; a recursing walk would hang."""
    documents = _documents(
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
        pytest.param("$/.github/workflows/child.yml", "child.yml", id="the-self-repository-form"),
        pytest.param("leynos/shared-actions/.github/workflows/x.yml@v1", None, id="foreign"),
        pytest.param("./.github/actions/setup", None, id="an-action-directory"),
    ],
)
def test_a_local_call_is_recognised_by_its_shape(uses: str, target: str | None) -> None:
    """Local by shape: one leading `./` or `$/` removed, a workflow-directory path.

    GitHub accepts both prefixes for a call into this repository, so a reader
    of `./` alone would drop a `$/` child while a pull request still ran it.
    """
    assert local_workflow_target(uses) == target


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
        pytest.param("on:\n  pull_request_review:\n", True, id="a-review"),
        pytest.param("on: pull_request_review_comment\n", True, id="a-review-comment"),
        pytest.param("on: [merge_group]\n", True, id="the-merge-queue"),
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
        "pull_request" in declaration.replace("pull_request_", "")
    )
