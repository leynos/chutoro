"""Contracts cancelling superseded pull-request runs.

Every push to a pull request starts a fresh run of each gate, and the run
already in flight is answering a question about a commit nobody will merge.
Left alone it holds a paid runner until it finishes, so the branch pays twice
for one answer. A concurrency group keyed on the pull request makes the newer
run cancel the older one.

Cancellation has to stay conditioned on the event. A literal
``cancel-in-progress: true`` would also cancel a push to `main`, a schedule,
and a dispatch, none of which has a successor waiting: the run that writes the
warm cache on `main` would be killed by the next merge. The condition is
therefore part of the contract, and
`test_cancellation_is_conditioned_on_the_event` fails on the literal.

The group keeps two pushes to one pull request together and every other run
apart. When there is no pull request its fallback is ``github.run_id``, so two
pushes to `main` or two dispatches never share a group: a shared ref group
lets a third run replace a still-pending second one, and that commit never
gets CI (estate rule "PR-lane concurrency fallback"). The run identifier is
allowed only in that fallback position. Rather than search the group's text,
the contract renders it for a set of run contexts and compares the results.

Only `pull_request` is in scope. A `pull_request_target` workflow runs against
the base repository to carry a token, and the one here merges Dependabot pull
requests; cancelling a merge mid-flight is a hazard with no minutes to win.

Workflows are read at test setup through `workflow_support`, whose parser
refuses duplicate mapping keys.
PyYAML otherwise keeps the last of two `concurrency:` blocks and says nothing,
so a correct block could sit above a wrong one and pass.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pr_concurrency_groups import (
    FIRST_PUSH,
    MUST_PART,
    MUST_SHARE,
    fallback_problems,
    render_group,
    shared_groups,
)
from pr_concurrency_triggers import pull_request_workflows, trigger_names
from strict_yaml import parse_workflow_text
from workflow_support import WorkflowReadError, all_workflow_documents

Document = dict[object, object]

#: Parsed workflows keyed by file name, as `all_workflow_documents` returns them.
Workflows = dict[str, Document]

#: The exact `cancel-in-progress` expression every pull-request workflow
#: carries. Comparing against one string rather than searching for a substring
#: is what makes the literal `true` mutation fail: `true` is a YAML boolean and
#: never equals this.
CANCEL_EXPRESSION = "${{ github.event_name == 'pull_request' }}"


#: Workflows known to start on `pull_request`. Discovery below is dynamic so a
#: new workflow is covered the day it lands, but a dynamic list that silently
#: empties turns every contract into a vacuous pass. This names the floor
#: discovery must still reach.
KNOWN_PULL_REQUEST_WORKFLOWS: frozenset[str] = frozenset(
    {
        "benchmark-regressions.yml",
        "ci.yml",
        "kani-pr.yml",
        "property-tests.yml",
    }
)


def _parse(text: str, name: str) -> Document:
    """Parse workflow text strictly, refusing anything but a mapping."""
    document = parse_workflow_text(text)
    if not isinstance(document, dict):
        message = f"{name} must parse as a mapping"
        raise AssertionError(message)
    return document


def _concurrency(document: Document) -> Document:
    """Return a workflow's top-level concurrency mapping.

    The shorthand string form cannot carry `cancel-in-progress` at all, so it
    reads as empty here, as does a workflow declaring no concurrency.
    """
    declared = document.get("concurrency")
    return declared if isinstance(declared, dict) else {}


def _group(document: Document) -> str:
    """Return a workflow's concurrency group as written, or an empty string."""
    return str(_concurrency(document).get("group", ""))


def _workflow_name(name: str, document: Document) -> str:
    """Return the name Actions gives a workflow: its `name:`, else its path."""
    declared = document.get("name")
    return declared if isinstance(declared, str) else f".github/workflows/{name}"


@pytest.fixture
def workflows() -> Workflows:
    """Read this repository's workflows at setup rather than at import.

    A file that cannot be read or parsed, including one declaring a mapping
    key twice, fails the test that asked for it with the reason, instead of
    breaking collection of the whole module.
    """
    try:
        return all_workflow_documents()
    except (WorkflowReadError, yaml.YAMLError) as error:
        pytest.fail(f"cannot read the workflows under .github/workflows: {error}")


@pytest.fixture
def pr_workflows(workflows: Workflows) -> Workflows:
    """Return the repository's workflows a pull request can start."""
    return pull_request_workflows(workflows)


def test_discovery_still_finds_the_known_pull_request_workflows(
    pr_workflows: Workflows,
) -> None:
    """Discovery reaches its floor, so the contracts below are not empty.

    If the read broke, or the `on:` key changed shape, the set would empty
    and each contract below would pass having asserted nothing.
    """
    missing = sorted(KNOWN_PULL_REQUEST_WORKFLOWS - set(pr_workflows))
    assert not missing, (
        f"these workflows start on pull_request but discovery missed them: "
        f"{', '.join(missing)}; the contracts below would pass without "
        "asserting anything about them"
    )


def test_every_workflow_declares_a_trigger_set_this_reader_models(
    workflows: Workflows,
) -> None:
    """No workflow's `on:` defeats the reader that decides what is in scope.

    A workflow the reader cannot model is dropped from discovery, and every
    contract below would pass while saying nothing about it.
    """
    unreadable = sorted(
        name for name, document in workflows.items() if trigger_names(document) is None
    )
    assert not unreadable, (
        f"these workflows declare an `on:` this reader does not model: "
        f"{', '.join(unreadable)}; each is dropped from discovery, so every "
        "contract below would pass without asserting anything about it"
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("on: pull_request\n", frozenset({"pull_request"})),
        ("on: [push, pull_request]\n", frozenset({"push", "pull_request"})),
        ("on:\n  push:\n  pull_request:\n", frozenset({"push", "pull_request"})),
        ("'on': pull_request\n", frozenset({"pull_request"})),
        ("'on': push\non: pull_request\n", None),
        ("on: 3\n", None),
        ("on: [push, 3]\n", None),
        ("name: no trigger\n", frozenset()),
    ],
    ids=[
        "bare",
        "list",
        "mapping",
        "quoted",
        "both-keys",
        "number",
        "mixed-list",
        "absent",
    ],
)
def test_the_trigger_reader_models_every_shape_github_accepts(
    text: str, expected: frozenset[str] | None
) -> None:
    """The reader drives discovery, so each `on:` shape is read, or refused.

    The repository's own workflows use the mapping form only, so a list or
    bare-name reading that broke would leave every file-driven contract green.
    This drives the reader directly with each shape.
    """
    actual = trigger_names(_parse(text, "shape.yml"))
    assert actual == expected, (
        f"the trigger reader read {text!r} as {actual!r}, expected {expected!r}"
    )


def test_a_duplicated_key_is_refused_rather_than_resolved() -> None:
    """The loader refuses a second `concurrency:` block instead of keeping it.

    PyYAML's own loader keeps the last value, so a correct block followed by
    a wrong one would reach these contracts as the wrong one only, and a
    wrong one followed by a correct one would pass. This drives the loader
    the contracts use, not the files it guards.
    """
    text = "on:\n  pull_request:\nconcurrency:\n  group: a\nconcurrency:\n  group: b\n"
    with pytest.raises(
        yaml.constructor.ConstructorError, match="found duplicate key 'concurrency'"
    ):
        _parse(text, "duplicated.yml")


def test_a_workflow_directory_is_read_through_the_shared_loader(
    tmp_path: Path,
) -> None:
    """Discovery reads any directory it is given, not only this repository's.

    The contracts below take their workflows from `all_workflow_documents`, so the
    scope decision is a pure function of the documents. This drives it over a
    directory holding one pull-request workflow, one push workflow and one
    `.yaml` file, and checks that only the two pull-request ones are in scope.
    """
    (tmp_path / "ci.yml").write_text("on: pull_request\n", encoding="utf-8")
    (tmp_path / "main.yml").write_text("on: push\n", encoding="utf-8")
    (tmp_path / "other.yaml").write_text("on: [pull_request]\n", encoding="utf-8")
    in_scope = sorted(pull_request_workflows(all_workflow_documents(tmp_path)))
    assert in_scope == ["ci.yml", "other.yaml"], (
        f"discovery over {tmp_path} found {in_scope}"
    )


def test_every_pull_request_workflow_declares_a_concurrency_group(
    pr_workflows: Workflows,
) -> None:
    """A workflow a pull request starts declares a concurrency group.

    Without one, every push to the branch leaves its predecessor running to
    completion on a paid runner.
    """
    missing = sorted(
        name
        for name, document in pr_workflows.items()
        if not _group(document).strip()
    )
    assert not missing, (
        f"{', '.join(missing)} start on pull_request and must declare "
        "concurrency.group; without it a superseded run holds a runner until "
        "it finishes"
    )


def _split_pairs(group: str) -> list[tuple[str, str]]:
    """Return the `MUST_SHARE` pairs a group renders differently."""
    return [
        (render_group(group, first), render_group(group, second))
        for first, second in MUST_SHARE
        if render_group(group, first) != render_group(group, second)
    ]


def test_two_pushes_to_one_pull_request_share_a_group(
    pr_workflows: Workflows,
) -> None:
    """The newer push lands in its predecessor's group, so it can cancel it.

    A group built from the run identifier alone, the SHA, or the run
    identifier ahead of the pull-request number renders differently for each
    push and cancels nothing.
    """
    split = {
        name: pairs
        for name, pairs in (
            (name, _split_pairs(_group(document)))
            for name, document in pr_workflows.items()
        )
        if pairs
    }
    assert not split, (
        f"these groups render differently for two pushes to one pull request: "
        f"{split}"
    )


def test_no_other_two_runs_share_a_group(pr_workflows: Workflows) -> None:
    """No run cancels another pull request's, and none replaces a pending run.

    The runs are a pull request, a fork's pull request from a branch of the
    same name, two pushes to `main`, and two dispatches of another branch. A
    ``github.head_ref`` group collides on the forks; a ``github.ref`` or
    ``github.base_ref`` fallback collides on the trunk pushes, where a third
    push would replace the pending second.
    """
    rendered = {
        name: [render_group(_group(document), run) for run in MUST_PART]
        for name, document in pr_workflows.items()
    }
    colliding = {
        name: groups
        for name, groups in rendered.items()
        if len(set(groups)) != len(MUST_PART)
    }
    assert not colliding, (
        f"these groups collide across runs that must stay apart: {colliding}"
    )


def test_the_run_identifier_is_only_the_fallback(pr_workflows: Workflows) -> None:
    """``github.run_id`` appears once, behind the pull-request number.

    The estate rule allows a run-unique value only there. Anywhere else it
    either splits one pull request's pushes or hides a ref-keyed fallback.
    """
    problems = {
        name: found
        for name, found in (
            (name, fallback_problems(_group(document)))
            for name, document in pr_workflows.items()
        )
        if found
    }
    assert not problems, f"these groups break the fallback rule: {problems}"


def test_no_two_workflows_share_a_group_for_one_pull_request(
    pr_workflows: Workflows,
) -> None:
    """Two workflows on one pull request never cancel each other.

    Each workflow is rendered under its own name. A group that leaves the
    workflow out, such as ``pr-${{ github.event.pull_request.number }}``,
    would put the CI run and every other pull-request workflow in one group,
    and whichever started last would cancel the rest.
    """
    rendered = {
        name: render_group(
            _group(document),
            {**FIRST_PUSH, "github.workflow": _workflow_name(name, document)},
        )
        for name, document in pr_workflows.items()
    }
    shared = shared_groups(rendered)
    assert not shared, (
        f"these workflows share a concurrency group for one pull request, "
        f"compared without case as GitHub does: {shared}"
    )


def test_cancellation_is_conditioned_on_the_event(pr_workflows: Workflows) -> None:
    """Cancellation applies to pull requests only, not to pushes or schedules.

    A literal `true` reads as a stricter setting and is a regression: it would
    cancel a run on `main`, a schedule, or a dispatch, none of which has a
    successor that repeats its work.
    """
    wrong = {
        name: declared
        for name, declared in (
            (name, _concurrency(document).get("cancel-in-progress"))
            for name, document in pr_workflows.items()
        )
        if declared != CANCEL_EXPRESSION
    }
    assert not wrong, (
        f"these workflows must set cancel-in-progress to {CANCEL_EXPRESSION!r}: "
        f"{wrong}; a missing value leaves superseded runs in flight and a "
        "literal true also cancels pushes to main, schedules, and dispatches"
    )
