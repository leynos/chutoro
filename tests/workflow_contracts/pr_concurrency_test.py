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

Workflows are read through `strict_yaml`, which refuses duplicate mapping keys.
PyYAML otherwise keeps the last of two `concurrency:` blocks and says nothing,
so a correct block could sit above a wrong one and pass.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pr_concurrency_groups import (
    ESTATE_FALLBACK,
    FIRST_PUSH,
    MUST_PART,
    MUST_SHARE,
    expressions,
    render_group,
)
from pr_concurrency_triggers import trigger_names
from strict_yaml import parse_workflow_text

#: The repository's workflow directory. The module sits two levels below the
#: repository root, in `tests/workflow_contracts/`.
WORKFLOW_DIR = Path(__file__).resolve().parents[2] / ".github" / "workflows"

#: Extensions GitHub accepts for a workflow file. Scanning only `.yml` would
#: silently exempt a `.yaml` workflow from every contract here.
WORKFLOW_SUFFIXES: tuple[str, ...] = (".yml", ".yaml")

#: The exact `cancel-in-progress` expression every pull-request workflow
#: carries. Comparing against one string rather than searching for a substring
#: is what makes the literal `true` mutation fail: `true` is a YAML boolean and
#: never equals this.
CANCEL_EXPRESSION = "${{ github.event_name == 'pull_request' }}"

#: The trigger that puts a workflow in scope. `pull_request_target` is
#: deliberately absent; see the module docstring.
PULL_REQUEST = "pull_request"

#: Context values unique to one run. The estate rule allows one only as the
#: fallback behind the pull-request number.
RUN_UNIQUE: frozenset[str] = frozenset(
    {"github.run_id", "github.run_number", "github.run_attempt", "github.sha"}
)


#: Workflows known to start on `pull_request`. Discovery below is dynamic so a
#: new workflow is covered the day it lands, but a dynamic list that silently
#: empties turns every parametrized test into a vacuous pass. This names the
#: floor discovery must still reach.
KNOWN_PULL_REQUEST_WORKFLOWS: frozenset[str] = frozenset(
    {
        "benchmark-regressions.yml",
        "ci.yml",
        "kani-pr.yml",
        "property-tests.yml",
    }
)


def _parse(text: str, name: str) -> dict[object, object]:
    """Parse workflow text strictly, refusing anything but a mapping."""
    document = parse_workflow_text(text)
    if not isinstance(document, dict):
        message = f"{name} must parse as a mapping"
        raise AssertionError(message)
    return document


def _load(path: Path) -> dict[object, object]:
    """Read and strictly parse one workflow file."""
    return _parse(path.read_text(encoding="utf-8"), path.name)


def _workflow_paths() -> list[Path]:
    """Return every workflow file, sorted for stable test identifiers."""
    return sorted(
        path
        for path in WORKFLOW_DIR.iterdir()
        if path.is_file() and path.suffix in WORKFLOW_SUFFIXES
    )


def _pull_request_workflows() -> list[Path]:
    """Return every workflow a pull request can start, in name order."""
    return [
        path
        for path in _workflow_paths()
        if PULL_REQUEST in (trigger_names(_load(path)) or frozenset())
    ]


def _concurrency(path: Path) -> dict[object, object]:
    """Return a workflow's top-level concurrency mapping.

    The shorthand string form cannot carry `cancel-in-progress` at all, so it
    reads as empty here, as does a workflow declaring no concurrency.
    """
    declared = _load(path).get("concurrency")
    return declared if isinstance(declared, dict) else {}


PULL_REQUEST_WORKFLOWS = _pull_request_workflows()
WORKFLOW_IDS = [path.name for path in PULL_REQUEST_WORKFLOWS]


def test_discovery_still_finds_the_known_pull_request_workflows() -> None:
    """Discovery reaches its floor, so the parametrized contracts are not empty.

    If the read broke, or the `on:` key changed shape, the list would empty
    and each contract below would pass having asserted nothing.
    """
    missing = sorted(KNOWN_PULL_REQUEST_WORKFLOWS - set(WORKFLOW_IDS))
    assert not missing, (
        f"these workflows start on pull_request but discovery missed them: "
        f"{', '.join(missing)}; the contracts below would pass without "
        "asserting anything about them"
    )


def test_every_workflow_declares_a_trigger_set_this_reader_models() -> None:
    """No workflow's `on:` defeats the reader that decides what is in scope.

    A workflow the reader cannot model is dropped from discovery, and every
    contract below would pass while saying nothing about it.
    """
    unreadable = sorted(
        path.name for path in _workflow_paths() if trigger_names(_load(path)) is None
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


@pytest.mark.parametrize("workflow", PULL_REQUEST_WORKFLOWS, ids=WORKFLOW_IDS)
def test_every_pull_request_workflow_declares_a_concurrency_group(
    workflow: Path,
) -> None:
    """A workflow a pull request starts declares a concurrency group.

    Without one, every push to the branch leaves its predecessor running to
    completion on a paid runner.
    """
    group = _concurrency(workflow).get("group")
    message = (
        f"{workflow.name} starts on pull_request and must declare "
        "concurrency.group; without it a superseded run holds a runner until "
        "it finishes"
    )
    assert isinstance(group, str), message
    assert group.strip(), message


@pytest.mark.parametrize("workflow", PULL_REQUEST_WORKFLOWS, ids=WORKFLOW_IDS)
def test_two_pushes_to_one_pull_request_share_a_group(workflow: Path) -> None:
    """The newer push lands in its predecessor's group, so it can cancel it.

    A group built from the run identifier alone, the SHA, or the run
    identifier ahead of the pull-request number renders differently for each
    push and cancels nothing.
    """
    group = str(_concurrency(workflow).get("group", ""))
    split = [
        (render_group(group, first), render_group(group, second))
        for first, second in MUST_SHARE
        if render_group(group, first) != render_group(group, second)
    ]
    assert not split, (
        f"{workflow.name}'s group renders differently for two pushes to one "
        f"pull request: {split}"
    )


@pytest.mark.parametrize("workflow", PULL_REQUEST_WORKFLOWS, ids=WORKFLOW_IDS)
def test_no_other_two_runs_share_a_group(workflow: Path) -> None:
    """No run cancels another pull request's, and none replaces a pending run.

    The runs are a pull request, a fork's pull request from a branch of the
    same name, two pushes to `main`, and two dispatches of another branch. A
    ``github.head_ref`` group collides on the forks; a ``github.ref`` or
    ``github.base_ref`` fallback collides on the trunk pushes, where a third
    push would replace the pending second.
    """
    group = str(_concurrency(workflow).get("group", ""))
    rendered = [render_group(group, run) for run in MUST_PART]
    assert len(set(rendered)) == len(MUST_PART), (
        f"{workflow.name}'s group collides across runs that must stay apart: {rendered}"
    )


@pytest.mark.parametrize("workflow", PULL_REQUEST_WORKFLOWS, ids=WORKFLOW_IDS)
def test_the_run_identifier_is_only_the_fallback(workflow: Path) -> None:
    """``github.run_id`` appears once, behind the pull-request number.

    The estate rule allows a run-unique value only there. Anywhere else it
    either splits one pull request's pushes or hides a ref-keyed fallback.
    """
    group = str(_concurrency(workflow).get("group", ""))
    found = expressions(group)
    misplaced = [
        operands
        for operands in found
        if operands != ESTATE_FALLBACK and any(name in RUN_UNIQUE for name in operands)
    ]
    assert found.count(ESTATE_FALLBACK) == 1, (
        f"{workflow.name}'s group must contain exactly one "
        f"`${{{{ {' || '.join(ESTATE_FALLBACK)} }}}}`, found {found}"
    )
    assert not misplaced, (
        f"{workflow.name}'s group uses a run-unique value outside the fallback: "
        f"{misplaced}"
    )


def _workflow_name(workflow: Path) -> str:
    """Return the name Actions gives a workflow: its `name:`, else its path."""
    declared = _load(workflow).get("name")
    return (
        declared if isinstance(declared, str) else f".github/workflows/{workflow.name}"
    )


def test_no_two_workflows_share_a_group_for_one_pull_request() -> None:
    """Two workflows on one pull request never cancel each other.

    Each workflow is rendered under its own name. A group that leaves the
    workflow out, such as ``pr-${{ github.event.pull_request.number }}``,
    would put the CI run and every other pull-request workflow in one group,
    and whichever started last would cancel the rest.
    """
    rendered = {
        workflow.name: render_group(
            str(_concurrency(workflow).get("group", "")),
            {**FIRST_PUSH, "github.workflow": _workflow_name(workflow)},
        )
        for workflow in PULL_REQUEST_WORKFLOWS
    }
    assert len(set(rendered.values())) == len(rendered), (
        f"these workflows share a concurrency group for one pull request: {rendered}"
    )


@pytest.mark.parametrize("workflow", PULL_REQUEST_WORKFLOWS, ids=WORKFLOW_IDS)
def test_cancellation_is_conditioned_on_the_event(workflow: Path) -> None:
    """Cancellation applies to pull requests only, not to pushes or schedules.

    A literal `true` reads as a stricter setting and is a regression: it would
    cancel a run on `main`, a schedule, or a dispatch, none of which has a
    successor that repeats its work.
    """
    declared = _concurrency(workflow).get("cancel-in-progress")
    assert declared == CANCEL_EXPRESSION, (
        f"{workflow.name} must set cancel-in-progress to "
        f"{CANCEL_EXPRESSION!r}, not {declared!r}; a missing value leaves "
        "superseded runs in flight and a literal true also cancels pushes to "
        "main, schedules, and dispatches"
    )
