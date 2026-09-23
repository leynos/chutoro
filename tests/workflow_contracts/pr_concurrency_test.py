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

The group also has to distinguish one pull request from another. A group
derived from ``github.run_id`` is unique per run and so cancels nothing, while
a constant group would let one branch cancel another's gates, and so would one
keyed on ``github.head_ref``, which two forks' `patch-1` branches share. Rather
than search the group's text for a name, the contract renders it against a
small set of run contexts and compares the results.

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
    FIRST_PUSH,
    MUST_PART,
    MUST_SHARE,
    UnmodelledGroupError,
    keeps_runs_together_and_apart,
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

#: The group every workflow here uses unless it already had its own.
ESTATE_GROUP = (
    "${{ github.workflow }}-${{ github.event.pull_request.number || github.ref }}"
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
    assert trigger_names(_parse(text, "shape.yml")) == expected


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
def test_runs_that_must_queue_together_share_a_group(workflow: Path) -> None:
    """Runs that must cancel or queue behind one another render one group.

    Two pushes to one pull request, a re-run of the first, and two pushes to
    `main` each form a pair. A group built from the run identifier, the SHA
    or the run attempt renders differently within a pair, so it cancels
    nothing, and on `main` it would let two cache writers race.
    """
    group = str(_concurrency(workflow).get("group", ""))
    split = [
        (render_group(group, first), render_group(group, second))
        for first, second in MUST_SHARE
        if render_group(group, first) != render_group(group, second)
    ]
    assert not split, (
        f"{workflow.name}'s group renders differently for runs that must share "
        f"one: {split}"
    )


@pytest.mark.parametrize("workflow", PULL_REQUEST_WORKFLOWS, ids=WORKFLOW_IDS)
def test_no_two_pull_requests_or_main_share_a_group(workflow: Path) -> None:
    """A push to one pull request never cancels another's run, or `main`'s.

    The second pull request comes from a fork whose branch has the same name,
    so a group keyed on ``github.head_ref`` fails here as a constant one does.
    """
    group = str(_concurrency(workflow).get("group", ""))
    rendered = [render_group(group, run) for run in MUST_PART]
    assert len(set(rendered)) == len(MUST_PART), (
        f"{workflow.name}'s group collides across runs that must stay apart: {rendered}"
    )


@pytest.mark.parametrize(
    ("template", "verdict"),
    [
        (ESTATE_GROUP, "accept"),
        ("kani-pr-${{ github.ref }}", "accept"),
        ("${{ github.workflow }}-${{ github.run_id }}", "refuse"),
        ("${{ github.workflow }}-${{ github.sha }}", "refuse"),
        ("${{ github.workflow }}-${{ github.head_ref }}", "refuse"),
        ("pr-${{ github.event.pull_request.number || github.run_id }}", "refuse"),
        ("pr-${{ github.event.pull_request.number || github.base_ref }}", "refuse"),
        ("${{ github.ref }}-${{ github.run_attempt }}", "refuse"),
        ("one-group-for-everything", "refuse"),
    ],
    ids=[
        "estate",
        "ref-keyed",
        "run-id",
        "sha",
        "head-ref",
        "run-id-on-main",
        "base-ref-on-main",
        "run-attempt",
        "constant",
    ],
)
def test_the_group_rules_accept_and_refuse_the_known_shapes(
    template: str, verdict: str
) -> None:
    """Drive both group rules directly, so neither passes for want of a case.

    The repository's own groups are all acceptable, so they alone cannot show
    that the rules refuse anything.
    """
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
