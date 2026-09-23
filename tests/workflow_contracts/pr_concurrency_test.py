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
a constant group would let one branch cancel another's gates.

Only `pull_request` is in scope. A `pull_request_target` workflow runs against
the base repository to carry a token, and the one here merges Dependabot pull
requests; cancelling a merge mid-flight is a hazard with no minutes to win.

Workflows are read through a loader that refuses duplicate mapping keys.
PyYAML otherwise keeps the last of two `concurrency:` blocks and says nothing,
so a correct block could sit above a wrong one and pass.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

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

#: Expressions that are unique to a single run. A group built from one of these
#: can never match another run, so it cancels nothing while looking exactly
#: like a concurrency control.
RUN_UNIQUE_EXPRESSIONS: tuple[str, ...] = (
    "github.run_id",
    "github.run_number",
    "github.run_attempt",
    "github.sha",
)

#: Expressions that differ between two pull requests. A group naming none of
#: them is shared by every branch, so one pull request's push would cancel
#: another's gates.
PER_PULL_REQUEST_EXPRESSIONS: tuple[str, ...] = (
    "github.event.pull_request.number",
    "github.head_ref",
    "github.ref",
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


class DuplicateKeyError(yaml.constructor.ConstructorError):
    """Raised when a workflow mapping declares the same key twice."""


class _StrictLoader(yaml.SafeLoader):
    """Safe loader that refuses a mapping declaring one key twice."""

    def construct_mapping(
        self,
        node: yaml.MappingNode,
        deep: bool = False,  # noqa: FBT001, FBT002 - PyYAML's own signature
    ) -> dict[object, object]:
        """Build a mapping, refusing the second occurrence of any key."""
        seen: set[object] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise DuplicateKeyError(
                    None,
                    None,
                    f"duplicate key {key!r}",
                    key_node.start_mark,
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def _parse(text: str, name: str) -> dict[object, object]:
    """Parse workflow text strictly, refusing anything but a mapping."""
    document = yaml.load(text, Loader=_StrictLoader)  # noqa: S506 - SafeLoader subclass
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


def _trigger_names(document: dict[object, object]) -> frozenset[str] | None:
    """Return the event names a workflow declares under `on:`.

    GitHub accepts a mapping of event to configuration, a list of event
    names, and a bare event name; all three are read. PyYAML resolves an
    unquoted `on:` key to the boolean ``True``, so both spellings of the key
    are read, and a document carrying both is refused because GitHub would
    see one trigger set and this reader another.

    Returns ``None`` for a shape this reader does not model, so that
    `test_every_workflow_declares_a_trigger_set_this_reader_models` can name
    the workflow rather than let discovery drop it in silence.
    """
    if "on" in document and True in document:
        return None
    declared = document.get("on", document.get(True))
    if declared is None:
        return frozenset()
    if isinstance(declared, dict):
        return frozenset(key for key in declared if isinstance(key, str))
    if isinstance(declared, list):
        return frozenset(event for event in declared if isinstance(event, str))
    if isinstance(declared, str):
        return frozenset({declared})
    return None


def _pull_request_workflows() -> list[Path]:
    """Return every workflow a pull request can start, in name order."""
    return [
        path
        for path in _workflow_paths()
        if PULL_REQUEST in (_trigger_names(_load(path)) or frozenset())
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
        path.name for path in _workflow_paths() if _trigger_names(_load(path)) is None
    )
    assert not unreadable, (
        f"these workflows declare an `on:` this reader does not model: "
        f"{', '.join(unreadable)}; each is dropped from discovery, so every "
        "contract below would pass without asserting anything about it"
    )


def test_a_duplicated_key_is_refused_rather_than_resolved() -> None:
    """The loader refuses a second `concurrency:` block instead of keeping it.

    PyYAML's own loader keeps the last value, so a correct block followed by
    a wrong one would reach these contracts as the wrong one only, and a
    wrong one followed by a correct one would pass. This drives the loader
    the contracts use, not the files it guards.
    """
    text = "on:\n  pull_request:\nconcurrency:\n  group: a\nconcurrency:\n  group: b\n"
    with pytest.raises(DuplicateKeyError, match="duplicate key 'concurrency'"):
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
def test_the_group_is_not_unique_to_one_run(workflow: Path) -> None:
    """The group is shared by successive runs of the same pull request.

    A group built from the run identifier or the commit SHA matches no other
    run, so it cancels nothing while reading as a concurrency control.
    """
    group = str(_concurrency(workflow).get("group", ""))
    offenders = [name for name in RUN_UNIQUE_EXPRESSIONS if name in group]
    assert not offenders, (
        f"{workflow.name} builds its concurrency group from "
        f"{', '.join(offenders)}, which is unique to one run; the group would "
        "never match a superseded run and would cancel nothing"
    )


@pytest.mark.parametrize("workflow", PULL_REQUEST_WORKFLOWS, ids=WORKFLOW_IDS)
def test_the_group_distinguishes_one_pull_request_from_another(
    workflow: Path,
) -> None:
    """The group varies with the pull request, so branches do not cancel each other.

    A constant group would put every open pull request in one queue, and the
    first push anywhere would cancel the gates running everywhere else.
    """
    group = str(_concurrency(workflow).get("group", ""))
    assert any(name in group for name in PER_PULL_REQUEST_EXPRESSIONS), (
        f"{workflow.name} must key its concurrency group on the pull request, "
        f"by naming one of {', '.join(PER_PULL_REQUEST_EXPRESSIONS)}; a group "
        "shared by every branch would cancel unrelated pull requests"
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
