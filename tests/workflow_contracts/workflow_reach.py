"""Which workflows a pull request can cause to run.

A pull request reaches a workflow in two ways: by a trigger the workflow
declares, and through a local reusable-workflow call made by a workflow it
already reaches. The second is the one a trigger-based reading misses, because
a reusable child declares `workflow_call`, not `pull_request`, and so reads as
unreachable while a pull request runs it with whatever secrets its caller
forwarded.

These readers take parsed documents rather than reading files, so the contracts
can drive graphs and trigger spellings this repository does not have.

Run via ``make test-workflow-contracts``.
"""

import collections.abc as cabc
import typing as typ

PULL_REQUEST_TRIGGER: typ.Final[str] = "pull_request"

#: The variant that runs in the base repository's context and therefore *can*
#: read its secrets, unlike `pull_request`. A coverage step here would be worse
#: than one in an ordinary pull-request job, not equivalent to it.
PULL_REQUEST_TARGET_TRIGGER: typ.Final[str] = "pull_request_target"

#: The trigger that resumes a run with the base repository's privileges.
SUBMISSION_TRIGGER: typ.Final[str] = "workflow_run"

#: The triggers a pull request's review activity or its merge fires. Each runs
#: on the pull request's behalf and can report a check on it, so a workflow
#: declaring one alone is as reachable as one declaring `pull_request`.
REVIEW_AND_MERGE_TRIGGERS: typ.Final[tuple[str, ...]] = (
    "pull_request_review",
    "pull_request_review_comment",
    "merge_group",
)

REACHABLE_TRIGGERS: typ.Final[tuple[str, ...]] = (
    PULL_REQUEST_TRIGGER,
    PULL_REQUEST_TARGET_TRIGGER,
    SUBMISSION_TRIGGER,
    *REVIEW_AND_MERGE_TRIGGERS,
)

#: The spellings GitHub accepts for a call into this repository: `./` and the
#: `$/` self-repository form, which resolves at the running commit.
_LOCAL_CALL_PREFIXES: typ.Final[tuple[str, ...]] = ("./", "$/")

#: The keys `on:` can arrive under. A resolving loader reads a bare `on` as
#: the boolean True; a quoted `'on'` stays a string. Both are read, because a
#: document could carry both and a reader of one would miss the other's
#: triggers.
_TRIGGER_KEYS: typ.Final[tuple[object, ...]] = ("on", True)

#: Where a same-repository reusable workflow lives, relative to the root.
WORKFLOW_DIRECTORY: typ.Final[str] = ".github/workflows/"


def declared_triggers(document: dict[typ.Any, typ.Any]) -> list[str]:
    """Return every trigger name a parsed workflow declares.

    Parameters
    ----------
    document : dict[typing.Any, typing.Any]
        One parsed workflow document.

    Returns
    -------
    list[str]
        The trigger names from the scalar, sequence and mapping forms `on:`
        accepts, read under both the string key and the boolean one.

    Examples
    --------
    >>> declared_triggers({True: ["push", "pull_request"]})
    ['push', 'pull_request']
    >>> declared_triggers({"on": "pull_request"})
    ['pull_request']
    """
    names: list[str] = []
    for key in _TRIGGER_KEYS:
        match document.get(key):
            case str() as scalar:
                names.append(scalar)
            case list() as sequence:
                names.extend(item for item in sequence if isinstance(item, str))
            case dict() as mapping:
                names.extend(item for item in mapping if isinstance(item, str))
            case _:
                pass
    return names


def declares_both_trigger_keys(document: dict[typ.Any, typ.Any]) -> bool:
    """Return whether a workflow declares `on` under both of its keys.

    GitHub merges the two, so a reader that picks either is blind to the
    other's triggers. Refused outright rather than merged here, because a
    document that spells its triggers twice has one spelling nobody meant.

    Examples
    --------
    >>> declares_both_trigger_keys({"on": "push", True: "pull_request"})
    True
    >>> declares_both_trigger_keys({True: "push"})
    False
    """
    return all(key in document for key in _TRIGGER_KEYS)


def declares_trigger(document: dict[typ.Any, typ.Any], trigger: str) -> bool:
    """Return whether a parsed workflow declares the given trigger.

    Examples
    --------
    >>> declares_trigger({True: {"pull_request": None}}, "pull_request")
    True
    >>> declares_trigger({"on": "push"}, "pull_request")
    False
    """
    return trigger in declared_triggers(document)


def is_reachable_by_a_pull_request(document: dict[typ.Any, typ.Any]) -> bool:
    """Return whether a pull request can cause this workflow to run directly.

    Every reachable trigger counts. `pull_request_target` and `workflow_run`
    resume in the base repository's context, so a step under either reads a
    credential in a run a pull request's contents influenced; the review and
    merge-queue triggers run on the pull request's behalf.

    Examples
    --------
    >>> is_reachable_by_a_pull_request({True: {"workflow_run": None}})
    True
    >>> is_reachable_by_a_pull_request({True: {"push": None}})
    False
    """
    return any(declares_trigger(document, name) for name in REACHABLE_TRIGGERS)


def local_workflow_target(uses: str) -> str | None:
    """Return the workflow file a job-level `uses` names in this repository.

    A call is local by its shape: with one leading `./` or `$/` removed, the
    remainder is a path under the workflow directory. Anything else is a call
    into another repository and is not followed, since a foreign workflow is
    not ours to read.

    Examples
    --------
    >>> local_workflow_target("./.github/workflows/child.yml")
    'child.yml'
    >>> local_workflow_target("$/.github/workflows/child.yml")
    'child.yml'
    >>> local_workflow_target("leynos/shared-actions/.github/workflows/x.yml@v1") is None
    True
    """
    prefix = next((p for p in _LOCAL_CALL_PREFIXES if uses.startswith(p)), "")
    path = uses.removeprefix(prefix)
    if not path.startswith(WORKFLOW_DIRECTORY):
        return None
    return path.removeprefix(WORKFLOW_DIRECTORY)


def local_workflows_called_by(document: dict[typ.Any, typ.Any]) -> list[str]:
    """Return the local reusable workflows one document calls, by file name.

    Examples
    --------
    >>> local_workflows_called_by(
    ...     {"jobs": {"a": {"uses": "./.github/workflows/child.yml"}}}
    ... )
    ['child.yml']
    """
    declared = document.get("jobs")
    definitions = declared.values() if isinstance(declared, dict) else ()
    targets = (
        local_workflow_target(definition["uses"])
        for definition in definitions
        if isinstance(definition, dict) and isinstance(definition.get("uses"), str)
    )
    return [target for target in targets if target is not None]


class Reach(typ.NamedTuple):
    """The workflows one entry reaches, and the local calls that name nothing."""

    reached: list[str]
    missing: list[str]


def reachable_workflows(
    entry: str, documents: cabc.Mapping[str, dict[typ.Any, typ.Any]]
) -> Reach:
    """Return the entry and every local workflow it reaches, transitively.

    The walk tracks what it has seen, so a cycle, which GitHub rejects but a
    half-finished edit can produce, terminates rather than recursing. A local
    call to a file that is absent, or that does not parse to a mapping, is
    returned as missing rather than skipped: a skipped target is a workflow
    no rule was ever asked about.

    Examples
    --------
    >>> reachable_workflows(
    ...     "a.yml",
    ...     {
    ...         "a.yml": {"jobs": {"x": {"uses": "./.github/workflows/b.yml"}}},
    ...         "b.yml": {"jobs": {"y": {"uses": "./.github/workflows/gone.yml"}}},
    ...     },
    ... )
    Reach(reached=['a.yml', 'b.yml'], missing=['gone.yml'])
    """
    reached: list[str] = []
    missing: list[str] = []
    pending = [entry]
    while pending:
        current = pending.pop(0)
        if current in reached or current in missing:
            continue
        document = documents.get(current)
        if document is None:
            missing.append(current)
            continue
        reached.append(current)
        pending.extend(local_workflows_called_by(document))
    return Reach(reached, missing)
