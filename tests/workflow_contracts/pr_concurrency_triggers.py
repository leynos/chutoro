"""Read the event names a workflow declares under `on:`.

The pull-request cancel contract decides which workflows are in scope from
their triggers, so a reader that misread a shape would drop a workflow from
every contract in silence. GitHub accepts a mapping of event to
configuration, a list of event names, and a bare event name; all three are
read. PyYAML resolves an unquoted `on:` key to the boolean ``True``, so both
spellings of the key are read, and a document carrying both is refused
because GitHub would see one trigger set and this reader another.
"""

from __future__ import annotations

#: The trigger that puts a workflow in scope. `pull_request_target` is
#: deliberately absent: it runs against the base repository to carry a token,
#: and cancelling one mid-flight is a hazard with no minutes to win.
PULL_REQUEST = "pull_request"


def trigger_names(document: dict[object, object]) -> frozenset[str] | None:
    """Return the event names a workflow declares under `on:`.

    Parameters
    ----------
    document
        One parsed workflow.

    Returns
    -------
    frozenset of str, or None
        The declared events, empty when there is no `on:`, and ``None`` for a
        shape this reader does not model, so the caller can name the workflow
        rather than let discovery drop it.

    Examples
    --------
    >>> sorted(trigger_names({True: ["push", "pull_request"]}))
    ['pull_request', 'push']
    >>> trigger_names({"on": "push", True: "pull_request"}) is None
    True
    """
    if "on" in document and True in document:
        return None
    declared = document.get("on", document.get(True))
    return frozenset() if declared is None else event_names(declared)


def event_names(declared: object) -> frozenset[str] | None:
    """Return the event names in one `on:` value, or ``None`` for another shape.

    Iterating a mapping yields its keys and a list its items, so both shapes
    share one reading; a bare string names a single event. A list or mapping
    holding anything but names is refused whole rather than filtered, since
    dropping an entry would read a different trigger set from GitHub's.

    Parameters
    ----------
    declared
        The value of the `on:` key.

    Returns
    -------
    frozenset of str, or None
        The event names, or ``None`` when the shape is not modelled.

    Examples
    --------
    >>> event_names(["push", 3]) is None
    True
    """
    match declared:
        case str():
            return frozenset({declared})
        case dict() | list() if all(isinstance(name, str) for name in declared):
            return frozenset(declared)
        case _:
            return None


def pull_request_workflows(
    workflows: dict[str, dict[object, object]],
) -> dict[str, dict[object, object]]:
    """Return the workflows a pull request can start, keyed by file name.

    Parameters
    ----------
    workflows
        Parsed workflows keyed by file name, as ``all_workflow_documents`` returns.

    Returns
    -------
    dict of str to dict
        The workflows whose `on:` names `pull_request`, in the same order.

    Examples
    --------
    >>> pull_request_workflows({"a.yml": {True: "push"}, "b.yml": {True: "pull_request"}})
    {'b.yml': {True: 'pull_request'}}
    """
    return {
        name: document
        for name, document in workflows.items()
        if PULL_REQUEST in (trigger_names(document) or frozenset())
    }
