"""What the one workflow allowed to publish coverage must itself hold to.

`coverage-main.yml` owns the CodeScene upload and the ratchet baseline that
every pull request compares against. Moving the upload there is only a boundary
if the publisher cannot be made to publish from anywhere else, and cannot be
cancelled halfway through writing what the next pull request will read.

Three things are read here. The upload step's condition must require the trunk
ref as one of its conjuncts, so a manual dispatch from a branch cannot upload
that branch's coverage as the project's. The push filter must name the trunk
alone. And no concurrency setting may cancel a run in progress, because a
cancelled publisher abandons both its upload and its baseline write, where a
queued one publishes later.

Run via ``make test-workflow-contracts``.
"""

import re
import typing as typ

#: The conjunct that confines the upload to the trunk. Compared exactly after
#: whitespace is normalised, because a looser match accepts a condition that
#: tests a sibling field.
TRUNK_REF_GUARD: typ.Final[str] = "github.ref == 'refs/heads/main'"

#: The branch the publisher's push trigger must name, and name alone.
TRUNK_BRANCH: typ.Final[str] = "main"

#: A single-quoted literal in an expression. Blanked before the operators are
#: read, so an `||` inside a quoted string is not taken for the operator.
_QUOTED_LITERAL: typ.Final[re.Pattern[str]] = re.compile(r"'(?:[^']|'')*'")

#: The wrapper GitHub accepts around an `if:` condition, and does not require.
_EXPRESSION_WRAPPER: typ.Final[re.Pattern[str]] = re.compile(
    r"^\$\{\{\s*(?P<body>.*?)\s*\}\}$", re.DOTALL
)


def _normalised(text: str) -> str:
    """Return an expression fragment with runs of whitespace collapsed."""
    return " ".join(text.split())


def upload_condition_offences(condition: object) -> list[str]:
    """Return why an upload condition fails to confine it to the trunk.

    The condition is split on `&&` and the trunk guard must be one of the
    conjuncts. An unquoted `||` is refused outright rather than reasoned
    about: `a && trunk || b` makes every conjunct optional, and a substring
    test for the guard passes it.

    Parameters
    ----------
    condition : object
        The step's `if:` value as parsed, which may be absent or not a string.

    Returns
    -------
    list[str]
        Empty when the condition requires the trunk ref; otherwise one reason
        per defect.

    Examples
    --------
    >>> upload_condition_offences(
    ...     "env.CS_ACCESS_TOKEN != '' && github.ref == 'refs/heads/main'"
    ... )
    []
    >>> reasons = upload_condition_offences(
    ...     "github.ref == 'refs/heads/main' || github.event_name == 'workflow_dispatch'"
    ... )
    >>> any("unquoted '||'" in reason for reason in reasons)
    True
    """
    if not isinstance(condition, str) or not condition.strip():
        return [f"the upload has no condition, so it does not require {TRUNK_REF_GUARD}"]
    wrapped = _EXPRESSION_WRAPPER.match(condition.strip())
    body = wrapped.group("body") if wrapped else condition.strip()
    offences: list[str] = []
    if "||" in _QUOTED_LITERAL.sub("''", body):
        offences.append(
            "the upload condition contains an unquoted '||', which makes the "
            "trunk guard optional"
        )
    conjuncts = [_normalised(part) for part in body.split("&&")]
    if TRUNK_REF_GUARD not in conjuncts:
        offences.append(
            f"the upload condition does not require {TRUNK_REF_GUARD} as one of "
            f"its conjuncts; it reads {condition!r}"
        )
    return offences


def push_branches(document: dict[typ.Any, typ.Any]) -> list[str] | None:
    """Return the branch filter on a workflow's push trigger, if it declares one.

    Examples
    --------
    >>> push_branches({"on": {"push": {"branches": ["main"]}}})
    ['main']
    >>> push_branches({True: {"push": None}}) is None
    True
    """
    for key in ("on", True):
        declared = document.get(key)
        push = declared.get("push") if isinstance(declared, dict) else None
        branches = push.get("branches") if isinstance(push, dict) else None
        if isinstance(branches, list):
            return [str(branch) for branch in branches]
    return None


def _cancels(concurrency: object) -> bool:
    """Return whether one concurrency setting can cancel a run in progress."""
    # A group given as a bare string never cancels. Anything other than an
    # absent or false `cancel-in-progress` is refused, an expression included,
    # because an expression that reads false today is one edit from true.
    if not isinstance(concurrency, dict):
        return False
    return concurrency.get("cancel-in-progress", False) not in (False, "false")


def cancelling_scopes(document: dict[typ.Any, typ.Any]) -> list[str]:
    """Return every scope in a workflow whose concurrency cancels in progress.

    Examples
    --------
    >>> cancelling_scopes(
    ...     {"concurrency": {"group": "g", "cancel-in-progress": True}, "jobs": {}}
    ... )
    ['the workflow']
    >>> cancelling_scopes({"jobs": {"a": {"concurrency": "g"}}})
    []
    """
    scopes = ["the workflow"] if _cancels(document.get("concurrency")) else []
    declared = document.get("jobs")
    for name, definition in (declared if isinstance(declared, dict) else {}).items():
        if isinstance(definition, dict) and _cancels(definition.get("concurrency")):
            scopes.append(f"job {name}")
    return scopes
