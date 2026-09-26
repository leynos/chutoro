"""Render a workflow's concurrency group for a given run, as Actions would.

`pr_concurrency_test.py` decides whether a group keeps together the runs that
should queue or cancel one another and keeps every other run apart. Searching
the group's text for a context name cannot answer that: ``github.head_ref``
names the pull request's branch and still collides across forks, and a group
can name the pull-request number and then discard it. So the contract renders
each group against the run contexts below and compares the strings.

Runs that must share a group: two pushes to one pull request, so the newer
cancels the older. Runs that must not: that pull request, another pull request
from a fork whose branch has the same name, two pushes to `main`, and two
dispatches of one other branch. A ref-keyed fallback would put the trunk
pushes in one group, where a third push replaces a still-pending second one
and that commit never gets CI; overlapping trunk runs are the cheaper risk,
because compiler-cache writes are content-addressed and a cache save of an
existing key is refused harmlessly. So the fallback is ``github.run_id``
(estate rule "PR-lane concurrency fallback").

Only context paths joined by ``||`` are modelled, which is every form the
estate's groups use. Anything else is refused rather than guessed at.
"""

from __future__ import annotations

import re

#: One `${{ ... }}` expression inside a group template.
_EXPRESSION = re.compile(r"\$\{\{\s*(.*?)\s*\}\}")

#: A bare context path such as ``github.event.pull_request.number``.
_CONTEXT_PATH = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*")


class UnmodelledGroupError(AssertionError):
    """Raised when a group uses an expression the renderer does not model."""


def pull_request_run(
    number: str, head_ref: str, run_id: str, attempt: str = "1"
) -> dict[str, str]:
    """Return the context of one run of a pull request's workflow.

    Parameters
    ----------
    number
        The pull request number, or ``""`` for a run with no pull request.
    head_ref
        The source branch name.
    run_id
        The run identifier, which also stands in for the run number and SHA.
    attempt
        The run attempt, ``"2"`` or more for a re-run.

    Returns
    -------
    dict of str to str
        Context paths mapped to the values Actions would supply.

    Examples
    --------
    >>> pull_request_run("7", "patch-1", "101")["github.ref"]
    'refs/pull/7/merge'
    """
    return {
        "github.workflow": "CI",
        "github.event_name": "pull_request",
        "github.event.pull_request.number": number,
        "github.ref": f"refs/pull/{number}/merge",
        "github.head_ref": head_ref,
        "github.base_ref": "main",
        "github.run_id": run_id,
        "github.run_number": run_id,
        "github.run_attempt": attempt,
        "github.sha": f"sha-{run_id}",
    }


def main_push(run_id: str) -> dict[str, str]:
    """Return the context of one push to `main`, which has no pull request.

    Parameters
    ----------
    run_id
        The run identifier.

    Returns
    -------
    dict of str to str
        Context paths mapped to the values Actions would supply; the
        pull-request fields, ``head_ref`` and ``base_ref`` are empty.

    Examples
    --------
    >>> main_push("104")["github.base_ref"]
    ''
    """
    return {
        **pull_request_run("", "", run_id),
        "github.event_name": "push",
        "github.ref": "refs/heads/main",
        "github.base_ref": "",
    }


FIRST_PUSH = pull_request_run("7", "patch-1", "101")
SECOND_PUSH = pull_request_run("7", "patch-1", "102")
OTHER_FORK = pull_request_run("8", "patch-1", "103")
MAIN_PUSH = main_push("104")
NEXT_MAIN_PUSH = main_push("105")
BRANCH_DISPATCH = {
    **main_push("106"),
    "github.event_name": "workflow_dispatch",
    "github.ref": "refs/heads/feature",
}
NEXT_BRANCH_DISPATCH = {**BRANCH_DISPATCH, "github.run_id": "107"}

#: Runs that must render one group: the newer cancels the older.
MUST_SHARE: tuple[tuple[dict[str, str], dict[str, str]], ...] = (
    (FIRST_PUSH, SECOND_PUSH),
)

#: Runs that must render distinct groups: none may cancel or replace another.
MUST_PART: tuple[dict[str, str], ...] = (
    FIRST_PUSH,
    OTHER_FORK,
    MAIN_PUSH,
    NEXT_MAIN_PUSH,
    BRANCH_DISPATCH,
    NEXT_BRANCH_DISPATCH,
)

#: The one expression the estate rule allows a run-unique value in: the
#: pull-request number first, the run identifier as its fallback.
ESTATE_FALLBACK: tuple[str, ...] = ("github.event.pull_request.number", "github.run_id")

#: Context values unique to one run. The estate rule allows one only as the
#: fallback behind the pull-request number.
RUN_UNIQUE: frozenset[str] = frozenset(
    {"github.run_id", "github.run_number", "github.run_attempt", "github.sha"}
)


def expressions(template: str) -> list[tuple[str, ...]]:
    """Return the ``||`` operands of each `${{ ... }}` expression in a group.

    Parameters
    ----------
    template
        The group as written in the workflow.

    Returns
    -------
    list of tuple of str
        One tuple of stripped operands per expression, in order.

    Examples
    --------
    >>> expressions("${{ github.workflow }}-${{ a || b }}")
    [('github.workflow',), ('a', 'b')]
    """
    return [
        tuple(operand.strip() for operand in match.group(1).split("||"))
        for match in _EXPRESSION.finditer(template)
    ]


def render_group(template: str, context: dict[str, str]) -> str:
    """Render a concurrency group for `context`.

    ``||`` yields its first non-empty operand, as in Actions, where a missing
    pull-request number is null and falls through to the ref. Every operand
    is checked before any is chosen, so an unmodelled right-hand operand is
    refused even when the left one would have been used.

    Parameters
    ----------
    template
        The group as written in the workflow.
    context
        Context paths mapped to their values for one run.

    Returns
    -------
    str
        The group Actions would compute.

    Raises
    ------
    UnmodelledGroupError
        If an expression holds anything but known context paths joined by
        ``||``, or a ``${{`` opener is left unclosed.

    Examples
    --------
    >>> template = "pr-${{ github.event.pull_request.number || github.ref }}"
    >>> render_group(template, FIRST_PUSH)
    'pr-7'
    >>> render_group(template, MAIN_PUSH)
    'pr-refs/heads/main'
    """

    def evaluate(match: re.Match[str]) -> str:
        paths = [operand.strip() for operand in match.group(1).split("||")]
        unmodelled = [
            path
            for path in paths
            if not _CONTEXT_PATH.fullmatch(path) or path not in context
        ]
        if unmodelled:
            message = f"unmodelled expression {match.group(0)!r} in {template!r}"
            raise UnmodelledGroupError(message)
        return next((context[path] for path in paths if context[path]), "")

    rendered = _EXPRESSION.sub(evaluate, template)
    if "${{" in rendered:
        message = f"unclosed expression in {template!r}"
        raise UnmodelledGroupError(message)
    return rendered


def keeps_runs_together_and_apart(template: str) -> bool:
    """Report whether a group joins the runs in `MUST_SHARE` and parts the rest.

    Parameters
    ----------
    template
        The group as written in the workflow.

    Returns
    -------
    bool
        True when every `MUST_SHARE` pair renders one group and the
        `MUST_PART` runs render distinct ones.

    Examples
    --------
    >>> group = "${{ github.event.pull_request.number || github.run_id }}"
    >>> keeps_runs_together_and_apart(group)
    True
    >>> keeps_runs_together_and_apart("kani-pr-${{ github.ref }}")
    False
    >>> keeps_runs_together_and_apart("${{ github.workflow }}-${{ github.head_ref }}")
    False
    """
    shares = all(
        render_group(template, first) == render_group(template, second)
        for first, second in MUST_SHARE
    )
    rendered = {render_group(template, run) for run in MUST_PART}
    return shares and len(rendered) == len(MUST_PART)


def fallback_problems(group: str) -> list[str]:
    """Return why a group breaks the run-identifier rule, or nothing.

    The estate rule allows a run-unique value exactly once, as the fallback
    behind the pull-request number. Anywhere else it either splits one pull
    request's pushes or hides a ref-keyed fallback.

    Parameters
    ----------
    group
        The group as written in the workflow.

    Returns
    -------
    list of str
        One reason per broken clause; empty when the group complies.

    Examples
    --------
    >>> fallback_problems("${{ github.workflow }}-${{ github.event.pull_request.number || github.run_id }}")
    []
    >>> len(fallback_problems("${{ github.workflow }}-${{ github.run_id }}"))
    2
    """
    found = expressions(group)
    misplaced = [
        operands
        for operands in found
        if operands != ESTATE_FALLBACK and any(name in RUN_UNIQUE for name in operands)
    ]
    problems: list[str] = []
    if found.count(ESTATE_FALLBACK) != 1:
        problems.append(
            f"must contain exactly one `${{{{ {' || '.join(ESTATE_FALLBACK)} }}}}`,"
            f" found {found}"
        )
    if misplaced:
        problems.append(f"uses a run-unique value outside the fallback: {misplaced}")
    return problems


def shared_groups(rendered: dict[str, str]) -> dict[str, list[str]]:
    """Return each group that more than one workflow renders.

    GitHub compares concurrency group names case-insensitively, so ``CI-7``
    and ``ci-7`` are one group and either run can cancel the other. Groups
    are therefore compared casefolded.

    Parameters
    ----------
    rendered
        Each workflow's file name mapped to the group it renders.

    Returns
    -------
    dict of str to list of str
        Each casefolded group claimed more than once, mapped to the files
        claiming it; empty when every group is distinct.

    Examples
    --------
    >>> shared_groups({"a.yml": "CI-7", "b.yml": "ci-7", "c.yml": "Lint-7"})
    {'ci-7': ['a.yml', 'b.yml']}
    """
    owners: dict[str, list[str]] = {}
    for name, group in rendered.items():
        owners.setdefault(group.casefold(), []).append(name)
    return {group: names for group, names in owners.items() if len(names) > 1}
