"""Render a workflow's concurrency group for a given run, as Actions would.

`pr_concurrency_test.py` decides whether a group keeps one pull request's
runs together and every other run apart. Searching the group's text for a
context name cannot answer that: ``github.head_ref`` names the pull request's
branch and still collides across forks, and a group can name the pull-request
number and then discard it. So the contract renders each group against the
run contexts below and compares the strings.

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


def pull_request_run(number: str, head_ref: str, run_id: str) -> dict[str, str]:
    """Return the context of one run of a pull request's workflow.

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
        "github.run_attempt": "1",
        "github.sha": f"sha-{run_id}",
    }


#: Two pushes to pull request 7, then pull request 8 from a fork whose branch
#: has the same name, then a push to `main`, which has no pull request.
FIRST_PUSH = pull_request_run("7", "patch-1", "101")
SECOND_PUSH = pull_request_run("7", "patch-1", "102")
OTHER_FORK = pull_request_run("8", "patch-1", "103")
MAIN_PUSH = {
    **pull_request_run("", "", "104"),
    "github.event_name": "push",
    "github.ref": "refs/heads/main",
}


def render_group(template: str, context: dict[str, str]) -> str:
    """Render a concurrency group for `context`.

    ``||`` yields its first non-empty operand, as in Actions, where a missing
    pull-request number is null and falls through to the ref.

    Examples
    --------
    >>> template = "pr-${{ github.event.pull_request.number || github.ref }}"
    >>> render_group(template, FIRST_PUSH)
    'pr-7'
    >>> render_group(template, MAIN_PUSH)
    'pr-refs/heads/main'
    """

    def evaluate(match: re.Match[str]) -> str:
        for operand in match.group(1).split("||"):
            path = operand.strip()
            if not _CONTEXT_PATH.fullmatch(path) or path not in context:
                message = f"unmodelled expression {match.group(0)!r} in {template!r}"
                raise UnmodelledGroupError(message)
            if context[path]:
                return context[path]
        return ""

    return _EXPRESSION.sub(evaluate, template)


def keeps_runs_together_and_apart(template: str) -> bool:
    """Report whether a group joins one pull request's pushes and parts the rest.

    Examples
    --------
    >>> keeps_runs_together_and_apart("kani-pr-${{ github.ref }}")
    True
    >>> keeps_runs_together_and_apart("${{ github.workflow }}-${{ github.head_ref }}")
    False
    """
    shares = render_group(template, FIRST_PUSH) == render_group(template, SECOND_PUSH)
    rendered = {
        render_group(template, run) for run in (FIRST_PUSH, OTHER_FORK, MAIN_PUSH)
    }
    return shares and len(rendered) == 3
