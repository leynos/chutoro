"""Test the reading of the shell that executes a CI step's script.

GitHub accepts a shell declaration at three scopes, and a custom template can
execute the step's script, only parse it, or run it without ``errexit``. The
reading lives in ``scheduling_shell_reading``; these tests pin its answers.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import pytest
from scheduling_shell_reading import effective_shell, runs_the_script


def _scope(shell: str | None) -> dict[str, object]:
    """Return a workflow or job declaring `shell` as its run default, if any.

    Examples
    --------
    >>> _scope("bash")
    {'defaults': {'run': {'shell': 'bash'}}}
    >>> _scope(None)
    {}
    """
    if shell is None:
        return {}
    return {"defaults": {"run": {"shell": shell}}}


@pytest.mark.parametrize(
    ("workflow_shell", "job_shell", "step_shell", "expected"),
    [
        pytest.param(None, None, None, None, id="the-runner-default"),
        pytest.param("sh", None, None, "sh", id="a-workflow-default"),
        pytest.param("sh", "bash", None, "bash", id="a-job-override"),
        pytest.param(None, None, "bash", "bash", id="a-step-default"),
        pytest.param(
            "sh",
            "bash",
            "bash -n {0}",
            "bash -n {0}",
            id="a-step-override",
        ),
    ],
)
def test_the_effective_shell_follows_override_precedence(
    workflow_shell: str | None,
    job_shell: str | None,
    step_shell: str | None,
    expected: str | None,
) -> None:
    """A shell declared at any scope decides how the step's script is run.

    A step's own ``shell`` wins over the job's default, which wins over the
    workflow's. Reading only the step would miss a default declared higher up,
    and reading only the defaults would miss the step that overrides them.
    """
    workflow = _scope(workflow_shell)
    definition = _scope(job_shell)
    step = {} if step_shell is None else {"shell": step_shell}
    effective = effective_shell(step, definition, workflow)
    assert effective == expected, (
        "the step's shell must be read from the nearest scope declaring one; "
        f"workflow={workflow_shell!r}, job={job_shell!r}, "
        f"step={step_shell!r}, read {effective!r}"
    )


@pytest.mark.parametrize("scope", ["workflow", "job", "step"])
def test_a_syntax_only_shell_is_rejected_at_every_scope(scope: str) -> None:
    """``bash -n`` parses the script and runs none of the gate's command.

    GitHub accepts a shell template at all three scopes, so a syntax-only
    template declared at any of them would leave the step green with no
    evidence produced.
    """
    shell = "bash -n {0}"
    workflow = _scope(shell) if scope == "workflow" else {}
    definition = _scope(shell) if scope == "job" else {}
    step = {"shell": shell} if scope == "step" else {}
    effective = effective_shell(step, definition, workflow)
    assert effective == shell, (
        f"a {scope}-level shell must override the scopes above it; "
        f"read {effective!r}"
    )
    assert not runs_the_script(effective), (
        f"a {scope}-level syntax-only shell must be rejected; accepted "
        f"{effective!r}"
    )


@pytest.mark.parametrize(
    ("shell", "expected"),
    [
        pytest.param("bash {0}", False, id="bash-without-errexit"),
        pytest.param("sh {0}", False, id="sh-without-errexit"),
        pytest.param("bash -e {0}", True, id="errexit-in-its-own-word"),
        pytest.param("bash -o errexit {0}", True, id="errexit-named-by-o"),
        pytest.param(
            "bash --noprofile --norc -eo pipefail {0}",
            True,
            id="the-runner-default-spelled-out",
        ),
        pytest.param("bash -en {0}", False, id="syntax-only-in-a-cluster"),
        pytest.param("bash -o noexec {0}", False, id="noexec-named-by-o"),
        pytest.param("bash -eo noexec {0}", False, id="noexec-after-errexit"),
        pytest.param('bash -e "{0}"', True, id="a-quoted-operand"),
        pytest.param("bash -- -e {0}", False, id="an-option-terminator"),
        pytest.param('bash -et "{0}"', False, id="a-quoted-one-command"),
        pytest.param("bash -et {0}", False, id="one-command-in-a-cluster"),
        pytest.param(
            "bash -o onecmd -e {0}", False, id="one-command-named-by-o"
        ),
        pytest.param("bash -c run {0}", False, id="a-command-string"),
        pytest.param("bash -eD {0}", False, id="dump-strings-in-a-cluster"),
        pytest.param(
            "bash --dump-strings -e {0}", False, id="a-dump-strings-request"
        ),
        pytest.param(
            "bash --dump-po-strings -e {0}",
            False,
            id="a-dump-po-strings-request",
        ),
        pytest.param("bash --help -e {0}", False, id="a-help-report"),
        pytest.param("bash --version -e {0}", False, id="a-version-report"),
        pytest.param("python {0}", False, id="an-unmodelled-interpreter"),
    ],
)
def test_a_template_must_run_the_script_under_errexit(
    shell: str, expected: bool
) -> None:
    """A template that runs the script without ``errexit`` proves nothing.

    The scheduling contract reads a failed command as failing the step only
    while ``errexit`` holds. ``bash {0}`` runs the script but lets a later
    success absorb the failure, so it is no more evidence than ``bash -n {0}``;
    the same applies to combined syntax-only flags, to ``-o noexec``, to ``-D``
    and its long forms, which imply ``-n`` while dumping strings, to ``-t``
    and ``-o onecmd``, which run one command and exit, and to ``--help`` or
    ``--version``, which exit successfully without running it.
    """
    assert runs_the_script(shell) is expected, (
        f"runs_the_script must return {expected} for a custom {shell!r} template"
    )
