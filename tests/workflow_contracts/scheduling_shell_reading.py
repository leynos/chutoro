"""Read which shell executes a step's script, and whether it runs it.

GitHub lets a workflow, a job, or a step name the shell a ``run`` script is
handed to. A custom template such as ``bash -n {0}`` parses that script and
executes none of it, so a step carrying one would satisfy every other check
while the gate never ran. Resolving the shell that actually applies means
reading all three scopes in override order and then reading the template for
an option that only checks syntax.

This module holds that reading. ``scheduling_ci_test`` holds the assertions
over it, so neither file outgrows the 400-line limit ``AGENTS.md`` sets.

Re-use policy: import from ``scheduling_ci_test`` only. A new shell-template
rule belongs here with the GitHub behaviour it models stated beside it, so a
later reader can check the rule against the runner rather than against the
tests.
"""

from __future__ import annotations

#: Shell keywords GitHub runs with its default execution flags.
EXECUTING_SHELLS = frozenset({"bash", "sh"})

#: Shell options that parse a script and execute none of it.
SYNTAX_ONLY_OPTIONS = frozenset({"-n", "--noexec"})


def _default_run_shell(scope: dict[str, object]) -> str | None:
    """Return the ``defaults.run.shell`` a workflow or job declares, if any.

    Examples
    --------
    >>> _default_run_shell({"defaults": {"run": {"shell": "bash"}}})
    'bash'
    >>> _default_run_shell({}) is None
    True
    """
    defaults = scope.get("defaults")
    run = defaults.get("run") if isinstance(defaults, dict) else None
    shell = run.get("shell") if isinstance(run, dict) else None
    return shell if isinstance(shell, str) else None


def effective_shell(
    step: dict[str, object],
    job_definition: dict[str, object],
    workflow: dict[str, object],
) -> str | None:
    """Return the shell `step` runs under, resolving the three scopes.

    A step's own ``shell`` overrides its job's ``defaults.run.shell``, which
    overrides the workflow's. With none of the three declared, the runner's
    default shell executes the script.

    Examples
    --------
    >>> effective_shell({"shell": "bash -n {0}"}, {}, {})
    'bash -n {0}'
    >>> effective_shell({}, {"defaults": {"run": {"shell": "bash"}}}, {})
    'bash'
    >>> effective_shell({}, {}, {"defaults": {"run": {"shell": "sh"}}})
    'sh'
    >>> effective_shell({}, {}, {}) is None
    True
    """
    for shell in (
        step.get("shell"),
        _default_run_shell(job_definition),
        _default_run_shell(workflow),
    ):
        if isinstance(shell, str):
            return shell
    return None


def runs_the_script(shell: str | None) -> bool:
    """Return whether `shell` executes the step's script.

    A custom shell template must contain ``{0}``, where GitHub substitutes the
    script's path, and must not hand the shell an option that only parses the
    script: ``bash -n {0}`` and ``sh --noexec {0}`` check syntax and run
    nothing.

    Examples
    --------
    >>> runs_the_script(None)
    True
    >>> runs_the_script("bash")
    True
    >>> runs_the_script("bash --noprofile --norc -eo pipefail {0}")
    True
    >>> runs_the_script("bash -n {0}")
    False
    >>> runs_the_script("bash -n")
    False
    """
    if shell is None or shell in EXECUTING_SHELLS:
        return True
    if "{0}" not in shell:
        return False
    command = shell.split("{0}", 1)[0]
    return not any(option in SYNTAX_ONLY_OPTIONS for option in command.split())
