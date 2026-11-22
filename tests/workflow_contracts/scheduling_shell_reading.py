"""Resolve which shell a CI step's script is handed to.

GitHub lets a workflow, a job, or a step name the shell a ``run`` script is
handed to, and the nearest scope wins: a step's own ``shell`` overrides its
job's ``defaults.run.shell``, which overrides the workflow's. With none of the
three declared, the runner's default shell executes the script.

This module holds that resolution. The template reading that decides whether
the resolved shell runs the script under ``errexit`` lives in
``scheduling_template_reading``, and the split keeps each file inside the
400-line limit ``AGENTS.md`` sets.

Re-use policy: import from ``scheduling_shell_test`` and ``scheduling_ci_test``
only, which read both the resolved shell and the template verdict.
"""

from __future__ import annotations

from scheduling_template_reading import EXECUTING_SHELLS, runs_the_script

__all__ = ["EXECUTING_SHELLS", "effective_shell", "runs_the_script"]


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
