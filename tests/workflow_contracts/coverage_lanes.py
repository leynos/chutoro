"""Reads every coverage lane out of the workflow files.

Separated from ``timeout_budgets`` so the workflow reading and the
nextest arithmetic stay legible apart, and so neither module outgrows
the 400-line limit ``AGENTS.md`` sets.
"""

import collections.abc as cabc
import typing as typ

import yaml
from timeout_budgets import COVERAGE_ACTION, WATCHDOG_VARIABLE, _optional_seconds
from workflow_support import workflow_paths


class CoverageLane(typ.NamedTuple):
    """One coverage step, with the job budget that encloses it.

    Attributes
    ----------
    workflow : str
        The workflow file name.
    job : str
        The job the step belongs to.
    step : str
        The step's declared name.
    watchdog : float | None
        The step's watchdog budget in seconds, or ``None`` when it sets
        none and so inherits the action's default.
    job_timeout : float | None
        The enclosing job's ``timeout-minutes`` in seconds, or ``None``
        when the job declares none.
    condition : tuple[object, object]
        The ``if`` on the coverage step and on its job. A skipped step
        runs no ``cargo``, so its watchdog never arms and every budget
        here says nothing about it; the condition is part of what
        identifies a lane rather than incidental to it.
    """

    workflow: str
    job: str
    step: str
    watchdog: float | None
    job_timeout: float | None
    condition: tuple[object, object] = (None, None)

    def __str__(self) -> str:
        """Return a location suitable for a failure message.

        Returns
        -------
        str
            ``workflow:job:step`` for this lane.
        """
        return f"{self.workflow}:{self.job}:{self.step!r}"


def _watchdog_seconds(raw: object) -> float | None:
    """Return one source's watchdog budget, or None when it sets none.

    A blank or whitespace-only value is not a budget of zero, it is a
    source that says nothing, so it falls through to the next one. That
    is what a workflow writes when it interpolates an expression that
    resolved to nothing, and converting it directly raises before the
    contract can name the lane at fault.

    Zero and negative values are refused rather than returned. The
    shared action reads them as no timeout at all, so a lane carrying
    one has no third tier while appearing to declare one, which is the
    inversion this contract exists to catch rather than to propagate.

    Parameters
    ----------
    raw : object
        The value the workflow set, as the YAML parser returned it.

    Returns
    -------
    float | None
        The budget in seconds, or None when the source sets none.

    Raises
    ------
    ValueError
        If the value is present and non-blank but not a positive number
        of seconds.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    seconds = float(text)
    if seconds <= 0:
        message = (
            f"{WATCHDOG_VARIABLE} must be a positive number of seconds; "
            f"{raw!r} would leave the cargo invocation unbounded while "
            f"appearing to bound it"
        )
        raise ValueError(message)
    return seconds


def _watchdog_of(
    document: dict[str, typ.Any], job: dict[str, typ.Any], step: dict[str, typ.Any]
) -> float | None:
    """Return the watchdog budget in force for one coverage step.

    All three environment levels are read, innermost first, as GitHub
    resolves them. Both lanes here set the value on the step, so a
    contract reading only that scope agrees with this one today and
    would stop agreeing the moment a lane moved it to the job, reporting
    a lane that is bounded as inheriting the action's default.

    Parameters
    ----------
    document : dict[str, typ.Any]
        The whole workflow document.
    job : dict[str, typ.Any]
        The enclosing job.
    step : dict[str, typ.Any]
        The coverage step.

    Returns
    -------
    float | None
        The budget in seconds, or None when no level sets one.
    """
    for owner in (step, job, document):
        environment = owner.get("env")
        if not isinstance(environment, dict):
            continue
        seconds = _watchdog_seconds(environment.get(WATCHDOG_VARIABLE))
        if seconds is not None:
            return seconds
    return None


def _job_lanes(
    workflow: str,
    job_name: str,
    job: dict[str, typ.Any],
    document: dict[str, typ.Any] | None = None,
) -> cabc.Iterator[CoverageLane]:
    """Yield one lane per coverage step in a single job."""
    job_timeout = _optional_seconds(job.get("timeout-minutes"))
    for step in job.get("steps") or []:
        if COVERAGE_ACTION not in str(step.get("uses", "")):
            continue
        yield CoverageLane(
            workflow=workflow,
            job=str(job_name),
            step=str(step.get("name", "")) or str(job_name),
            watchdog=_watchdog_of(document or {}, job, step),
            job_timeout=job_timeout,
            condition=(step.get("if"), job.get("if")),
        )


def _lanes() -> cabc.Iterator[CoverageLane]:
    """Yield every step that invokes the shared coverage action.

    Both workflow extensions are scanned. A coverage lane in a ``.yaml``
    file would otherwise inherit the action's default watchdog without
    failing anything here.

    Yields
    ------
    CoverageLane
        One lane per coverage step, across every workflow.
    """
    for path in sorted(workflow_paths()):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        for job_name, job in (document.get("jobs") or {}).items():
            yield from _job_lanes(path.name, job_name, job, document)
