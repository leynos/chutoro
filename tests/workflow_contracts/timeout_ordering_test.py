"""Contract for the four timers that can end a test run.

Four independent budgets bound a coverage lane, and each is set in a
different place: a per-test ``slow-timeout`` and a whole-run
``global-timeout`` in ``.config/nextest.toml``, a wall-clock watchdog on
the ``cargo`` invocation in the workflow, and the job's own
``timeout-minutes``. They only work if each sits above the one inside it.

The ordering was inverted here and nothing said so. The shared coverage
action kills ``cargo`` after 1,800 s by default, this repository never
set the value, and nextest is configured for a 40 m run. A cold compile
would have been killed by a budget the repository had not chosen, does
not mention, and could not see, and the failure would have named
``cargo`` rather than the test still running. rstest-bdd hit exactly that
on 2026-09-05.

Two of the four timers do not start with the others, which the ordering
has to allow for. The watchdog starts when ``cargo`` starts and covers
the build; nextest's global timeout starts only once tests begin. The job
timer starts when the job starts, before the linting and formatting that
precede coverage. Comparing the configured numbers alone would call an
inverted lane correct, so the allowances below are measured.

See "Test timeouts: four tiers, outermost last" in
``docs/developers-guide.md``.
"""

import collections.abc as cabc
import re
import typing as typ
from pathlib import Path

import pytest
import yaml
from workflow_support import ROOT, WORKFLOW_DIR

NEXTEST_CONFIG: typ.Final[Path] = ROOT / ".config" / "nextest.toml"

#: The environment variable the shared coverage action reads.
WATCHDOG_VARIABLE: typ.Final[str] = "RUN_RUST_CARGO_WAIT_TIMEOUT"

#: The action whose steps must declare a watchdog budget.
COVERAGE_ACTION: typ.Final[str] = "shared-actions/.github/actions/generate-coverage"

#: Build time inside the `cargo` invocation, before nextest starts its
#: own clock. The watchdog covers it; the global timeout does not.
#: Fifteen minutes is far above anything measured here, where the whole
#: coverage step runs in under four.
COLD_BUILD_ALLOWANCE_SECONDS: typ.Final[float] = 15 * 60.0

#: Everything in the job that is not the coverage step. The job timer
#: covers it; the watchdog does not. Measured at 6 m 08 s before and
#: 16 s after on run 33939048036.
NON_COVERAGE_ALLOWANCE_SECONDS: typ.Final[float] = 15 * 60.0

#: ``30s``, ``5m``, ``20 m``: the durations nextest accepts here.
_DURATION: typ.Final[re.Pattern[str]] = re.compile(
    r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>ms|s|m|h)\s*$"
)

_UNIT_SECONDS: typ.Final[dict[str, float]] = {
    "ms": 0.001,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
}


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
    """

    workflow: str
    job: str
    step: str
    watchdog: float | None
    job_timeout: float | None

    def __str__(self) -> str:
        """Return a location suitable for a failure message.

        Returns
        -------
        str
            ``workflow:job:step`` for this lane.
        """
        return f"{self.workflow}:{self.job}:{self.step!r}"


def _seconds(duration: str) -> float:
    """Convert a nextest duration to seconds.

    Parameters
    ----------
    duration : str
        A duration as nextest spells it, such as ``"40m"``.

    Returns
    -------
    float
        The duration in seconds.
    """
    match = _DURATION.match(duration)
    assert match is not None, f"unrecognized nextest duration {duration!r}"
    return float(match["value"]) * _UNIT_SECONDS[match["unit"]]


def _optional_seconds(value: object) -> float | None:
    """Return a ``timeout-minutes`` value in seconds, or None.

    Parameters
    ----------
    value : object
        The declared value, or ``None`` when the job declares none.

    Returns
    -------
    float | None
        The budget in seconds.
    """
    return None if value is None else float(str(value)) * 60.0


def _lanes() -> cabc.Iterator[CoverageLane]:
    """Yield every step that invokes the shared coverage action.

    Yields
    ------
    CoverageLane
        One lane per coverage step, across every workflow.
    """
    for path in sorted(WORKFLOW_DIR.glob("*.yml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        for job_name, job in (document.get("jobs") or {}).items():
            for step in job.get("steps") or []:
                if COVERAGE_ACTION not in str(step.get("uses", "")):
                    continue
                raw = (step.get("env") or {}).get(WATCHDOG_VARIABLE)
                yield CoverageLane(
                    workflow=path.name,
                    job=str(job_name),
                    step=str(step.get("name", "")) or str(job_name),
                    watchdog=None if raw is None else float(str(raw)),
                    job_timeout=_optional_seconds(job.get("timeout-minutes")),
                )


@pytest.fixture(scope="module")
def lanes() -> tuple[CoverageLane, ...]:
    """Return every coverage lane in the repository.

    Returns
    -------
    tuple[CoverageLane, ...]
        One entry per coverage step.
    """
    return tuple(_lanes())


@pytest.fixture(scope="module")
def nextest_config() -> str:
    """Return the nextest configuration file's text.

    Returns
    -------
    str
        The file's contents.
    """
    return NEXTEST_CONFIG.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def global_timeout(nextest_config: str) -> float:
    """Return the default profile's ``global-timeout`` in seconds.

    Read textually rather than through a TOML parser, because the value
    must be matched to the profile it belongs to and the file declares
    more than one profile.

    Returns
    -------
    float
        The default profile's whole-run budget, in seconds.
    """
    blocks = re.split(r"^\[profile\.", nextest_config, flags=re.MULTILINE)
    default = next((block for block in blocks if block.startswith("default]")), None)
    assert default is not None, (
        "nextest.toml must declare a [profile.default] section; the ordering "
        "contract has nothing to compare against without one"
    )
    match = re.search(r'^global-timeout\s*=\s*"([^"]+)"', default, re.MULTILINE)
    assert match is not None, (
        "[profile.default] must set global-timeout; without it the whole-run "
        "budget is unbounded and the watchdog becomes the only limit"
    )
    return _seconds(match[1])


@pytest.fixture(scope="module")
def largest_slow_timeout(nextest_config: str) -> float:
    """Return the longest single-test allowance in seconds.

    Returns
    -------
    float
        The longest per-test budget.
    """
    periods = re.findall(r'period\s*=\s*"([^"]+)"', nextest_config)
    assert periods, "nextest.toml must set at least one slow-timeout period"
    return max(_seconds(period) for period in periods)


def test_every_coverage_step_declares_a_watchdog_budget(
    lanes: tuple[CoverageLane, ...],
) -> None:
    """The default is invisible, so every step must write it down.

    This repository set no value at all, so its lanes ran under a budget
    nobody here had chosen. Asserting that some step sets it would not
    do: one step losing its override is enough to bring that back.
    """
    assert lanes, (
        f"no workflow invokes {COVERAGE_ACTION}; this contract has nothing to "
        f"assert against"
    )
    missing = [str(lane) for lane in lanes if lane.watchdog is None]
    assert not missing, (
        f"these coverage steps do not set {WATCHDOG_VARIABLE} and so inherit "
        f"the action's undocumented 1,800 s default: {missing}"
    )


def test_the_watchdog_covers_the_nextest_budget_and_the_build(
    lanes: tuple[CoverageLane, ...],
    global_timeout: float,
) -> None:
    """Tier three must not pre-empt tier two.

    The two clocks do not start together. The watchdog starts with
    ``cargo`` and covers the build; nextest's global timeout starts only
    once tests begin. A watchdog merely above the global timeout still
    pre-empts it whenever the build takes longer than the difference.
    """
    required = global_timeout + COLD_BUILD_ALLOWANCE_SECONDS
    for lane in lanes:
        assert lane.watchdog is not None, str(lane)
        assert lane.watchdog >= required, (
            f"{lane} sets {WATCHDOG_VARIABLE}={lane.watchdog:.0f}s, below the "
            f"{required:.0f}s needed to cover the {global_timeout:.0f}s nextest "
            f"budget plus {COLD_BUILD_ALLOWANCE_SECONDS:.0f}s of cold build"
        )


def test_the_nextest_global_timeout_sits_above_the_largest_slow_timeout(
    global_timeout: float,
    largest_slow_timeout: float,
) -> None:
    """Tier two must not pre-empt tier one.

    A global timeout below the longest per-test allowance kills the run
    before the test that allowance exists for can finish.
    """
    assert global_timeout > largest_slow_timeout, (
        f"the {global_timeout:.0f}s global-timeout is not above the "
        f"{largest_slow_timeout:.0f}s largest per-test slow-timeout; the run "
        f"would end before that test could use its budget"
    )


def test_the_job_timeout_covers_the_watchdog_and_the_rest_of_the_job(
    lanes: tuple[CoverageLane, ...],
) -> None:
    """Tier four must not pre-empt tier three.

    Compared per job, not against the tightest budget in the repository:
    the Verus job's own ceiling has nothing to do with the coverage
    lane's, and comparing them would either fail honestly-sized jobs or
    force unrelated budgets to move together.
    """
    for lane in lanes:
        assert lane.watchdog is not None, str(lane)
        assert lane.job_timeout is not None, (
            f"{lane} runs cargo under a {lane.watchdog:.0f}s watchdog in a job "
            f"with no timeout-minutes; the outermost tier is missing"
        )
        required = lane.watchdog + NON_COVERAGE_ALLOWANCE_SECONDS
        assert lane.job_timeout >= required, (
            f"{lane} has a job timeout of {lane.job_timeout:.0f}s, below the "
            f"{required:.0f}s needed to cover its {lane.watchdog:.0f}s watchdog "
            f"plus {NON_COVERAGE_ALLOWANCE_SECONDS:.0f}s of work outside it; an "
            f"overrun would be cancelled rather than reported"
        )
