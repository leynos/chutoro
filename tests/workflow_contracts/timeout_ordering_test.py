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

import typing as typ

import pytest
from coverage_lanes import CoverageLane, _lanes
from timeout_budgets import (
    CEILING_MARGIN_SECONDS,
    COVERAGE_ACTION,
    COLD_BUILD_ALLOWANCE_SECONDS,
    NEXTEST_CONFIG,
    NON_COVERAGE_ALLOWANCE_SECONDS,
    WATCHDOG_VARIABLE,
    _seconds,
    bounds_a_single_test,
    largest_slow_timeout_of,
    parse_config,
    required_ceiling,
    termination_allowance_of,
)

#: The condition each coverage lane legitimately carries, keyed by
#: workflow and job, as the step's ``if`` and its job's.
#:
#: Neither lane carries one, and both are pinned at ``None`` rather than
#: merely unchecked. A skipped step runs no `cargo`, so its watchdog
#: never arms and every assertion below says nothing about it:
#: `if: false` on the step or on its job would leave a lane that looks
#: bounded and is not, and so would a plausible condition that quietly
#: excluded the event the lane exists for.
REQUIRED_CONDITIONS: typ.Final[dict[tuple[str, str], tuple[object, object]]] = {
    ("ci.yml", "build-test"): (None, None),
    ("coverage-main.yml", "coverage-upload"): (None, None),
}


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

    Parameters
    ----------
    nextest_config : str
        The nextest configuration file's text.

    Returns
    -------
    float
        The default profile's whole-run budget, in seconds.
    """
    profiles = parse_config(nextest_config).get("profile") or {}
    default = profiles.get("default")
    assert isinstance(default, dict), (
        "nextest.toml must declare a [profile.default] section; the ordering "
        "contract has nothing to compare against without one"
    )
    budget = default.get("global-timeout")
    assert isinstance(budget, str), (
        "[profile.default] must set global-timeout; without it the whole-run "
        "budget is unbounded and the watchdog becomes the only limit"
    )
    return _seconds(budget)


@pytest.fixture(scope="module")
def largest_slow_timeout(nextest_config: str) -> float:
    """Return the longest single-test allowance in seconds.

    Parameters
    ----------
    nextest_config : str
        The nextest configuration file's text.

    Returns
    -------
    float
        The longest per-test budget.
    """
    return largest_slow_timeout_of(nextest_config)


@pytest.fixture(scope="module")
def termination_allowance(nextest_config: str) -> float:
    """Return the time nextest may take to stop the run, in seconds.

    Read from the configuration rather than fixed, because a profile that
    raised its grace period past a hard-coded allowance would drift out
    of the requirement this contract exists to hold. The canonical
    section this repository copies says to take the allowance from
    ``slow-timeout.grace-period`` where one is set.

    Parameters
    ----------
    nextest_config : str
        The nextest configuration file's text.

    Returns
    -------
    float
        The largest configured grace period, or the floor when that is
        smaller or absent.
    """
    return termination_allowance_of(nextest_config)


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
    termination_allowance: float,
) -> None:
    """Tier three must not pre-empt tier two.

    The two clocks do not start together. The watchdog starts with
    ``cargo`` and covers the build; nextest's global timeout starts only
    once tests begin. A watchdog merely above the global timeout still
    pre-empts it whenever the build takes longer than the difference.

    The far end matters too, though less than it first appears. Hitting
    the global timeout starts nextest's termination procedure rather than
    stopping the run: on Unix it signals the process group and waits a
    grace period before killing it. That allowance is seconds, not
    minutes, but it is not zero.
    """
    required = global_timeout + termination_allowance + COLD_BUILD_ALLOWANCE_SECONDS
    for lane in lanes:
        assert lane.watchdog is not None, str(lane)
        assert lane.watchdog >= required, (
            f"{lane} sets {WATCHDOG_VARIABLE}={lane.watchdog:.0f}s, below the "
            f"{required:.0f}s needed to cover the {global_timeout:.0f}s nextest "
            f"budget, {termination_allowance:.0f}s for nextest to "
            f"terminate the run, and {COLD_BUILD_ALLOWANCE_SECONDS:.0f}s of "
            f"cold build"
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
        required = required_ceiling(lane.watchdog, NON_COVERAGE_ALLOWANCE_SECONDS)
        assert lane.job_timeout >= required, (
            f"{lane} has a job timeout of {lane.job_timeout:.0f}s, below the "
            f"{required:.0f}s needed to cover its {lane.watchdog:.0f}s watchdog "
            f"plus {NON_COVERAGE_ALLOWANCE_SECONDS:.0f}s of work outside it and "
            f"a {CEILING_MARGIN_SECONDS:.0f}s margin above that sum; an overrun "
            f"would be cancelled rather than reported"
        )


def test_each_coverage_lane_carries_the_condition_it_is_meant_to(
    lanes: tuple[CoverageLane, ...],
) -> None:
    """A skipped step runs no `cargo`, so its watchdog never arms.

    Every assertion above reads a lane's declared budgets and says
    nothing about whether the step runs. `if: false` on the step or on
    its job would leave a lane that looks bounded and is not, and this
    contract would certify it. So would a plausible condition that
    quietly excluded the event the lane exists for, which is why the
    conditions are pinned by value rather than checked for falsity:
    YAML parses `false` to a boolean, and enumerating spellings would
    miss the plausible ones anyway.

    Neither lane here carries a condition, so both are pinned at
    ``None``. The coordinates are compared both ways first, so a new
    lane with no entry fails rather than passing unexamined, and a lane
    that disappeared fails rather than being skipped.

    Proved by mutation: `if: false` on the coverage step, the same on
    its job, a push-only condition on the job, and a coordinate dropped
    from ``REQUIRED_CONDITIONS`` each fail this test.
    """
    found: dict[tuple[str, str], set[tuple[object, object]]] = {}
    for lane in lanes:
        found.setdefault((lane.workflow, lane.job), set()).add(lane.condition)
    assert set(found) == set(REQUIRED_CONDITIONS), (
        f"the coverage lanes are not the ones this contract pins: "
        f"unlisted {sorted(set(found) - set(REQUIRED_CONDITIONS))}, missing "
        f"{sorted(set(REQUIRED_CONDITIONS) - set(found))}; a lane with no "
        f"entry here is a lane whose condition nobody has judged"
    )
    wrong = {
        coordinate: (expected, found[coordinate])
        for coordinate, expected in REQUIRED_CONDITIONS.items()
        if found[coordinate] != {expected}
    }
    assert not wrong, (
        f"these coverage lanes do not carry the conditions the developers' "
        f"guide records, as expected versus found: {wrong}; a lane that is "
        f"skipped runs no cargo, so its watchdog never arms"
    )


def test_the_default_profile_bounds_a_test_no_override_matches(
    nextest_config: str,
) -> None:
    """An override bounds its filter's tests; the profile bounds the rest.

    `largest_slow_timeout_of` reports the largest budget anywhere in the
    file, so deleting `[profile.default]`'s own `slow-timeout` and
    leaving the fifteen overrides behind still reports a comfortable
    number while every test none of them matches runs with no bound at
    all. Nothing else here would notice.

    `[profile.ci]` declares none of its own and does not need to:
    nextest's other profiles inherit the default profile's own keys.

    Proved by mutation: commenting out the default profile's own
    `slow-timeout` fails this test and nothing else.
    """
    assert bounds_a_single_test(nextest_config), (
        "[profile.default] itself must set slow-timeout with terminate-after; "
        "an override satisfies the file as a whole while leaving every test it "
        "does not match unbounded"
    )
