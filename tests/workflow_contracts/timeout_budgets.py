"""Reading the timers that can end a test run.

The contract in ``timeout_ordering_test`` compares budgets written down
in three different files. Turning those files into comparable seconds is
the part that can be wrong without any file being wrong, so it lives
here where it can be read on its own, and is driven past this
repository's own numbers in ``timeout_derivation_test``.

See "Test timeouts: four tiers, outermost last" in
``docs/developers-guide.md``.
"""

import re
import tomllib
import typing as typ
from pathlib import Path

from workflow_support import ROOT

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

#: What nextest allows a test between `SIGTERM` and `SIGKILL` when the
#: configuration names no `grace-period`.
NEXTEST_DEFAULT_GRACE_PERIOD_SECONDS: typ.Final[float] = 10.0

#: Added to that grace period to cover the teardown and report writing
#: that follow it. A separate term rather than a floor over the two, so
#: raising a grace period raises the requirement instead of vanishing
#: into it.
TERMINATION_SAFETY_MARGIN_SECONDS: typ.Final[float] = 60.0

#: Everything in the job that is not the coverage step. The job timer
#: covers it; the watchdog does not. Measured at 6 m 08 s before and
#: 16 s after on run 33939048036.
NON_COVERAGE_ALLOWANCE_SECONDS: typ.Final[float] = 15 * 60.0

#: How far a ceiling must sit above the sum it contains, rather than
#: merely reaching it. A ceiling equal to that sum cancels the job at
#: the moment the watchdog would have reported the overrun, and the
#: report is the only thing that makes an overrun actionable, so
#: equality buys nothing: it converts a legible failure into a
#: cancellation with no log. This lane is always the cold writer, so
#: that is the likely case rather than the remote one.
CEILING_MARGIN_SECONDS: typ.Final[float] = 15 * 60.0

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


def required_ceiling(watchdog: float, allowance: float) -> float:
    """Return the smallest acceptable job ceiling, in seconds.

    Three terms. The watchdog is what one `cargo` invocation may
    legitimately spend. The allowance is the measured work either side
    of it, which the job timer covers and the watchdog does not. The
    margin is added because a ceiling equal to that sum cancels the job
    at the moment the watchdog would have reported the overrun, and the
    report is the only thing that makes an overrun actionable.

    Parameters
    ----------
    watchdog : float
        The coverage step's watchdog budget, in seconds.
    allowance : float
        The measured work outside that window, in seconds.

    Returns
    -------
    float
        The smallest acceptable ceiling, in seconds.
    """
    return watchdog + allowance + CEILING_MARGIN_SECONDS


def _slow_timeouts(config: dict[str, typ.Any]) -> list[dict[str, typ.Any]]:
    """Return every ``slow-timeout`` table the configuration sets.

    Both the profiles' own and their overrides', because an override is
    where the longest allowances live.

    Parameters
    ----------
    config : dict[str, typ.Any]
        A parsed nextest configuration.

    Returns
    -------
    list[dict[str, typ.Any]]
        The inline tables, in no particular order.
    """
    return [
        table
        for section in _budget_sections(config)
        if isinstance(table := section.get("slow-timeout"), dict)
    ]


def _budget_sections(config: dict[str, typ.Any]) -> list[dict[str, typ.Any]]:
    """Return every section that may declare a budget.

    A profile and each of its overrides are the same shape as far as
    this contract is concerned: a mapping that may carry a
    ``slow-timeout``. Flattening them here is what lets the reading
    above be one comprehension rather than a loop inside a loop.

    Parameters
    ----------
    config : dict[str, typ.Any]
        A parsed nextest configuration.

    Returns
    -------
    list[dict[str, typ.Any]]
        Each profile followed by its overrides, in no particular order.
    """
    profiles = [
        profile
        for profile in (config.get("profile") or {}).values()
        if isinstance(profile, dict)
    ]
    return [
        section
        for profile in profiles
        for section in (profile, *(profile.get("overrides") or []))
        if isinstance(section, dict)
    ]


def parse_config(config_text: str) -> dict[str, typ.Any]:
    """Parse a nextest configuration.

    Parsed rather than matched. A commented-out
    ``grace-period = "30m"`` reads as an active value to a regular
    expression, so a line nobody meant would inflate the termination
    allowance and fail this contract without changing what nextest does.
    TOML is the only reading that distinguishes the two.

    Parameters
    ----------
    config_text : str
        A nextest configuration file's text.

    Returns
    -------
    dict[str, typ.Any]
        The parsed document.
    """
    return tomllib.loads(config_text)


def largest_slow_timeout_of(config_text: str) -> float:
    """Return the longest per-test allowance a configuration sets.

    The budget a test gets is ``period`` multiplied by
    ``terminate-after``: nextest warns once per period and terminates
    after that many of them. Every multiplier here is one, so reading the
    period alone gives the same answer against this file and a different
    one the moment somebody raises a multiplier.

    Separated from the fixture so it can be driven with configurations
    this repository does not have.

    Parameters
    ----------
    config_text : str
        A nextest configuration file's text.

    Returns
    -------
    float
        The longest per-test budget.
    """
    budgets: list[float] = []
    for table in _slow_timeouts(parse_config(config_text)):
        period = table.get("period")
        if not isinstance(period, str):
            continue
        terminate = table.get("terminate-after")
        multiplier = terminate if isinstance(terminate, int) else 1
        budgets.append(_seconds(period) * multiplier)
    assert budgets, "nextest.toml must set at least one slow-timeout period"
    return max(budgets)


def bounds_a_single_test(config_text: str, profile: str = "default") -> bool:
    """Return whether a profile's own table terminates a slow test.

    Only the profile's own ``slow-timeout`` counts. An override bounds
    the tests its filter matches; the profile's own bounds the rest, so
    a profile whose only ``terminate-after`` sits in an override leaves
    every unmatched test running with no bound at all while
    :func:`largest_slow_timeout_of` still reports a comfortable number.

    nextest's other profiles inherit ``[profile.default]``'s own keys,
    so a profile that declares no ``slow-timeout`` of its own is bounded
    by the default's rather than unbounded; only the default profile's
    absence is a hole.

    Parameters
    ----------
    config_text : str
        The nextest configuration file's text.
    profile : str
        The profile to read.

    Returns
    -------
    bool
        True when that profile's own ``slow-timeout`` is a table setting
        ``terminate-after``.
    """
    profiles = parse_config(config_text).get("profile")
    own = profiles.get(profile) if isinstance(profiles, dict) else None
    table = own.get("slow-timeout") if isinstance(own, dict) else None
    return isinstance(table, dict) and table.get("terminate-after") is not None


def termination_allowance_of(config_text: str) -> float:
    """Return the termination allowance a configuration implies.

    Two terms, not one: what nextest promises a test after ``SIGTERM``,
    plus a margin for the teardown and report writing that follow it. A
    single floor over the two would absorb every grace period below the
    margin, so raising this file's five seconds to thirty would demand
    nothing more of the watchdog above it.

    Separated from the fixture so it can be driven with configurations
    this repository does not have. Every ``grace-period`` here is five
    seconds, so the fixture only ever sees one value, and a contract
    that sees only that cannot tell this rule from one that ignored the
    configuration entirely.

    Parameters
    ----------
    config_text : str
        A nextest configuration file's text.

    Returns
    -------
    float
        The largest configured grace period, or nextest's default when
        none is set, plus the safety margin.
    """
    periods = [
        table["grace-period"]
        for table in _slow_timeouts(parse_config(config_text))
        if isinstance(table.get("grace-period"), str)
    ]
    largest = max(
        (_seconds(period) for period in periods),
        default=NEXTEST_DEFAULT_GRACE_PERIOD_SECONDS,
    )
    return largest + TERMINATION_SAFETY_MARGIN_SECONDS
