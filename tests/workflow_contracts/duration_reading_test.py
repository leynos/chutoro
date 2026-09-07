"""How a nextest duration is read, driven past this repository's values.

Every period in `.config/nextest.toml` is a whole number of seconds
written with a single unit, so the reading agrees with several wrong
ones against the real file. nextest parses durations with `humantime`,
which reads a sequence of value-and-unit pairs and sums them, so these
drive the reading with the composite and long-spelled forms the runner
accepts and this repository does not use.

The refusal of a ``slow-timeout`` that terminates nothing is here for
the same reason: every table in this repository sets ``terminate-after``
explicitly, so the case can only be driven with a configuration written
for it.
"""

import pytest
from nextest_durations import NextestDurationError, _seconds
from timeout_budgets import UnboundedTestError, largest_slow_timeout_of


@pytest.mark.parametrize(
    ("duration", "expected"),
    [
        pytest.param("2h 37min", 2 * 3600.0 + 37 * 60.0, id="composite-with-a-space"),
        pytest.param("1m30s", 90.0, id="composite-without-a-space"),
        pytest.param("500ms", 0.5, id="milliseconds"),
        pytest.param("15sec", 15.0, id="a-long-unit-spelling"),
        pytest.param("2minutes", 120.0, id="a-plural-unit-spelling"),
        pytest.param("1h 0m 30s", 3630.0, id="three-terms"),
    ],
)
def test_every_duration_humantime_accepts_is_read(
    duration: str, expected: float
) -> None:
    """nextest parses durations with `humantime`, which sums its terms.

    A parser reading one value and one unit refuses `2h 37min`, which is
    valid configuration, so the contract would fail a repository whose
    timeouts were fine. The long and plural unit spellings are here for
    the same reason: `humantime` accepts them and a reader that did not
    would call a working file broken.
    """
    assert _seconds(duration) == pytest.approx(expected), (
        f"{duration!r} must read as {expected}s, as humantime reads it"
    )


@pytest.mark.parametrize(
    "duration",
    [
        pytest.param("1.5m", id="a-decimal-humantime-refuses"),
        pytest.param("30", id="no-unit"),
        pytest.param("30d 5", id="a-trailing-value-with-no-unit"),
        pytest.param("30 fortnights", id="a-unit-humantime-does-not-know"),
        pytest.param("", id="empty"),
        pytest.param("   ", id="whitespace-only"),
    ],
)
def test_a_duration_humantime_would_refuse_is_refused(duration: str) -> None:
    """What the runner cannot parse, this contract must not guess at.

    `humantime` takes whole numbers only, so `1.5m` is not a shorter way
    of writing 90 seconds; it is a configuration nextest rejects. Giving
    it a value here would compare the tiers against a budget that never
    applies.
    """
    with pytest.raises(NextestDurationError):
        _seconds(duration)


@pytest.mark.parametrize(
    "config_text",
    [
        pytest.param(
            '[profile.default]\nslow-timeout = { period = "180s" }\n',
            id="a-table-without-terminate-after",
        ),
        pytest.param(
            "[profile.default]\n"
            'slow-timeout = { period = "180s", grace-period = "5s" }\n',
            id="a-table-with-only-a-grace-period",
        ),
        pytest.param(
            "[profile.default]\n"
            'slow-timeout = { period = "60s", terminate-after = 1 }\n'
            "\n[[profile.default.overrides]]\n"
            "filter = 'binary(slow)'\n"
            'slow-timeout = { period = "600s" }\n',
            id="an-override-without-terminate-after",
        ),
    ],
)
def test_a_slow_timeout_that_terminates_nothing_is_refused(config_text: str) -> None:
    """`terminate-after` is optional, and without it nothing is bounded.

    cargo-nextest treats an omitted `terminate-after` as no termination:
    the test is reported slow, once per period, and runs on. Reading
    that as a single period would put a number on the tier that is
    missing, so the ordering above it would be compared against a budget
    nextest never applies and would pass.

    Every table in `.config/nextest.toml` sets it explicitly, so no
    value in this repository changes; this is what stops one appearing.
    """
    with pytest.raises(UnboundedTestError, match=r"terminate-after"):
        largest_slow_timeout_of(config_text)
