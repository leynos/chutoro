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
from timeout_budgets import (
    TerminateAfterError,
    UnboundedTestError,
    largest_slow_timeout_of,
)


@pytest.mark.parametrize(
    ("duration", "expected"),
    [
        pytest.param("2h 37min", 2 * 3600.0 + 37 * 60.0, id="composite-with-a-space"),
        pytest.param("1m30s", 90.0, id="composite-without-a-space"),
        pytest.param("500ms", 0.5, id="milliseconds"),
        pytest.param("15sec", 15.0, id="a-long-unit-spelling"),
        pytest.param("2minutes", 120.0, id="a-plural-unit-spelling"),
        pytest.param("1h 0m 30s", 3630.0, id="three-terms"),
        pytest.param("1.5m", 90.0, id="a-fractional-value"),
        pytest.param("0.5s", 0.5, id="a-fraction-below-one"),
        pytest.param("1 . 5 m", 90.0, id="a-fraction-spaced-around-the-point"),
        pytest.param("1wk", 604800.0, id="the-abbreviated-week"),
        pytest.param("2wks", 1209600.0, id="the-abbreviated-plural-week"),
        pytest.param("1yr", 31557600.0, id="the-abbreviated-year"),
        pytest.param("3yrs", 94672800.0, id="the-abbreviated-plural-year"),
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

    The fractional values and the `wk`, `wks`, `yr` and `yrs` aliases
    were measured against humantime 2.4.0, the version cargo-nextest
    resolves through humantime_serde, rather than assumed: this reader
    had refused all of them.
    """
    assert _seconds(duration) == pytest.approx(expected), (
        f"{duration!r} must read as {expected}s, as humantime reads it"
    )


@pytest.mark.parametrize(
    "duration",
    [
        pytest.param("30", id="no-unit"),
        pytest.param("30d 5", id="a-trailing-value-with-no-unit"),
        pytest.param("30 fortnights", id="a-unit-humantime-does-not-know"),
        pytest.param(".5s", id="a-fraction-with-no-whole-part"),
        pytest.param("1.s", id="a-point-with-no-fraction-after-it"),
        pytest.param("1.5.5m", id="two-points"),
        pytest.param("-1m", id="a-negative-value"),
        pytest.param("1_000s", id="a-digit-separator"),
        pytest.param("", id="empty"),
        pytest.param("   ", id="whitespace-only"),
    ],
)
def test_a_duration_humantime_would_refuse_is_refused(duration: str) -> None:
    """What the runner cannot parse, this contract must not guess at.

    Reading a duration nextest rejects would compare the tiers against a
    budget that never applies. Each of these was checked against
    humantime 2.4.0 and refused there: a fraction needs a whole part
    before the point and a digit after it, values are unsigned, and the
    only separators are whitespace.
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


@pytest.mark.parametrize(
    "terminate_after",
    [
        pytest.param("0", id="zero"),
        pytest.param("-1", id="a-negative"),
        pytest.param("1.5", id="a-fraction"),
        pytest.param("true", id="a-boolean"),
        pytest.param('"5"', id="a-quoted-number"),
    ],
)
def test_a_terminate_after_nextest_would_refuse_is_refused(
    terminate_after: str,
) -> None:
    """cargo-nextest reads the field as `Option<NonZeroUsize>`.

    Converting the text of whatever the document held accepted every one
    of these, and each names a configuration the runner refuses to start
    with. Zero is the dangerous one: read as a multiplier it makes the
    per-test allowance vanish, and every comparison above it then passes
    against nothing.
    """
    config_text = (
        "[profile.default]\n"
        f'slow-timeout = {{ period = "60s", terminate-after = {terminate_after} }}\n'
    )

    with pytest.raises(TerminateAfterError, match=r"positive integer"):
        largest_slow_timeout_of(config_text)
