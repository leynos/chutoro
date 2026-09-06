"""The termination allowance, driven past what this repository reaches.

Every ``grace-period`` in ``.config/nextest.toml`` is five seconds, so the
ordering contract's allowance always lands on its 60 s floor. A contract
that only ever sees the floor cannot tell the rule it states from one
that ignored the configuration: deleting the reading entirely would leave
the watchdog at 4,200 s and every assertion passing.

These drive the derivation with controlled configurations instead, so
each branch is exercised where this repository's own numbers never reach.

See "Test timeouts: four tiers, outermost last" in
``docs/developers-guide.md``.
"""

import pytest
from timeout_ordering_test import (
    largest_slow_timeout_of,
    termination_allowance_of,
)


@pytest.mark.parametrize(
    ("config_text", "expected"),
    [
        pytest.param("[profile.default]\n", 60.0, id="no-grace-period-at-all"),
        pytest.param(
            '[profile.default]\ngrace-period = "5s"\n',
            60.0,
            id="grace-period-below-the-floor",
        ),
        pytest.param(
            '[profile.default]\ngrace-period = "60s"\n',
            60.0,
            id="grace-period-at-the-floor",
        ),
        pytest.param(
            '[profile.default]\ngrace-period = "90s"\n',
            90.0,
            id="grace-period-above-the-floor",
        ),
        pytest.param(
            '[profile.default]\ngrace-period = "3m"\n',
            180.0,
            id="grace-period-in-minutes",
        ),
        pytest.param(
            '[profile.default]\ngrace-period = "5s"\n'
            '[profile.long]\ngrace-period = "2m"\n',
            120.0,
            id="largest-of-several-profiles",
        ),
    ],
)
def test_the_termination_allowance_follows_the_configured_grace_period(
    config_text: str, expected: float
) -> None:
    """A raised grace period raises the allowance; the floor catches the rest.

    This is the whole point of reading the value rather than fixing it. A
    profile that gives nextest three minutes to stop the run needs three
    minutes of watchdog to cover it, and the hard-coded 60 s this
    replaces would silently stop covering the case it exists for.
    """
    assert termination_allowance_of(config_text) == pytest.approx(expected), (
        f"{config_text!r} must yield a {expected:.0f}s termination allowance"
    )


def test_the_termination_allowance_ignores_the_per_test_period() -> None:
    """``period`` and ``grace-period`` share an inline table.

    Reading the wrong one would put a fifteen-minute per-test budget where
    a five-second termination belongs.
    """
    config_text = '[profile.default]\nslow-timeout = { period = "900s" }\n'
    assert termination_allowance_of(config_text) == pytest.approx(60.0), (
        "the termination allowance read a per-test period as a grace period"
    )


def test_the_largest_slow_timeout_ignores_the_grace_period() -> None:
    """The per-test ceiling must not read a grace period either.

    The two keys differ by a prefix, so a substring match returns
    whichever is larger. Here the grace period is deliberately the larger,
    which would hold the global timeout to a ceiling no test can spend.
    """
    config_text = (
        '[profile.default]\nslow-timeout = { period = "60s", grace-period = "30m" }\n'
    )
    assert largest_slow_timeout_of(config_text) == pytest.approx(60.0), (
        "the per-test ceiling read a grace period as a slow-timeout"
    )
