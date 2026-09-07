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

import typing as typ

import pytest
from coverage_lanes import _watchdog_of
from hypothesis import given
from hypothesis import strategies as st
from nextest_durations import (
    _seconds,
)
from timeout_budgets import (
    CEILING_MARGIN_SECONDS,
    NEXTEST_DEFAULT_GRACE_PERIOD_SECONDS,
    TERMINATION_SAFETY_MARGIN_SECONDS,
    largest_slow_timeout_of,
    required_ceiling,
    termination_allowance_of,
)


@pytest.mark.parametrize(
    ("config_text", "expected"),
    [
        pytest.param(
            '[profile.default]\nslow-timeout = { period = "60s" }\n',
            70.0,
            id="no-grace-period-falls-back-to-nextest-s-default",
        ),
        pytest.param(
            "[profile.default]\n"
            'slow-timeout = { period = "60s", grace-period = "5s" }\n',
            65.0,
            id="a-short-grace-period-still-counts",
        ),
        pytest.param(
            "[profile.default]\n"
            'slow-timeout = { period = "60s", grace-period = "60s" }\n',
            120.0,
            id="a-grace-period-equal-to-the-margin",
        ),
        pytest.param(
            "[profile.default]\n"
            'slow-timeout = { period = "60s", grace-period = "90s" }\n',
            150.0,
            id="a-long-grace-period",
        ),
        pytest.param(
            "[profile.default]\n"
            'slow-timeout = { period = "60s", grace-period = "3m" }\n',
            240.0,
            id="grace-period-in-minutes",
        ),
        pytest.param(
            "[profile.default]\n"
            'slow-timeout = { period = "60s", grace-period = "5s" }\n'
            "[profile.long]\n"
            'slow-timeout = { period = "60s", grace-period = "2m" }\n',
            180.0,
            id="largest-of-several-profiles",
        ),
        pytest.param(
            "[profile.default]\n"
            'slow-timeout = { period = "60s", grace-period = "5s" }\n'
            "[[profile.default.overrides]]\n"
            'filter = "all()"\n'
            'slow-timeout = { period = "60s", grace-period = "4m" }\n',
            300.0,
            id="the-longest-lives-in-an-override",
        ),
        pytest.param(
            "[profile.default]\n"
            'slow-timeout = { period = "60s", grace-period = "5s" }\n'
            '# slow-timeout = { period = "60s", grace-period = "30m" }\n',
            65.0,
            id="a-commented-grace-period-is-not-a-value",
        ),
    ],
)
def test_the_termination_allowance_follows_the_configured_grace_period(
    config_text: str, expected: float
) -> None:
    """A raised grace period raises the allowance, by the same amount.

    This is the whole point of reading the value rather than fixing it. A
    profile that gives nextest three minutes to stop the run needs three
    minutes of watchdog to cover it, plus the margin, and the hard-coded
    60 s this replaces would silently stop covering the case it exists
    for. The allowance is the two terms added, so a grace period below
    the margin is not absorbed by it.

    The configurations are shaped as nextest accepts them, with the grace
    period inside the ``slow-timeout`` table rather than beside it, so
    the reading is exercised against the structure it will meet. The last
    case is the one a regular expression got wrong: a commented-out grace
    period is not a value.
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
    assert termination_allowance_of(config_text) == pytest.approx(70.0), (
        "the termination allowance read a per-test period as a grace period"
    )


@pytest.mark.parametrize(
    ("config_text", "expected"),
    [
        pytest.param(
            "[profile.default]\n"
            'slow-timeout = { period = "180s", terminate-after = 1 }\n',
            180.0,
            id="a-multiplier-of-one",
        ),
        pytest.param(
            "[profile.default]\n"
            'slow-timeout = { period = "60s", terminate-after = 5 }\n',
            300.0,
            id="five-warning-periods",
        ),
        pytest.param(
            "[profile.default]\n"
            'slow-timeout = { period = "2m", terminate-after = 3 }\n',
            360.0,
            id="minutes-times-three",
        ),
    ],
)
def test_the_largest_slow_timeout_counts_the_multiplier(
    config_text: str, expected: float
) -> None:
    """``terminate-after`` scales the period; the budget is their product.

    nextest warns once per period and terminates after that many of them,
    so a test given five sixty-second periods may run for five minutes.
    Every multiplier in this repository is one, so a reading that ignored
    it entirely would give the same answer against the real file and a
    wrong one the moment somebody raises a multiplier. That is why this
    is driven with configurations this repository does not have.
    """
    assert largest_slow_timeout_of(config_text) == pytest.approx(expected), (
        f"{config_text!r} must yield a {expected:.0f}s largest per-test "
        f"allowance; terminate-after scales the period"
    )


def test_the_largest_slow_timeout_ignores_the_grace_period() -> None:
    """The per-test ceiling must not read a grace period either.

    The two keys differ by a prefix, so a substring match returns
    whichever is larger. Here the grace period is deliberately the larger,
    which would hold the global timeout to a ceiling no test can spend.
    """
    config_text = (
        "[profile.default]\n"
        'slow-timeout = { period = "60s", terminate-after = 1, '
        'grace-period = "30m" }\n'
    )
    assert largest_slow_timeout_of(config_text) == pytest.approx(60.0), (
        "the per-test ceiling read a grace period as a slow-timeout"
    )


_UNITS: typ.Final[dict[str, float]] = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}

_GRACE_SECONDS = st.integers(min_value=0, max_value=7200)
_PERIOD_SECONDS = st.integers(min_value=1, max_value=7200)
_MULTIPLIERS = st.integers(min_value=1, max_value=20)


def _profile_text(entries: list[tuple[int, int, int | None]]) -> str:
    """Render slow-timeout tables as a nextest configuration.

    Parameters
    ----------
    entries : list[tuple[int, int, int | None]]
        One tuple per table: period in seconds, terminate-after, and a
        grace period in seconds or None for a table that sets none.

    Returns
    -------
    str
        A configuration nextest would accept.
    """
    lines = ["[profile.default]"]
    for index, (period, multiplier, grace) in enumerate(entries):
        if index:
            lines.extend(["", "[[profile.default.overrides]]", 'filter = "all()"'])
        table = f'period = "{period}s", terminate-after = {multiplier}'
        if grace is not None:
            table += f', grace-period = "{grace}s"'
        lines.append(f"slow-timeout = {{ {table} }}")
    return "\n".join(lines) + "\n"


@given(
    value=st.integers(min_value=0, max_value=100_000),
    unit=st.sampled_from(sorted(_UNITS)),
)
def test_a_duration_converts_to_its_unit_times_its_value(value: int, unit: str) -> None:
    """Every duration nextest accepts converts by its unit alone.

    Stated as a property because the four units are a fixed set and the
    values are not: a table of examples fixes which numbers were tried,
    and the failure this guards against is a unit applied to the wrong
    magnitude, which any single example can miss.
    """
    assert _seconds(f"{value}{unit}") == pytest.approx(value * _UNITS[unit]), (
        f"{value}{unit} must convert to {value * _UNITS[unit]}s"
    )


@given(
    entries=st.lists(
        st.tuples(_PERIOD_SECONDS, _MULTIPLIERS, st.none() | _GRACE_SECONDS),
        min_size=1,
        max_size=6,
    )
)
def test_the_termination_allowance_is_the_grace_period_plus_the_margin(
    entries: list[tuple[int, int, int | None]],
) -> None:
    """Two terms added, and the property is that neither is absorbed.

    A single floor over the grace period and the margin would swallow
    every grace period below the margin, so raising one would look free
    until the run it cancelled. Adding them keeps a raised grace period
    visible in the requirement, and the generator produces the cases
    where the two readings differ without anyone having to think of
    them: several tables, some naming no grace period at all.
    """
    allowance = termination_allowance_of(_profile_text(entries))
    graces = [grace for _, _, grace in entries if grace is not None]
    largest = max(graces) if graces else NEXTEST_DEFAULT_GRACE_PERIOD_SECONDS
    expected = largest + TERMINATION_SAFETY_MARGIN_SECONDS
    assert allowance == pytest.approx(expected), (
        f"the allowance must be the largest configured grace period, or "
        f"nextest's default when none is set, plus the safety margin; got "
        f"{allowance} for {entries}"
    )


@given(
    entries=st.lists(
        st.tuples(_PERIOD_SECONDS, _MULTIPLIERS, st.none() | _GRACE_SECONDS),
        min_size=1,
        max_size=6,
    )
)
def test_the_largest_per_test_allowance_is_the_largest_product(
    entries: list[tuple[int, int, int | None]],
) -> None:
    """The budget is a product, and the answer is the largest of them.

    Two things can go wrong and only one is visible in examples: reading
    the period without its multiplier, and taking the largest period
    rather than the largest product. A table with a long period and a
    multiplier of one, beside a short period with a large multiplier,
    tells them apart, and the generator produces that case without anyone
    having to think of it.
    """
    largest = largest_slow_timeout_of(_profile_text(entries))
    expected = max(period * multiplier for period, multiplier, _ in entries)
    assert largest == pytest.approx(expected), (
        f"the largest per-test allowance must be the largest period times its "
        f"terminate-after; got {largest} for {entries}"
    )


@pytest.mark.parametrize(
    ("step", "job", "document", "expected"),
    [
        pytest.param(
            {"env": {"RUN_RUST_CARGO_WAIT_TIMEOUT": "4200"}},
            {"env": {"RUN_RUST_CARGO_WAIT_TIMEOUT": "3600"}},
            {"env": {"RUN_RUST_CARGO_WAIT_TIMEOUT": "1800"}},
            4200.0,
            id="the-step-wins",
        ),
        pytest.param(
            {},
            {"env": {"RUN_RUST_CARGO_WAIT_TIMEOUT": "3600"}},
            {"env": {"RUN_RUST_CARGO_WAIT_TIMEOUT": "1800"}},
            3600.0,
            id="then-the-job",
        ),
        pytest.param(
            {},
            {},
            {"env": {"RUN_RUST_CARGO_WAIT_TIMEOUT": "1800"}},
            1800.0,
            id="then-the-workflow",
        ),
        pytest.param(
            {"env": {"RUN_RUST_CARGO_WAIT_TIMEOUT": "   "}},
            {"env": {"RUN_RUST_CARGO_WAIT_TIMEOUT": "3600"}},
            {},
            3600.0,
            id="a-blank-step-value-falls-through",
        ),
        pytest.param({}, {}, {}, None, id="nothing-sets-one"),
    ],
)
def test_the_watchdog_resolves_innermost_first(
    step: dict[str, object],
    job: dict[str, object],
    document: dict[str, object],
    expected: float | None,
) -> None:
    """Step, then job, then workflow, as GitHub resolves them.

    Both lanes here set the value on the step, so a reading that
    consulted only that scope agrees with this one against the tree and
    would stop agreeing the moment a lane moved the value to the job,
    reporting a bounded lane as inheriting the action's default.

    A blank source is not a budget of zero. It is what a workflow writes
    when it interpolates an expression that resolved to nothing, and
    converting it directly raises before the contract can name the lane.
    """
    resolved = _watchdog_of(document, job, step)
    if expected is None:
        assert resolved is None, f"no scope sets one, got {resolved!r}"
    else:
        assert resolved == pytest.approx(expected), (
            f"step={step!r} job={job!r} document={document!r} must resolve to "
            f"{expected}, got {resolved!r}"
        )


@pytest.mark.parametrize("value", ["0", "-1", " -30 "], ids=str)
def test_a_non_positive_watchdog_is_refused(value: str) -> None:
    """Zero is not a watchdog, it is the absence of one.

    The shared action reads a non-positive value as no timeout, so a
    lane carrying one has no third tier while appearing to declare one.
    Returning it would let the ceiling arithmetic certify a lane whose
    cargo invocation is unbounded.
    """
    with pytest.raises(ValueError, match="positive number of seconds"):
        _watchdog_of({}, {}, {"env": {"RUN_RUST_CARGO_WAIT_TIMEOUT": value}})


def test_the_required_ceiling_carries_all_three_terms() -> None:
    """Watchdog, measured work outside it, and the margin above them.

    Both lanes here now sit fifteen minutes above their requirement, so
    dropping the margin from the derivation changes nothing the
    assertion over the workflows can see: the ceiling still clears the
    smaller number. Driving the derivation with controlled values is
    what makes the missing term visible.
    """
    assert required_ceiling(4200.0, 900.0) == pytest.approx(
        4200.0 + 900.0 + CEILING_MARGIN_SECONDS
    ), "all three terms are added"
    assert required_ceiling(4200.0, 0.0) == pytest.approx(
        4200.0 + CEILING_MARGIN_SECONDS
    ), "the margin applies even when nothing runs outside the watchdog"
    assert required_ceiling(0.0, 0.0) == pytest.approx(CEILING_MARGIN_SECONDS), (
        "the margin is a term of its own, not a fraction of the others"
    )
