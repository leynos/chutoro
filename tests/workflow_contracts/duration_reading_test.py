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

import re

import pytest
from nextest_durations import (
    _DIGIT_CHARS,
    _SPACE_CHARS,
    NextestDurationError,
    _digits,
    _seconds,
)
from timeout_budgets import (
    TerminateAfterError,
    UnboundedTestError,
    bounds_a_single_test,
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
        pytest.param("0.5s 0.5s", 1.0, id="two-halves-summing-to-a-whole-second"),
        pytest.param("1 . 5 m", 90.0, id="a-fraction-spaced-around-the-point"),
        pytest.param("1wk", 604800.0, id="the-abbreviated-week"),
        pytest.param("2wks", 1209600.0, id="the-abbreviated-plural-week"),
        pytest.param("1yr", 31557600.0, id="the-abbreviated-year"),
        pytest.param("3yrs", 94672800.0, id="the-abbreviated-plural-year"),
        pytest.param("1 0s", 10.0, id="whitespace-inside-the-number"),
        pytest.param("0", 0.0, id="a-bare-zero-with-no-unit"),
        pytest.param("500nanos", 5e-7, id="the-long-nanosecond-spelling"),
        pytest.param("250millis", 0.25, id="the-long-millisecond-spelling"),
        pytest.param("750\u00b5s", 0.00075, id="the-micro-sign"),
        pytest.param("1.5h", 5400.0, id="a-fraction-of-an-hour-in-whole-seconds"),
        pytest.param("0.123s", 0.123, id="a-fraction-of-a-second-in-nanoseconds"),
        pytest.param("0.000000001m", 6e-8, id="a-fraction-of-a-minute-in-nanoseconds"),
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

    The fractional values, the `wk`, `wks`, `yr` and `yrs` aliases, the
    number carrying whitespace and the bare zero were measured against
    humantime 2.3.0, the version the lockfile of the pinned
    cargo-nextest release resolves through humantime_serde, rather than
    assumed: this reader had refused all of them. humantime's parser
    ignores whitespace while it accumulates a number, so `1 0s` is ten
    seconds, and it special-cases `0` before reading a character, so a
    zero duration needs no unit.

    `nanos` and `millis` come from humantime's own unit table, which
    takes three spellings each where this reader had two; the micro
    sign is its one non-ASCII spelling, U+00B5 not U+03BC.
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
        pytest.param("1.5ns", id="a-fractional-nanosecond"),
        pytest.param("0.5ns", id="half-a-nanosecond"),
        pytest.param("0.123h", id="a-fraction-of-an-hour-below-a-second"),
        pytest.param("0.0000000001s", id="a-fraction-below-a-nanosecond"),
        pytest.param(" 0 ", id="a-padded-bare-zero"),
        pytest.param("00", id="a-repeated-bare-zero"),
    ],
)
def test_a_duration_humantime_would_refuse_is_refused(duration: str) -> None:
    """What the runner cannot parse, this contract must not guess at.

    Reading a duration nextest rejects would compare the tiers against a
    budget that never applies. Each of these was checked against
    humantime 2.3.0 and refused there: a fraction needs a whole part
    before the point and a digit after it, values are unsigned, and the
    only separators are whitespace. `0` is the one value that may carry
    no unit, so `30` stays refused, and the special case is the exact
    text, so `" 0 "` and `"00"` are not it.

    The four fractions are the same fault seen from the other side.
    humantime carries a fraction as a numerator over a power of ten and
    divides with a remainder check, so it has no step below a
    nanosecond and refuses a fractional one outright; and for hours and
    longer it divides whole seconds, which is why `0.123h` is refused
    where `0.123s` is exact. Read as floats these four became 5e-10,
    1.5e-09, 442.8 and 1e-10, none of which nextest would have started
    with.
    """
    with pytest.raises(NextestDurationError):
        _seconds(duration)


@pytest.mark.parametrize(
    "duration",
    [
        pytest.param("1\x1cs", id="a-file-separator-inside-a-number"),
        pytest.param("\x1c45m", id="a-file-separator-leading"),
        pytest.param("45m\x1f", id="a-unit-separator-trailing"),
        pytest.param("1\x1d0s", id="a-group-separator-between-digits"),
    ],
)
def test_c0_separators_are_not_whitespace_and_are_refused(duration: str) -> None:
    """Python calls U+001C to U+001F whitespace; humantime does not.

    Rust's `char::is_whitespace` is the Unicode White_Space property,
    which excludes the file, group, record and unit separators. Python's
    `\\s`, `str.strip` and `str.split` all include them, so a reader
    written the obvious way reads the first of these as one second and
    the second as forty-five minutes, and reports a budget for a
    configuration nextest refuses at startup.

    The last is the one nobody would notice: a separator between two
    digits of an ordinary number, which `str.split` silently removes.
    """
    with pytest.raises(NextestDurationError):
        _seconds(duration)


def test_the_whitespace_class_is_rusts_own() -> None:
    """The class is the Unicode White_Space property, and nothing more.

    Pins the reasoning rather than the consequence. The four separators
    the previous test refuses are refused because they are absent from
    this set, and this asserts both directions of that: Python's `\\s`
    exceeds the set by exactly those four, and the set exceeds `\\s` by
    nothing. If either language's notion of whitespace moves, this fails
    here rather than in a runner.
    """
    ours = set(_SPACE_CHARS)
    pythons = {c for c in map(chr, range(0x11000)) if re.fullmatch(r"\s", c)}
    assert pythons - ours == set("\x1c\x1d\x1e\x1f"), (
        "Python's whitespace must exceed humantime's by exactly the four "
        f"C0 separators, exceeds it by {sorted(pythons - ours)!r}"
    )
    assert not ours - pythons, (
        "every character this reader treats as whitespace must be one "
        f"Python agrees is whitespace, {sorted(ours - pythons)!r} are not"
    )


def test_the_digit_join_removes_only_what_the_pattern_tolerated() -> None:
    """Driven directly, because the reader cannot reach this case.

    The pattern refuses a separator between two digits, so the join
    never sees one while the two agree. It is written with the reader's
    own class rather than `str.split` so that a later widening of the
    pattern cannot turn a refusal into a silently different number, and
    that property has to be asserted where it can be: here.
    """
    assert _digits("1 0") == "10", (
        "an ordinary space is whitespace to humantime, so the join must "
        "remove it and read the digits as one number"
    )
    assert _digits("1\u20080") == "10", (
        "U+2008 is in the Unicode White_Space property, so humantime skips "
        "it and the join must remove it too"
    )
    assert _digits("1\x1d0") == "1\x1d0", (
        "U+001D is not whitespace to humantime, so the join must leave it "
        "in place; removing it is what `str.split` would do and is what "
        "turns a duration the runner refuses into ten seconds"
    )


def test_the_digit_class_is_humantimes_own() -> None:
    """The class is `0` to `9`, and nothing more.

    Pins the reasoning rather than the consequence, as the whitespace
    contract does. Python's `\\d` is every Unicode decimal digit;
    humantime matches `'0'..='9'`. Asserting both directions means a
    change in either language's notion of a digit fails here rather
    than in a runner.
    """
    ours = set(_DIGIT_CHARS)
    pythons = {c for c in map(chr, range(0x11000)) if re.fullmatch(r"\d", c)}
    assert ours == set("0123456789"), (
        f"humantime reads the ASCII digits and nothing else, {sorted(ours)!r}"
    )
    assert ours < pythons, (
        "Python's digits must be a strict superset of humantime's; this "
        f"reader treats {sorted(ours - pythons)!r} as digits Python does not"
    )


@pytest.mark.parametrize(
    "duration",
    [
        pytest.param("\u0665s", id="an-arabic-indic-numeral"),
        pytest.param("\u096ams", id="a-devanagari-numeral"),
        pytest.param("1\u0660s", id="an-arabic-indic-numeral-inside-a-number"),
        pytest.param("1.\u0665s", id="an-arabic-indic-numeral-in-a-fraction"),
    ],
)
def test_non_ascii_digits_are_refused(duration: str) -> None:
    """Python calls these digits; humantime does not.

    A `\\d` reader converts all four and nextest refuses all four at
    startup. The third is the shape nobody would notice in a file: an
    ASCII digit followed by an Arabic-Indic one, which reads as ten.
    """
    with pytest.raises(NextestDurationError):
        _seconds(duration)


@pytest.mark.parametrize(
    "duration",
    [
        pytest.param("18446744073709551616s", id="seconds-past-the-ceiling"),
        pytest.param("307445734561825861m", id="a-product-past-the-ceiling"),
        pytest.param("584542046091y", id="a-year-product-past-the-ceiling"),
        pytest.param("18446744073709551615s 1s", id="a-sum-past-the-ceiling"),
        pytest.param(
            "1.00000000000000000000s", id="a-fraction-denominator-past-the-ceiling"
        ),
    ],
)
def test_values_beyond_64_bits_are_refused(duration: str) -> None:
    """humantime accumulates in `u64` and checks every step; Python does not.

    Its parser checks each multiplication and each addition, so a value
    that leaves the range is an error there and not a large number. A
    reader on Python's unbounded integers accepts all of these and
    reports budgets nextest refuses at startup, and the numbers it
    invents are enormous and plausible rather than obviously wrong.

    The denominator is checked too, and is the least obvious of these:
    it is a power of ten built one digit at a time, so a fraction of
    twenty digits overflows where one of nineteen does not, whatever
    the digits are.
    """
    with pytest.raises(NextestDurationError):
        _seconds(duration)


def test_a_nineteen_digit_fraction_is_still_read() -> None:
    """The denominator check is a ceiling, not a ban on long fractions.

    Without this the range contract above would pass just as well
    against a reader that refused every fraction over some shorter
    length, or every fraction at all.
    """
    assert _seconds("1.0000000000000000000s") == pytest.approx(1.0), (
        "nineteen fractional digits give a denominator of 10^19, which fits "
        "in u64, so humantime reads this as one second and so must this"
    )


@pytest.mark.parametrize(
    "duration",
    [
        pytest.param(
            "18446744073709551615ns 18446744073709551615ns",
            id="two-nanosecond-parts-overflowing-their-accumulator",
        ),
        pytest.param("18446744073709551615s 500ms 500ms", id="a-carry-past-the-ceiling"),
        pytest.param("18446744073709551615s 1000ms", id="a-whole-carry-past-the-ceiling"),
    ],
)
def test_a_carry_out_of_the_nanosecond_part_is_checked(duration: str) -> None:
    """humantime keeps seconds and nanoseconds apart, and carries between them.

    A reader accumulating one count of nanoseconds accepts the first of
    these: two maximal nanosecond values are about 1,169 years, nowhere
    near the seconds ceiling. humantime refuses it because the second
    value overflows the nanosecond accumulator before anything is
    carried. The other two overflow on the carry itself, which is why
    the carry is checked and not only the parts.
    """
    with pytest.raises(NextestDurationError):
        _seconds(duration)


def test_the_parts_are_summed_in_humantimes_order() -> None:
    """One input separates the two orders, and this is it.

    `18446744073709551615ns 1ns` is accepted only when each part is
    added and carried as it is read, which is what humantime does: the
    first part carries out of the nanosecond accumulator into seconds
    immediately, leaving room for the second. Summing the parts first
    and carrying once overflows the nanosecond accumulator and refuses
    it. Without this case the order is unasserted and either reader
    passes every other input.
    """
    # 18446744073709551616 nanoseconds, which humantime reports as
    # 18446744073 seconds and 709551616 nanoseconds. Written as the pair
    # rather than as a decimal, so the expectation is the carry itself.
    assert _seconds("18446744073709551615ns 1ns") == pytest.approx(
        18446744073 + 709551616 / 1_000_000_000
    ), (
        "the first part must carry into seconds as it is read, leaving the "
        "nanosecond accumulator room for the second; summing the parts first "
        "overflows it and refuses a duration humantime accepts"
    )


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
        pytest.param(
            '[profile.default]\nslow-timeout = "180s"\n',
            id="a-base-profile-declared-as-a-bare-duration",
        ),
        pytest.param(
            "[profile.default]\n"
            'slow-timeout = { period = "60s", terminate-after = 1 }\n'
            "\n[[profile.default.overrides]]\n"
            "filter = 'binary(slow)'\n"
            'slow-timeout = "600s"\n',
            id="an-override-declared-as-a-bare-duration",
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

    A bare `slow-timeout = "600s"` is the same hole written shorthand:
    nextest reads it as that period with no `terminate-after`. A reading
    that kept only inline tables would drop it, and a sibling section
    holding a bounded table would then supply a finite maximum in its
    place, which is the last two cases here.

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


def test_a_base_profile_declared_as_a_bare_duration_bounds_nothing() -> None:
    """The shorthand carries no `terminate-after`, so it bounds no test.

    `bounds_a_single_test` reads the default profile's own declaration,
    and a reading that accepted any declaration at all would call this
    profile bounded while every test it covers runs on after the period
    elapses.
    """
    config_text = '[profile.default]\nslow-timeout = "60s"\n'

    assert not bounds_a_single_test(config_text)
