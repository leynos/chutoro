"""Reading a nextest duration the way the runner reads it.

Separated from ``timeout_budgets`` so the duration parsing and the
budget arithmetic stay legible apart, and so neither module outgrows
the 400-line limit ``AGENTS.md`` sets.

nextest parses durations with `humantime`, which reads a sequence of
value-and-unit pairs and sums them. A parser reading one pair would
refuse configuration the runner accepts, and this contract would fail a
repository whose timeouts were fine.
"""

import re
import string
import typing as typ


class NextestDurationError(ValueError):
    """Raised when a duration is not one `humantime` would accept.

    Distinguished from a budget in the wrong order. A duration nextest
    cannot parse leaves the runner with no configuration to apply, so
    there is nothing for this contract to compare rather than something
    compared wrongly.
    """


#: The whitespace humantime skips, written out rather than abbreviated.
#:
#: humantime skips on Rust's `char::is_whitespace`, which is the Unicode
#: White_Space property. Python's `\s` is that property plus U+001C to
#: U+001F, the file, group, record and unit separators, and `str.strip`
#: and `str.split` carry the same four-character excess. A reader
#: spelling the class `\s` therefore reads `1\x1cs` as one second and
#: `\x1c45m` as forty-five minutes, both of which nextest refuses at
#: startup. That is the accept-what-the-runner-refuses direction this
#: module exists to avoid, and it is why the class is written out.
#: `test_the_whitespace_class_is_rusts_own` pins the difference in both
#: directions, so a change in either language's notion of whitespace
#: fails there rather than in a runner.
_SPACE_CHARS: typ.Final[str] = (
    "\t\n\v\f\r \x85\xa0\u1680"
    "\u2000\u2001\u2002\u2003\u2004\u2005"
    "\u2006\u2007\u2008\u2009\u200a"
    "\u2028\u2029\u202f\u205f\u3000"
)

#: The same set as a regular-expression character class.
_SPACE: typ.Final[str] = f"[{re.escape(_SPACE_CHARS)}]"

#: The digits humantime reads, enumerated for the same reason the
#: whitespace above is.
#:
#: Python's `\d` matches every Unicode decimal digit. humantime matches
#: `'0'..='9'` and nothing else, so an Arabic-Indic or Devanagari
#: numeral is a duration a `\d` reader converts happily and nextest
#: refuses at startup: the same accept-what-the-runner-refuses
#: direction, and `1\u0660s` is the shape nobody would notice in a
#: file. `test_the_digit_class_is_humantimes_own` pins this set in both
#: directions against Python's.
_DIGIT_CHARS: typ.Final[str] = string.digits

#: The same set as a regular-expression character class.
_DIGIT: typ.Final[str] = f"[{re.escape(_DIGIT_CHARS)}]"

#: Digits with whitespace tolerated between them. humantime's parser
#: ignores whitespace while it accumulates a number, so `1 0s` is ten
#: seconds rather than a malformed duration.
_SPACED_DIGITS: typ.Final[str] = rf"{_DIGIT}(?:{_SPACE}*{_DIGIT})*"

#: One value-and-unit pair of a humantime duration. nextest parses its
#: durations with `humantime`, which takes a sequence of these and sums
#: them, so `2h 37min` and `1m30s` are both valid and a parser reading
#: one pair would refuse configuration the runner accepts.
#:
#: The value may carry a fractional part, and humantime tolerates
#: whitespace around the point, so `1.5m` and `1 . 5 m` both read as 90
#: seconds. Measured against humantime 2.3.0, the version the lockfile
#: of the pinned cargo-nextest release resolves through humantime_serde.
#: A leading point, a trailing point and a second point are all refused
#: there and are refused here. The whole part and the fraction are
#: captured apart because humantime scales them differently.
_DURATION_TOKEN: typ.Final[re.Pattern[str]] = re.compile(
    rf"(?P<whole>{_SPACED_DIGITS})"
    rf"(?:{_SPACE}*\.{_SPACE}*(?P<fraction>{_SPACED_DIGITS}))?"
    rf"{_SPACE}*(?P<unit>[A-Za-z\u00b5]+){_SPACE}*"
)

#: The one duration humantime accepts with no unit. Its parser
#: special-cases the exact text before reading a single character, so
#: the comparison is against the raw value rather than a stripped one:
#: `" 0 "` is not this case and nextest refuses it.
_BARE_ZERO: typ.Final[str] = "0"

#: Nanoseconds in a second.
_SECOND: typ.Final[int] = 1_000_000_000

#: The largest value humantime's parser can hold. Its accumulators and
#: its intermediate products are all `u64`, checked at every step, and
#: Python's integers are not, so the checks are made explicitly here.
#: Without them the reader accepts fourteen of the differential's range
#: inputs that nextest refuses at startup. The least obvious of those is
#: a fraction's denominator, a power of ten built one digit at a time:
#: twenty digits overflow it where nineteen do not, whatever the digits
#: are, which is why `1.0000000000000000000s` is one second and
#: `1.00000000000000000000s` is refused.
_U64_MAX: typ.Final[int] = 2**64 - 1


class _Unit(typ.NamedTuple):
    """One humantime unit, as its parser treats it.

    Attributes
    ----------
    nanoseconds : int
        One of this unit in nanoseconds. A value's whole part is
        multiplied by this.
    fraction_scale : int or None
        What a fraction's numerator is multiplied by before the exact
        division humantime requires, or None when the unit admits no
        fraction at all. ``ns`` is that case: humantime refuses a
        fractional nanosecond outright rather than rounding it.
    fraction_in_seconds : bool
        Whether that division yields seconds rather than nanoseconds.
        humantime divides whole seconds for hours and longer, so
        ``0.123h`` is refused where ``0.123s`` is exact. A reader
        working in nanoseconds throughout would accept durations
        nextest rejects, and one working in floats accepts every
        inexact fraction at every unit.
    """

    nanoseconds: int
    fraction_scale: int | None
    fraction_in_seconds: bool


#: Every spelling `humantime` accepts, grouped by the unit it names,
#: using humantime's own definitions: a month is 30.44 days and a year
#: 365.25, which is why neither is expressed in terms of the other.
#: Spelt out in full rather than trimmed to the plausible ones, because
#: refusing a unit nextest accepts would fail a configuration the runner
#: is happy with, which is the fault this table exists to avoid.
_UNIT_SPELLINGS: typ.Final[tuple[tuple[tuple[str, ...], _Unit], ...]] = (
    (
        ("nanos", "nsec", "ns"),
        _Unit(nanoseconds=1, fraction_scale=None, fraction_in_seconds=False),
    ),
    (
        ("usec", "us", "\u00b5s"),
        _Unit(nanoseconds=1_000, fraction_scale=1_000, fraction_in_seconds=False),
    ),
    (
        ("millis", "msec", "ms"),
        _Unit(
            nanoseconds=1_000_000,
            fraction_scale=1_000_000,
            fraction_in_seconds=False,
        ),
    ),
    (
        ("seconds", "second", "secs", "sec", "s"),
        _Unit(nanoseconds=_SECOND, fraction_scale=_SECOND, fraction_in_seconds=False),
    ),
    (
        ("minutes", "minute", "mins", "min", "m"),
        _Unit(
            nanoseconds=60 * _SECOND,
            fraction_scale=60 * _SECOND,
            fraction_in_seconds=False,
        ),
    ),
    (
        ("hours", "hour", "hrs", "hr", "h"),
        _Unit(
            nanoseconds=3_600 * _SECOND,
            fraction_scale=3_600,
            fraction_in_seconds=True,
        ),
    ),
    (
        ("days", "day", "d"),
        _Unit(
            nanoseconds=86_400 * _SECOND,
            fraction_scale=86_400,
            fraction_in_seconds=True,
        ),
    ),
    (
        ("weeks", "week", "wks", "wk", "w"),
        _Unit(
            nanoseconds=604_800 * _SECOND,
            fraction_scale=604_800,
            fraction_in_seconds=True,
        ),
    ),
    (
        ("months", "month", "M"),
        _Unit(
            nanoseconds=2_630_016 * _SECOND,
            fraction_scale=2_630_016,
            fraction_in_seconds=True,
        ),
    ),
    (
        ("years", "year", "yrs", "yr", "y"),
        _Unit(
            nanoseconds=31_557_600 * _SECOND,
            fraction_scale=31_557_600,
            fraction_in_seconds=True,
        ),
    ),
)

_UNITS: typ.Final[dict[str, _Unit]] = {
    spelling: unit for spellings, unit in _UNIT_SPELLINGS for spelling in spellings
}

#: Each unit's length in seconds, for callers comparing budgets.
_UNIT_SECONDS: typ.Final[dict[str, float]] = {
    spelling: unit.nanoseconds / _SECOND for spelling, unit in _UNITS.items()
}


def _u64(duration: str, value: int) -> int:
    """Return a value humantime could hold, or refuse it as it does.

    Every multiplication and addition in humantime's parser is checked
    against this bound and Python's integers are not, so each of those
    steps passes through here.

    Parameters
    ----------
    duration : str
        The whole duration, named in any message raised here.
    value : int
        The result of one step.

    Returns
    -------
    int
        The value unchanged, when humantime could hold it.

    Raises
    ------
    NextestDurationError
        If the value exceeds what a 64-bit unsigned integer holds, as
        every one of humantime's own checked steps would.

    Examples
    --------
    >>> _u64("1s", 1)
    1
    """
    if value > _U64_MAX:
        message = (
            f"unrecognized nextest duration {duration!r}: humantime "
            f"accumulates in 64-bit integers and this exceeds their range"
        )
        raise NextestDurationError(message)
    return value


def _whole_scaling(unit: _Unit) -> tuple[int, bool]:
    """Return how a whole value scales, and which part it lands in.

    humantime adds a second-or-longer unit to its seconds accumulator
    and a shorter one to its nanosecond accumulator, so the two overflow
    at different values. Derived from the unit's length rather than
    stored beside it, so the two cannot drift apart.

    Examples
    --------
    >>> _whole_scaling(_UNITS["m"])
    (60, True)
    >>> _whole_scaling(_UNITS["ms"])
    (1000000, False)
    """
    if unit.nanoseconds >= _SECOND:
        return unit.nanoseconds // _SECOND, True
    return unit.nanoseconds, False


class _Total:
    """The running total humantime keeps: whole seconds and nanoseconds.

    Both are 64-bit unsigned there and every step is checked, so this
    carries the pair rather than one count of nanoseconds. The two are
    not the same claim. `18446744073709551615ns` twice over names about
    36.9 billion seconds, some 1,169 years, nowhere near the seconds
    ceiling, and humantime refuses it because the second value overflows
    the nanosecond accumulator before it is carried. A reader checking
    only an accumulated total finds that comfortably in range and
    accepts it.

    Examples
    --------
    >>> total = _Total()
    >>> total.add("1s 1s", 1, 0)
    >>> total.add("1s 1s", 1, 0)
    >>> total.as_seconds()
    2.0
    """

    def __init__(self) -> None:
        self.seconds = 0
        self.nanoseconds = 0

    def add(self, duration: str, seconds: int, nanoseconds: int) -> None:
        """Add one part, refusing what humantime's checks would refuse.

        Every sum and the carry go through `_u64`, so a part that does
        not fit is refused here rather than accumulated.
        """
        nanos = _u64(duration, self.nanoseconds + nanoseconds)
        running = _u64(duration, self.seconds + seconds)
        # humantime carries in two places, not one. Its parser
        # normalizes on a strict `>`, so a nanosecond part of exactly
        # one second survives the loop untouched and reaches
        # `Duration::new`, which carries on `>=` and aborts the process
        # rather than erroring when that carry overflows. nextest cannot
        # run either way, so one refusal here answers both. Written out
        # separately at first and the strict branch proved unreachable:
        # collapsing them changed no answer over the differential's
        # seventy-one inputs, and an unfalsifiable guard is worse than
        # none. This single `>=` is what the evidence supports, and it
        # is what makes `0.5s 0.5s` one second while
        # `18446744073709551615s 500ms 500ms` is refused.
        if nanos >= _SECOND:
            running = _u64(duration, running + nanos // _SECOND)
            nanos %= _SECOND
        self.seconds = running
        self.nanoseconds = nanos

    def as_seconds(self) -> float:
        """Return the total in seconds.

        Returns
        -------
        float
            Whole seconds and the nanosecond part together.
        """
        return self.seconds + self.nanoseconds / _SECOND


def _add_landed(duration: str, total: _Total, amount: int, in_seconds: bool) -> None:
    """Add an already-scaled amount to whichever part it belongs in."""
    if in_seconds:
        total.add(duration, amount, 0)
    else:
        total.add(duration, 0, amount)


def _read_pair(duration: str, text: str, position: int, total: _Total) -> int:
    """Add one value-and-unit pair to the total, and return where it ends."""
    match = _DURATION_TOKEN.match(text, position)
    if match is None:
        message = (
            f"unrecognized nextest duration {duration!r}: humantime reads a "
            f"sequence of values, each optionally fractional and each "
            f"followed by a unit, or a bare {_BARE_ZERO!r}"
        )
        raise NextestDurationError(message)
    unit = _UNITS.get(match["unit"])
    if unit is None:
        message = (
            f"unrecognized nextest duration {duration!r}: {match['unit']!r} is "
            f"not a unit humantime accepts"
        )
        raise NextestDurationError(message)
    # humantime ignores whitespace while it accumulates a number and
    # around the fractional point, so the matched digits can read "1 0"
    # or "1 . 5"; int cannot.
    whole = _u64(duration, int(_digits(match["whole"])))
    scale, in_seconds = _whole_scaling(unit)
    _add_landed(duration, total, _u64(duration, whole * scale), in_seconds)
    if match["fraction"] is not None:
        amount, fraction_in_seconds = _fraction_amount(
            duration, match["fraction"], unit
        )
        _add_landed(duration, total, amount, fraction_in_seconds)
    return match.end()


def _digits(matched: str) -> str:
    """Return a matched digit run with its internal whitespace removed.

    Removes exactly the characters the pattern tolerated. `str.split`
    would also remove U+001C to U+001F, which the pattern refuses, so
    while the two cannot disagree today, a later widening of the pattern
    would turn a refusal into a silently different number. This site is
    unreachable through the reader for that reason, and is tested
    directly.
    """
    return re.sub(_SPACE, "", matched)


def _fraction_amount(duration: str, matched: str, unit: _Unit) -> tuple[int, bool]:
    """Return a fractional part and the accumulator it lands in.

    humantime carries the fraction as a numerator over a power of ten
    and divides with a remainder check, so a fraction that is not a
    whole number of the unit's smallest step is an error rather than a
    rounded value.

    Parameters
    ----------
    duration : str
        The whole duration, named in any message raised here rather
        than the fraction, which is not what anybody wrote.
    matched : str
        The digits after the point, whitespace and all.
    unit : _Unit
        The unit the fraction belongs to, which decides both the scale
        and whether the division is over seconds or nanoseconds.

    Returns
    -------
    tuple of (int, bool)
        The fraction's contribution and whether it lands in seconds.
        humantime divides whole seconds for minutes and longer, so the
        two are reported apart rather than reduced to nanoseconds: an
        amount in seconds would overflow the nanosecond accumulator it
        does not belong in.

    Raises
    ------
    NextestDurationError
        If humantime would refuse the fraction: on a nanosecond, which
        has no smaller step, where the division leaves a remainder, or
        where any step leaves 64 bits.

    Examples
    --------
    Minutes and shorter divide into nanoseconds, hours and longer into
    whole seconds, which is what the second element reports:

    >>> _fraction_amount("1.5m", "5", _UNITS["m"])
    (30000000000, False)
    >>> _fraction_amount("1.5h", "5", _UNITS["h"])
    (1800, True)

    A thousandth of an hour does not divide, because that division is
    over whole seconds and 3.6 is not one:

    >>> _fraction_amount("1.001h", "001", _UNITS["h"])  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    nextest_durations.NextestDurationError: unrecognized nextest duration '1.001h': ...
    """
    if unit.fraction_scale is None:
        message = (
            f"unrecognized nextest duration {duration!r}: humantime has no "
            f"step below a nanosecond, so a fractional one is an error"
        )
        raise NextestDurationError(message)
    digits = _digits(matched)
    numerator = _u64(duration, int(digits))
    denominator = _u64(duration, 10 ** len(digits))
    scaled = _u64(duration, numerator * unit.fraction_scale)
    if scaled % denominator:
        step = "second" if unit.fraction_in_seconds else "nanosecond"
        message = (
            f"unrecognized nextest duration {duration!r}: humantime divides "
            f"exactly, and this fraction is not a whole number of the unit's "
            f"{step}s"
        )
        raise NextestDurationError(message)
    return scaled // denominator, unit.fraction_in_seconds


def _seconds(duration: str) -> float:
    """Convert a nextest duration to seconds, or raise."""
    # nextest parses durations with `humantime`, which reads a sequence
    # of value-and-unit pairs and sums them, so "2h 37min" and "1m30s"
    # are as valid as "40m". A parser accepting one pair would refuse
    # configuration the runner accepts, and this contract would fail a
    # repository whose timeouts were fine. Anything humantime refuses
    # raises `NextestDurationError`, because a configuration nextest
    # cannot parse has no budgets to compare.
    if duration == _BARE_ZERO:
        return 0.0
    text = duration.strip(_SPACE_CHARS)
    if not text:
        message = f"unrecognized nextest duration {duration!r}: it is empty"
        raise NextestDurationError(message)
    total = _Total()
    position = 0
    while position < len(text):
        position = _read_pair(duration, text, position, total)
    return total.as_seconds()
