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
import typing as typ


class NextestDurationError(ValueError):
    """Raised when a duration is not one `humantime` would accept.

    Distinguished from a budget in the wrong order. A duration nextest
    cannot parse leaves the runner with no configuration to apply, so
    there is nothing for this contract to compare rather than something
    compared wrongly.
    """


#: Digits with whitespace tolerated between them. humantime's parser
#: ignores whitespace while it accumulates a number, so `1 0s` is ten
#: seconds rather than a malformed duration.
_SPACED_DIGITS: typ.Final[str] = r"\d(?:\s*\d)*"

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
    rf"(?:\s*\.\s*(?P<fraction>{_SPACED_DIGITS}))?"
    r"\s*(?P<unit>[A-Za-z\u00b5]+)\s*"
)

#: The one duration humantime accepts with no unit. Its parser
#: special-cases the exact text before reading a single character, so
#: the comparison is against the raw value rather than a stripped one:
#: `" 0 "` is not this case and nextest refuses it.
_BARE_ZERO: typ.Final[str] = "0"

#: Nanoseconds in a second, which is the scale this port works in.
_SECOND: typ.Final[int] = 1_000_000_000


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


def _read_pair(duration: str, text: str, position: int) -> tuple[int, int]:
    """Return one value-and-unit pair in nanoseconds, and where it ends."""
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
    nanoseconds = int(_digits(match["whole"])) * unit.nanoseconds
    if match["fraction"] is not None:
        nanoseconds += _fraction_nanoseconds(duration, match["fraction"], unit)
    return nanoseconds, match.end()


def _digits(matched: str) -> str:
    """Return a matched digit run with its internal whitespace removed."""
    return "".join(matched.split())


def _fraction_nanoseconds(duration: str, matched: str, unit: _Unit) -> int:
    """Return a fractional part in nanoseconds, as humantime computes it.

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
    int
        The fraction's contribution, in nanoseconds.

    Raises
    ------
    NextestDurationError
        If humantime would refuse the fraction: on a nanosecond, which
        has no smaller step, or where the division leaves a remainder.
    """
    digits = _digits(matched)
    numerator = int(digits)
    denominator = 10 ** len(digits)
    if unit.fraction_scale is None:
        message = (
            f"unrecognized nextest duration {duration!r}: humantime has no "
            f"step below a nanosecond, so a fractional one is an error"
        )
        raise NextestDurationError(message)
    scaled = numerator * unit.fraction_scale
    if scaled % denominator:
        step = "second" if unit.fraction_in_seconds else "nanosecond"
        message = (
            f"unrecognized nextest duration {duration!r}: humantime divides "
            f"exactly, and this fraction is not a whole number of the unit's "
            f"{step}s"
        )
        raise NextestDurationError(message)
    quotient = scaled // denominator
    return quotient * _SECOND if unit.fraction_in_seconds else quotient


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
    text = duration.strip()
    if not text:
        message = f"unrecognized nextest duration {duration!r}: it is empty"
        raise NextestDurationError(message)
    total = 0
    position = 0
    while position < len(text):
        nanoseconds, position = _read_pair(duration, text, position)
        total += nanoseconds
    return total / _SECOND
