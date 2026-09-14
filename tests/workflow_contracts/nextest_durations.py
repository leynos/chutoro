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
#: there and are refused here.
_DURATION_TOKEN: typ.Final[re.Pattern[str]] = re.compile(
    rf"(?P<value>{_SPACED_DIGITS}(?:\s*\.\s*{_SPACED_DIGITS})?)"
    r"\s*(?P<unit>[A-Za-z\u00b5]+)\s*"
)

#: The one duration humantime accepts with no unit. Its parser
#: special-cases the exact text before reading a single character, so
#: the comparison is against the raw value rather than a stripped one:
#: `" 0 "` is not this case and nextest refuses it.
_BARE_ZERO: typ.Final[str] = "0"

#: Every unit `humantime` accepts, with its length in seconds, using
#: humantime's own definitions: a month is 30.44 days and a year 365.25,
#: which is why neither is expressed in terms of the other. Spelt out in
#: full rather than trimmed to the plausible ones, because refusing a
#: unit nextest accepts would fail a configuration the runner is happy
#: with, which is the fault this table exists to avoid.
_UNIT_SECONDS: typ.Final[dict[str, float]] = {
    "nsec": 1e-9,
    "ns": 1e-9,
    "usec": 1e-6,
    "us": 1e-6,
    "\u00b5s": 1e-6,
    "msec": 0.001,
    "ms": 0.001,
    "seconds": 1.0,
    "second": 1.0,
    "secs": 1.0,
    "sec": 1.0,
    "s": 1.0,
    "minutes": 60.0,
    "minute": 60.0,
    "mins": 60.0,
    "min": 60.0,
    "m": 60.0,
    "hours": 3600.0,
    "hour": 3600.0,
    "hrs": 3600.0,
    "hr": 3600.0,
    "h": 3600.0,
    "days": 86400.0,
    "day": 86400.0,
    "d": 86400.0,
    "weeks": 604800.0,
    "week": 604800.0,
    "wks": 604800.0,
    "wk": 604800.0,
    "w": 604800.0,
    "months": 2630016.0,
    "month": 2630016.0,
    "M": 2630016.0,
    "years": 31557600.0,
    "year": 31557600.0,
    "yrs": 31557600.0,
    "yr": 31557600.0,
    "y": 31557600.0,
}


def _read_pair(duration: str, text: str, position: int) -> tuple[float, int]:
    """Return one value-and-unit pair's seconds and where it ends."""
    match = _DURATION_TOKEN.match(text, position)
    if match is None:
        message = (
            f"unrecognized nextest duration {duration!r}: humantime reads a "
            f"sequence of values, each optionally fractional and each "
            f"followed by a unit, or a bare {_BARE_ZERO!r}"
        )
        raise NextestDurationError(message)
    unit = match["unit"]
    if unit not in _UNIT_SECONDS:
        message = (
            f"unrecognized nextest duration {duration!r}: {unit!r} is not a "
            f"unit humantime accepts"
        )
        raise NextestDurationError(message)
    # humantime ignores whitespace while it accumulates a number and
    # around the fractional point, so the matched value can read "1 0" or
    # "1 . 5"; float cannot. The whole duration is named in the message
    # rather than the pair, since the pair is not what anybody wrote.
    value = "".join(match["value"].split())
    return float(value) * _UNIT_SECONDS[unit], match.end()


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
    total = 0.0
    position = 0
    while position < len(text):
        pair, position = _read_pair(duration, text, position)
        total += pair
    return total
