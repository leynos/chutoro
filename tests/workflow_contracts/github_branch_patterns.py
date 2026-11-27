"""Evaluate GitHub Actions branch filters the way GitHub does.

A trigger's ``branches`` and ``branches-ignore`` values are glob patterns, not
literal names. ``*`` matches any run of characters except ``/``, ``**``
matches across separators, ``?`` and ``+`` quantify the preceding character,
and ``[]`` is a character class with ``!`` for negation. A pattern beginning
with ``!`` removes branches selected earlier, and a later positive pattern can
select them again.

Reading the values as literal membership would accept
``branches-ignore: ['m*']``, which really does exclude ``main``, and treating
``?`` as a wildcard would accept ``ma?n``, which does not reach it. Both
mistakes would let a required job's trigger miss a pull request while the
contract said it was covered.

Re-use policy: import from :mod:`required_checks_test` only. The branch-filter
rules are GitHub's, not this repository's, so a second consumer should extend
this module rather than restate the syntax.
"""

from __future__ import annotations

import re
import typing as typ

#: The regular expression for a `*` wildcard: any run without a separator.
_WILDCARD = "[^/]*"

#: The regular expression for `**`: any run, separators included.
_ANY = ".*"


def branch_pattern_regex(pattern: str) -> re.Pattern[str]:
    """Translate one GitHub branch filter into a regular expression.

    GitHub's filter syntax is not literal membership: ``*`` matches any run of
    characters except ``/``, ``**`` matches across separators, ``?`` and ``+``
    quantify the preceding character, and ``[]`` is a character class with
    ``!`` for negation.

    Examples
    --------
    >>> bool(branch_pattern_regex("m*").fullmatch("main"))
    True
    >>> bool(branch_pattern_regex("ma?n").fullmatch("main"))
    False
    >>> bool(branch_pattern_regex("release/**").fullmatch("release/1.0"))
    True
    """
    parts: list[str] = []
    index = 0
    while index < len(pattern):
        atom, index = _pattern_atom(pattern, index)
        if atom in {"?", "+"} and parts:
            parts[-1] = f"{parts[-1]}{atom}"
        else:
            parts.append(atom)
    return re.compile("".join(parts))


def selected_by(patterns: typ.Iterable[str], branch: str) -> bool:
    """Return whether `patterns` select `branch`, honouring ordered exclusions.

    Examples
    --------
    >>> selected_by(["main", "!main"], "main")
    False
    >>> selected_by(["main", "!main", "mai*"], "main")
    True
    """
    selected = False
    for pattern in patterns:
        if pattern.startswith("!"):
            if branch_pattern_regex(pattern[1:]).fullmatch(branch):
                selected = False
        elif branch_pattern_regex(pattern).fullmatch(branch):
            selected = True
    return selected


def _pattern_atom(pattern: str, index: int) -> tuple[str, int]:
    """Return the next regular-expression atom and the offset after it."""
    character = pattern[index]
    if character == "*":
        if pattern[index : index + 2] == "**":
            return _ANY, index + 2
        return _WILDCARD, index + 1
    if character in "?+":
        return character, index + 1
    if character == "[":
        return _character_class(pattern, index)
    return re.escape(character), index + 1


def _character_class(pattern: str, index: int) -> tuple[str, int]:
    """Return one bracket expression's regex and the offset after it."""
    end = pattern.find("]", index + 1)
    if end < 0:
        return re.escape("["), index + 1
    body = pattern[index + 1 : end]
    if body.startswith("!"):
        body = "^" + body[1:]
    return f"[{body}]", end + 1