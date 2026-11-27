"""Decide whether a ``cfg`` guard excludes an item from the Kani gate's build.

The gate runs ``cargo kani --lib``, which compiles without ``cfg(test)`` and
with ``kani`` set. A proof behind a guard that is false in that configuration
is not compiled at all, so it cannot be verified even though its declaration
exists. Searching the attribute text for the word ``test`` is not enough:
``cfg(not(test))`` and ``cfg(any(kani, test))`` both compile, and
``cfg(all(not(feature = "opt"), test))`` does not. The predicate is therefore
evaluated with three-valued logic, and only a predicate that is definitely
false is treated as excluding the item.

Re-use policy: import from :mod:`kani_harness_reading` and
:mod:`kani_source_reading` only. The gate's configuration is this
repository's, so a second consumer should extend this module rather than
assume a different set of cfgs.
"""

from __future__ import annotations

import re
import typing as typ

#: A ``cfg`` guard and its predicate body.
_CFG_GUARD = re.compile(r"#!?\[\s*cfg\s*\((?P<body>.*)\)\s*\]$", re.DOTALL)

#: One predicate: a name, or a combinator over parenthesised predicates.
_COMBINATOR = re.compile(r"(?P<fn>all|any|not)\s*\((?P<args>.*)\)$", re.DOTALL)


def gated_by_test(attributes: typ.Iterable[str]) -> bool:
    """Return whether these guards exclude the item when the test cfg is off.

    Examples
    --------
    >>> gated_by_test(["#[cfg(all(kani, test))]"])
    True
    >>> gated_by_test(["#[cfg(not(test))]"])
    False
    >>> gated_by_test(["#[cfg(any(kani, test))]"])
    False
    """
    return any(_excluded_without_test(attribute) for attribute in attributes)


def _excluded_without_test(attribute: str) -> bool:
    """Return whether one ``cfg`` attribute excludes the item under the gate."""
    found = _CFG_GUARD.match(attribute.strip())
    if found is None:
        return False
    return _predicate(found["body"]) is False


def _predicate(body: str) -> bool | None:
    """Evaluate one cfg predicate with ``test`` off and ``kani`` on.

    Unknown configuration -- a feature, a target, an OS -- is unknown rather
    than assumed, so only a predicate that is definitely false excludes an
    item from the gate's build.
    """
    body = body.strip()
    if body == "test":
        return False
    if body == "kani":
        return True
    found = _COMBINATOR.match(body)
    if found is None:
        return None
    values = [_predicate(argument) for argument in _arguments(found["args"])]
    return _combine(found["fn"], values)


def _arguments(body: str) -> list[str]:
    """Split one combinator's body at its top-level commas."""
    parts: list[str] = []
    depth = 0
    start = 0
    for index, character in enumerate(body):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        elif character == "," and depth == 0:
            parts.append(body[start:index])
            start = index + 1
    parts.append(body[start:])
    return [part for part in (part.strip() for part in parts) if part]


def _combine(name: str, values: list[bool | None]) -> bool | None:
    """Combine sub-predicates under one combinator's three-valued logic."""
    if name == "not":
        return _negate(values[0] if values else None)
    if name == "all":
        if False in values:
            return False
        return None if None in values else True
    if True in values:
        return True
    return None if None in values else False


def _negate(value: bool | None) -> bool | None:
    """Return the three-valued negation of `value`."""
    return None if value is None else not value