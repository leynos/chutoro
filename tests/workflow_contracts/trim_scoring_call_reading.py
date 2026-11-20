"""Read a Rust method call's argument list, turbofish and all.

A dispatch marker such as ``.install(`` or ``.join(`` is a call, and whether a
body contains one decides whether it scores in place. Matching the name alone
is not enough: a turbofish may sit between the name and the argument list, and
its angle brackets may hold commas, parentheses or an arrow. ``.join`` is
worse still, because ``ThreadPool::join`` takes two operations while
``[T]::join`` and ``Path::join`` take one, so the two are told apart only by
reading the list itself.

This module holds those readings. ``trim_scoring_markers`` keeps the marker
table that uses them, so neither file outgrows the 400-line limit ``AGENTS.md``
sets.

Re-use policy: import from ``trim_scoring_markers`` only. The two public
predicates, :func:`pool_join` and :func:`calls_the_method`, are the whole
interface; the bracket and angle helpers exist to serve them and are not
contracts of their own. A new call shape belongs here with the Rust syntax it
models stated beside it.
"""

from __future__ import annotations

import re
import typing as typ

#: A test over one function body: true when the body holds the shape.
_Predicate = typ.Callable[[str], bool]

#: Brackets that nest an argument without ending the list.
_OPENING: typ.Final[frozenset[str]] = frozenset("([{")

#: Brackets that close an argument or the list.
_CLOSING: typ.Final[frozenset[str]] = frozenset(")]}")


def pool_join(body: str) -> bool:
    """Return whether `body` calls `ThreadPool::join`, which takes two ops.

    `[T]::join` and `Path::join` take one argument, so the two are told apart
    by the argument list rather than by what the first argument looks like: a
    top-level comma with a value after it means two arguments, whichever form
    they take. The first may be a closure, `move` or not, a bound callable, or
    a call that builds one. Reading the list also settles what a pattern
    could not: a trailing comma declares a single argument, and a comma inside
    a nested call or a string literal is not a separator. Literals reach here
    already blanked by :func:`rust_source_reading.function_body`.

    Examples
    --------
    >>> pool_join("pool.join(|| left(), || right())")
    True
    >>> pool_join("pool.join::<_, _, (), ()>(|| left(), || right())")
    True
    >>> pool_join("pool.join(make_job(), || ())")
    True
    >>> pool_join("names.join(separator,)")
    False
    >>> pool_join("names.join(separator)")
    False
    >>> pool_join("pool.join(f(a, b))")
    False
    """
    return any(
        _has_second_argument(body, start)
        for start in _argument_starts(body, "join")
    )


def _has_second_argument(text: str, start: int) -> bool:
    """Return whether the argument list opening before `start` holds two.

    The scan runs from just inside the opening parenthesis, counting nesting
    so a comma inside a nested call is not read as a separator, and stops at
    the matching close. A turbofish's type arguments are skipped for the same
    reason: ``make_path::<A, B>()`` is one value, not two. Only a comma found
    at the list's own level counts, and only when a value rather than the
    close follows it.

    Examples
    --------
    >>> _has_second_argument("left, right)", 0)
    True
    >>> _has_second_argument("left,)", 0)
    False
    >>> _has_second_argument("f(a, b))", 0)
    False
    >>> _has_second_argument("make_path::<A, B>())", 0)
    False
    """
    depth = 1
    index = start
    while index < len(text):
        generics_end = _after_generics(text, index, start)
        if generics_end is not None:
            index = generics_end
            continue
        if text[index] == "," and _separates_arguments(depth, text, index):
            return True
        depth += _bracket_change(text[index])
        if depth == 0:
            return False
        index += 1
    return False


def _after_generics(text: str, index: int, start: int) -> int | None:
    """Return the index past a ``::<...>`` list opening at `index`, if any.

    Expression-position generics are written ``::<...>``, so the commas in
    ``make_path::<A, B>()`` separate type arguments rather than the values the
    call is given. The ``::`` requirement is what keeps a comparison such as
    ``a < b`` from opening a list that never closes.

    Examples
    --------
    >>> _after_generics("::<A> f", 2, 0)
    5
    >>> _after_generics("a < b", 2, 0) is None
    True
    """
    if text[index] != "<":
        return None
    if text[max(index - 2, start):index] != "::":
        return None
    return _after_turbofish(text, index)


def _separates_arguments(depth: int, text: str, index: int) -> bool:
    """Return whether the comma at `index` separates two arguments.

    Only a comma at the list's own level separates arguments, and a trailing
    comma before the close separates none.

    Examples
    --------
    >>> _separates_arguments(1, "a, b)", 1)
    True
    >>> _separates_arguments(2, "f(a, b), c)", 3)
    False
    >>> _separates_arguments(1, "a,)", 1)
    False
    """
    if depth != 1:
        return False
    return _follows_a_value(text, index + 1)


def _bracket_change(char: str) -> int:
    """Return how `char` changes argument-list nesting.

    Examples
    --------
    >>> _bracket_change("(")
    1
    >>> _bracket_change(")")
    -1
    >>> _bracket_change(",")
    0
    """
    if char in _OPENING:
        return 1
    if char in _CLOSING:
        return -1
    return 0


def _angle_change(text: str, index: int) -> int:
    """Return how the character at `index` changes generic-list depth.

    Examples
    --------
    >>> _angle_change("a<b", 1)
    1
    >>> _angle_change("a>b", 1)
    -1
    >>> _angle_change("a-b", 1)
    0
    >>> _angle_change("->", 1)
    0
    """
    if text[index] == "<":
        return 1
    if _is_angle_close(text, index):
        return -1
    return 0


def _is_angle_close(text: str, index: int) -> bool:
    """Return whether the character at `index` closes a generic list.

    An arrow in a function type closes nothing, since its ``>`` follows a
    ``-``.

    Examples
    --------
    >>> _is_angle_close("a>b", 1)
    True
    >>> _is_angle_close("->", 1)
    False
    >>> _is_angle_close("a-b", 1)
    False
    """
    if text[index] != ">":
        return False
    return text[max(index - 1, 0)] != "-"


def _after_turbofish(text: str, start: int) -> int:
    """Return the index just past the generic list whose ``<`` is at `start`.

    A list may itself be generic: in ``make_path::<Outer<A, B>, C>()`` the
    commas inside ``Outer<A, B>`` belong to the inner list. Stopping at the
    first ``>`` ended the skip inside that inner list, leaving its comma to be
    read as an argument separator, so the angles are counted instead. An
    unclosed list ends at the end of the text.

    Examples
    --------
    >>> _after_turbofish("::<A, B>())", 2)
    8
    >>> _after_turbofish("::<Outer<A, B>, C>())", 2)
    18
    >>> _after_turbofish("::<fn() -> u8>())", 2)
    14
    >>> _after_turbofish("::<unclosed", 2)
    11
    """
    depth = 1
    for index in range(start + 1, len(text)):
        depth += _angle_change(text, index)
        if depth == 0:
            return index + 1
    return len(text)


def _skip_whitespace(text: str, index: int) -> int:
    """Return the index of the first non-space character at or after `index`.

    Examples
    --------
    >>> _skip_whitespace("  x", 0)
    2
    >>> _skip_whitespace("x", 1)
    1
    """
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def _argument_start(body: str, name_end: int) -> int | None:
    """Return the index just inside a call's argument list, if it follows.

    A method may carry a turbofish between its name and its arguments, and the
    type list may hold a unit or tuple type whose parentheses no fixed-depth
    pattern can count. The angles are counted instead, so
    ``pool.install::<_, ()>(...)`` is still read as a call and does not slip
    past the contract.

    Examples
    --------
    >>> _argument_start("(args)", 0)
    1
    >>> _argument_start("::<_, ()>(args)", 0)
    10
    >>> _argument_start("::<_, ()>", 0) is None
    True
    """
    index = _skip_whitespace(body, name_end)
    if body.startswith("::<", index):
        index = _skip_whitespace(body, _after_turbofish(body, index + 2))
    if body.startswith("(", index):
        return index + 1
    return None


def _argument_starts(body: str, name: str) -> typ.Iterator[int]:
    """Yield the index just inside each call to the method `name`.

    Examples
    --------
    >>> list(_argument_starts("pool.join(a, b)", "join"))
    [10]
    >>> list(_argument_starts("pool. join(a)", "join"))
    [11]
    >>> list(_argument_starts("pool.join_context(a)", "join"))
    []
    """
    pattern = re.compile(rf"\.\s*{re.escape(name)}\b")
    for match in pattern.finditer(body):
        start = _argument_start(body, match.end())
        if start is not None:
            yield start


def calls_the_method(name: str) -> _Predicate:
    """Return a predicate reporting whether a body calls the method `name`.

    Examples
    --------
    >>> calls_the_method("install")("pool.install::<_, ()>(|| ())")
    True
    >>> calls_the_method("install")("pool.installed(|| ())")
    False
    """
    return lambda body: next(_argument_starts(body, name), None) is not None


def _follows_a_value(text: str, start: int) -> bool:
    """Return whether a value rather than a closing bracket follows `start`.

    A trailing comma is legal before the close, so its comma is not evidence
    of a second argument.

    Examples
    --------
    >>> _follows_a_value(" right)", 0)
    True
    >>> _follows_a_value(")", 0)
    False
    >>> _follows_a_value("  )", 0)
    False
    """
    rest = text[start:].lstrip()
    return bool(rest) and rest[0] not in _CLOSING