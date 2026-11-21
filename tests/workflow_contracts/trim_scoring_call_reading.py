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

Re-use policy: import from ``trim_scoring_markers`` only. The predicates
:func:`pool_join` and :func:`calls_the_method` serve the marker table, and
:func:`after_turbofish` and :func:`skip_whitespace` serve the table's readings
of a call through its type, as in ``ThreadPool::install::<_, ()>(pool, job)``.
The remaining helpers exist for these and are not contracts of their own. A
new call shape belongs here with the Rust syntax it models stated beside it.
"""

from __future__ import annotations

import re
import typing as typ

from trim_scoring_syntax_reading import (
    CLOSING,
    after_generics,
    after_turbofish,
    bracket_change,
    opens_a_value,
    skip_whitespace,
    turbofish_end,
)

#: A test over one function body: true when the body holds the shape.
_Predicate = typ.Callable[[str], bool]


def pool_join(body: str) -> bool:
    """Return whether `body` calls `ThreadPool::join`, which takes two ops.

    `[T]::join` and `Path::join` take one argument, so the two are told apart
    by the argument list rather than by what the first argument looks like: a
    top-level comma with a value after it means two arguments, whichever form
    they take; a trailing comma declares a single argument, and a comma inside
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
    >>> pool_join("path.join(<Paths<A, B> as HasPath>::path())")
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
    reason: ``make_path::<A, B>()`` is one value, not two, and neither is a
    qualified path such as ``<Paths<A, B> as HasPath>::path()``. Only a comma
    found at the list's own level counts, and only when a value rather than
    the close follows it.

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
        skipped = after_generics(text, index, start)
        if skipped is None:
            skipped = _after_qualified_path(text, index, start)
        if skipped is not None:
            index = skipped
            continue
        if text[index] == "," and _separates_arguments(depth, text, index):
            return True
        depth += bracket_change(text[index])
        if depth == 0:
            return False
        index += 1
    return False


def _after_qualified_path(text: str, index: int, start: int) -> int | None:
    """Return the index past a ``<Type as Trait>::`` path at `index`, if any.

    A qualified path is the other place a type list appears in an expression,
    and it opens with a bare ``<`` rather than ``::<``. Its commas separate
    type arguments, so ``join(<Paths<A, B> as HasPath>::path())`` passes one
    value rather than two. Only a path beginning a value is skipped, and a
    real path closes into ``::``, which the ``<`` of a comparison such as
    ``a < b`` does not.

    Examples
    --------
    >>> _after_qualified_path("<A as B>::c()", 0, 0)
    10
    >>> _after_qualified_path("a < b > ::c", 2, 0) is None
    True
    >>> _after_qualified_path("<A as B>", 0, 0) is None
    True
    """
    if text[index] != "<":
        return None
    if not opens_a_value(text, index, start):
        return None
    rest = skip_whitespace(text, after_turbofish(text, index))
    if not text.startswith("::", rest):
        return None
    return rest + 2


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
    index = skip_whitespace(body, name_end)
    turbofish = turbofish_end(body, index)
    if turbofish is not None:
        index = skip_whitespace(body, turbofish)
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
    return bool(rest) and rest[0] not in CLOSING
