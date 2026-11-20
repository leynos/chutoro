"""The Rayon entry points the trim-scoring contract rejects.

The insertion critical section scores trim jobs on the calling thread, so any
body on the scoring path that hands work to the shared pool is the regression
the contract exists to catch. These are the shapes that do it, as (label,
test) pairs, and the argument-list reader that tells a `ThreadPool::join`
from a slice or path join.

This module holds the marker set alone. `trim_scoring_reading` holds the
pinned function list and applies these tests to the named bodies, and
`trim_scoring_samples` holds the synthetic sources that separate a correct
test from a weak one, so no one of the three outgrows the 400-line limit
`AGENTS.md` sets.

Re-use policy: import from `trim_scoring_reading` only, which applies the set.
A new marker belongs here with the shape it names stated in its comment, and a
sample that proves it belongs in `trim_scoring_samples`.
"""

from __future__ import annotations

import re
import typing as typ

#: A test over one function body: true when the body holds the shape.
_Predicate = typ.Callable[[str], bool]


def _matches(expression: str) -> _Predicate:
    """Return a predicate reporting whether a body matches `expression`.

    Nearly every marker is one regular expression, compiled once here rather
    than on every body. The pool join needs its argument list read and gets a
    predicate of its own, so the table has one entry shape either way.

    Examples
    --------
    >>> _matches(r"\\binto_par_iter\\b")("jobs.into_par_iter()")
    True
    >>> _matches(r"\\binto_par_iter\\b")("jobs.into_iter()")
    False
    """
    pattern = re.compile(expression)
    return lambda body: pattern.search(body) is not None


#: A `.join(` call, with `ThreadPool::join`'s optional turbofish allowed for.
_JOIN_CALL: typ.Final[re.Pattern[str]] = re.compile(
    r"\.join\s*(?:::<[^()]*>)?\s*\("
)

#: Brackets that nest an argument without ending the list.
_OPENING: typ.Final[frozenset[str]] = frozenset("([{")

#: Brackets that close an argument or the list.
_CLOSING: typ.Final[frozenset[str]] = frozenset(")]}")


def _pool_join(body: str) -> bool:
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
    >>> _pool_join("pool.join(|| left(), || right())")
    True
    >>> _pool_join("pool.join(make_job(), || ())")
    True
    >>> _pool_join("names.join(separator,)")
    False
    >>> _pool_join("names.join(separator)")
    False
    >>> _pool_join("pool.join(f(a, b))")
    False
    """
    return any(
        _has_second_argument(body, match.end())
        for match in _JOIN_CALL.finditer(body)
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
        char = text[index]
        if char == "<" and text[max(index - 2, start):index] == "::":
            index = _after_turbofish(text, index)
            continue
        if char in _OPENING:
            depth += 1
        elif char in _CLOSING:
            depth -= 1
            if depth == 0:
                return False
        elif char == "," and depth == 1 and _follows_a_value(text, index + 1):
            return True
        index += 1
    return False


def _after_turbofish(text: str, start: int) -> int:
    """Return the index just past the generic list opening at `start`.

    Expression-position generics are written ``::<...>``, so the commas in
    ``make_path::<A, B>()`` separate type arguments rather than the values
    the call is given. The ``::`` requirement is what keeps a comparison such
    as ``a < b`` from opening a list that never closes.

    A list may itself be generic: in ``make_path::<Outer<A, B>, C>()`` the
    commas inside ``Outer<A, B>`` belong to the inner list. Stopping at the
    first ``>`` ended the skip inside that inner list, leaving its comma to be
    read as an argument separator, so the angles are counted instead. An
    arrow in a function type closes nothing, since its ``>`` follows a ``-``.

    Examples
    --------
    >>> _after_turbofish("::<A, B>())", 0)
    8
    >>> _after_turbofish("::<Outer<A, B>, C>())", 0)
    18
    >>> _after_turbofish("::<fn() -> u8>())", 0)
    14
    >>> _after_turbofish("::<unclosed", 0)
    11
    """
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "<":
            depth += 1
        elif char == ">" and text[max(index - 1, 0)] != "-":
            depth -= 1
            if depth == 0:
                return index + 1
    return len(text)


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


#: Rayon entry points that hand a job to the shared pool, as (label, test)
#: pairs. A body matching any of these is dispatching rather than scoring in
#: place; the label is what a failure reports, so it names the shape that
#: matched rather than echoing the whole body.
#:
#: The patterns are boundary-aware because a raw substring is wrong in both
#: directions. `join(` appears in `names.join(", ")`, which is a slice join,
#: and `scope(` appears in `with_scope(`, an ordinary identifier; a body that
#: built an error message with either would fail the contract for neither. A
#: prelude import can also call `.par_windows()`, `.par_split()`, `.par_lines()`,
#: `.par_chars()`, `.par_drain()`, `broadcast()`, or `spawn_broadcast()`, and a
#: substring set without them reads such a body as serial. Free functions are
#: anchored so a path-qualified call is left to the `rayon::` marker and a
#: method of the same name to its own marker.
#:
#: `scope` and `spawn` have method forms on `ThreadPool` that run work in the
#: pool, so each is recognised with and without a receiver. The method patterns
#: are separate because a dot before the name rules out the free-function
#: anchor.
#:
#: Three method patterns are narrower than the rest. `ThreadPool::install`
#: and `ParallelExtend::par_extend` are generic, so a turbofish
#: (`pool.install::<_, _>(...)`, `out.par_extend::<_>(items)`) may sit between
#: the name and the argument list, and both forms are allowed for. `ThreadPool::join` takes two
#: arguments, whereas `[T]::join` and `Path::join` take one, so the receiver
#: form is read by walking its argument list: a comma at the list's own level
#: with a value after it means two arguments, whichever form the first one
#: takes, and a slice or path join is left alone however its single argument
#: is spelled or whether it carries a trailing comma.
#:
#: `into_par_iter` precedes `par_iter` so the more specific name is reported
#: for a parallel iterator, and `rayon::` precedes the bare calls so a
#: qualified call names its namespace rather than only its function.
DISPATCH_MARKERS: typ.Final[tuple[tuple[str, _Predicate], ...]] = (
    ("into_par_iter", _matches(r"\binto_par_iter\b")),
    ("par_iter_mut", _matches(r"\bpar_iter_mut\b")),
    ("par_iter", _matches(r"\bpar_iter\b")),
    (".par_bridge(", _matches(r"\.par_bridge\s*\(")),
    (".par_chunks", _matches(r"\.par_chunks")),
    # The prefix carries the reverse-chunk family too: `par_rchunks`,
    # `par_rchunks_mut` and both `_exact` forms all start with it, just as
    # `.par_chunks` carries its mutable and `_exact` forms.
    (".par_rchunks", _matches(r"\.par_rchunks")),
    (".par_sort", _matches(r"\.par_sort")),
    # `ParallelExtend::par_extend` is generic, so a turbofish may sit between
    # the name and the argument list, as it may for `install`.
    (".par_extend(", _matches(r"\.par_extend\s*(?:::<[^()]*>)?\s*\(")),
    (".par_windows", _matches(r"\.par_windows")),
    (".par_split", _matches(r"\.par_split")),
    (".par_lines", _matches(r"\.par_lines")),
    (".par_chars", _matches(r"\.par_chars")),
    (".par_drain", _matches(r"\.par_drain")),
    ("rayon::", _matches(r"rayon\s*::")),
    (".install(", _matches(r"\.install\s*(?:::<[^()]*>)?\s*\(")),
    ("scope(", _matches(r"(?<![.\w:])scope\s*\(")),
    (".scope(", _matches(r"\.scope\s*\(")),
    ("scope_fifo(", _matches(r"(?<![.\w:])scope_fifo\s*\(")),
    (".scope_fifo(", _matches(r"\.scope_fifo\s*\(")),
    ("spawn(", _matches(r"(?<![.\w:])spawn\s*\(")),
    (".spawn(", _matches(r"\.spawn\s*\(")),
    ("spawn_fifo(", _matches(r"(?<![.\w:])spawn_fifo\s*\(")),
    (".spawn_fifo(", _matches(r"\.spawn_fifo\s*\(")),
    ("spawn_broadcast(", _matches(r"(?<![.\w:])spawn_broadcast\s*\(")),
    (".spawn_broadcast(", _matches(r"\.spawn_broadcast\s*\(")),
    (".join(", _pool_join),
    ("join(", _matches(r"(?<![.\w:])join\s*\(")),
    ("join_context(", _matches(r"(?<![.\w:])join_context\s*\(")),
    ("broadcast(", _matches(r"(?<![.\w:])broadcast\s*\(")),
    (".broadcast(", _matches(r"\.broadcast\s*\(")),
)
