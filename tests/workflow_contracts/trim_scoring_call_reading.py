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
        skipped = _after_generics(text, index, start)
        if skipped is None:
            skipped = _after_qualified_path(text, index, start)
        if skipped is not None:
            index = skipped
            continue
        if text[index] == "," and _separates_arguments(depth, text, index):
            return True
        depth += _bracket_change(text[index])
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
    if not _opens_a_value(text, index, start):
        return None
    rest = skip_whitespace(text, after_turbofish(text, index))
    if not text.startswith("::", rest):
        return None
    return rest + 2


def _opens_a_value(text: str, index: int, start: int) -> bool:
    """Return whether the character at `index` begins a value of its own.

    A value begins the scanned list, follows the opening bracket, or follows
    the comma that ended the previous argument. Anywhere else the character
    follows an operand, as the ``<`` of ``a < b`` does. A reference prefix
    leaves the value a value: ``join(&<Paths<A, B> as HasPath>::path())``
    passes one borrowed value, and ``&mut`` marks its mutability, so the angle
    still opens it.

    Examples
    --------
    >>> _opens_a_value("<A>::b()", 0, 0)
    True
    >>> _opens_a_value("a < b", 2, 0)
    False
    >>> _opens_a_value("x, <A>::b", 3, 0)
    True
    >>> _opens_a_value("&<A>::b", 1, 0)
    True
    >>> _opens_a_value("&mut <A>::b", 5, 0)
    True
    """
    before = text[start:index].rstrip()
    if not before:
        return True
    return before[-1] in "(," or before.endswith(("&", "&mut"))


def turbofish_end(body: str, index: int) -> int | None:
    """Return the index past a turbofish starting at `index`, if one does.

    Rust permits whitespace between ``::`` and ``<``, so the two tokens are
    read separately; the type list itself is read by counting its angles.

    Examples
    --------
    >>> turbofish_end("::<_, ()>(x)", 0)
    9
    >>> turbofish_end(":: <_, ()>(x)", 0)
    10
    >>> turbofish_end("(x)", 0) is None
    True
    """
    if not body.startswith("::", index):
        return None
    angle = skip_whitespace(body, index + 2)
    if not body.startswith("<", angle):
        return None
    return after_turbofish(body, angle)


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
    if not text[start:index].rstrip().endswith("::"):
        return None
    return after_turbofish(text, index)


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


def after_turbofish(text: str, start: int) -> int:
    """Return the index just past the generic list whose ``<`` is at `start`.

    A list may itself be generic, so in ``make_path::<Outer<A, B>, C>()`` the
    commas inside ``Outer<A, B>`` belong to the inner list and the angles are
    counted rather than the skip stopping at the first ``>``. A bracketed
    region holds expressions rather than type delimiters: the ``<<`` of
    ``[u8; { 1 << 2 }]`` is a shift, so the region is skipped whole, while a
    function type's arrow is still not a close. An unclosed list ends at the
    end of the text.

    Examples
    --------
    >>> after_turbofish("::<A, B>())", 2)
    8
    >>> after_turbofish("::<Outer<A, B>, C>())", 2)
    18
    >>> after_turbofish("::<fn() -> u8>())", 2)
    14
    >>> after_turbofish("::<_, [u8; { 1 << 2 }]>(x)", 2)
    23
    >>> after_turbofish("::<unclosed", 2)
    11
    """
    depth = 1
    brackets = 0
    for index in range(start + 1, len(text)):
        if brackets:
            brackets += _bracket_change(text[index])
            continue
        if text[index] in _OPENING:
            brackets = 1
            continue
        depth += _angle_change(text, index)
        if depth == 0:
            return index + 1
    return len(text)


def skip_whitespace(text: str, index: int) -> int:
    """Return the index of the first non-space character at or after `index`.

    Examples
    --------
    >>> skip_whitespace("  x", 0)
    2
    >>> skip_whitespace("x", 1)
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
    return bool(rest) and rest[0] not in _CLOSING
