"""Read the Rust syntax that decides where a call's argument list starts.

A call may carry a turbofish whose type arguments hold commas, brackets and
nested angle lists; the list may hold a unit or tuple type, an array with a
const-expression shift, or a function type's arrow. Counting these correctly
is what lets the trim-scoring scan find a call's own parenthesis and tell a
two-argument pool join from a one-argument slice or path join.

This module holds that syntax reading; ``trim_scoring_call_reading`` applies
it to the methods the marker set names. The split keeps both files inside the
400-line limit ``AGENTS.md`` sets.

Re-use policy: import from ``trim_scoring_call_reading`` and
``trim_scoring_markers`` only. A new syntax rule belongs here with the Rust
grammar it models stated beside it.
"""

from __future__ import annotations

import typing as typ

#: Brackets that nest an argument without ending the list.
OPENING: typ.Final[frozenset[str]] = frozenset("([{")

#: Brackets that close an argument or the list.
CLOSING: typ.Final[frozenset[str]] = frozenset(")]}")


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


def opens_a_value(text: str, index: int, start: int) -> bool:
    """Return whether the character at `index` begins a value of its own.

    A value begins the scanned list, follows the opening bracket, or follows
    the comma that ended the previous argument. Anywhere else the character
    follows an operand, as the ``<`` of ``a < b`` does. A reference prefix
    leaves the value a value: ``join(&<Paths<A, B> as HasPath>::path())``
    passes one borrowed value, and ``&mut`` marks its mutability, so the angle
    still opens it.

    Examples
    --------
    >>> opens_a_value("<A>::b()", 0, 0)
    True
    >>> opens_a_value("a < b", 2, 0)
    False
    >>> opens_a_value("x, <A>::b", 3, 0)
    True
    >>> opens_a_value("&<A>::b", 1, 0)
    True
    >>> opens_a_value("&mut <A>::b", 5, 0)
    True
    """
    before = text[start:index].rstrip()
    if not before:
        return True
    return before[-1] in "(," or before.endswith(("&", "&mut"))


def bracket_change(char: str) -> int:
    """Return how `char` changes argument-list nesting.

    Examples
    --------
    >>> bracket_change("(")
    1
    >>> bracket_change(")")
    -1
    >>> bracket_change(",")
    0
    """
    if char in OPENING:
        return 1
    if char in CLOSING:
        return -1
    return 0


def angle_change(text: str, index: int) -> int:
    """Return how the character at `index` changes generic-list depth.

    Examples
    --------
    >>> angle_change("a<b", 1)
    1
    >>> angle_change("a>b", 1)
    -1
    >>> angle_change("a-b", 1)
    0
    >>> angle_change("->", 1)
    0
    """
    if text[index] == "<":
        return 1
    if is_angle_close(text, index):
        return -1
    return 0


def is_angle_close(text: str, index: int) -> bool:
    """Return whether the character at `index` closes a generic list.

    An arrow in a function type closes nothing, since its ``>`` follows a
    ``-``.

    Examples
    --------
    >>> is_angle_close("a>b", 1)
    True
    >>> is_angle_close("->", 1)
    False
    >>> is_angle_close("a-b", 1)
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
            brackets += bracket_change(text[index])
            continue
        if text[index] in OPENING:
            brackets = 1
            continue
        depth += angle_change(text, index)
        if depth == 0:
            return index + 1
    return len(text)


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


def after_generics(text: str, index: int, start: int) -> int | None:
    """Return the index past a ``::<...>`` list opening at `index`, if any.

    Expression-position generics are written ``::<...>``, so the commas in
    ``make_path::<A, B>()`` separate type arguments rather than the values the
    call is given. The ``::`` requirement is what keeps a comparison such as
    ``a < b`` from opening a list that never closes.

    Examples
    --------
    >>> after_generics("::<A> f", 2, 0)
    5
    >>> after_generics("a < b", 2, 0) is None
    True
    """
    if text[index] != "<":
        return None
    if not text[start:index].rstrip().endswith("::"):
        return None
    return after_turbofish(text, index)
