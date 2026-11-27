"""Find the literal, comment and bracket regions of Rust source.

A Rust reader that searches raw text will read a proof declaration inside a
comment, a string literal, or a macro body as though it were code. This module
locates the regions where code stops being code: line and block comments, raw
and ordinary strings, character literals, and the brace depths that decide
where a block opens and closes. Callers blank those regions before scanning
for declarations, so a mention in text cannot be mistaken for syntax.

Re-use policy: import from :mod:`rust_source_reading`,
:mod:`rust_declaration_reading` and the Kani readers only. The lexing rules are
Rust's, so a second consumer should extend this module rather than restate
them.
"""

from __future__ import annotations

import collections.abc as cabc
import re
import typing as typ

def _block_comment(text: str, index: int) -> int:
    """Return the position after a block comment starting at `index`.

    Nested block comments are counted, because Rust permits them and a scan
    that stopped at the first `*/` would resume inside a comment.
    """
    depth = 0
    position = index
    while position < len(text):
        pair = text[position : position + 2]
        if pair in {"/*", "*/"}:
            depth += 1 if pair == "/*" else -1
            position += 2
            if depth == 0:
                return position
            continue
        position += 1
    return len(text)


def _line_comment(text: str, index: int) -> int:
    """Return the position after a line comment starting at `index`."""
    end = text.find("\n", index)
    return len(text) if end == -1 else end


def _raw_string(text: str, index: int) -> int:
    """Return the position after a raw string starting at `index`.

    A raw string has no escapes: it ends at the first quote followed by as many
    hashes as opened it. Reading it as an ordinary string would treat a
    backslash as an escape and a contained quote as a terminator, either of
    which leaves the scan inside code it should have skipped.
    """
    position = index + 1
    hashes = 0
    while position < len(text) and text[position] == "#":
        hashes += 1
        position += 1
    if position >= len(text) or text[position] != '"':
        return index + 1
    closing = '"' + "#" * hashes
    end = text.find(closing, position + 1)
    return len(text) if end == -1 else end + len(closing)


def _quoted(text: str, index: int, quote: str) -> int:
    """Return the position after a quoted literal, honouring backslash escapes."""
    position = index + 1
    while position < len(text) and text[position] != quote:
        position += 2 if text[position] == "\\" else 1
    return min(position + 1, len(text))


def _string_literal(text: str, index: int) -> int:
    """Return the position after a string or byte-string literal."""
    offset = 1 if text[index] == "b" else 0
    return _quoted(text, index + offset, '"')


#: The shape of a character literal, which is what tells one from a lifetime.
#: Rust admits one character, a simple escape, a byte escape or a unicode
#: escape, and nothing longer; `b'x'` is the byte form.
#:
#: Matched by shape rather than by "closes before the end of the line",
#: because a signature's lifetime and a later character literal sit on one
#: line as a matter of course. `fn f<'a>(x: &'a str) { assert_eq!(c, '}'); }`
#: is the case: reading from `'a` to the `'}'` puts the function's opening
#: brace inside a literal span, the body scan then skips that brace, and the
#: costly test is discovered by nothing.
_CHAR_LITERAL = re.compile(
    r"b?'(?:\\(?:x[0-9A-Fa-f]{2}|u\{[0-9A-Fa-f]{1,6}\}|.)|[^\\'\n])'"
)


def _char_literal(text: str, index: int) -> int:
    """Return the position after a character literal, or `index` plus one.

    A lone quote is Rust's lifetime marker, not a literal, so anything that is
    not shaped like a literal consumes one character and the text after it is
    read as the code it is.
    """
    found = _CHAR_LITERAL.match(text, index)
    return found.end() if found is not None else index + 1


class _Region(typ.NamedTuple):
    """One span of a source that is not plain code.

    Attributes
    ----------
    start
        Index of the region's first character.
    end
        Index one past its last.
    is_comment
        Whether the region is a comment rather than a literal. The two are
        blanked for different questions: braces must not be counted inside
        either, and a cost marker must still be found inside a literal,
        because one of the markers contains a string.
    """

    start: int
    end: int
    is_comment: bool


#: How a region reader is called: the text and the index its opener sits at,
#: answering the index just past what it consumed. Named so the table cannot
#: hold a reader of a different shape.
type _RegionReader = cabc.Callable[[str, int], int]

#: Openers for the regions a Rust source has that are not code, each with the
#: reader that consumes what it opens and whether it opens a comment. Order
#: matters: `r#"` and `b"` are tried before a bare quote, so `r#"` is not read
#: as an identifier followed by a string.
_OPENERS: typ.Final[tuple[tuple[str, _RegionReader, bool], ...]] = (
    ("/*", _block_comment, True),
    ("//", _line_comment, True),
    ('r"', _raw_string, False),
    ("r#", _raw_string, False),
    ('b"', _string_literal, False),
    ("b'", _char_literal, False),
    ('"', _string_literal, False),
    ("'", _char_literal, False),
)


def _region_at(text: str, index: int) -> _Region | None:
    """Return the region opening at `index`, or None when code does.

    A lone quote that opens nothing is a lifetime, and its reader returns the
    next position rather than a region; that reads as code, which is what it
    is.

    Returns
    -------
    _Region or None
        The region, or None when no opener matches or the match was a lifetime.
    """
    for opener, read, is_comment in _OPENERS:
        if not text.startswith(opener, index):
            continue
        end = read(text, index)
        if end > index + 1 or is_comment:
            return _Region(index, end, is_comment)
        return None
    return None


def _regions(text: str) -> list[_Region]:
    """Return every comment and literal region of a Rust source, in order.

    One scan serves two questions. Braces are counted only outside every
    region, because a brace in a literal or a comment closes nothing; cost
    markers are matched outside comments but inside literals, because
    `Command::new(env!("CARGO"))` contains one.

    Returns
    -------
    list[_Region]
        The regions, non-overlapping and in source order.
    """
    found: list[_Region] = []
    index = 0
    while index < len(text):
        region = _region_at(text, index)
        if region is None:
            index += 1
            continue
        found.append(region)
        index = max(region.end, index + 1)
    return found


def _blank(text: str, regions: cabc.Iterable[_Region]) -> str:
    """Return the text with the given regions replaced by spaces.

    Newlines are kept so a line comment does not swallow the line break, and
    positions are preserved so a body's extent is unchanged by blanking.
    """
    out = list(text)
    for region in regions:
        for index in range(region.start, region.end):
            if out[index] != "\n":
                out[index] = " "
    return "".join(out)


def _without_comments(text: str) -> str:
    """Return the source with comments blanked and literals kept.

    A marker inside a comment is a mention, not a cost. A marker inside a
    literal is part of a call: `Command::new(env!("CARGO"))` is one.
    """
    return _blank(text, (region for region in _regions(text) if region.is_comment))


def _literal_spans(regions: cabc.Iterable[_Region]) -> list[tuple[int, int]]:
    """Return the extents of the regions that are literals rather than comments.

    Brace counting skips these. A brace in a string, raw string, byte string or
    character literal closes nothing, and counting one would end a function
    body early and hide every marker after it.
    """
    return [
        (region.start, region.end) for region in regions if not region.is_comment
    ]


def _outside(index: int, spans: list[tuple[int, int]]) -> bool:
    """Return whether a position lies outside every given span."""
    return not any(start <= index < end for start, end in spans)


#: One brace's effect on nesting depth. Everything outside this mapping leaves
#: the depth alone, so a scan that consults it counts structure rather than
#: characters.
_BRACE_DELTA: typ.Final[dict[str, int]] = {"{": 1, "}": -1}


def _brace_depths(code: str) -> list[int]:
    """Return the nesting depth at every offset, counted from the file root.

    One pass serves every caller, so a depth is a list lookup rather than a
    rescan per declaration. ``code`` must already have its comments and
    literals blanked: a brace inside either opens or closes nothing, and
    counting one moves every later declaration to the wrong depth.

    Examples
    --------
    >>> _brace_depths("ab{cd}")
    [0, 0, 0, 1, 1, 1]
    """
    depths: list[int] = []
    depth = 0
    for character in code:
        depths.append(depth)
        depth += _BRACE_DELTA.get(character, 0)
    return depths


def _matching_brace(code: str, depths: list[int], open_offset: int) -> int | None:
    """Return the offset of the ``}`` closing the ``{`` at ``open_offset``.

    Depth counting is what stops a nested module or an ``impl`` from ending the
    span early, which would hide every declaration after it. An unbalanced
    source returns None, which the caller reads as "no such extent" rather than
    guessing at one.

    Examples
    --------
    >>> _matching_brace("ab{cd}", _brace_depths("ab{cd}"), 2)
    5
    >>> _matching_brace("ab{cd", _brace_depths("ab{cd"), 2) is None
    True
    """
    inside_depth = depths[open_offset] + 1
    return next(
        (
            offset
            for offset in range(open_offset + 1, len(depths))
            if code[offset] == "}" and depths[offset] == inside_depth
        ),
        None,
    )


