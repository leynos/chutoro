"""Read a Rust source for the tests that pay a nested build's cost.

`trybuild_override_test` states the policy; this module supplies the reading it
states the policy in. They are separated because the reading has its own cases,
a Rust source having comments, four kinds of literal and a lifetime marker that
looks like a fifth, and because the policy module reached the 400-line limit
`AGENTS.md` sets once those cases arrived.

Nothing here asserts anything. A reader that fails closed returns nothing and
leaves the caller to decide whether that is an offence.

It is not a Rust parser and does not try to be. It knows where code stops and a
comment or a literal begins, which is all that is needed to count braces and to
tell a call from a mention. Everything it cannot read reads as code, so the
failure direction is over-reporting, which the pinned set in the policy module
catches.
"""

import collections.abc as cabc
import pathlib
import re
import typing as typ

from workflow_support import ROOT as REPO_ROOT

#: Sources of the cost this contract is about. A constructed ``TestCases``
#: compiles a scratch crate; a spawned ``cargo`` builds a fixture workspace.
#: Naming the crate is not enough for either: an import that is never used, or
#: a paragraph explaining why a harness was removed, costs nothing.
COST_MARKERS: typ.Final[tuple[str, ...]] = (
    "TestCases::new(",
    'Command::new(env!("CARGO"))',
    'Command::new("cargo")',
)

#: Whitespace Rust permits around a path separator or before a call's
#: parenthesis, which the formatter removes. Normalised rather than depended
#: upon, so the reading does not rest on the formatter having run.
_SPACED_SYNTAX = re.compile(r"\s*(::)\s*|\s+(\()")

#: A test function's declaration. `#[test]` and its cfg attributes may sit
#: between the attribute and the signature, so the name is taken from the
#: signature and the attribute is found by walking backwards.
#:
#: Every shape a test may take, because the narrow ones fail in the
#: under-reporting direction this module exists to close. `async fn` is the
#: only way a `#[tokio::test]` or `#[async_std::test]` test is written, and
#: both attributes are named below, so a matcher without `async` disagreed
#: with the attribute set it is paired with. A `fn` inside `mod tests { ... }`
#: is indented. A name may carry an uppercase character. A test in any of
#: those shapes was discovered by nothing, and a pinned set records only the
#: tests the tree holds today, so the missing override passed.
#:
#: The trailing `\(` stays: the body scan starts from the match's end.
_FUNCTION = re.compile(
    r"^[ \t]*"
    r"(?:pub(?:\s*\([^)]*\))?[ \t]+)?"
    r"(?:async[ \t]+)?"
    r"(?:unsafe[ \t]+)?"
    r"(?:extern[ \t]+\"[^\"]*\"[ \t]+)?"
    r"fn[ \t]+(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    # An `#[rstest]` function may be generic or carry a lifetime, and the list
    # nests (`<T: Into<String>>`), so it is consumed up to the last `>` before
    # the parameter list rather than by a bracket-counting pattern.
    r"(?:\s*<[^{;]*?>)?\s*\(",
    re.MULTILINE,
)


#: Openers for the regions a Rust source has that are not code, each with the
#: reader that consumes one. Raw strings come before ordinary ones so `r#"` is
#: not read as an identifier followed by a string.
def _normalised(text: str) -> str:
    """Return the text with spacing around `::` and `(` removed.

    Rust permits the spacing and this repository's formatter removes it, so the
    reading normalises rather than depending on the formatter having run.

    Applied to a body once its extent is known, never to the whole source:
    collapsing spacing changes the text's length, and every index used to find
    that extent is the source's own.

    Returns
    -------
    str
        The text with that spacing collapsed.
    """
    return _SPACED_SYNTAX.sub(lambda found: found.group(1) or found.group(2), text)


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


#: Each opener, the reader that consumes what it opens, and whether it opens a
#: comment. Order matters: `r#"` and `b"` are tried before a bare quote.
#: How a region reader is called: the text and the index its opener sits at,
#: answering the index just past what it consumed. Named so the table cannot
#: hold a reader of a different shape.
type _RegionReader = cabc.Callable[[str, int], int]

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


def _body_span(
    text: str, literals: list[tuple[int, int]], start: int
) -> tuple[int, int] | None:
    """Return one function body's extent, by matching braces from its signature.

    Braces inside a literal are skipped, and the text is already comment-free,
    so only a brace in code opens or closes the body. Counting one from a
    literal would end the body early and hide every marker after it, and the
    pinned set would then read as a repository with one fewer costly test
    rather than failing.

    Positions are the source's own throughout, because blanking preserves
    them; the text is not re-indexed at any point.

    Returns
    -------
    tuple[int, int] or None
        The half-open extent of the body, or None when no brace in code
        follows the signature. Unbalanced braces yield the rest of the source,
        which reads as a larger body and so can only over-report.
    """
    opening = next(
        (
            index
            for index in range(start, len(text))
            if text[index] == "{" and _outside(index, literals)
        ),
        None,
    )
    if opening is None:
        return None
    depth = 0
    for index in range(opening, len(text)):
        if not _outside(index, literals):
            continue
        depth += {"{": 1, "}": -1}.get(text[index], 0)
        if depth == 0:
            return opening, index + 1
    return opening, len(text)


#: Attributes that make a function a test nextest will run and therefore
#: bound. `#[test]` alone is not enough: this repository declares tests with
#: `rstest` as a matter of course, and a costly test written that way would be
#: discovered by nothing while the pinned set stayed unchanged.
#:
#: Matched on the attribute's path rather than the whole attribute, so
#: `#[rstest]`, `#[rstest(...)]` and `#[tokio::test]` are all recognised and
#: `#[test_only_helper]` is not.
_TEST_ATTRIBUTES: typ.Final[frozenset[str]] = frozenset({
    "test",
    "rstest",
    "tokio::test",
    "async_std::test",
})

_ATTRIBUTE_PATH = re.compile(r"^#\[\s*(?P<path>[A-Za-z_][A-Za-z0-9_:]*)")


def _is_test(text: str, start: int) -> bool:
    """Return whether the declaration at `start` is attributed as a test.

    Attributes are read backwards from the signature, because `#[cfg(...)]`
    and doc comments may sit between the two.

    Returns
    -------
    bool
        True when one of the preceding attributes names a test attribute.
    """
    for line in reversed(text[:start].rstrip().split("\n")):
        stripped = line.strip()
        # The caller walks the comment-blanked source, so a comment between the
        # attribute and the signature arrives here as a line of spaces. Nothing
        # tests for a comment prefix, because none survives the blanking; the
        # walk steps over the blank instead. A walk that stopped at it found no
        # attribute at all, and the test it was reading paid a nested build's
        # cost with nothing naming an allowance for it.
        if not stripped:
            continue
        # Rust permits spacing around `::`, so `#[tokio :: test]` is the same
        # attribute as `#[tokio::test]`. Normalising first keeps the attribute
        # set and the recognition of it from disagreeing over whitespace.
        found = _ATTRIBUTE_PATH.match(_normalised(stripped))
        if found is not None and found["path"] in _TEST_ATTRIBUTES:
            return True
        if not stripped.startswith("#["):
            return False
    return False


def compile_contract_tests(text: str) -> list[str]:
    """Return every test in the source that pays a nested build's cost.

    A test pays it by constructing a `trybuild::TestCases` harness, which
    compiles a scratch crate against this workspace's dependency graph, or by
    spawning a nested `cargo`, which builds a fixture workspace. Both cost what
    a cold build costs and both overrun the base allowance.

    The reading is per test function rather than per file, because the defect
    this contract exists to catch is a sibling: one test of a pair named in an
    override and the other left on the base allowance.

    Comments are excluded and literals are not. A marker in a comment is a
    mention, and this repository has a module whose documentation explains at
    length why a trybuild harness was removed; a marker inside a literal is
    part of a call, because one of the markers contains a string.

    Parameters
    ----------
    text
        The contents of one Rust source file.

    Examples
    --------
    >>> compile_contract_tests(
    ...     "#[test]\\nfn a() { let c = trybuild::TestCases::new(); }"
    ... )
    ['a']
    >>> compile_contract_tests("#[test]\\nfn b() { assert!(true); }")
    []
    >>> compile_contract_tests("fn c() { let c = TestCases::new(); }")
    []
    >>> compile_contract_tests(
    ...     "#[rstest]\\nfn d() { let c = trybuild::TestCases::new(); }"
    ... )
    ['d']

    Returns
    -------
    list[str]
        The name of each test whose body constructs a trybuild harness or
        spawns a nested cargo, in declaration order.
    """
    regions = _regions(text)
    # One text, with comments blanked and literals kept, so every index below
    # is the source's own. Normalising the whole source first would change its
    # length and the spans would no longer line up.
    source = _blank(text, (region for region in regions if region.is_comment))
    literals = _literal_spans(regions)
    found = []
    for match in _FUNCTION.finditer(source):
        if not _is_test(source, match.start()):
            continue
        span = _body_span(source, literals, match.end())
        if span is None:
            continue
        body = _normalised(source[span[0] : span[1]])
        if any(marker in body for marker in COST_MARKERS):
            found.append(match["name"])
    return found


def discovered_tests(root: pathlib.Path = REPO_ROOT) -> dict[str, str]:
    """Return each compile-contract test, mapped to the file declaring it.

    Parameters
    ----------
    root : pathlib.Path
        The tree to read. Defaults to this repository, which is what every
        caller here wants; it is a parameter so the sweep can be driven over a
        tree built for the purpose. Parameterized over this repository's own
        sources the sweep agrees with itself whatever it does, because the
        answer it gives is the answer the assertion is written from.

    Returns
    -------
    dict[str, str]
        Test name to the path of its source file, relative to `root`.

    Notes
    -----
    An unreadable or undecodable source raises rather than being skipped. A
    sweep that swallowed the error would report a smaller set, and a smaller
    set is exactly how a costly test goes missing from the override list
    without anything failing. Raising names the file.
    """
    return {
        name: path.relative_to(root).as_posix()
        for path in sorted(root.rglob("tests/**/*.rs"))
        if "target" not in path.parts
        for name in compile_contract_tests(path.read_text(encoding="utf-8"))
    }
