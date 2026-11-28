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

from rust_scanning import (
    _Region,
    _blank,
    _brace_depths,
    _literal_spans,
    _matching_brace,
    _outside,
    _regions,
    _without_comments,
)

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


def function_body(text: str, name: str) -> str | None:
    """Return one named function's body as code, or None when it is absent.

    The extent runs from the opening brace through its matching brace, with
    comments and literals blanked and spacing around `::` and `(` collapsed.
    Blanking both is what makes this reading answer where a *call* is rather
    than where a name is written: a marker in a comment or a string literal is
    a mention, and a mention is not a dispatch. Blanking preserves positions,
    so the extent the brace scan finds is still the source's own.

    A source holding several functions is read function by function, so a
    marker in a sibling function is not attributed to this one. Names are
    compared exactly; a call, a mention in a comment and a declaration in a
    nested module are all a different name.

    Parameters
    ----------
    text
        The contents of one Rust source file.
    name
        The function whose body is wanted, without any qualifier.

    Examples
    --------
    >>> function_body("fn a() { work(); }", "a")
    '{ work(); }'
    >>> function_body("fn b() { work(); }", "a") is None
    True
    >>> 'work' in function_body("fn a() { // work\\n run(); }", "a")
    False
    >>> 'run' in function_body("fn a() { // work\\n run(); }", "a")
    True
    >>> 'into_par_iter' in function_body(
    ...     'fn a() { let note = "into_par_iter"; run(); }', "a"
    ... )
    False

    Returns
    -------
    str or None
        The body as code, or None when no function of that name is declared.
    """
    regions = _regions(text)
    # The declaration is found on the comment-blanked source, whose positions
    # are the text's own; the extent is then taken from the fully blanked
    # source so a brace or marker inside a literal cannot end the body early.
    comments = (region for region in regions if region.is_comment)
    declarations = _blank(text, comments)
    code = _blank(text, regions)
    literals = _literal_spans(regions)
    for match in _FUNCTION.finditer(declarations):
        if match["name"] != name:
            continue
        span = _body_span(declarations, literals, match.end())
        if span is None:
            return None
        return _normalised(code[span[0] : span[1]]).strip()
    return None


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
