"""Match `actions/cache` path patterns against a concrete path.

`actions/cache` resolves its `path` input through `@actions/glob`, so a
contract that compares those values as plain strings can be walked straight
past: `~/.cargo/bin/*` archives `cargo-kani` while never spelling its name.
The translation lives here, apart from the contracts that use it, because it
is a small grammar with its own edge cases and its own tests.

`@actions/glob` runs with `implicitDescendants` on, so matching a directory
sweeps its children in. `covers` folds that in, together with the case where
the pattern's fixed prefix already sits inside the path in question.
"""

from __future__ import annotations

import re


def normalize(path: str) -> str:
    """Strip a trailing separator so equivalent spellings compare equal.

    `actions/cache` accepts a directory named with or without a trailing
    separator, so `~/.kani` and `~/.kani/` describe the same archive. A
    comparison that treats them as different values reports no coverage for
    one of the two spellings.

    Parameters
    ----------
    path : str
        A cache path or pattern, as written in a workflow.

    Returns
    -------
    str
        The same path without its trailing separator, or ``"/"`` when the
        path was nothing but separators.

    Examples
    --------
    >>> normalize("~/.kani/")
    '~/.kani'
    >>> normalize("~/.kani")
    '~/.kani'
    """
    return path.rstrip("/") or "/"


#: Characters that make an `actions/cache` path a pattern rather than a
#: literal. The action resolves its paths through `@actions/glob`, so a
#: contract that compares them as plain strings can be walked straight past:
#: `~/.cargo/bin/*` archives `cargo-kani` while never spelling its name.
GLOB_METACHARACTERS = "*?["


def _has_glob(pattern: str) -> bool:
    """Report whether a declared path is a glob pattern."""
    return any(character in pattern for character in GLOB_METACHARACTERS)


#: One glob token, as a regex fragment and the number of characters it
#: consumed. Splitting the translation into matchers keeps the loop below a
#: single statement; a `while` with a branch per token kind is the shape
#: that turns a small grammar into an unreadable one.
GlobToken = tuple[str, int]


def _match_recursive_wildcard(pattern: str, index: int) -> GlobToken | None:
    """Match `**`, which crosses path separators."""
    return (".*", 2) if pattern.startswith("**", index) else None


def _match_wildcard(pattern: str, index: int) -> GlobToken | None:
    """Match `*`, which stays within one path segment."""
    return ("[^/]*", 1) if pattern[index] == "*" else None


def _match_single_character(pattern: str, index: int) -> GlobToken | None:
    """Match `?`, which stands for one non-separator character."""
    return ("[^/]", 1) if pattern[index] == "?" else None


def _match_bracket_expression(pattern: str, index: int) -> GlobToken | None:
    """Match `[...]`, rewriting `!` negation to the regex spelling."""
    if pattern[index] != "[":
        return None
    close = pattern.find("]", index + 1)
    if close == -1:
        return None
    body = pattern[index + 1 : close]
    if body.startswith("!"):
        body = f"^{body[1:]}"
    return (f"[{body}]", close + 1 - index)


def _match_literal(pattern: str, index: int) -> GlobToken:
    """Match anything else, as itself."""
    return (re.escape(pattern[index]), 1)


#: Ordered because `**` must be tried before `*`, and an unterminated `[`
#: must fall through to the literal matcher rather than being dropped.
GLOB_TOKEN_MATCHERS = (
    _match_recursive_wildcard,
    _match_wildcard,
    _match_single_character,
    _match_bracket_expression,
    _match_literal,
)


def _glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Translate an `@actions/glob` pattern into an equivalent regex."""
    fragments: list[str] = []
    index = 0
    while index < len(pattern):
        fragment, width = next(
            token
            for matcher in GLOB_TOKEN_MATCHERS
            if (token := matcher(pattern, index)) is not None
        )
        fragments.append(fragment)
        index += width
    return re.compile("".join(fragments) + r"\Z")


def _literal_prefix(pattern: str) -> str:
    """Return the leading segments of a pattern that contain no wildcard.

    Everything a pattern can match lives under this directory, which is what
    makes it the right thing to compare against when asking whether the
    archive would reach inside a rejected tree.
    """
    kept: list[str] = []
    for segment in pattern.split("/"):
        if _has_glob(segment):
            break
        kept.append(segment)
    return "/".join(kept) or "/"


def _ancestors(path: str) -> list[str]:
    """Return every proper ancestor directory of a path, shallowest first."""
    segments = path.split("/")
    return ["/".join(segments[:count]) for count in range(1, len(segments))]


def covers(declared: str, rejected: str) -> bool:
    """Report whether a declared cache path would archive a rejected one.

    Exact equality is not enough, in three separate ways. `actions/cache`
    accepts a directory with or without a trailing separator. Caching a
    parent sweeps its children in, so `~/.cargo/bin` archives `cargo-kani`
    just as surely as naming it; `@actions/glob` sets `implicitDescendants`,
    so a pattern matching that parent does the same. And a child counts too:
    `~/.kani-rustup/toolchains` is most of the 1.3 GB.

    The answer is deliberately conservative. A pattern whose fixed prefix
    lands inside the rejected tree counts even when the wildcard part might
    match nothing, because a contract that guesses in the permissive
    direction is one that lets the archive back in.

    Parameters
    ----------
    declared : str
        The `path` value a cache step declares. May be a glob pattern.
    rejected : str
        A concrete path that must never be archived.

    Returns
    -------
    bool
        True when `declared` would archive `rejected`, in whole or in part.

    Examples
    --------
    >>> covers("~/.cargo/bin/*", "~/.cargo/bin/cargo-kani")
    True
    >>> covers("~/.kani-rustup/toolchains", "~/.kani-rustup")
    True
    >>> covers("~/.cargo/bin/whitaker-*", "~/.cargo/bin/cargo-kani")
    False
    """
    left, right = normalize(declared), normalize(rejected)
    matches = _glob_to_regex(left).fullmatch
    if matches(right):
        return True
    if any(matches(ancestor) for ancestor in _ancestors(right)):
        return True
    prefix = normalize(_literal_prefix(left))
    return prefix == right or prefix.startswith(f"{right}/")
