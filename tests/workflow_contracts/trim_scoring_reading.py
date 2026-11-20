"""Read the trim-scoring path and report any Rayon dispatch in its bodies.

The insertion critical section holds ``insert_mutex`` while it scores trim
jobs. Scoring used to run through ``into_par_iter()``, which let Rayon park the
mutex holder on a latch that only a worker blocked on the same mutex could
satisfy, so a parallel build hung outright.

This module is the reading: the pinned function list, the marker set and the
source-reading helpers that apply them. The synthetic sources it must accept
or reject live in ``trim_scoring_samples``, and the assertions over both live
in ``trim_scoring_dispatch_test``, so no one of the three outgrows the
400-line limit ``AGENTS.md`` sets.

The reading is lexical. It cannot see dispatch through a helper that is not
one of the five pinned functions, through an aliased import such as
``use rayon::spawn as run``, or inside a ``DataSource`` implementation. Those
limits are stated beside ``SCANNED_FILES`` so a passing scan is not over-read.
"""

from __future__ import annotations

import re
import typing as typ
from dataclasses import dataclass
from pathlib import Path

from rust_source_reading import function_body
from workflow_support import ROOT as REPO_ROOT

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
    the matching close. Only a comma found at the list's own level counts, and
    only when a value rather than the close follows it.

    Examples
    --------
    >>> _has_second_argument("left, right)", 0)
    True
    >>> _has_second_argument("left,)", 0)
    False
    >>> _has_second_argument("f(a, b))", 0)
    False
    """
    depth = 1
    for index, char in enumerate(text[start:], start):
        if char in _OPENING:
            depth += 1
        elif char in _CLOSING:
            depth -= 1
            if depth == 0:
                return False
        elif char == "," and depth == 1 and _follows_a_value(text, index + 1):
            return True
    return False


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
#: Two method patterns are narrower than the rest. `ThreadPool::install` is
#: generic, so its turbofish (`pool.install::<_, _>(...)`) sits between the
#: name and the argument list and is allowed for. `ThreadPool::join` takes two
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
    (".par_extend(", _matches(r"\.par_extend\s*\(")),
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


@dataclass(frozen=True)
class ScannedFile:
    """One source file and the functions on the scoring path it declares.

    Keeping the functions beside the path is what makes the scan a pinned
    list rather than a whole-file search: a dispatch in a function this
    record does not name belongs to some other concern and must not be
    attributed to scoring.

    Attributes
    ----------
    path
        Repository-relative source file the contract reads.
    functions
        Functions whose bodies must contain no dispatch marker.
    """

    path: Path
    functions: tuple[str, ...]


#: The direct helpers between the insertion guard and the distance callbacks.
#:
#: `insert_with_collector` is the transaction that holds `insert_mutex`;
#: `score_trim_jobs` and its metrics wrapper are the serial scorer; the
#: crate-private `run_trim_job` computes one job; `batch_distances_for_trim`
#: reaches the source. A dispatch cannot enter the critical section through
#: these five without appearing here.
#:
#: Blind spots, stated so a passing scan is not over-read:
#:
#: - A helper these functions call that is not listed here may dispatch.
#: - An aliased Rayon import can dispatch under another name.
#: - A `DataSource` implementation can dispatch on its own account; ADR-006
#:   records that limit and the `DataSource` documentation states it.
SCANNED_FILES: typ.Final[tuple[ScannedFile, ...]] = (
    ScannedFile(
        REPO_ROOT / "chutoro-core" / "src" / "hnsw" / "cpu" / "mod.rs",
        ("insert_with_collector",),
    ),
    ScannedFile(
        REPO_ROOT / "chutoro-core" / "src" / "hnsw" / "cpu" / "trim.rs",
        (
            "score_trim_jobs",
            "score_trim_jobs_with_metrics",
            "run_trim_job",
        ),
    ),
    ScannedFile(
        REPO_ROOT / "chutoro-core" / "src" / "hnsw" / "helpers.rs",
        ("batch_distances_for_trim",),
    ),
)

#: The source file that declares the current scorer.
TRIM_SOURCE: typ.Final[Path] = (
    REPO_ROOT / "chutoro-core" / "src" / "hnsw" / "cpu" / "trim.rs"
)

#: The function every fixture and the real source is read for.
SCORER: typ.Final[str] = "score_trim_jobs"

class MissingFunctionError(LookupError):
    """Raised when a function this contract must read is not in the source.

    Failing closed is the point. A reading that returned no marker for a
    source it could not find the function in would report the scorer as serial
    whenever the function was renamed or moved, which is exactly when the
    guard is worth having. The missing function is the one signal that the
    contract has stopped reading the code it claims to police.
    """


def dispatch_marker(source: str, name: str) -> str | None:
    """Return the first Rayon dispatch marker in `name`'s body, if any.

    The body is read through :func:`rust_source_reading.function_body`, so a
    marker written in a comment or a string literal is a mention rather than a
    dispatch, and a marker in a sibling function is not attributed to this
    one. A source that does not declare the function raises rather than
    reading as clean.

    Parameters
    ----------
    source
        The contents of one Rust source file.
    name
        The function to read.

    Examples
    --------
    >>> dispatch_marker(
    ...     "fn score_trim_jobs() { jobs.into_par_iter().collect() }",
    ...     "score_trim_jobs",
    ... )
    'into_par_iter'
    >>> dispatch_marker(
    ...     "fn score_trim_jobs() { work() }", "score_trim_jobs"
    ... ) is None
    True

    Returns
    -------
    str or None
        The first marker found, in `DISPATCH_MARKERS` order, or None when the
        body dispatches nothing.
    """
    body = function_body(source, name)
    if body is None:
        message = (
            f"{name} is not declared in this source; the contract cannot "
            "prove its body scores serially"
        )
        raise MissingFunctionError(message)
    return next(
        (label for label, matches in DISPATCH_MARKERS if matches(body)),
        None,
    )


def dispatch_markers(source: str, names: typ.Sequence[str]) -> dict[str, str | None]:
    """Return each named function's first dispatch marker.

    Every name is read, even after one reports a marker, so a single failing
    assertion can describe the whole pinned list. A missing declaration raises
    through :func:`dispatch_marker` rather than being skipped.

    Parameters
    ----------
    source
        The contents of one Rust source file.
    names
        The functions to read, in the order they appear on the scoring path.

    Examples
    --------
    >>> dispatch_markers(
    ...     "fn a() { work() }\\nfn b() { jobs.par_iter() }", ("a", "b")
    ... )
    {'a': None, 'b': 'par_iter'}

    Returns
    -------
    dict[str, str or None]
        One entry per name, in `names` order.
    """
    return {name: dispatch_marker(source, name) for name in names}


def scan_pinned_functions() -> dict[tuple[str, str], str | None]:
    """Read every pinned file and return a marker per (file, function).

    A file that cannot be read fails the scan: a contract that skipped an
    unreadable source would report the scoring path clean because it had read
    nothing.

    Returns
    -------
    dict[tuple[str, str], str or None]
        Keyed by repository-relative path and function name.
    """
    found: dict[tuple[str, str], str | None] = {}
    for entry in SCANNED_FILES:
        source = entry.path.read_text(encoding="utf-8")
        for name, marker in dispatch_markers(source, entry.functions).items():
            found[(str(entry.path.relative_to(REPO_ROOT)), name)] = marker
    return found
