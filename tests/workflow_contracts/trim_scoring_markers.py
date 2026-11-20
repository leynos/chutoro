"""The Rayon entry points the trim-scoring contract rejects.

The insertion critical section scores trim jobs on the calling thread, so any
body on the scoring path that hands work to the shared pool is the regression
the contract exists to catch. These are the shapes that do it, as (label,
test) pairs.

This module holds the marker set alone. `trim_scoring_call_reading` reads the
calls that need their argument lists understood, `trim_scoring_reading` holds
the pinned function list and applies these tests to the named bodies, and
`trim_scoring_samples` holds the synthetic sources that separate a correct
test from a weak one, so no one of the four outgrows the 400-line limit
`AGENTS.md` sets.

Re-use policy: import from `trim_scoring_reading` only, which applies the set.
A new marker belongs here with the shape it names stated in its comment, and a
sample that proves it belongs in `trim_scoring_samples`.
"""

from __future__ import annotations

import re
import typing as typ

from trim_scoring_call_reading import calls_the_method, pool_join

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
#: the name and the argument list, and both forms are allowed for. A generic
#: type may itself be parenthesized, as `install::<_, ()>` and
#: `join::<_, _, (), ()>` are, so the type list is skipped by counting angles
#: rather than by a pattern that rules parentheses out. `ThreadPool::join`
#: takes two arguments, whereas `[T]::join` and `Path::join` take one, so the
#: receiver form is read by walking its argument list: a comma at the list's
#: own level with a value after it means two arguments, whichever form the
#: first one takes, and a slice or path join is left alone however its single
#: argument is spelled or whether it carries a trailing comma. Both readings
#: live in `trim_scoring_call_reading`.
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
    (".par_extend(", calls_the_method("par_extend")),
    (".par_windows", _matches(r"\.par_windows")),
    (".par_split", _matches(r"\.par_split")),
    (".par_lines", _matches(r"\.par_lines")),
    (".par_chars", _matches(r"\.par_chars")),
    (".par_drain", _matches(r"\.par_drain")),
    ("rayon::", _matches(r"rayon\s*::")),
    (".install(", calls_the_method("install")),
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
    (".join(", pool_join),
    ("join(", _matches(r"(?<![.\w:])join\s*\(")),
    ("join_context(", _matches(r"(?<![.\w:])join_context\s*\(")),
    ("broadcast(", _matches(r"(?<![.\w:])broadcast\s*\(")),
    (".broadcast(", _matches(r"\.broadcast\s*\(")),
)