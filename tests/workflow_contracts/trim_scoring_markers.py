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

from trim_scoring_call_reading import (
    calls_the_method,
    pool_join,
    skip_whitespace,
    turbofish_end,
)

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
#: `ThreadPool`'s associated-function entry points that dispatch to the pool.
_ASSOCIATED_ENTRY_POINTS = (
    "install|join|scope|scope_fifo|in_place_scope|in_place_scope_fifo"
    "|spawn|spawn_fifo|spawn_broadcast|broadcast"
)
_ASSOCIATED_CALL = re.compile(
    rf"\bThreadPool\s*::\s*(?:{_ASSOCIATED_ENTRY_POINTS})\b"
)


def pool_associated_call(body: str) -> bool:
    """Return whether `body` calls a pool entry point through its type.

    Rust lets an inherent method be called through its type, so
    ``ThreadPool::scope(pool, |_| ())`` runs the same work as
    ``pool.scope(|_| ())``. The receiver patterns require a dot and the
    free-function anchors reject the ``::``, so without this reading the call
    meets no marker at all. An entry point may be generic and its type list
    may hold a unit type, so ``ThreadPool::install::<_, ()>(pool, job)`` is
    read by skipping the list with :func:`after_turbofish` before requiring
    the call's own parenthesis. ``ThreadPool::new`` builds a pool without
    dispatching to it, so only the entry points are named.

    Examples
    --------
    >>> pool_associated_call("ThreadPool::scope(pool, |_| score(jobs))")
    True
    >>> pool_associated_call("ThreadPool::install::<_, ()>(pool, job)")
    True
    >>> pool_associated_call("ThreadPool::new()")
    False
    >>> pool_associated_call("myThreadPool::scope(pool, |_| ())")
    False
    >>> pool_associated_call("ThreadPool :: scope (pool, |_| ())")
    True
    """
    for found in _ASSOCIATED_CALL.finditer(body):
        index = skip_whitespace(body, found.end())
        turbofish = turbofish_end(body, index)
        if turbofish is not None:
            index = skip_whitespace(body, turbofish)
        if body.startswith("(", skip_whitespace(body, index)):
            return True
    return False


#: Words that may stand before a free call without being its receiver. A
#: path or method ends in an identifier, and the word before a call is a
#: keyword rather than part of its name: ``return join(a, b)`` calls
#: Rayon's ``join`` while ``cache_dir.join(a)`` does not.
_CALL_PRECEDING_WORDS = frozenset(
    {
        "return",
        "break",
        "continue",
        "yield",
        "await",
        "else",
        "in",
        "move",
        "let",
        "match",
        "if",
        "while",
        "for",
        "as",
        "mut",
        "ref",
    }
)


def _ends_a_receiver(body: str, start: int) -> bool:
    """Return whether the text before a call at `start` names its receiver.

    The character before the name is read back past any spaces, because Rust
    permits whitespace around the dot. A ``.`` or ``:`` there leaves the call
    a method of something else, and a preceding identifier is the same
    unless the whole word is a keyword that cannot be a receiver.

    Examples
    --------
    >>> _ends_a_receiver("rejoin(a)", 2)
    True
    >>> _ends_a_receiver("cache_dir . join(a)", 12)
    True
    >>> _ends_a_receiver("return join(a)", 7)
    False
    >>> _ends_a_receiver("break 'outer join(a)", 13)
    False
    >>> _ends_a_receiver("join(a)", 0)
    False
    """
    before = body[:start].rstrip()
    if not before:
        return False
    last = before[-1]
    if last in "_.:":
        return True
    if not last.isalnum():
        return False
    words = before.split()
    if words[-1].startswith("'"):
        words = words[:-1]
    word = words[-1].rsplit(".", 1)[-1] if words else ""
    return word not in _CALL_PRECEDING_WORDS


def _past_group(body: str, start: int, index: int) -> int:
    """Return the index past a closing parenthesis that groups the name.

    ``(join)(a, b)`` calls the free function through a grouped operand: the
    parenthesis after the name closes the group rather than opening an
    argument list, so the list that follows it is the call's. A ``)`` that
    does not close a group opened before the name ends nothing here.

    Examples
    --------
    >>> _past_group("(join)(a)", 1, 5)
    6
    >>> _past_group("join)(a)", 0, 4)
    4
    """
    if not body[:start].rstrip().endswith("("):
        return index
    if body[index : index + 1] != ")":
        return index
    return index + 1


def free_call(name: str) -> _Predicate:
    """Return a reading of the free function `name`'s call.

    A free function is called without a receiver, so ``join(a, b)`` is
    Rayon's while ``cache_dir.join(a)`` and ``Path::join(a, b)`` are not. Rust
    permits whitespace around the dot, so ``cache_dir . join(a)`` is the same
    path join and has to be rejected too: the character before the name is
    read back past any spaces, and a ``.`` or ``::`` there leaves the call a
    method of something else. A keyword before the name is not a receiver, so
    ``return join(a, b)`` and ``break 'outer join(a, b)`` still read as the
    dispatches they are. Rayon's free functions are generic, so a turbofish
    may sit between the name and the argument list, as
    ``join::<_, _, (), ()>(|| (), || ())`` shows.

    Examples
    --------
    >>> calls_free_join = free_call("join")
    >>> calls_free_join("join(|| (), || ())")
    True
    >>> calls_free_join("join::<_, _, (), ()>(|| (), || ())")
    True
    >>> calls_free_join("join:: <_, _, (), ()>(|| (), || ())")
    True
    >>> calls_free_join("return join(|| (), || ())")
    True
    >>> calls_free_join("break 'outer join(|| (), || ())")
    True
    >>> calls_free_join("cache_dir.join(cache_file)")
    False
    >>> calls_free_join("cache_dir . join(cache_file)")
    False
    >>> calls_free_join("Path::join(a, b)")
    False
    >>> calls_free_join("rejoin(a)")
    False
    >>> calls_free_join("(join)(|| (), || ())")
    True
    """
    pattern = re.compile(rf"\b{name}\b")

    def calls(body: str) -> bool:
        """Return whether `body` calls `name` without a receiver.

        A candidate preceded by an identifier, `.` or `:` is a method,
        associated function, longer name or receiver-qualified call rather
        than the free function, unless that identifier is a keyword or a loop
        label after one, and one followed by a turbofish is read past it
        before the argument list is required.

        Examples
        --------
        >>> free_call("join")("join(a, b)")
        True
        >>> free_call("join")("cache_dir . join(a)")
        False
        """
        for found in pattern.finditer(body):
            if _ends_a_receiver(body, found.start()):
                continue
            index = _past_group(body, found.start(), found.end())
            turbofish = turbofish_end(body, skip_whitespace(body, index))
            if turbofish is not None:
                index = turbofish
            if body.startswith("(", skip_whitespace(body, index)):
                return True
        return False

    return calls


#: `into_par_iter` precedes `par_iter` so the more specific name is reported
#: for a parallel iterator, and `rayon::` precedes the bare calls so a
#: qualified call names its namespace rather than only its function.
#:
#: Rust permits whitespace between a receiver and its `.`, and the reading
#: normalises only the spacing around `::` and before `(`, so every receiver
#: pattern allows `\s*` after the dot: `jobs . par_chunks(2)` dispatches just
#: as `jobs.par_chunks(2)` does.
DISPATCH_MARKERS: typ.Final[tuple[tuple[str, _Predicate], ...]] = (
    ("into_par_iter", _matches(r"\binto_par_iter\b")),
    ("par_iter_mut", _matches(r"\bpar_iter_mut\b")),
    ("par_iter", _matches(r"\bpar_iter\b")),
    (".par_bridge(", _matches(r"\.\s*par_bridge\s*\(")),
    (".par_chunks", _matches(r"\.\s*par_chunks")),
    # The prefix carries the reverse-chunk family too: `par_rchunks`,
    # `par_rchunks_mut` and both `_exact` forms all start with it, just as
    # `.par_chunks` carries its mutable and `_exact` forms.
    (".par_rchunks", _matches(r"\.\s*par_rchunks")),
    # `par_chunk_by` starts with neither `par_chunks` nor `par_rchunks`, so
    # the chunk entries above leave it unread.
    (".par_chunk_by", _matches(r"\.\s*par_chunk_by")),
    (".par_sort", _matches(r"\.\s*par_sort")),
    # `ParallelExtend::par_extend` is generic, so a turbofish may sit between
    # the name and the argument list, as it may for `install`.
    (".par_extend(", calls_the_method("par_extend")),
    (".par_windows", _matches(r"\.\s*par_windows")),
    # `par_array_windows` contains neither `par_windows` nor `par_chunks`, so
    # it needs its own entry.
    (".par_array_windows", _matches(r"\.\s*par_array_windows")),
    (".par_split", _matches(r"\.\s*par_split")),
    (".par_lines", _matches(r"\.\s*par_lines")),
    (".par_chars", _matches(r"\.\s*par_chars")),
    # The three string adapters below are distinct words: `par_char_indices`
    # is not `par_chars` followed by text, and `par_matches` is not a prefix
    # of `par_match_indices`.
    (".par_char_indices", _matches(r"\.\s*par_char_indices")),
    (".par_bytes", _matches(r"\.\s*par_bytes")),
    (".par_encode_utf16", _matches(r"\.\s*par_encode_utf16")),
    (".par_matches", _matches(r"\.\s*par_matches")),
    (".par_match_indices", _matches(r"\.\s*par_match_indices")),
    (".par_drain", _matches(r"\.\s*par_drain")),
    ("rayon::", _matches(r"rayon\s*::")),
    ("ThreadPool::", pool_associated_call),
    (".install(", calls_the_method("install")),
    # Each receiver form precedes its free-function form so the more specific
    # entry reports the call. A free call is read by `free_call`, which rejects
    # a receiver even when whitespace separates it from its dot and accepts a
    # generic call's turbofish.
    (".scope(", calls_the_method("scope")),
    ("scope(", free_call("scope")),
    (".in_place_scope(", calls_the_method("in_place_scope")),
    ("in_place_scope(", free_call("in_place_scope")),
    (".in_place_scope_fifo(", calls_the_method("in_place_scope_fifo")),
    ("in_place_scope_fifo(", free_call("in_place_scope_fifo")),
    (".scope_fifo(", calls_the_method("scope_fifo")),
    ("scope_fifo(", free_call("scope_fifo")),
    (".spawn(", calls_the_method("spawn")),
    ("spawn(", free_call("spawn")),
    (".spawn_fifo(", calls_the_method("spawn_fifo")),
    ("spawn_fifo(", free_call("spawn_fifo")),
    (".spawn_broadcast(", calls_the_method("spawn_broadcast")),
    ("spawn_broadcast(", free_call("spawn_broadcast")),
    (".join(", pool_join),
    ("join(", free_call("join")),
    ("join_context(", free_call("join_context")),
    (".broadcast(", calls_the_method("broadcast")),
    ("broadcast(", free_call("broadcast")),
)
