"""The trim-scoring marker set, one entry at a time.

The scan applies every marker in ``trim_scoring_markers`` to a pinned body;
this module proves each entry is load-bearing. An entry nobody detects would
sit in the set looking like coverage while the dispatch it names went unread,
so each case builds a body from the call the entry exists to recognise and
requires that entry's label back.

It lives apart from ``trim_scoring_dispatch_test`` because the table of calls
and the tests that read the real source would otherwise push one file past the
400-line limit ``AGENTS.md`` sets.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import pytest
from trim_scoring_reading import SCORER, dispatch_marker


@pytest.mark.parametrize(
    ("marker", "call"),
    [
        pytest.param("into_par_iter", "jobs.into_par_iter()", id="into-par-iter"),
        pytest.param("par_iter_mut", "jobs.par_iter_mut()", id="par-iter-mut"),
        pytest.param("par_iter", "jobs.par_iter()", id="par-iter"),
        pytest.param(".par_bridge(", "jobs.par_bridge()", id="par-bridge"),
        pytest.param(".par_chunks", "items.par_chunks(2)", id="par-chunks"),
        pytest.param(
            ".par_chunks", "items . par_chunks(2)", id="a-spaced-par-chunks"
        ),
        pytest.param(".par_sort", "items.par_sort()", id="par-sort"),
        pytest.param(".par_extend(", "out.par_extend(items)", id="par-extend"),
        pytest.param(
            ".par_extend(",
            "out.par_extend::<_>(items)",
            id="generic-par-extend",
        ),
        pytest.param(
            ".par_windows", "items.par_windows(2)", id="par-windows"
        ),
        pytest.param(".par_split", "items.par_split()", id="par-split"),
        pytest.param(".par_lines", "text.par_lines()", id="par-lines"),
        pytest.param(".par_chars", "text.par_chars()", id="par-chars"),
        pytest.param(
            ".par_char_indices",
            "text.par_char_indices()",
            id="par-char-indices",
        ),
        pytest.param(".par_bytes", "text.par_bytes()", id="par-bytes"),
        pytest.param(
            ".par_encode_utf16", "text.par_encode_utf16()", id="par-encode-utf16"
        ),
        pytest.param(
            ".par_matches", "text.par_matches('a')", id="par-matches"
        ),
        pytest.param(
            ".par_match_indices",
            "text.par_match_indices('a')",
            id="par-match-indices",
        ),
        pytest.param(
            ".par_chunk_by",
            "items.par_chunk_by(|a, b| a == b)",
            id="par-chunk-by",
        ),
        pytest.param(
            ".par_array_windows",
            "items.par_array_windows::<2>()",
            id="par-array-windows",
        ),
        pytest.param(".par_drain", "items.par_drain(..)", id="par-drain"),
        pytest.param(
            "rayon::",
            "rayon::current_thread_index()",
            id="a-qualified-rayon-path",
        ),
        pytest.param(".install(", "pool.install(|| run())", id="install"),
        pytest.param(
            ".install(",
            "pool.install::<_, _>(|| run())",
            id="generic-install",
        ),
        pytest.param(
            ".install(",
            "pool.install::<_, ()>(|| run())",
            id="unit-type-install",
        ),
        pytest.param(
            ".par_extend(",
            "out.par_extend::<()>(items)",
            id="unit-type-par-extend",
        ),
        pytest.param("scope(", "scope(|scope| run())", id="scope"),
        pytest.param(".scope(", "pool.scope(|scope| run())", id="method-scope"),
        pytest.param(
            ".in_place_scope(",
            "pool.in_place_scope(|scope| run(scope))",
            id="method-in-place-scope",
        ),
        pytest.param(
            "in_place_scope(",
            "in_place_scope(|scope| run(scope))",
            id="in-place-scope",
        ),
        pytest.param(
            ".install(", "pool.install:: <_, ()>(|| ())", id="spaced-turbofish"
        ),
        pytest.param(
            "join(", "(join)(|| (), || ())", id="grouped-free-join"
        ),
        pytest.param(
            "join(",
            "(join::<_, _, (), ()>)(|| (), || ())",
            id="grouped-generic-free-join",
        ),
        pytest.param(
            "join(", "((join))(|| (), || ())", id="nested-grouped-free-join"
        ),
        pytest.param(
            ".in_place_scope_fifo(",
            "pool.in_place_scope_fifo(|scope| run(scope))",
            id="method-in-place-scope-fifo",
        ),
        pytest.param(
            "in_place_scope_fifo(",
            "in_place_scope_fifo(|scope| run(scope))",
            id="in-place-scope-fifo",
        ),
        pytest.param(
            "ThreadPool::",
            "ThreadPool::in_place_scope(pool, |_| ())",
            id="an-associated-in-place-scope",
        ),
        pytest.param(
            "ThreadPool::",
            "<ThreadPool>::scope(pool, |_| ())",
            id="a-qualified-type-scope",
        ),
        pytest.param(
            ".scope(", "pool . scope(|scope| run())", id="a-spaced-method-scope"
        ),
        pytest.param(
            "ThreadPool::",
            "ThreadPool::scope(pool, |_| ())",
            id="an-associated-function-scope",
        ),
        pytest.param(
            "ThreadPool::",
            "ThreadPool::install::<_, ()>(pool, || ())",
            id="a-generic-associated-install",
        ),
        pytest.param("scope_fifo(", "scope_fifo(|scope| run())", id="scope-fifo"),
        pytest.param(
            ".scope_fifo(", "pool.scope_fifo(|scope| run())", id="method-scope-fifo"
        ),
        pytest.param("spawn(", "spawn(run())", id="spawn"),
        pytest.param(".spawn(", "scope.spawn(|_| run())", id="method-spawn"),
        pytest.param("spawn_fifo(", "spawn_fifo(run())", id="spawn-fifo"),
        pytest.param(
            ".spawn_fifo(", "scope.spawn_fifo(|_| run())", id="method-spawn-fifo"
        ),
        pytest.param(
            "spawn_broadcast(",
            "spawn_broadcast(|ctx| run(ctx))",
            id="spawn-broadcast",
        ),
        pytest.param(
            ".spawn_broadcast(",
            "pool.spawn_broadcast(|ctx| run(ctx))",
            id="method-spawn-broadcast",
        ),
        pytest.param(
            ".join(", "pool.join(|| left(), || right())", id="method-join"
        ),
        pytest.param(
            ".join(",
            "pool.join::<_, _, (), ()>(|| left(), || right())",
            id="unit-type-join",
        ),
        pytest.param("join(", "join(left, right)", id="join"),
        pytest.param("join_context(", "join_context(left, right)", id="join-context"),
        pytest.param("broadcast(", "broadcast(|ctx| run(ctx))", id="broadcast"),
        pytest.param(
            ".broadcast(", "pool.broadcast(|ctx| run(ctx))", id="method-broadcast"
        ),
    ],
)
def test_every_marker_in_the_set_is_detected(marker: str, call: str) -> None:
    """The marker set is the rule, so each entry has to be load-bearing.

    An entry nobody detects would sit in the tuple looking like coverage while
    the dispatch it names went unread. A body is built from each call, so the
    parameter and the detection cannot drift apart.
    """
    source = f"fn score_trim_jobs() {{ {call}; }}"
    assert dispatch_marker(source, SCORER) == marker, (
        f"{call!r} must be reported as {marker!r}, the entry that names it"
    )
