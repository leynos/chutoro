"""The insertion critical section never hands its work to the Rayon pool.

``CpuHnsw::insert_with_collector`` holds ``insert_mutex`` while it scores trim
jobs. Scoring used to run them through ``into_par_iter()``, which let Rayon park
the mutex holder on a latch that only a worker blocked on the same mutex could
satisfy; a parallel build then hung outright, and a benchmark child that hit it
had to be killed after twenty-five minutes.

Two runtime regressions in ``hnsw::cpu::trim_tests`` observe where scoring runs.
The insertion-path test drives a real insertion and records the thread that
serves each post-planning distance call; the direct test scores a synthetic
batch. Both are scheduling observations rather than proofs: if no idle worker
steals before the bounded wait expires, the caller can finish a parallel batch
itself and the observation still looks sequential.

This contract is the deterministic guard. It reads the five functions on the
scoring path out of the source and rejects every Rayon entry point in their
bodies, so restoring ``into_par_iter()`` fails here whatever the scheduler
does. The reading and its pinned function list live in
``trim_scoring_reading``, the marker set it applies lives in
``trim_scoring_markers``, the synthetic sources it must tell apart live in
``trim_scoring_samples``, and the limits of the reading are stated beside the
function list.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import pytest
from trim_scoring_reading import (
    SCANNED_FILES,
    SCORER,
    TRIM_SOURCE,
    MissingFunctionError,
    dispatch_marker,
    dispatch_markers,
    scan_pinned_functions,
)
from trim_scoring_samples import (
    ASSOCIATED_SCOPE,
    BORROWED_QUALIFIED_PATH_JOIN,
    BOUND_JOIN,
    COMMENTED_MENTION,
    CONST_EXPRESSION_INSTALL,
    DISPATCHING,
    FACTORY_JOIN,
    GENERIC_ASSOCIATED_INSTALL,
    GENERIC_FREE_JOIN,
    GENERIC_METHOD_SCOPE,
    GENERIC_PATH_JOIN,
    GROUPED_FREE_JOIN,
    HISTORICAL_SCORE_TRIM_JOBS,
    IN_PLACE_SCOPE,
    LISTED_HELPER_DISPATCH,
    LITERAL_MENTION,
    NESTED_GENERIC_PATH_JOIN,
    PAR_ARRAY_WINDOWS,
    PAR_BYTES,
    PAR_CHAR_INDICES,
    PAR_CHUNK_BY,
    PAR_ENCODE_UTF16,
    PAR_MATCHES,
    PAR_MATCH_INDICES,
    PATH_JOIN,
    PATH_JOIN_TRAILING_COMMA,
    POOL_JOIN,
    QUALIFIED_PATH_JOIN,
    RETURNED_FREE_JOIN,
    REVERSE_CHUNKS,
    SCOPED_HELPER_NAME,
    SEQUENTIAL,
    SLICE_JOIN,
    SLICE_JOIN_TRAILING_COMMA,
    SLICE_JOIN_VARIABLE,
    SPACED_TURBOFISH_INSTALL,
    SPACED_RECEIVER_JOIN,
    TUPLE_JOIN,
    UNIT_INSTALL,
    UNIT_TYPE_LOCAL,
    UNLISTED_SIBLING,
)

#: The number of functions the pinned list must cover.
PINNED_FUNCTIONS = sum(len(entry.functions) for entry in SCANNED_FILES)


def test_the_scoring_path_dispatches_nothing_to_the_pool() -> None:
    """The former regression must not return without a failing check.

    The runtime regressions observe the same property, but only when a worker
    happens to steal a job. This reading is the deterministic guard for the
    specific regression: restoring ``into_par_iter()`` in any listed function
    fails here whatever the scheduler does with the pool.
    """
    markers = scan_pinned_functions()
    dispatching = {where: marker for where, marker in markers.items() if marker}
    assert not dispatching, (
        "the insertion critical section must score in place; these functions "
        f"hand work to the pool: {dispatching}"
    )
    assert len(markers) == PINNED_FUNCTIONS == 5, (
        "the pinned list must cover the five functions on the scoring path, "
        f"read {sorted(markers)}"
    )


def test_the_historical_scorer_is_the_dispatch_the_contract_rejects() -> None:
    """The checked-in fixture pins both sides of the guard.

    The historical scorer must read as a dispatch, and the current scorer must
    read as serial. If either reading changed, the fixture would no longer
    demonstrate that the contract distinguishes the old path from the fixed
    one.
    """
    historical = dispatch_marker(HISTORICAL_SCORE_TRIM_JOBS, SCORER)
    current = dispatch_marker(TRIM_SOURCE.read_text(encoding="utf-8"), SCORER)
    assert historical == "into_par_iter", (
        "the historical fixture must carry the dispatch it was copied to "
        f"record, found {historical!r}"
    )
    assert current is None, (
        f"the current scorer must score in place, found {current!r}"
    )


@pytest.mark.parametrize(
    ("source", "marker"),
    [
        pytest.param(SEQUENTIAL, None, id="a-serial-body"),
        pytest.param(DISPATCHING, "into_par_iter", id="a-parallel-iterator"),
        pytest.param(COMMENTED_MENTION, None, id="a-mention-in-a-comment"),
        pytest.param(LITERAL_MENTION, None, id="a-mention-in-a-literal"),
        pytest.param(SLICE_JOIN, None, id="a-slice-join"),
        pytest.param(SLICE_JOIN_VARIABLE, None, id="a-variable-slice-join"),
        pytest.param(
            SLICE_JOIN_TRAILING_COMMA, None, id="a-slice-join-trailing-comma"
        ),
        pytest.param(PATH_JOIN, None, id="a-path-join"),
        pytest.param(
            SPACED_RECEIVER_JOIN, None, id="a-spaced-receiver-join"
        ),
        pytest.param(GENERIC_FREE_JOIN, "join(", id="a-generic-free-join"),
        pytest.param(
            GROUPED_FREE_JOIN, "join(", id="a-grouped-free-join"
        ),
        pytest.param(
            SPACED_TURBOFISH_INSTALL, ".install(", id="a-spaced-turbofish"
        ),
        pytest.param(IN_PLACE_SCOPE, ".in_place_scope(", id="in-place-scope"),
        pytest.param(
            RETURNED_FREE_JOIN, "join(", id="a-returned-free-join"
        ),
        pytest.param(
            GENERIC_METHOD_SCOPE, ".scope(", id="a-generic-method-scope"
        ),
        pytest.param(
            PATH_JOIN_TRAILING_COMMA, None, id="a-path-join-trailing-comma"
        ),
        pytest.param(GENERIC_PATH_JOIN, None, id="a-generic-path-join"),
        pytest.param(
            NESTED_GENERIC_PATH_JOIN,
            None,
            id="a-nested-generic-path-join",
        ),
        pytest.param(
            QUALIFIED_PATH_JOIN, None, id="a-qualified-path-join"
        ),
        pytest.param(
            BORROWED_QUALIFIED_PATH_JOIN,
            None,
            id="a-borrowed-qualified-path-join",
        ),
        pytest.param(POOL_JOIN, ".join(", id="a-pool-join"),
        pytest.param(BOUND_JOIN, ".join(", id="a-bound-callable-join"),
        pytest.param(FACTORY_JOIN, ".join(", id="a-factory-call-join"),
        pytest.param(UNIT_INSTALL, ".install(", id="a-unit-type-install"),
        pytest.param(
            CONST_EXPRESSION_INSTALL,
            ".install(",
            id="a-const-expression-install",
        ),
        pytest.param(TUPLE_JOIN, ".join(", id="a-tuple-type-join"),
        pytest.param(UNIT_TYPE_LOCAL, None, id="a-unit-and-tuple-type-local"),
        pytest.param(REVERSE_CHUNKS, ".par_rchunks", id="reverse-chunks"),
        pytest.param(PAR_CHUNK_BY, ".par_chunk_by", id="par-chunk-by"),
        pytest.param(
            PAR_ARRAY_WINDOWS, ".par_array_windows", id="par-array-windows"
        ),
        pytest.param(PAR_CHAR_INDICES, ".par_char_indices", id="par-char-indices"),
        pytest.param(PAR_BYTES, ".par_bytes", id="par-bytes"),
        pytest.param(
            PAR_ENCODE_UTF16, ".par_encode_utf16", id="par-encode-utf16"
        ),
        pytest.param(PAR_MATCHES, ".par_matches", id="par-matches"),
        pytest.param(
            PAR_MATCH_INDICES, ".par_match_indices", id="par-match-indices"
        ),
        pytest.param(SCOPED_HELPER_NAME, None, id="a-helper-named-with-scope"),
        pytest.param(
            ASSOCIATED_SCOPE, "ThreadPool::", id="an-associated-function-scope"
        ),
        pytest.param(
            GENERIC_ASSOCIATED_INSTALL,
            "ThreadPool::",
            id="a-generic-associated-install",
        ),
        pytest.param(
            UNLISTED_SIBLING, None, id="a-dispatch-in-an-unlisted-sibling"
        ),
    ],
)
def test_the_reading_answers_for_the_named_body_alone(
    source: str, marker: str | None
) -> None:
    """Over the real scorer every weak reading agrees, so these separate them.

    The real body is serial, so a reading that scanned the whole file, or that
    kept comments and literals intact, would still pass it. These sources make
    the scoping and the blanking load-bearing: a comment explaining the
    deadlock, a string literal warning about it and a dispatch in a function
    nobody pinned all have to read as clean.
    """
    assert dispatch_marker(source, SCORER) == marker, (
        f"the reading must return {marker!r} for this source; it returned "
        f"{dispatch_marker(source, SCORER)!r}"
    )


def test_a_listed_helper_dispatch_is_attributed_to_that_helper() -> None:
    """Scoping must not let a listed helper hide its own dispatch.

    The scorer stays serial, so a scan that read only ``score_trim_jobs``
    would pass. Reading the pinned helper list is what catches the dispatch
    in ``run_trim_job``, which reaches the pool from the same critical section.
    """
    markers = dispatch_markers(
        LISTED_HELPER_DISPATCH, ("score_trim_jobs", "run_trim_job")
    )
    assert markers == {"score_trim_jobs": None, "run_trim_job": "into_par_iter"}


def test_an_unlisted_sibling_is_not_attributed_to_a_pinned_function() -> None:
    """A dispatch outside the pinned list belongs to another concern.

    ``build_worker_rngs`` legitimately dispatches during construction and is
    not on the insertion scoring path. Attributing its marker to the scorer
    would fail a correct body, which is the opposite of what the scan is for.
    """
    assert dispatch_markers(UNLISTED_SIBLING, ("score_trim_jobs",)) == {
        "score_trim_jobs": None
    }


def test_a_missing_function_fails_rather_than_reading_as_clean() -> None:
    """A renamed or moved function must fail the contract, not pass it.

    With no body to read there is no marker to report, so a reading that
    shrugged would keep calling the scorer serial after it had moved out of
    the file. Raising is what makes the silence loud.
    """
    with pytest.raises(MissingFunctionError, match="score_trim_jobs"):
        dispatch_marker(
            "fn build_worker_rngs() { jobs.into_par_iter() }", "score_trim_jobs"
        )
