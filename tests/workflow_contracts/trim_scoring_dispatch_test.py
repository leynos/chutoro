"""The trim scorer never hands its jobs to the Rayon pool.

``CpuHnsw::insert_with_collector`` holds ``insert_mutex`` while it scores trim
jobs. Scoring used to run them through ``into_par_iter()``, which let Rayon
park the mutex holder on a latch that only a worker blocked on the same mutex
could satisfy; a parallel build then hung outright, and a benchmark child that
hit it had to be killed after twenty-five minutes.

A runtime regression test records the Rayon worker that serves each distance
call and waits, briefly, for a call from another worker. That catches the
dispatch in practice, but it is a scheduling observation rather than a proof:
if no idle worker steals before the wait expires, the caller can finish the
parallel batch itself and the observation still looks sequential.

This contract is the deterministic guard for the specific former regression,
the one a review asked for rather than a second runtime test. It reads
``score_trim_jobs`` from the source and rejects every Rayon entry point in its
body, so restoring ``into_par_iter()`` fails here whatever the scheduler does.
It cannot see indirect dispatch -- a helper that dispatches on the scorer's
behalf -- which is why the runtime test stays as broader coverage.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import typing as typ

import pytest
from rust_source_reading import function_body

from workflow_support import ROOT as REPO_ROOT

#: The scorer whose body must stay serial. The runtime regression exercises
#: the same function; this contract reads it rather than running it.
SCORED_FUNCTION: typ.Final[str] = "score_trim_jobs"

TRIM_SOURCE = REPO_ROOT / "chutoro-core" / "src" / "hnsw" / "cpu" / "trim.rs"

#: Rayon entry points that hand a job to the shared pool. A body containing
#: any of these is dispatching rather than scoring in place. `into_par_iter`
#: precedes `par_iter` so the more specific name is reported for a parallel
#: iterator, and `rayon::` precedes the bare calls so a qualified call names
#: its namespace rather than only its function.
DISPATCH_MARKERS: typ.Final[tuple[str, ...]] = (
    "into_par_iter",
    "par_iter",
    "rayon::",
    ".install(",
    "scope(",
    "spawn(",
    "join(",
)

#: A body that scores in place: the shape the contract requires.
SEQUENTIAL = """
fn score_trim_jobs(jobs: Vec<Job>) -> Vec<Scored> {
    jobs.into_iter().map(score).collect()
}
"""

#: The former regression, restored. This is the shape the contract exists to
#: reject, and the shape a benchmark once spent twenty-five minutes inside.
DISPATCHING = """
fn score_trim_jobs(jobs: Vec<Job>) -> Vec<Scored> {
    jobs.into_par_iter().map(score).collect()
}
"""

#: A comment in the scored body naming the dispatch it must not contain. A
#: reading that scanned raw text would report this and fail a correct body,
#: so the existing comment explaining the deadlock must stay harmless.
COMMENTED_MENTION = """
fn score_trim_jobs(jobs: Vec<Job>) -> Vec<Scored> {
    // into_par_iter() would strand the insert mutex holder.
    jobs.into_iter().map(score).collect()
}
"""

#: The same mention inside a string literal, which a reader that blanked only
#: comments would still report. A warning about the dispatch is not one.
LITERAL_MENTION = """
fn score_trim_jobs(jobs: Vec<Job>) -> Vec<Scored> {
    let warning = "into_par_iter() strands the insert mutex holder";
    jobs.into_iter().map(score).collect()
}
"""

#: A dispatch in a sibling function. Scoping the reading to the named body is
#: what keeps this out of the verdict; a file-level search would report it and
#: fail a scorer that is itself serial.
SIBLING_DISPATCH = """
fn score_trim_jobs(jobs: Vec<Job>) -> Vec<Scored> {
    jobs.into_iter().map(score).collect()
}

fn build_worker_rngs(jobs: Vec<Job>) -> Vec<Scored> {
    jobs.into_par_iter().map(score).collect()
}
"""


class MissingFunctionError(LookupError):
    """Raised when the function this contract must read is not in the source.

    Failing closed is the point. A reading that returned no marker for a
    source it could not find the function in would report the scorer as serial
    whenever the function was renamed or moved, which is exactly when the
    guard is worth having. The missing function is the one signal that the
    contract has stopped reading the code it claims to police.
    """


def dispatch_marker(source: str, name: str = SCORED_FUNCTION) -> str | None:
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
        The function to read. Defaults to the trim scorer this contract is
        about, and is a parameter so the reading can be driven with the
        synthetic sources above.

    Examples
    --------
    >>> dispatch_marker(
    ...     "fn score_trim_jobs() { jobs.into_par_iter().collect() }"
    ... )
    'into_par_iter'
    >>> dispatch_marker("fn score_trim_jobs() { work() }") is None
    True

    Returns
    -------
    str or None
        The first marker found, in `DISPATCH_MARKERS` order, or None when the
        body dispatches nothing.
    """
    body = function_body(source, name)
    if body is None:
        raise MissingFunctionError(
            f"{name} is not declared in this source; the contract cannot "
            "prove its body scores serially"
        )
    return next((marker for marker in DISPATCH_MARKERS if marker in body), None)


def test_the_scorer_dispatches_nothing_to_the_pool() -> None:
    """The former regression must not return without a failing check.

    The runtime regression observes the same property, but only when a worker
    happens to steal a job. This reading is the deterministic guard for the
    specific regression: restoring ``into_par_iter()`` in the scorer fails
    here whatever the scheduler does with the pool.
    """
    marker = dispatch_marker(TRIM_SOURCE.read_text(encoding="utf-8"))
    assert marker is None, (
        "score_trim_jobs must score jobs serially while it holds insert_mutex; "
        f"found {marker!r}, which hands work to a pool that may already be "
        "blocked on that same mutex"
    )


@pytest.mark.parametrize(
    ("source", "marker"),
    [
        pytest.param(SEQUENTIAL, None, id="a-serial-body"),
        pytest.param(DISPATCHING, "into_par_iter", id="a-parallel-iterator"),
        pytest.param(COMMENTED_MENTION, None, id="a-mention-in-a-comment"),
        pytest.param(LITERAL_MENTION, None, id="a-mention-in-a-literal"),
        pytest.param(SIBLING_DISPATCH, None, id="a-dispatch-in-a-sibling"),
    ],
)
def test_the_reading_answers_for_the_named_body_alone(
    source: str, marker: str | None
) -> None:
    """Over the real scorer every weak reading agrees, so these separate them.

    The real body is serial, so a reading that scanned the whole file, or that
    kept comments and literals intact, would still pass it. These sources make
    the scoping and the blanking load-bearing: a comment explaining the
    deadlock, a string literal warning about it and a dispatch in a sibling
    function all have to read as clean.
    """
    assert dispatch_marker(source) == marker, (
        f"the reading must return {marker!r} for this source; it returned "
        f"{dispatch_marker(source)!r}"
    )


@pytest.mark.parametrize(
    ("marker", "call"),
    [
        pytest.param(
            "into_par_iter", "jobs.into_par_iter()", id="into-par-iter"
        ),
        pytest.param("par_iter", "jobs.par_iter()", id="par-iter"),
        pytest.param(
            "rayon::",
            "rayon::current_thread_index()",
            id="a-qualified-rayon-path",
        ),
        pytest.param(".install(", "pool.install(|| run())", id="install"),
        pytest.param("scope(", "scope(|scope| run())", id="scope"),
        pytest.param("spawn(", "spawn(run())", id="spawn"),
        pytest.param("join(", "join(left, right)", id="join"),
    ],
)
def test_every_marker_in_the_set_is_detected(marker: str, call: str) -> None:
    """The marker set is the rule, so each entry has to be load-bearing.

    An entry nobody detects would sit in the tuple looking like coverage while
    the dispatch it names went unread. A body is built from each call, so the
    parameter and the detection cannot drift apart.
    """
    source = f"fn score_trim_jobs() {{ {call}; }}"
    assert dispatch_marker(source) == marker, (
        f"{call!r} must be reported as {marker!r}, the entry that names it"
    )


def test_a_missing_scorer_fails_rather_than_reading_as_clean() -> None:
    """A renamed or moved function must fail the contract, not pass it.

    With no body to read there is no marker to report, so a reading that
    shrugged would keep calling the scorer serial after it had moved out of
    the file. Raising is what makes the silence loud.
    """
    with pytest.raises(MissingFunctionError, match=SCORED_FUNCTION):
        dispatch_marker("fn build_worker_rngs() { jobs.into_par_iter() }")