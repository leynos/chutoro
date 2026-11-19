"""Read the trim-scoring path and report any Rayon dispatch in its bodies.

The insertion critical section holds ``insert_mutex`` while it scores trim
jobs. Scoring used to run through ``into_par_iter()``, which let Rayon park the
mutex holder on a latch that only a worker blocked on the same mutex could
satisfy, so a parallel build hung outright.

This module is the reading: the pinned function list, the marker set, the
checked-in historical fixture, the source-reading helpers, and the synthetic
cases that keep the reading honest. The assertions that use them live in
``trim_scoring_dispatch_test``, so neither module outgrows the 400-line limit
``AGENTS.md`` sets.

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

#: Rayon entry points that hand a job to the shared pool, as (label, pattern)
#: pairs. A body containing any of these is dispatching rather than scoring in
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
#: `into_par_iter` precedes `par_iter` so the more specific name is reported
#: for a parallel iterator, and `rayon::` precedes the bare calls so a
#: qualified call names its namespace rather than only its function.
DISPATCH_MARKERS: typ.Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("into_par_iter", re.compile(r"\binto_par_iter\b")),
    ("par_iter_mut", re.compile(r"\bpar_iter_mut\b")),
    ("par_iter", re.compile(r"\bpar_iter\b")),
    (".par_bridge(", re.compile(r"\.par_bridge\s*\(")),
    (".par_chunks", re.compile(r"\.par_chunks")),
    (".par_sort", re.compile(r"\.par_sort")),
    (".par_extend(", re.compile(r"\.par_extend\s*\(")),
    (".par_windows", re.compile(r"\.par_windows")),
    (".par_split", re.compile(r"\.par_split")),
    (".par_lines", re.compile(r"\.par_lines")),
    (".par_chars", re.compile(r"\.par_chars")),
    (".par_drain", re.compile(r"\.par_drain")),
    ("rayon::", re.compile(r"rayon\s*::")),
    (".install(", re.compile(r"\.install\s*\(")),
    ("scope(", re.compile(r"(?<![.\w:])scope\s*\(")),
    (".scope(", re.compile(r"\.scope\s*\(")),
    ("scope_fifo(", re.compile(r"(?<![.\w:])scope_fifo\s*\(")),
    (".scope_fifo(", re.compile(r"\.scope_fifo\s*\(")),
    ("spawn(", re.compile(r"(?<![.\w:])spawn\s*\(")),
    (".spawn(", re.compile(r"\.spawn\s*\(")),
    ("spawn_fifo(", re.compile(r"(?<![.\w:])spawn_fifo\s*\(")),
    (".spawn_fifo(", re.compile(r"\.spawn_fifo\s*\(")),
    ("spawn_broadcast(", re.compile(r"(?<![.\w:])spawn_broadcast\s*\(")),
    ("join(", re.compile(r"(?<![.\w:])join\s*\(")),
    ("join_context(", re.compile(r"(?<![.\w:])join_context\s*\(")),
    ("broadcast(", re.compile(r"(?<![.\w:])broadcast\s*\(")),
    (".broadcast(", re.compile(r"\.broadcast\s*\(")),
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

#: The former scorer, copied verbatim from commit
#: 56d6909c7019ac7caa6d485d11ebf5708b47528e,
#: ``chutoro-core/src/hnsw/cpu/trim.rs`` lines 35-48.
#:
#: It is checked in rather than read from Git history so the contract keeps
#: its evidence when history is shallow. The marker it carries is the
#: regression the contract exists to reject.
HISTORICAL_SCORE_TRIM_JOBS: typ.Final[str] = """
    pub(crate) fn score_trim_jobs<D: DataSource + Sync>(
        &self,
        trim_jobs: Vec<TrimJob>,
        source: &D,
    ) -> Result<Vec<TrimResult>, HnswError> {
        if trim_jobs.is_empty() {
            return Ok(Vec::new());
        }

        trim_jobs
            .into_par_iter()
            .map(|job| self.run_trim_job(job, source))
            .collect::<Result<Vec<_>, HnswError>>()
    }
"""

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

#: A slice join in the scored body. `join(` is a Rayon free function, but
#: `names.join(", ")` is `[String]::join`, which dispatches nothing; a
#: substring matcher reported it and failed a correct body.
SLICE_JOIN = """
fn score_trim_jobs(jobs: Vec<Job>) -> String {
    names.join(", ")
}
"""

#: A helper whose name ends in `scope`. A substring match on `scope(` read it
#: as Rayon's `scope`; the boundary anchor leaves it alone.
SCOPED_HELPER_NAME = """
fn score_trim_jobs(jobs: Vec<Job>) -> Vec<Scored> {
    with_scope(|| jobs.into_iter().map(score).collect())
}
"""

#: A dispatch in a sibling function that the pinned list does not name. A
#: file-level search would report it and fail a scorer that is itself serial.
UNLISTED_SIBLING = """
fn score_trim_jobs(jobs: Vec<Job>) -> Vec<Scored> {
    jobs.into_iter().map(score).collect()
}

fn build_worker_rngs(jobs: Vec<Job>) -> Vec<Scored> {
    jobs.into_par_iter().map(score).collect()
}
"""

#: A dispatch in a helper the pinned list does name. Scoping the reading to
#: named bodies must not become a way to miss a listed helper: `run_trim_job`
#: computes the distances a trim job needs, so dispatch there reaches the pool
#: from inside the critical section just as surely as dispatch in the scorer.
LISTED_HELPER_DISPATCH = """
fn score_trim_jobs(jobs: Vec<Job>) -> Vec<Scored> {
    jobs.into_iter().map(|job| run_trim_job(job)).collect()
}

fn run_trim_job(job: Job) -> Scored {
    batch.into_par_iter().map(score).collect()
}
"""


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
        (label for label, pattern in DISPATCH_MARKERS if pattern.search(body)),
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
