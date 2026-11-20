"""Synthetic Rust sources that keep the trim-scoring reading honest.

Every real scorer on the fixed tree is serial, so a reading that scanned the
whole file, kept comments intact, or matched a bare substring would still pass
it. These samples are the bodies that separate those readings from a correct
one: each is a shape the marker set has to report, or has to leave alone.

They are plain source text with no reader of their own. ``trim_scoring_reading``
holds the markers, the pinned function list and the scan; this module holds the
inputs that prove them. Keeping the two apart keeps each file inside the
400-line limit ``AGENTS.md`` sets.

Re-use policy: import these from
``tests/workflow_contracts/trim_scoring_dispatch_test.py`` only, the one place
that asserts over them. A new sample belongs here with the shape it separates
stated in its comment, so a later reader knows which wrong reading it exists to
catch.
"""

from __future__ import annotations

import typing as typ

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

#: A path join, which is not a dispatch: `Path::join` takes a path, and the
#: single argument is what stops the receiver form reporting it as a pool join.
PATH_JOIN = """
fn score_trim_jobs(jobs: Vec<Job>) -> PathBuf {
    cache_dir.join(&cache_file)
}
"""

#: A slice join whose separator is a variable. Here no comma follows the
#: argument at all, which is what the receiver form requires, so a separator
#: assembled at run time stays a slice join rather than a pool join.
SLICE_JOIN_VARIABLE = """
fn score_trim_jobs(jobs: Vec<Job>) -> String {
    names.join(separator)
}
"""

#: A pool join, which is a dispatch: `ThreadPool::join` runs both closures on
#: the pool, unlike the slice join above. The receiver form needed its own
#: marker, and the two-argument shape is what keeps the two apart.
POOL_JOIN = """
fn score_trim_jobs(jobs: Vec<Job>) -> Scored {
    pool.join(|| score(left), || score(right))
}
"""

#: A pool join of bound callables. `ThreadPool::join` accepts anything that
#: implements `FnOnce`, so `pool.join(left, right)` dispatches exactly as the
#: inline-closure form does; a pattern that required `|` after the parenthesis
#: read this body as serial.
BOUND_JOIN = """
fn score_trim_jobs(jobs: Vec<Job>) -> Scored {
    pool.join(score_left, score_right)
}
"""

#: A reverse-chunk iterator, which dispatches as surely as `par_chunks`.
#: `par_rchunks` does not contain `par_chunks`, so the ordinary chunk marker
#: missed this body.
REVERSE_CHUNKS = """
fn score_trim_jobs(jobs: Vec<Job>) -> Vec<Scored> {
    jobs.par_rchunks(2).map(score).collect()
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