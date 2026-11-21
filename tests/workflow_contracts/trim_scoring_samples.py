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

#: A path join written with whitespace around the dot. Rust permits it, so
#: `cache_dir . join(x)` is the same method call as `cache_dir.join(x)`; a
#: free-function reading that only required the name to start a word reported
#: the body as a pool join, because the character before the name is a space.
SPACED_RECEIVER_JOIN = """
fn score_trim_jobs(jobs: Vec<Job>) -> PathBuf {
    cache_dir . join(&cache_file)
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

#: A slice join written with a trailing comma. Rust permits the comma after a
#: single argument, so a marker that read any comma as a second argument
#: reported this correct body as a pool join.
SLICE_JOIN_TRAILING_COMMA = """
fn score_trim_jobs(jobs: Vec<Job>) -> String {
    names.join(separator,)
}
"""

#: A path join written with a trailing comma, the same shape one receiver
#: over. The argument list, not the receiver, is what the reading trusts.
PATH_JOIN_TRAILING_COMMA = """
fn score_trim_jobs(jobs: Vec<Job>) -> PathBuf {
    cache_dir.join(&cache_file,)
}
"""

#: A path join whose single argument is a call with its own type arguments.
#: The comma between `A` and `B` separates type arguments, not values, so this
#: is one argument; a scan that counted every comma at the list's level read
#: the body as a pool join and failed a correct scorer.
GENERIC_PATH_JOIN = """
fn score_trim_jobs(jobs: Vec<Job>) -> PathBuf {
    cache_dir.join(make_path::<A, B>())
}
"""

#: A path join whose single argument is generic in its own right. The comma
#: between `A` and `B` sits inside `Outer<A, B>`, so only the comma after the
#: inner list is at the outer list's level and even that one belongs to the
#: turbofish; a skip that stopped at the first `>` ended inside the inner list
#: and read the body as a pool join.
NESTED_GENERIC_PATH_JOIN = """
fn score_trim_jobs(jobs: Vec<Job>) -> PathBuf {
    cache_dir.join(make_path::<Outer<A, B>, C>())
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

#: A free call to a generic entry point. Rayon's free functions are generic, so
#: `join::<_, _, (), ()>(left, right)` dispatches exactly as `join(left, right)`
#: does; a pattern that expected the parenthesis straight after the name read
#: this body as serial.
GENERIC_FREE_JOIN = """
fn score_trim_jobs(jobs: Vec<Job>) -> Scored {
    join::<_, _, (), ()>(|| score(left), || score(right))
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

#: A pool join whose first closure is built by a call. `ThreadPool::join`
#: takes any two `FnOnce` values, so a factory call returning one dispatches
#: like any other operand; a pattern that allowed only a closure or a bare
#: name as the first argument read this body as serial.
FACTORY_JOIN = """
fn score_trim_jobs(jobs: Vec<Job>) -> Scored {
    pool.join(make_job(), || ())
}
"""

#: A pool install whose turbofish holds a unit type. `ThreadPool::install` is
#: generic in its closure and its result, and a result of `()` needs
#: parentheses inside the turbofish; a pattern that ruled parentheses out did
#: not see the call at all, so this dispatch read as serial.
UNIT_INSTALL = """
fn score_trim_jobs(jobs: Vec<Job>) {
    pool.install::<_, ()>(|| score(jobs))
}
"""

#: A pool join whose turbofish holds tuple types. Both results of
#: `ThreadPool::join` are spelled as parenthesized types here, so the call
#: never reached the argument-list check and the body read as a slice join.
TUPLE_JOIN = """
fn score_trim_jobs(jobs: Vec<Job>) -> (Scored, Scored) {
    pool.join::<_, _, (), ()>(|| score(left), || score(right))
}
"""

#: A local of unit and tuple type, which dispatches nothing. Parenthesised
#: types are ordinary Rust, so recognising them inside a turbofish must not
#: turn every tuple in a body into a call.
UNIT_TYPE_LOCAL = """
fn score_trim_jobs(jobs: Vec<Job>) -> () {
    let pair: (u8, u8) = (1, 2);
    ()
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

#: `par_chunk_by` dispatches a slice by a predicate. It shares its prefix with
#: neither `par_chunks` nor `par_rchunks`, so the chunk entries miss it.
PAR_CHUNK_BY = """
fn score_trim_jobs(jobs: &[Job]) -> usize {
    jobs.par_chunk_by(|left, right| left == right).count()
}
"""

#: `par_array_windows` dispatches constant-size windows. It is not spelled
#: with `par_windows`, so the window entry misses it.
PAR_ARRAY_WINDOWS = """
fn score_trim_jobs(jobs: &[Job]) -> usize {
    jobs.par_array_windows::<2>().count()
}
"""

#: `par_char_indices` dispatches a string by character index. It is not
#: `par_chars` followed by more text, so the character entry misses it.
PAR_CHAR_INDICES = """
fn score_trim_jobs(text: &str) -> usize {
    text.par_char_indices().map(|(index, _)| index).sum()
}
"""

#: `par_bytes` dispatches a string's bytes, a method no other entry names.
PAR_BYTES = """
fn score_trim_jobs(text: &str) -> usize {
    text.par_bytes().map(|byte| byte as usize).sum()
}
"""

#: `par_encode_utf16` dispatches a string's UTF-16 encoding.
PAR_ENCODE_UTF16 = """
fn score_trim_jobs(text: &str) -> usize {
    text.par_encode_utf16().count()
}
"""

#: `par_matches` dispatches a string's matches of a pattern.
PAR_MATCHES = """
fn score_trim_jobs(text: &str) -> usize {
    text.par_matches('a').count()
}
"""

#: `par_match_indices` dispatches the positions of a string's matches. Its
#: spelling is not a prefix of `par_matches`, so it needs its own entry.
PAR_MATCH_INDICES = """
fn score_trim_jobs(text: &str) -> usize {
    text.par_match_indices('a').count()
}
"""

#: A pool install whose type arguments include a const expression. The `1 << 2`
#: in `[u8; { 1 << 2 }]` spells two angle brackets that open no type list, so a
#: scan that counted them ran past the call and read this dispatch as serial.
CONST_EXPRESSION_INSTALL = """
fn score_trim_jobs(jobs: Vec<Job>) {
    pool.install::<_, [u8; { 1 << 2 }]>(|| score(jobs));
}
"""

#: A path join whose single argument is a qualified path. The comma in
#: `<Paths<A, B> as HasPath>::path()` separates type arguments rather than
#: values, so this is one argument; a scan that read it as two reported a
#: parallel join and failed a correct scorer.
QUALIFIED_PATH_JOIN = """
fn score_trim_jobs(jobs: Vec<Job>) -> PathBuf {
    cache_dir.join(<Paths<A, B> as HasPath>::path())
}
"""

#: A borrowed qualified path as a join's single argument. The `&` begins the
#: value rather than ending an operand, so the angle brackets still hold type
#: arguments and the call passes one argument.
BORROWED_QUALIFIED_PATH_JOIN = """
fn score_trim_jobs(jobs: Vec<Job>) -> PathBuf {
    cache_dir.join(&<Paths<A, B> as HasPath>::path())
}
"""

#: An associated-function call to a pool entry point. Rust lets an inherent
#: method be called through its type, so `ThreadPool::scope(pool, ...)` runs
#: the closure on the pool exactly as `pool.scope(...)` does. The free-function
#: anchor rejects the `::` and the receiver pattern requires a dot, so neither
#: entry read this dispatch.
ASSOCIATED_SCOPE = """
fn score_trim_jobs(jobs: Vec<Job>) {
    ThreadPool::scope(pool, |_| score(jobs));
}
"""

#: An associated-function call whose entry point carries a turbofish. The type
#: list sits between the name and the argument list, so a pattern expecting the
#: parenthesis straight after the name read this dispatch as serial.
GENERIC_ASSOCIATED_INSTALL = """
fn score_trim_jobs(jobs: Vec<Job>) {
    ThreadPool::install::<_, ()>(pool, || score(jobs));
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

#: A free call written after a keyword. `return join(left, right)` dispatches
#: exactly as `join(left, right)` does, but the word before the name ends in an
#: identifier character, so a reading that treated every preceding identifier as
#: a receiver reported this body as serial.
RETURNED_FREE_JOIN = """
fn score_trim_jobs(jobs: Vec<Job>) -> Scored {
    return join(|| score(left), || score(right));
}
"""

#: A receiver-form call whose entry point carries a turbofish. `ThreadPool`'s
#: `scope` is generic in its operation and its result, so the type list may sit
#: between the name and the argument list; a pattern expecting the parenthesis
#: straight after the name read this dispatch as serial.
GENERIC_METHOD_SCOPE = """
fn score_trim_jobs(jobs: Vec<Job>) {
    pool.scope::<_, ()>(|_| score(jobs));
}
"""

#: An in-place scope call. `in_place_scope` runs its closure on the shared
#: pool and waits for it, exactly as `scope` does, but the longer name matches
#: neither the `scope` receiver pattern nor its free-call anchor.
IN_PLACE_SCOPE = """
fn score_trim_jobs(jobs: Vec<Job>) {
    pool.in_place_scope(|scope| run(scope));
}
"""
