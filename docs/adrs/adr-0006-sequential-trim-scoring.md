# Architecture Decision Record (ADR) 006: sequential trim scoring under the insertion mutex

## Status

Accepted - 2026-11-14: trim scoring stays inside the insertion critical section
and runs sequentially on the thread that holds `insert_mutex`.

## Date

2026-11-20

## Context and problem statement

`CpuHnsw` parallelizes construction across Rayon workers, and each insertion
serializes its plan-apply-score-commit transaction through `insert_mutex`
(`chutoro-core/src/hnsw/cpu/mod.rs`). After applying a staged insertion, the
worker holding that mutex scored the resulting trim jobs with
`trim_jobs.into_par_iter().map(...).collect::<Result<Vec<_>, _>>()`.

Rayon's parallel `collect` bridges through a latch. The calling thread may park
on that latch instead of draining the work itself, which is safe only while
another worker can steal. Here every other worker was already blocked acquiring
`insert_mutex`, which the parked thread held, so none could steal. The pool
then made no progress at all: `CpuHnsw::build` and `CpuHnsw::build_with_edges`
never returned, and nothing was logged or reported as an error.

The stall was observed on 2026-11-08 while listing benchmark targets.
`cargo nextest list` builds bench binaries eagerly, and the extraction bench
builds a 1,000-point HNSW index at registration, so listing it really does
build an index. One child hung for 25 minutes with 9 CPU ticks across 19
threads: eighteen Rayon workers were parked in `Mutex::lock` inside
`insert_with_collector`, and the thread that should have drained the nested
work was parked in Rayon's latch. `make test-workflow-contracts` hung twice on
the same path. A standalone probe reproduced the stall with a four-worker pool.

The question this decision answers is how to keep the apply-score-commit
transaction correct while removing the nested dispatch that turns ordinary lock
contention into a permanent stall.

## Decision drivers

- The insertion transaction must stay atomic with respect to the graph. The
  `prepared` state produced by `apply` is derived from a graph snapshot and is
  committed later; another insertion must not interleave between the two.
- Correctness must not depend on scheduler timing. A fix that holds only when
  an extra worker happens to be free would leave consumers with a hang that
  reproduces intermittently.
- The parallel build is the point of the CPU backend, so a fix must not remove
  the outer parallelism that motivates it.
- Trim batches are small. Scoring produces at most one job per staged
  neighbour update, so per-insertion scoring is not a throughput bottleneck.

## Options considered

- **Score trim jobs outside the insertion critical section.** Rejected. The
  `prepared` state from `apply` is only valid while no other insertion mutates
  the graph before `commit`. Releasing `insert_mutex` for scoring would let
  another insertion apply and commit against the same graph, so the snapshot
  the commit relies on could no longer be shown valid without revalidation and
  a new ordering protocol. The transaction therefore keeps apply, score, and
  commit under one guard.
- **Dispatch sequentially only when the caller is already inside a Rayon
  worker or the pool is committed to an outer insert.** Rejected. Detecting
  that condition is fragile, and it is also the wrong test: in a parallel build
  the mutex holder usually is a Rayon worker, and the stall depends on whether
  any other worker can steal, not on how the caller entered. The conditional
  form also leaves two scoring paths to keep in step.
- **Restrict the outer parallel insert so it never calls code that spawns
  nested Rayon work, for example by scoring with the executor that drives the
  outer loop.** Rejected. It either removes the outer parallelism the build
  exists for or introduces a second executor boundary for one small nested
  batch, adding complexity without improving correctness.
- **Score the jobs sequentially on the calling thread, unconditionally.**
  Accepted. It removes core-owned nested dispatch from the critical section
  entirely, so no scheduling assumption remains, and the batch is small enough
  that the lost intra-insertion parallelism is negligible.

## Decision outcome

`CpuHnsw::score_trim_jobs` scores its jobs in input order on the calling thread:

```rust
trim_jobs
    .into_iter()
    .map(|job| self.run_trim_job(job, source))
    .collect()
```

The empty-batch early return is retained. Job validation, ranking, and the
`max_connections` cap are unchanged, and results stay in input order. The
critical section still spans apply, score, and commit, so the snapshot the
commit relies on remains valid. The mutex, the staging, the outer build
dispatch, and the RNG selection are otherwise untouched.

Because scoring is now sequential, first-error reporting becomes deterministic:
scoring stops at the first invalid job in input order, whereas the parallel
`collect` returned whichever error it reached first, which is not guaranteed to
be the leftmost in input order. Distance-cache reads and source
`batch_distances` calls (which may delegate to `distance_batch` and `distance`)
still happen at the same point in the transaction, after apply and before
commit under the mutex, but the order between jobs is now input order.

The change also adds two label-free instruments behind the existing optional
`metrics` feature, `chutoro.hnsw.trim.jobs_submitted_total` and
`chutoro.hnsw.trim.duration_seconds`, so the cost of serial scoring inside the
critical section stays observable.

## Architectural rationale

The lock protects a transaction, not just a data structure. Once scoring is
recognized as a step inside that transaction, the question stops being "can
this step be parallel?" and becomes "may anything inside the transaction wait
for work that needs the same lock?". It may not. Sequential scoring states that
rule directly in the code, rather than relying on a dispatcher to notice that
its workers are unavailable.

This extends the existing HNSW scoring invariant: distance evaluation runs
before graph write-lock mutation, and no code inside the insertion critical
section may block on, or dispatch work to, the Rayon pool.

## Known risks and limitations

- **Lost intra-insertion parallelism.** Trim batches no longer fan out. This is
  accepted because batches are small, at most one job per staged neighbour
  update, and the new metrics let a future regression in batch size surface.
- **Arbitrary `DataSource` callbacks are not made safe.** The fix removes
  nested dispatch owned by the core, not dispatch owned by a source
  implementation. A `DataSource` whose `distance`, `distance_batch`,
  `batch_distances`, or `metric_descriptor` blocks on, or dispatches to, the
  same Rayon pool while called under `insert_mutex` can still deadlock; trim
  batch construction reads the metric descriptor inside the critical section.
  The trait documentation now states this, together with the prohibition on
  re-entering the same index.
- **The deterministic guard reads source text.**
  `tests/workflow_contracts/trim_scoring_dispatch_test.py` pins the five
  functions on the scoring path — `insert_with_collector` in `cpu/mod.rs`;
  `score_trim_jobs`, `score_trim_jobs_with_metrics`, and `run_trim_job` in
  `cpu/trim.rs`; and `batch_distances_for_trim` in `hnsw/helpers.rs` — and
  rejects every recognized Rayon entry point in any of their bodies, including
  the `into_par_iter()`, `par_iter_mut()`, and `par_iter()` adapters, the
  parallel slice and string adapters (`.par_bridge()`, `.par_chunks()`,
  `.par_rchunks()`, `.par_chunk_by()`, `.par_windows()`, `.par_array_windows()`,
  `.par_split()`, `.par_lines()`, `.par_chars()`, `.par_char_indices()`,
  `.par_bytes()`, `.par_encode_utf16()`, `.par_matches()`,
  `.par_match_indices()`, `.par_drain()`, `.par_sort()`, and `.par_extend()`,
  each standing for its mutable and `_exact` forms as well) and the
  `spawn_fifo`, `scope_fifo`, `in_place_scope`, `in_place_scope_fifo`,
  `join_context`, `broadcast`, and `spawn_broadcast` calls with their
  `ThreadPool` method forms such as `pool.scope(...)` and `scope.spawn(...)`,
  and with the calls Rust lets an inherent method take through its type, so
  `ThreadPool::scope(pool, ...)` and `ThreadPool::install::<_, ()>(pool, ...)`
  are dispatches, turbofish and all, while `ThreadPool::new()` builds a pool
  without dispatching to it. The generic `pool.install::<_, _>(...)` and
  `out.par_extend::<_>(...)` are recognized through their turbofish, whose type
  list may hold a parenthesized unit or tuple type as in
  `pool.install::<_, ()>(...)`, `out.par_extend::<()>(items)` and
  `pool.join::<_, _, (), ()>(...)`, and whose angles are counted rather than
  matched, so a nested list such as `::<Outer<A, B>, C>` and an arrow in a
  function type such as `::<fn() -> u8>` are still read as one list. Rust
  permits whitespace and comments between the `.` and the method name, so
  `pool. install(...)` is read as the call it is. `pool.join(...)` is matched
  because it takes two arguments where a slice or path join takes one. The join
  is read from its argument list rather than from what the first argument looks
  like: a top-level comma with a value after it means two arguments, so an
  inline closure, a bound callable such as `pool.join(left, right)`, and a call
  that builds one such as `pool.join(make_job(), || ())` are all caught, while
  a trailing comma, or a comma between turbofish type arguments — including one
  inside a nested list such as `dir.join(make_path::<Outer<A, B>, C>())` —
  leaves `names.join(separator,)` and `dir.join(make_path::<A, B>())` slice and
  path joins. A qualified path's type arguments comma the same way and it may
  be borrowed, so `dir.join(<Paths<A, B> as HasPath>::path())` and
  `dir.join(&<Paths<A, B> as HasPath>::path())` stay path joins. Each remaining
  entry is a boundary-aware pattern, so a slice `names.join(", ")` is not read
  as a Rayon join, and a free call preceded by a keyword or a loop label — as
  `return join(left, right)` or `break 'outer join(left, right)` — is still the
  dispatch it is, and so is a call through a grouped operand such as
  `(join)(left, right)`, `(join::<_, _, (), ()>)(left, right)` or
  `((join))(left, right)`, with whitespace allowed between the closing
  parentheses too; a keyword attached to the preceding expression, as in
  `fn f(){return join(a)}`, is still a keyword. A value may follow `*` or an
  expression keyword, so `join(*<Paths<A, B> as HasPath>::path())` and
  `join(match <Paths<A, B> as HasPath>::path() { p => p })` pass one argument.
  A turbofish may put whitespace between `::` and `<`, as in
  `pool.install:: <_, ()>(...)`, and a qualified type path such as
  `<ThreadPool>::scope(pool, |_| ())` or a trait call such as
  `ParallelSliceMut::par_sort(&mut weights[..])` names the parallel operation
  it is. The iterator adapters need a call rather than a mention, so a body
  that only binds a name such as `par_iter` stays serial. The synthetic sources
  these readings are checked against live in `fixtures/trim_scoring`, loaded by
  `trim_scoring_samples.py`. Dispatch in a listed helper is therefore caught;
  dispatch it cannot see still evades it: an unlisted helper called by one of
  the five, an aliased import such as `use rayon::spawn as run; run(...)`, or a
  `DataSource` that dispatches on its own account. The scan fails closed when a
  pinned function is missing, so a rename or a move cannot read as clean. The
  runtime scheduling tests in `hnsw::cpu::trim_tests` remain as broader
  coverage, and they may catch unlisted helper dispatch, but they are
  observations rather than proofs: the caller's first distance callback waits
  up to two seconds for another worker, and if no worker steals in that
  interval a dispatching implementation can still finish the batch on the
  caller and pass.
- **A watchdog stress test was considered and rejected.** An earlier
  watchdog-based stress test passed three times out of three on the unfixed
  code, because the stall is a scheduling race rather than a deterministic
  failure.
- **Broader scheduling changes need stronger evidence than this ADR's.**
  Neither the lexical scan nor the bounded runtime observations establish
  deadlock freedom, and none of them should be extended into a structural
  harness that claims to. A future change to scheduling beyond this critical
  section — a narrower lock, a second mutex, work stealing across insertions,
  or any path that may wait while holding a graph or cache lock — needs a
  bounded scheduler or lock model that covers the new interaction *and*
  production-path regressions. The concurrency limitations in
  [ADR-0002](adr-0002-adoption-of-kani-formal-verification.md) explain why the
  existing Kani harnesses do not model lock waits or deadlock freedom and
  cannot be stretched to cover this.

## Consequences

- Parallel builds of graphs large enough to produce trim jobs return instead of
  hanging. `cargo nextest list --all-targets`, and any gate that lists or runs
  benchmark binaries, no longer inherit the stall.
- Trim scoring is deterministic in input order, including first-error
  selection.
- Two new operational metrics describe the serial scoring cost.
- A future change that reintroduces one of the recognized Rayon entry-point
  markers into any of the five pinned functions fails the source contract
  deterministically, regardless of scheduler timing. Dispatch reached through
  an unlisted helper, an alias, or a source implementation is caught, if at
  all, only by the runtime observation.

## Validation evidence

The fix merged as PR #386 on 2026-11-14, in merge commit
`28c27bb9fa22599b68189655f42b393b129720ff`.

The checked-in evidence now has three parts.

- **Historical fixture.** The workflow contract carries a verbatim copy of the
  former `score_trim_jobs` from commit
  `56d6909c7019ac7caa6d485d11ebf5708b47528e`, `cpu/trim.rs` lines 35-48, and
  asserts that it reads as `into_par_iter` while the current scorer reads with
  no marker. The fixture is checked in rather than read from Git history, so a
  shallow checkout does not retire the evidence, and it cannot drift into
  agreement with the fixed code.
- **Direct runtime regression.**
  `direct_trim_scoring_stays_on_the_calling_worker` scores a synthetic batch in
  a dedicated four-worker pool and records the Rayon worker that serves each
  distance callback; every callback must be served by the caller's worker. The
  test holds no lock and probes no mutex, which is what isolates the scorer's
  own scheduling from the rest of the insertion transaction.
- **Insertion-path runtime regression.**
  `insertion_trim_scoring_runs_on_the_insert_mutex_holder` seeds an index with
  twelve nodes at `M = 4`, `ef_construction = 8`, and a one-entry distance
  cache, then inserts the thirteenth through `insert_with_collector` in a
  dedicated four-worker pool. A `WANTS_EDGES = true` collector marks the end of
  planning; every later distance callback must report the same thread, the same
  Rayon worker, and `WouldBlock` from a non-blocking `try_lock`. The one-entry
  cache forces at least two trim jobs to reach the source, so the regression
  observes real scoring rather than an empty batch. It shares the two-second
  steal window with the direct test.

`make test-scheduling-contract` runs one nextest invocation per regression,
each with a fully qualified exact-match filter, `--no-tests=fail`, and default
features, and the `build-test` CI job runs that target unconditionally beside
`make test-workflow-contracts`.
`tests/workflow_contracts/scheduling_contract_test.py`, `scheduling_ci_test.py`
and `scheduling_shell_test.py`, reading through `scheduling_reading.py`,
`makefile_reading.py`, `scheduling_command_reading.py`,
`scheduling_invocation_reading.py`, `scheduling_script_reading.py`,
`scheduling_segment_reading.py`, `scheduling_errexit_reading.py`,
`scheduling_line_reading.py`, `scheduling_here_document_reading.py`,
`scheduling_shell_reading.py`, `scheduling_selector_cases.py`,
`rust_declaration_reading.py`, and `workflow_support.py`, pin the target
recipe, one invocation per fully qualified test name, each invocation's whole
filter expression as exactly one selector — every spelling of a filter is read,
including `--filterset`, the `=` forms and an attached `-Eall()`, and a second
filter is rejected because nextest unions them — the zero-test failure flag on
each, the absence of every feature-selection flag (`--all-features`,
`--no-default-features`, `--features` and its `=` form, and `-F`), the absence
of `--no-run` (which compiles the selected tests without executing them), of
`--help` and `-h` (which print usage), the unconditional and blocking execution
of the step and its nextest installer within an unconditional job, the order
that installs `cargo-nextest` before the target runs, and that the installer
appears as a command in its own right rather than as an argument to `echo` or
inside a comment, certain to run — no earlier `||` can skip it
(`true || installer`), and an earlier `&&` runs it only when everything before
it succeeded, so `false && installer` never installs it and only `;` starts a
command list that runs whatever came before it, while a `;` after the installer
ends the installer's own list, so its failure is not exempt from `errexit` and
`installer; true`, `installer ; make test` and a trailing `installer ;` all
count — in the foreground (a trailing `&` is rejected, whether it follows the
installer or a later command, as in `installer && echo done &`), with no
operator that would replace its status, neither directly after it (`|| true`)
nor further along the same command line
(`installer && echo installed || true`), nor after the chain's own line:
`errexit` exempts a failing command inside an AND/OR list unless it is the
command after the final `&&`, so `installer && echo installed` followed by a
later `true` leaves the step green and only the script's last command line
carries the status out. It also requires the recipe's commands to run nextest
in executable position without Make's error-ignoring `-` prefix or a
failure-masking shell suffix such as `|| true`. The zero-test flag is matched
as a whole token, so a redirection target cannot stand in for it: neither the
attached `>--no-tests=fail` nor the spaced `> --no-tests=fail`, whose target
arrives as a separate word, and a file-descriptor prefix such as `2>` marks a
redirection too. A physical line ending in `&&`, `||`, `|` or `|&` continues
the command list onto the next line, so `true ||` followed by an installer line
never runs the installer while an installer followed by `&&` carries its
failure into the next line; comments and blank lines between the two are
skipped as the shell skips them. A backslash-newline pair is deleted with
nothing inserted in its place, so a path split as `scripts/install-\` and
`nextest.sh` is the one word the shell runs; only an odd trailing backslash run
continues the line, and only its final backslash is deleted, so three leave two
for the lexer to pair, a doubled backslash is a literal one and ends the line,
and a backslash followed by a space escapes the space rather than the newline.
A comment's trailing backslash is comment text and continues nothing, so the
line after a comment is a command line of its own; a descriptor copy such as
`2>&1` is part of its command rather than a background operator, a redirection
is not an argument to a shell that runs the script, while a word that merely
ends in `>` — a quoted file name such as `'scheduling>'` — is not a redirection
operator and leaves the `&` after it a background operator. A `.PHONY`
declaration is read with its comments removed as well, so a target named only
after a `#` is not declared phony, and a comment continued with a trailing
backslash keeps the next line out of the declaration too; only an unescaped
trailing backslash continues a comment, so a doubled backslash ends it and the
next line declares as usual. A `define` body is not declaration text, so a
`.PHONY` written between `define` and `endef` declares nothing, a nested
definition is tracked to its own `endef` rather than ending the outer one, a
`private`/`override`/`export` modifier may precede `define`, a tab-prefixed
`endef` is body text rather than a directive, and a backslash quotes only the
characters Make reads specially, so an escaped space belongs to one target name
while `\gate` keeps its backslash. `define = unused` is a variable assignment
rather than a definition. `.PHONY:= gate` is a variable assignment rather than
a declaration, so is `.PHONY: gate = value`, and a rule header written
`build ::=` assigns rather than declaring; `.PHONY : gate` with a space before
the colon declares as usual, an assignment before the colon is a variable
rather than a rule, an inline recipe after `;` is not part of the declaration
header, a recipe prefix is one tab rather than the line's indentation, and a
recipe continues across comment lines, which are skipped as make skips them; a
tab-prefixed `.PHONY` is recipe text and declares nothing. A `.PHONY` inside a
conditional block declares only when that condition holds, the reading does not
evaluate conditions, so the block declares nothing either, a definition opened
inside the block hides its body from the conditional, and a continued line is
joined before any directive is read, so an `endif` after a continuation is part
of the line it continues. Comments are removed with shell word boundaries, so a
`#` inside a word, as in a redirect to `scheduling#log`, is part of that word
and the mask after it stays visible; an escaped operator ends no word either, so
`>scheduling\;#log` keeps its mask too. Quoted text is data, so
`echo '&&' installer` prints the path rather than running it; an escaped
operator is data for the same reason, so `echo \&\& installer` prints both
ampersands, and a Bash ANSI-C string (`$'...'`) honours its backslash escapes,
so an escaped `'` does not end it. A `#` after a completed expansion, as in
`>$(LOG)#1`, is part of that word rather than a comment. Commands are tokenized
with shell separators rather than whitespace alone, so an unspaced `||true`
reads as the operator it is, `|&` is refused as the pipeline it is, and feature
flags are read from an invocation's arguments rather than its text, so a file
named `nextest-F.log` is not a flag. Finally, it checks that each named
regression is still declared under `cpu/trim_tests` with a test attribute,
since an undiscovered function cannot trip `--no-tests=fail`, and that a
declaration quoted inside a string literal does not stand in for one. A
physical line whose quote is still open continues physically: a trailing
backslash is resolved, and otherwise the newline is part of the quoted text.
The step's effective shell is resolved through the workflow, job, and step
scopes, and a custom `bash` or `sh` template must execute the script under
`errexit`: `bash {0}` runs the script without it and is rejected, as is any
template that only syntax-checks the script, whether as `bash -n {0}`, in a
combined cluster such as `bash -en {0}`, or as `bash -o noexec {0}` or
`bash -eo noexec {0}`; `-D`, `--dump-strings` and `--dump-po-strings` are
syntax-only modes too, since `-D` implies `-n` while dumping translated
strings. A long option's value is not an option, so `bash --rcfile -e {0}`
names a file called `-e` and enables nothing, and `--` ends option parsing, so
`bash -- -e {0}` runs a file called `-e` rather than the script. `errexit` can
also be turned off inside the script, so `set +e` and `set +o errexit` — and a
cluster such as `+xo errexit` — before the installer stop its failure from
failing the step and `set -e` restores it, whether they are written directly or
through `builtin` or `command`; a lone `-` ends option processing as `--` does,
and `-eo noexec` names noexec through its cluster's final `o`. `builtin exit`
ends the script as `exit` does, `exec` with a command replaces the shell,
`exec` with only a redirection does not, and a `set` in a pipeline element
changes only that element's subshell. Redirections are removed before an option
or a `builtin` wrapper is read, both spaced and attached, so
`set > /dev/null +e`, `set >/dev/null +e` and `> /dev/null builtin set +e`
still disable errexit, while a quoted `>` stays an argument, a `command -p`
wrapper is read through to its builtin, and a negated pipeline is refused
rather than read as if the `!` were not there, `command --` ends the wrapper's
options, and a conditional change that cannot alter the state keeps the known
one. A script that installs an `ERR`, `EXIT` or signal-`0` trap is refused: the
handler can turn an installer failure into success, whether or not it is
written through `builtin`. The reading skips an option operand
(`set -o pipefail -e`), stops at `--` and at the first positional parameter
(`set -- -e` sets `$1`), treats an option change after `&&` or `||` as
uncertain rather than applied, ignores one written in a pipeline element or a
backgrounded list because that list runs in a subshell of its own, and takes
either an `exit` or a `set` that enables `noexec` before the installer as
ending the script's execution. A comment is classified on the line as a
continuation leaves it, so `#; installer` after a handled backslash is text and
its `;` starts a command. A backslash-newline pair inside ordinary single
quotes is preserved, because the shell preserves every character there, and a
literal newline inside one is kept rather than joined away; an escaped
apostrophe opens no quote, either quote closes on its own delimiter, and a line
still continued after its quote closes waits for the next physical line. A
dollar-double-quoted word is the option word it expands to. A short Unicode
escape is decoded as Bash decodes it, so `$'-\u6e'` is `-n`. A here-document's
body is stdin for the command that opened it, so a path written there installs
nothing; the body starts at the newline that ends the opening line even when a
list operator continues past it, and its closing word repeats the delimiter
exactly for `<<`, so a closing line indented with a space or a tab is body
text, while a `<<-` document ignores leading tabs on both body and closing
line. A quoted delimiter makes the body literal, so a trailing backslash there
is a character, while an unquoted body deletes backslash-newline pairs before
its closing word is compared, and a backslash that escapes a character of the
delimiter is removed before the closing line is compared. A script holding a
compound command — a branch, loop, case, brace group or parenthesized group —
is refused whole, because the reading tracks no block structure and cannot say
which body ran; a group written without spaces, as `(true`, is refused too; a
script that enables `lastpipe` is refused as well, because a pipeline element
then changes the parent's state. A quoted word is read as the shell reads it,
so a command substitution inside a double-quoted word keeps its own quotes and a
`#` in the substituted text opens nothing, while an ANSI-C string is data
throughout, a `$(` inside one opens no substitution, and adjacent fragments
such as `'a''b'` are one word. A shell template may not select `-t` or
`-o onecmd`, which runs one command and exits, so a script of several commands
would stop after the first; a quoted operand such as `bash -e "{0}"` is
accepted, because the runner substitutes the path inside the quotes before the
shell parses the template.

Positive evidence, on commit `c8952f3` on 2026-11-22: `make check-fmt`,
`make markdownlint` including spelling, `make interrogate` (at 100% documented
over 931 definitions), `make nixie`, `cs delta origin/main`, and
`make test-workflow-contracts` (1232 passed and 5 deselected, the deselected
module passing separately at 4 passed and 1 skipped, including its CV005
contracts check) all pass on this head. The module that compiles a probe crate
was run apart from the rest only to fit a background time limit; nothing else
was changed. Both runtime regressions pass, one nextest invocation each with
`--no-tests=fail`, 1 passed with 674 skipped apiece, observed directly on
`435a0a0` and re-run by the `build-test` job on every pushed head. `make lint`
(rustdoc, Clippy and Whitaker), `make typecheck`, `make nixie`, and `make test`
passed on earlier heads: 1481 nextest tests with 1 skipped and 141 doctests
with 24 ignored on `447a824`, and the lint gates on `f0b8bbb`. Those heads are
the last to change a Rust source, and the commits after them touch only these
contracts and the documentation this record describes.

Negative control (2026-11-19, scratch worktree detached at the branch head):
restoring `trim_jobs.into_par_iter()` in `score_trim_jobs` produced both
expected outcomes.

- The Python contract failed deterministically. Run from
  `tests/workflow_contracts` as `pytest -q trim_scoring_dispatch_test.py`, it
  collected 34 cases: two failed and thirty-two passed. The scan's failure named
  `('chutoro-core/src/hnsw/cpu/trim.rs', 'score_trim_jobs'): 'into_par_iter'`,
  and the historical-fixture test reported
  `the current scorer must score in place, found 'into_par_iter'`.
- An earlier run of the same mutation failed both runtime regressions: the
  direct test observed callbacks on four worker indices instead of the caller
  alone, and the insertion-path test observed a post-planning callback on a
  different thread from the `insert_mutex` holder. They failed in 0.024 s and
  0.009 s respectively because the dedicated pool had an idle worker to accept
  the steal offer.

The runtime outcome is scheduler-dependent by construction — a pool with no
idle worker can let the mutated caller finish the batch itself — which is why
the Python contract, not the runtime regressions, is the guarantee. The scratch
mutation was discarded and the candidate worktree was never changed.

## References

- [Deadlock report (#378)](https://github.com/leynos/chutoro/issues/378)
- [Fix (#386)](https://github.com/leynos/chutoro/pull/386)
- [Scheduling-contract assurance follow-up (#402)](https://github.com/leynos/chutoro/issues/402)
