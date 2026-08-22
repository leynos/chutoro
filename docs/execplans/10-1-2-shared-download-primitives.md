# Execution plan (ExecPlan): roadmap 10.1.2 shared download primitives

This ExecPlan is a living document. The sections `Constraints`, `Tolerances`,
`Risks`, `Progress`, `Surprises & discoveries`, `Decision log`, and
`Outcomes & retrospective` must be kept up to date as work proceeds.

Status: DRAFT

Roadmap item: 10.1.2. Requires 10.1.1 (complete). Blocks 10.1.3, 10.1.4,
10.1.5, and the whole of §10.3.

## Purpose / big picture

After this change, a benchmark dataset recipe can ask for a file from the
internet and get back bytes it is allowed to trust, without holding the whole
file in memory and without starting again from zero every time a connection
drops.

Concretely, three things become possible that are impossible today.

A recipe can declare a source with a pinned SHA-256 digest and a pinned byte
length, and the crate will refuse to hand over bytes that do not match. Today
`Checksum::parse` is a placeholder that always returns an error, and
`SourceSpec.checksum` is an `Option` that is provably always `None`, so no
recipe can express integrity at all.

A recipe can download a multi-gigabyte artefact through a streaming port that
writes to disk as bytes arrive, resuming from where it stopped if the transfer
is interrupted. Today the only fetch path is
`Fetcher::fetch_bytes(&SourceUrl, max_bytes) -> Bytes`, which materializes the
entire response in memory. The largest dataset in the benchmark suite is DEEP1B
at 89 to 477 gibibytes.

A recipe can unpack a `.tar`, `.gz`, `.bz2`, or `.xz` archive into a
destination directory, with entries that try to escape that directory refused
rather than silently skipped, and with a bound on how much data extraction may
produce.

Observable success: a novice runs `make test` and sees new behavioural
scenarios pass in which a scripted local HTTP server truncates a response
mid-body, the download resumes from the recorded offset on the next attempt,
and the assembled file matches its pinned digest. Running
`cargo check -p chutoro-bench-datasets --no-default-features` succeeds, proving
the domain layer compiles with no network, archive, or filesystem dependency.

## Constraints

Hard invariants. Violation requires escalation, not a workaround.

- Keep every source file at or below 400 lines. Split modules before
  exceeding the limit.
- `size_of::<RecipeError>() <= 32` is asserted at compile time in
  `chutoro-bench-datasets/src/error.rs:245`. This assertion must survive. It is
  already fully consumed: `align = 8` means the tag takes 8 bytes and the
  payload budget is 24, which `PortFailure` and `FetchSizeExceeded` already
  occupy exactly. A `Checksum` is 33 bytes on its own and cannot be stored
  inline under any arrangement.
- All filesystem access in this crate must go through `cap-std` /
  `cap_std::fs_utf8`. `chutoro_bench_datasets` is **not** in the
  `excluded_crates` list for `no_std_fs_operations` in the root `dylint.toml`
  (verified: the list contains `chutoro_benches`, `chutoro_test_support`,
  `kani_nightly_gate_cli`, `benchmark_regression_gate_cli`, `kani_nightly_gate`,
  `benchmark_regression_gate`, `chutoro_providers_dense`, and `chutoro_cli`).
  Whitaker will reject `std::fs` here.
- The workspace denies `float_arithmetic`, `integer_division`, and
  `integer_division_remainder_used`. Backoff and jitter arithmetic must use
  integer shifts and multiply-shift, never floating point, division, or the
  remainder operator.
- The workspace also denies `indexing_slicing`, `unwrap_used`, `expect_used`,
  `panic_in_result_fn`, `unreachable`, `missing_const_for_fn`,
  `must_use_candidate`, `shadow_reuse`, `shadow_same`, `shadow_unrelated`,
  `use_self`, `option_if_let_else`, `needless_pass_by_value`, and
  `result_large_err`. Parsing must use `.get()`, not `[..]`.
- `clippy.toml` sets `too-many-lines-threshold = 70`,
  `cognitive-complexity-threshold = 9`, `too-many-arguments-threshold = 4`
  (which counts `&self`), and `excessive-nesting-threshold = 4`.
- `RecipeContext::new` is a `pub const fn` with three positional arguments and
  eight call sites, including two `trybuild` fixtures and several doctests. Its
  signature must not change. New capabilities arrive through a builder method,
  not through additional constructor parameters.
- The recipe lifecycle stays synchronous, per ADR-004. Asynchronous code is
  permitted only if fully contained inside one adapter, and the 10.1.1
  Tolerances already pre-authorized that for this milestone. In practice no
  async is needed: `ureq` is natively blocking.
- Do not modify `chutoro-benches`, `chutoro-core`, `chutoro-cli`, or
  `chutoro-providers/*`.
- Do not define the serialized manifest file format. That is roadmap 10.1.3.
  This milestone ships in-memory pin types only, and adds no `serde` dependency.
- Do not build the cross-recipe cache index. That is roadmap 10.1.5.
- Follow guidance from `docs/chutoro-design.md`,
  `docs/benchmark-dataset-retrieval.md` §3.2, `docs/property-testing-design.md`,
  `docs/complexity-antipatterns-and-refactoring-strategies.md`,
  `docs/rust-testing-with-rstest-fixtures.md`, `docs/rust-doctest-dry-guide.md`,
  `docs/reliable-testing-in-rust-via-dependency-injection.md`, and
  `docs/documentation-style-guide.md`.

## Tolerances (exception triggers)

- Scope: if any single stage requires modifying more than 12 files outside
  `chutoro-bench-datasets/`, or more than 2,000 net lines within a stage, stop
  and escalate.
- Dependencies: the dependency set in `Interfaces and dependencies` is
  pre-authorized. Adding **any** crate not on that list requires stopping and
  escalating first.
- Interface: if `RecipeContext::new`, `DatasetRecipe`, `Fetcher`, `Storage`, or
  `Publisher` must change signature, stop and escalate. Additive builder
  methods and new trait definitions are fine.
- Error budget: if `size_of::<RecipeError>() <= 32` cannot be preserved, stop
  and escalate rather than raising the number.
- Iterations: if a gate still fails after 3 fix attempts, stop and escalate
  with the captured log path.
- Test wall clock: if any single test approaches 45 seconds under
  `make test` (the nextest slow-timeout is 60 seconds, `terminate-after = 1`),
  stop and either shrink the fixture or add a documented `.config/nextest.toml`
  override. Do not silently raise timeouts.
- Verus: if the single proof in Stage B is not discharged within 90 minutes of
  effort, stop and escalate. Do not add `assume` to close it.
- Network: no test may contact a host outside `127.0.0.1`. If a test appears
  to need real network access, stop and escalate.
- Ambiguity: if a design choice materially changes the public surface and this
  plan does not settle it, stop and present options with trade-offs.

## Risks

- Risk: a pin bump for a new upstream version at an unchanged URL causes every
  machine with a warm cache to resume new bytes onto an old prefix, fail the
  digest, and — under a naive "never retry a full-length mismatch" rule — wedge
  permanently and fleet-wide with an error that reads like tampering. Severity:
  high. Likelihood: high (one routine pull request triggers it). Mitigation:
  the staging sidecar records the expectation it was created for (`url`,
  `expected_digest`, `expected_size`, `validator`), and any difference yields
  `DiscardAndRestart(ExpectationChanged)`. The staging file is named by
  expected digest, so two incompatible partials cannot collide. A full-length
  mismatch on a *resumed* transfer earns exactly one clean restart before being
  declared fatal.
- Risk: SHA-256 hasher state cannot be persisted across a process restart, so a
  resumed transfer must re-read its prefix to rebuild the digest, making a
  large transfer quadratic in the number of resumes. Severity: high.
  Likelihood: high at GIST1M and above. Mitigation: chunked digests. The
  sidecar stores a digest per fixed-size chunk, so a resume re-hashes at most
  one partial chunk. Verified: `sha2 0.10.8` exposes no `SerializableState`, and
  `crypto-common 0.1.6` does not define that trait; it arrived in
  `digest 0.11`.
- Risk: abandoned `.part` files and half-extracted temporary directories are
  invisible to a content-addressed cache scan and accumulate until a runner
  fills its disk, surfacing as `ENOSPC` in unrelated jobs. Severity: high.
  Likelihood: high once §10.3 lands large datasets. Mitigation: sidecar
  timestamps plus `PartialStore::sweep(max_age)`, invoked automatically at the
  start of every download rather than as an opt-in command.
- Risk: resume silently stops working (upstream moves behind a content-delivery
  network that adds `Content-Encoding`, or downgrades to weak validators),
  everything still succeeds, and the only symptom is that transfers get slower
  and more expensive. Severity: medium. Likelihood: medium. Mitigation: emit
  `dataset_download_resume_rejected_total{dataset, reason}` and
  `dataset_download_bytes_total{dataset, kind}` so the degradation is countable.
- Risk: the streaming ports ship but recipes cannot reach them, because
  `RecipeContext` exposes only `fetcher()`, `storage()`, and `publisher()`.
  Severity: high. Likelihood: high if not designed in. Mitigation: Stage B adds
  `RecipeContext::with_transfer` and `RecipeContext::transfer`, and a
  behavioural test drives the whole path through `run_recipe` rather than by
  calling the driver directly.
- Risk: putting HTTP response types in the domain layer forces roadmap 10.1.4's
  `object_store` adapter to fabricate fake HTTP status codes and
  `Content-Range` strings to satisfy a domain type. Severity: high. Likelihood:
  certain if not designed in. Mitigation: the adapter classifies the outcome
  into a transport-neutral `RangeOutcome`; the domain parses header *values*
  but never branches on a status code.
- Risk: the `ScriptedHttpServer` becomes a flaky test dependency. The project
  already carries fourteen per-test timeout overrides in
  `.config/nextest.toml`, and the default profile runs eight test binaries in
  parallel with no retries. A blocking `accept()` after a panicked test is a
  guaranteed 60-second hang and a hard kill. Severity: medium. Likelihood:
  medium. Mitigation: non-blocking accept with an explicit deadline, an explicit
  `shutdown(Both)`, a shutdown signal wired into `Drop`, joined threads, and
  an explicit `ureq` read timeout.
- Risk: the crate goes from 5 dependencies to roughly 12 with no advisory or
  licence gate anywhere in the repository. Severity: medium. Likelihood: high.
  Mitigation: Stage 0 adds `deny.toml` and a CI job before any dependency lands.
- Risk: bzip2's pure-Rust backend `libbz2-rs-sys` ships under the non-SPDX
  `bzip2-1.0.6` licence and will be flagged by any downstream licence scanner.
  Severity: low. Likelihood: high. Mitigation: record the allowance with a
  rationale comment in `deny.toml`, matching the `dylint.toml` house style.

## Progress

- [ ] M0 (planning): this ExecPlan reviewed and approved by the user. **No
  implementation may begin before this box is ticked.**
- [ ] M1 (supply chain, Stage 0): `deny.toml`, a CI advisory and licence job,
  and the `bzip2-1.0.6` allowance recorded with a rationale.
- [ ] M2 (integrity domain, Stage A): `DigestAlgorithm`, inhabited `Checksum`
  with real hex parsing, `DigestSink` object-safe façade, `ChunkedDigest`.
- [ ] M3 (error redesign, Stage A): boxed `IntegrityFailure`,
  `TransferFailure`, `ArchiveFailure` sub-enums; `ChecksumUnsupported` deleted;
  `FailureClass` and `RecipeError::class()`.
- [ ] M4 (pin contract, Stage A): mandatory `Integrity` field on `SourceSpec`
  replacing `Option<Checksum>`; `audit-unpinned` Makefile target.
- [ ] M5 (Stage A validation): gates green; `coderabbit review --agent`; commit
  and push; open Stage A pull request.
- [ ] M6 (transfer domain, Stage B): `ResumeDecision`, `decide_resume`,
  `accept_response`, `ContentRange` value parsing, `RetrySchedule`,
  `RetryClass`.
- [ ] M7 (Verus, Stage B): the jitter-bound proof in
  `verus/download_backoff.rs`, wired into `scripts/run-verus.sh`.
- [ ] M8 (transfer adapters, Stage B): `ureq` `RangeFetcher`, cap-std
  `PartialStore`/`PartialSlot` with sidecar, chunked digests, sweep, free-space
  preflight, advisory lock.
- [ ] M9 (context wiring, Stage B): `RecipeContext::with_transfer` and
  `transfer()`; end-to-end scenario driven through `run_recipe`.
- [ ] M10 (Stage B tests): `ScriptedHttpServer`; resume sequence proptest; the
  `decide_resume` Kani harness; behavioural scenarios; observability.
- [ ] M11 (Stage B validation): gates green; `coderabbit review --agent`;
  commit, push, open Stage B pull request.
- [ ] M12 (archive domain, Stage C): `ArchiveFormat` sniffing including
  `ArchiveFormat::None`, `ContainedEntryPath`, `EntryPolicy`, `ExpansionBudget`/
  `ExpansionMeter`.
- [ ] M13 (archive adapters, Stage C): tar walker, decoder dispatch, cap-std
  extraction sink, adopt-in-place fast path.
- [ ] M14 (Stage C tests): format and path proptests; the `ExpansionMeter` Kani
  harness; extraction behavioural scenarios.
- [ ] M15 (documentation): three new ADRs, ADR-004 amendment, users' guide,
  developers' guide, `contents.md`, `repository-layout.md`,
  `benchmark-dataset-retrieval.md` §3.2.
- [ ] M16 (final validation): all gates; mark roadmap 10.1.2 `[x]`; final
  `coderabbit review --agent`; open Stage C pull request.

## Surprises & discoveries

Recorded during planning; extend during implementation.

- Observation: `Checksum` is currently an **uninhabited** type. Its only
  variant `Sha256` is gated behind `#[cfg(any())]`, which is never true.
  Evidence: `chutoro-bench-datasets/src/newtypes/keys.rs:70-76`. Impact:
  `Option<Checksum>` is provably always `None`, so `SourceSpec.checksum` is a
  field the type system guarantees can never hold a value. Making it inhabited
  changes `size_of::<SourceSpec>()`. Since no other workspace crate depends on
  `chutoro-bench-datasets` (verified: the only reference beyond the workspace
  member list is the crate's own dev-dependency self-reference), the blast
  radius is entirely in-crate and no deprecation dance is warranted.
- Observation: the tests that break when `Checksum` becomes real are **not**
  the ones an initial reading suggests. Evidence:
  `chutoro-bench-datasets/tests/error.rs` contains exactly one test,
  `recipe_error_other_wraps_arbitrary_source`, which never mentions checksums.
  The assertions that actually break are
  `chutoro-bench-datasets/tests/newtypes.rs:48`
  (`assert!(spec.checksum.is_none())`), and the doctests at
  `src/newtypes/mod.rs:204` and `src/newtypes/mod.rs:228`, plus the
  `Checksum::parse` doctest at `src/newtypes/keys.rs:92`. Impact: Stage A's
  concrete steps name these four sites explicitly.
- Observation: `sha2 0.10` cannot serialize hasher midstate. Evidence:
  `SerializableState` appears nowhere in `sha2-0.10.8/src/`, and
  `crypto-common 0.1.6` does not define it; the trait arrived in `digest 0.11`.
  Impact: resume in a fresh process would have to re-read the whole prefix to
  rebuild the digest. Chunked digests replace that with re-hashing at most one
  partial chunk.
- Observation: `cap-std` has no file locking. Evidence: `cap-std 3.4.5`
  (the resolved version) exposes only `into_std()` and `as_fd()` on
  `cap_std::fs::File`; there is no `lock` or `try_lock`. Impact: the obvious
  `std::fs::File::lock` route is unavailable, because `into_std()` returns the
  `std::fs::File` that Whitaker bans in this crate. Locking goes through
  `rustix::fs::flock` on the borrowed file descriptor. `rustix` is already in
  the lockfile via cap-std, so this adds no dependency.
- Observation: every upstream this project actually uses honours HTTP range
  requests, with strong validators and no content coding. Evidence: probes on
  2026-08-16 against `ann-benchmarks.com/sift-128-euclidean.hdf5`,
  `ann-benchmarks.com/glove-200-angular.hdf5`, and the in-repo MNIST mirror
  `storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz` all
  returned `206 Partial Content` with a correct `Content-Range`,
  `Accept-Ranges: bytes`, a strong `ETag` (no `W/` prefix), and no
  `Content-Encoding`. Impact: resume has a working primary path on real
  sources, not merely a fallback. The ANN-Benchmarks ETags carry an S3
  multipart suffix (`-63`, `-115`); they are still syntactically strong, so
  `If-Range` is valid, but they are not digests of content and must not be
  treated as such.
- Observation: no roadmap dataset ships `.bz2` or `.xz`, and one needs a format
  that is out of scope. Evidence: ANN-Benchmarks publishes bare `.hdf5`; SNAP
  com-Amazon, CIFAR-10, and 20 Newsgroups are `.gz`/`.tar.gz`; canonical GloVe
  is `glove.6B.zip` (862 MB, `Accept-Ranges: bytes`). Impact: `.bz2` and `.xz`
  are implemented because the roadmap requires them, but each sits behind its
  own off-by-default feature so neither is on the default build path. The
  `.zip` gap is raised as a note against roadmap 10.3.10 rather than silently
  absorbed here.
- Observation: the repository has no supply-chain gate at all. Evidence: no
  `deny.toml`, no `audit.toml`, and none of the eight workflows in
  `.github/workflows/` runs `cargo-deny`, `cargo-audit`, or any advisory check.
  Impact: Stage 0 exists.
- Observation: `make verus` is a blocking pull-request gate, while Kani is
  nightly only. Evidence: `.github/workflows/ci.yml:112` defines a
  `verus-proofs` job running `make verus`;
  `.github/workflows/nightly-kani.yml:4-5` is `schedule: cron '0 2 * * *'`.
  Impact: every Verus proof adds latency to every pull request forever, so the
  bar for adding one is higher than for a Kani harness.
- Observation: `PartialState::orphaned_cache_key` is structurally always
  `None`. Evidence: `PartialState::new` hardcodes it
  (`src/newtypes/keys.rs:167-173`) and `cleanup_after_error` is its only caller
  (`src/driver.rs:129`). Impact: it reads as populated and is a trap. Noted for
  a follow-up; this plan does not extend `PartialState`.
- Observation: `RecipeError::Cleanup` discards the original error. Evidence:
  `src/driver.rs:130-141` returns `RecipeError::cleanup(...)` built only from
  the cleanup failure, with the original surviving solely in a `warn!` field.
  Impact: noted for a follow-up; out of scope here.

## Decision log

- Decision: add streaming ports rather than widen `Fetcher`.
  Rationale: `Fetcher::fetch_bytes` returns an owned `Bytes` capped by
  `max_bytes`. Widening it to stream would break every existing implementor and
  both `trybuild` fixtures, and `Fetcher` remains the right shape for small
  whole artefacts and for `file://` sources. `RangeFetcher` could subsume
  `Fetcher`; the reverse is false. They are kept apart on materialization
  strategy, not on ranges. Date/Author: 2026-08-16, plan author.
- Decision: keep `RangeFetcher` and `PartialStore` as separate ports rather
  than one `ResumableTransfer`. Rationale: the copy loop between them is the
  *tee* — it feeds the digest, the staging slot, and the budget meter in one
  pass. Folding the ports together moves that loop into infrastructure and
  takes the resume decision with it, at which point `decide_resume` becomes a
  pure function nothing is obliged to consult and the Kani harness proves
  things about code off the critical path. The split is what keeps the
  verification load-bearing. They also differ in failure class, feature gate,
  and test double. Date/Author: 2026-08-16, plan author, after design review.
- Decision: HTTP protocol types do not enter the domain layer. The adapter
  classifies a ranged read into a transport-neutral `RangeOutcome` (`Full`,
  `Partial`, `NotSatisfiable`, `Unmodified`); the domain parses header *values*
  such as `Content-Range` but never branches on a status code. Rationale:
  roadmap 10.1.4 adds `object_store` adapters. S3 has ETags and byte ranges but
  no `Accept-Ranges` header and no numeric `416`. With `status: u16` in the
  port contract, every non-HTTP adapter would fabricate a status code for the
  domain to branch on. Three reviewers reached this independently.
  `SourceUrl::parse` already accepts `s3://` and `file://`, so the crate has
  committed to non-HTTP sources at the type level already. Date/Author:
  2026-08-16, plan author, after design review.
- Decision: `SourceSpec` gains a mandatory `Integrity` field, replacing
  `Option<Checksum>`. `Integrity::Unpinned { justification: &'static str }` is
  the escape hatch. Rationale: `Option::None` is unsearchable, and both current
  constructors hardcode it, so "unverified" is not merely the default but the
  only reachable state. A greppable `Integrity::Unpinned` makes an audit
  complete. A parallel `PinnedSource` type was rejected because
  `DatasetRecipe::sources` returns `&[SourceSpec]`, so a second type forks that
  contract across thirteen downstream recipes with no compile-time check that
  each chose correctly. Putting `size_bytes` in the same variant as `checksum`
  makes the length-before-digest retry rule true by construction. Date/Author:
  2026-08-16, plan author, after design review.
- Decision: keep the 32-byte `RecipeError` budget and box one struct pointer
  per domain, rather than raising the assertion. Rationale:
  `clippy::result_large_err` defaults to a 128-byte threshold and `clippy.toml`
  does not override it, so the existing 32-byte assertion is a self-imposed
  constraint four times stricter than the denied lint. That makes keeping it a
  free choice. Three new variants each costing 8 bytes of payload preserve the
  assertion, and second-tier `<= 128` assertions on each sub-enum bound the
  headroom rather than leaving it unmeasured. Date/Author: 2026-08-16, plan
  author, after design review.
- Decision: digests are computed per fixed-size chunk and the chunk digests are
  recorded in the staging sidecar. Rationale: `sha2 0.10` cannot persist hasher
  state, so a flat digest forces a full prefix re-read on every cross-process
  resume. `docs/benchmark-dataset-retrieval.md` §7.13 and §9 already call for
  "per-shard checksum manifests" and "sharded object keys to avoid very large
  single-object retries". One mechanism gives a linear resume, a restart floor
  of one chunk, and a natural granularity for 10.1.5's locking. Date/Author:
  2026-08-16, plan author, after design review.
- Decision: delete `RecipeError::ChecksumUnsupported` rather than repurpose it.
  Rationale: repurposing keeps a name saying "unsupported" attached to a
  failure that is usually "malformed hex", and — decisively — any
  `matches!(e, ChecksumUnsupported)` would silently change meaning instead of
  failing to compile. Deletion is a hard compile error at all three sites.
  Date/Author: 2026-08-16, plan author, after design review.
- Decision: ship the `SignatureVerifier` port, the `SignatureSpec` type, and
  the `Integrity::Signed` variant, but **no** signature adapter in this
  milestone. This reverses an earlier planning decision to ship
  `minisign-verify`. Rationale: ANN-Benchmarks, the actual upstream for SIFT1M,
  GIST1M, MNIST, Fashion-MNIST, and GloVe, publishes neither checksums nor
  signatures — its entire download implementation is
  `if not os.path.exists(dest): urlretrieve(url, dest)`. No dataset in §10.3
  ships a signature, so an adapter would have zero producers and zero
  consumers. Verification stays *expressible*, satisfying the roadmap wording,
  and roadmap 10.1.6 owns provenance, trust roots, expiry, and revocation —
  which is where a keyring policy belongs. `minisign-verify` remains the
  recommended crate when that lands: MIT, 93 KB, zero runtime dependencies,
  with a streaming verifier. Date/Author: 2026-08-16, plan author, after design
  review.
- Decision: no `backon` dependency. The retry loop is written in-crate over the
  pure `RetrySchedule` plus the `Sleeper` and `JitterSource` ports. Rationale:
  with a property-tested pure schedule function and injected sleep and jitter
  already required for determinism, `backon` contributes a `for` loop while
  creating a seam where two scheduling vocabularies must be reconciled in
  review. Date/Author: 2026-08-16, plan author, after design review.
- Decision: no `subtle` dependency; `Checksum` keeps its derived `PartialEq`.
  Rationale: a derived `==` beside a constant-time-comparison dependency is a
  manifest that lies about the threat model, and the derived operator is what
  callers would actually invoke. There is no secret and no repeated-query
  oracle here: the expected digest comes from an in-repo manifest and the
  attacker supplies the artefact, not the expectation. Date/Author: 2026-08-16,
  plan author, after design review.
- Decision: one Verus proof, on the jitter arithmetic — not on the resume
  sequence, and not on entry-path containment. Rationale: AGENTS.md requires a
  proof be substantive rather than a restatement of the assumed property. An
  entry-path containment lemma would prove a theorem about a handwritten model
  of `Path::join`, which Verus does not model, having assumed the very
  sanitizer under test; and it aims at the wrong bug class, since the recent
  tar advisories are parser differentials that occur *before* a component
  sequence exists. A resume-sequence lemma would rigorously prove the easy
  induction step while *assuming* the `Content-Range` parser where all the risk
  lives, and the realistic bugs (append before offset check, sidecar and file
  disagreeing after a crash) live in an adapter Verus never sees. The jitter
  bound is different: `(sample as u128 * span as u128) >> 64 <= span` is
  nonlinear, unbounded in both operands, cannot be exhaustively sampled by
  proptest, and is infeasible for Kani over a 2^64 × 2^64 space. It exists only
  because the workspace denies `float_arithmetic` and
  `integer_division_remainder_used`, forcing an unusual implementation, and if
  it is wrong the delay exceeds `max_delay` or overflows. It is small, stable,
  and will not drift. Date/Author: 2026-08-16, plan author, after design review.
- Decision: replace the dropped resume Verus proof with a proptest state
  machine over attempt *sequences*. Rationale: it drives the real
  `decide_resume` and `accept_response` rather than a transcription, it shrinks
  failures, and it runs in `make test` on every pull request. Verus proofs under
  `verus/` are standalone transcriptions with no build-level tie to `src/`, so
  drift is invisible to continuous integration; the resume machine must absorb
  `object_store` in 10.1.4 and lockfiles in 10.1.5 and would desynchronize
  quickly. Date/Author: 2026-08-16, plan author, after design review.
- Decision: leave the ad-hoc MNIST downloader in
  `chutoro-benches/src/source/mnist/mod.rs` alone. Rationale: roadmap 10.3.2
  already owns "MNIST digits: implement pinned IDX fetch + checksum
  validation". Migrating now would touch a crate with a divergent lint
  configuration that is excluded from `no_std_fs_operations`, and would perturb
  benchmark baselines inside a milestone that is already large. Recorded so the
  duplication is deliberate and visible. Date/Author: 2026-08-16, plan author.
- Decision: deliver 10.1.2 as three sequential pull requests under one roadmap
  checkbox. Rationale: roadmap 10.1.1 was roughly a third of this scope with no
  new dependencies, no network, and one property test, and it consumed nine
  milestones and eleven CodeRabbit rounds. A single 4,000-line pull request is
  not reviewable. There is precedent for staging inside one checkbox.
  Date/Author: 2026-08-16, plan author, after design review.
- Decision: `ArchiveFormat::None` is a first-class variant with an
  adopt-in-place path. Rationale: ANN-Benchmarks ships bare `.hdf5` and DEEP1B
  ships raw `.bvecs` and `.fbin`, so "not an archive" is the *common* case, not
  an edge case. Without it, preparing DEEP1B would copy roughly 450 gibibytes
  to produce a byte-identical file. Date/Author: 2026-08-16, plan author, after
  design review.

## Outcomes & retrospective

To be completed at each stage boundary and at M16. Compare the delivered
surface against `Purpose / big picture`, and record what would be done
differently.

## Context and orientation

You are working in a Rust workspace called `chutoro`, a clustering library. The
crate this plan changes is `chutoro-bench-datasets`, which prepares datasets
used to benchmark the clustering pipeline. It is `publish = false` and no other
workspace crate depends on it.

Some vocabulary, defined before use.

A **port** is a Rust trait describing something the crate needs from the
outside world (reading bytes, writing files). An **adapter** is a concrete
implementation of a port. **Domain** code is pure logic that depends on
neither. This separation is called *hexagonal architecture*; the rule is that
dependencies point inward, so domain code never mentions an adapter.

A **recipe** is one dataset's preparation procedure. `DatasetRecipe`
(`chutoro-bench-datasets/src/recipe.rs`) has four ordered phases — `fetch`,
`validate`, `prepare`, `publish` — where each phase's output type is the next
phase's input, so a caller cannot publish unvalidated bytes. `run_recipe`
(`src/driver.rs`) executes them in order and calls `cleanup` on failure.

**Pinning** means recording, in the repository, exactly which bytes a URL is
expected to yield — here, a SHA-256 digest and a byte length.

**Resuming** means asking a server for only the part of a file you do not yet
have, using an HTTP range request.

A **decompression bomb** is a small archive that expands to an enormous amount
of data, exhausting disk or memory.

### What exists today

`chutoro-bench-datasets/src/` contains: `lib.rs` (re-exports), `recipe.rs` (the
`DatasetRecipe` trait), `driver.rs` (`run_recipe`), `context.rs`
(`RecipeContext`, holding `&dyn Fetcher`, `&dyn Storage`, `&dyn Publisher`),
`error.rs` (`RecipeError` and the 32-byte assertion), `info.rs` (`DatasetInfo`),
`published.rs` (the sealed `PublishedArtefact` trait), `newtypes/` (`RecipeId`,
`RecipeVersion`, `SourceUrl`, `SourceSpec`, `CacheKey`, `ObjectKey`,
`Checksum`, `ManifestDigest`, `Phase`, `PortName`, `PartialState`), `ports/`
(the three port traits), and `testing/` (in-memory and filesystem test doubles,
behind the `testing` feature).

Three placeholders were deliberately left for this milestone. `Checksum` is an
uninhabited enum whose `parse` is a `const fn` always returning
`RecipeError::ChecksumUnsupported`. `SourceSpec.checksum` is
`Option<Checksum>`, documented "filled by roadmap item `10.1.2`". ADR-004
records that archive extraction is deferred and that 10.1.1 introduces no public
`Extractor` port.

### Relevant documentation to consult while implementing

- `docs/benchmark-dataset-retrieval.md`, especially §3.2 (this item's source of
  truth), §5 (the cost model), §7.13 (DEEP1B), and §9 (key risks).
- `docs/adr-004-bench-dataset-recipe-trait.md` for the existing port and phase
  decisions, which this milestone amends.
- `docs/chutoro-design.md` for project-wide design conventions.
- `docs/property-testing-design.md` §6.2 for property-test layout conventions,
  and Appendix A for how Verus candidates are framed here.
- `docs/rust-testing-with-rstest-fixtures.md` and
  `docs/rust-doctest-dry-guide.md` for test and doctest conventions.
- `docs/reliable-testing-in-rust-via-dependency-injection.md` for the injected
  clock and environment pattern that `Sleeper` and `JitterSource` follow.
- `docs/complexity-antipatterns-and-refactoring-strategies.md` before extracting
  helpers.
- `docs/developers-guide.md` §"Benchmark dataset recipes" and §"Verus proofs".
- `docs/documentation-style-guide.md` for the ADR template and Markdown rules.
- `AGENTS.md` for code style, the abstraction and port policy, error handling,
  and observability.

### Relevant skills to keep loaded while implementing

- `leta` for code navigation; prefer it to grep for symbol lookup.
- `rust-router` to route to the narrower Rust skills.
- `rust-types-and-apis` for the newtypes, the sealed and unforgeable types, and
  trait object safety.
- `rust-errors` for the boxed sub-enum design and failure classification.
- `rust-unit-testing` for rstest fixtures and assertion shape.
- `hexagonal-architecture` for the port and adapter boundaries.
- `proptest` for the property suites and the resume sequence state machine.
- `kani` for the two harnesses.
- `verus` for the single arithmetic proof.
- `arch-supply-chain` for Stage 0's `deny.toml`.
- `nextest` when a test approaches the slow-timeout.
- `execplans` for this document's envelope.
- `commit-message` and `pr-creation` at each stage boundary.

## Plan of work

Four stages. Each ends with `make check-fmt`, `make lint`, `make test`, then
`coderabbit review --agent`, then a commit. Stages A, B, and C each end with a
pull request. Do not begin a stage until the previous stage's gates are green.

### Stage 0: supply chain (M1)

Create `deny.toml` at the workspace root with `[advisories]`, `[licenses]`, and
`[bans]` sections. Under `[licenses]`, allow the workspace's existing set and
add an explicit `bzip2-1.0.6` allowance for `libbz2-rs-sys` with a rationale
comment in the style of `dylint.toml`'s `excluded_crates`. Under `[bans]`, add a
`deny` entry preventing two generations of `digest` from coexisting, so the
`sha2` version decision cannot be silently undone.

Add a `cargo-deny` job to `.github/workflows/ci.yml` alongside the existing
`verus-proofs` job. Add a `make deny` target and register it in `make all`. Run
`mbake validate Makefile` — it is a gate.

Because `sha2 0.10` is pinned deliberately against the newer 0.11, add a
Dependabot ignore entry with the rationale, or the daily bot will reopen that
pull request indefinitely.

Stage 0 validation: `make deny` exits 0 on the current tree;
`mbake validate Makefile` passes; `make check-fmt` and `make lint` pass.

### Stage A: integrity domain (M2, M3, M4, M5)

Everything here is pure. No network, no filesystem, no archives. Only one new
dependency, `sha2`. This stage should pass every gate first time.

**Red first.** Before touching production code, write the failing tests:

- `chutoro-bench-datasets/tests/integrity.rs` — rstest cases asserting
  `Checksum::parse("sha256:<64 hex>")` succeeds and round-trips through
  `Display`; that a wrong-length digest, non-hex characters, an unknown
  algorithm prefix, and a missing prefix each fail with the right
  `IntegrityFailure` variant.
- `chutoro-bench-datasets/tests/integrity_proptest.rs` — a round-trip property
  over arbitrary 32-byte arrays, and a rejection property over arbitrary
  strings that are not valid digests.
- Extend `chutoro-bench-datasets/tests/error.rs` with
  `integrity_failure_boxed_payload_preserves_source` and a
  `recipe_error_size_budget_is_respected` test asserting
  `size_of::<RecipeError>() <= 32`. The budget currently lives only in a
  `const _` at the bottom of `src/error.rs`; make it visible to a reviewer
  reading the test suite.

Run them; confirm they fail to compile or fail for the intended reason; capture
the log.

**Then implement.** In `src/integrity/digest.rs`, define `DigestAlgorithm`
(`Sha256`, `Blake3` reserved but not implemented — do not add `blake3` as a
dependency in this milestone) and an inhabited
`Checksum { algorithm, bytes: [u8; 32] }` with real hex parsing. Drop the
`const` from `parse`: `?` on `Result` is not permitted in a const fn on stable,
and `indexing_slicing` is denied so a hand-rolled const parser is not available.
`missing_const_for_fn` fires only when a function *could* be const, so it will
not object.

In `src/integrity/sink.rs`, define `DigestSink`. **`sha2::Digest` is not
object-safe** — `update<B: AsRef<[u8]>>` is generic and `new() -> Self` requires
`Sized`, so `Box<dyn Digest>` will not compile. `DigestSink` is the required
façade: `fn update(&mut self, chunk: &[u8])` and
`fn finish(self: Box<Self>) -> Checksum`.

In `src/integrity/chunked.rs`, define `ChunkedDigest`: a chunk size constant (8
MiB), a `Vec<[u8; 32]>` of per-chunk digests, and the whole-artefact digest.
Provide `resume_point(staged_len) -> (whole_chunks, partial_offset)` as a pure
function, and property-test that re-hashing from a chunk boundary yields the
same whole-artefact digest as hashing straight through.

In `src/error.rs`, replace `ChecksumUnsupported` with the three boxed variants
and add the second-tier size assertions. Add `FailureClass` (deliberately
**not** `#[non_exhaustive]`, so a new class is a compile error at every
dispatch site) and `RecipeError::class()`. Classify `Other` as `Contract`, not
`Transient`: an unclassified failure must never be retried by default.

In `src/newtypes/mod.rs`, add `Integrity` and change
`SourceSpec.checksum: Option<Checksum>` to `SourceSpec.integrity: Integrity`.
Update both constructors and both doctests. Add `TransferFailure` and
`ArchiveFailure` as empty-but-declared enums so Stages B and C only add
variants.

Add a `make audit-unpinned` target that greps for `Integrity::Unpinned` and
prints each occurrence with its justification.

The four sites that must change, exactly: `src/error.rs:19-21`,
`src/newtypes/keys.rs:83`, `src/newtypes/keys.rs:92`, `src/newtypes/keys.rs:96`
for `ChecksumUnsupported`; and `tests/newtypes.rs:48`,
`src/newtypes/mod.rs:204`, `src/newtypes/mod.rs:228` for `checksum.is_none()`.

Stage A validation: the red tests now pass; `make check-fmt`, `make lint`,
`make test` all green; `coderabbit review --agent` clean or findings resolved.

### Stage B: transfer (M6, M7, M8, M9, M10, M11)

The risky stage. Domain first, adapters second, wiring third.

**Domain** (`src/transfer/`), all pure and all property-testable:

`resume.rs` holds `ResumeContext`, `ResumeDecision`, and `decide_resume`. The
rules, each traceable to RFC 9110 and each stated in the module's `//!` comment
with its section number, because a reader without the citation will delete them
as missed optimizations:

- `Accept-Ranges` is advisory (§14.3, "A client MUST NOT assume…"), so resume
  is never gated on it and the whole-representation fallback is always handled.
- No strong validator means `StartFresh`, always. §13.1.5 forbids a weak
  validator in `If-Range`. Resuming without a validator is how a corrupt file
  gets built silently.
- A `Full` outcome in reply to a ranged read means the server ignored the range
  or the representation changed (§14.2, §13.1.5): truncate to zero and rewrite.
  **This is the normal path, not an error** — it reads like a bug to anyone who
  has not read the RFC, so say so in the comment.
- `NotSatisfiable` carries a remote total; if it equals the staged length the
  artefact is already whole, otherwise discard and restart (§15.5.17).
- Any content coding disables resume for that URL, because §14.1.2 makes range
  offsets refer to the *encoded* octet stream and on-the-fly compression is not
  byte-stable across requests.
- Any difference between the sidecar's recorded expectation and the current pin
  yields `DiscardAndRestart(ExpectationChanged)`.
- After K attempts adding zero bytes, `DiscardAndRestart(NoProgress)`; if a
  fresh attempt also stalls, terminal.

`response.rs` parses `Content-Range` *values* and enforces §14.4's invalidity
rules (last < first, or complete-length <= last, is invalid and must never be
recombined). It never sees a status code.

`backoff.rs` holds `RetryPolicy` and
`delay_for_attempt(policy, attempt, sample) -> Duration`. Exponential growth is
`base_millis << min(attempt, shift_cap)`, saturating, clamped to `max_delay`.
Jitter is multiply-shift, `(sample as u128 * span as u128) >> 64`, because
`integer_division_remainder_used` forbids the usual modulo and
`float_arithmetic` forbids the usual float. The retry budget counts **attempts
since forward progress**, not total attempts: an 11-hour transfer expects
roughly 11 drops, and a flat 5-attempt budget would kill it at hour 5.

`classify.rs` maps failures to `RetryClass`. `ENOSPC`, permission denied, and
TLS certificate expiry are terminal — retrying an expired certificate burns the
whole schedule waiting for time to run backwards.

**Verus** (M7): `verus/download_backoff.rs` proves the jitter bound and the
saturation property. Wire it into the `PROOF_FILES` array in
`scripts/run-verus.sh` — note that `edge_harvest_extract.rs` and
`edge_harvest_ordering.rs` are *not* listed there because they are pulled in as
`mod` declarations, so getting this wrong means the proof silently never runs.
Use `by(nonlinear_arith)` for the multiply-shift bound; start with `#![auto]`
triggers and read the note Verus prints, per the developers' guide.

**Adapters** (`src/transfer/adapters/`), not a top-level `adapters/` directory
— AGENTS.md says group by feature, not layer, and nothing here is shared
between features:

`http.rs` implements `RangeFetcher` over `ureq`. Build it
`default-features = false, features = ["rustls", "platform-verifier"]`: the
`gzip` feature decodes based on the response's content coding *regardless* of
what was requested, which would silently corrupt a resumed file. Configure
`.http_status_as_error(false)` so `416` and the whole-representation fallback
arrive as inspectable responses. Set an explicit read timeout and connect
timeout — all `ureq` timeouts default to `None`, and a missing read timeout is
how the truncation test becomes a 60-second hang. Pin this configuration with a
test against the scripted server, not a comment.

`staging.rs` implements `PartialStore` and the RAII `PartialSlot`. The staging
file is named by **expected digest**, so two incompatible partials cannot
collide. The sidecar records
`{url, expected_digest, expected_size, validator,
chunk_digests, created_at, last_progress_at}`.

Two durability invariants, both load-bearing:

1. Write ordering is data, then fsync data, then sidecar via atomic
   temp-and-rename, then fsync the directory. The sidecar records the last
   **durable** offset and deliberately lags the file.
2. Because the sidecar lags, the staged file's length is greater than or equal
   to the recorded offset. Resume must `set_len(recorded_offset)` **before**
   appending, or a partially written tail is silently incorporated. This is the
   single most important line in the feature.

Locking uses `rustix::fs::flock` on the borrowed file descriptor from
`cap_std::fs::Dir` — `cap-std` has no `lock`, and `into_std()` returns the
`std::fs::File` that Whitaker bans here. Use the non-blocking variant in a
bounded loop with a `warn!` on first contention and a hard timeout; a lock with
no timeout wedges every other job with no log line. Document that the cache
root must be a local filesystem, because advisory locks are unreliable on
network and overlay mounts.

Free-space preflight uses `rustix::fs::fstatvfs` against the pinned
`size_bytes` before a byte moves, failing fast with a typed error instead of
`ENOSPC` at hour six. `rustix` is already in the lockfile via cap-std.

`PartialStore::sweep(max_age)` removes aged staging pairs, orphaned sidecars,
and stale extraction temporaries, and is called **automatically at the start of
every download**, not as an opt-in command.

**Wiring** (M9): add
`RecipeContext::with_transfer(self, &TransferPorts) -> Self` and
`RecipeContext::transfer(&self) -> Option<&TransferPorts>`.
`RecipeContext::new` keeps its exact three-argument `const fn` signature, so
all eight existing call sites survive untouched. Without this step the ports
ship unreachable from `DatasetRecipe::fetch`, and both existing exemplar
recipes would continue to show new authors the in-memory path.

**Tests** (M10): `ScriptedHttpServer` under the `testing` feature in
`src/testing/http_server.rs`, on `std::net::TcpListener` with no new
dependency. Scriptable: honour range, ignore range, `416`, change the validator
mid-transfer, truncate the body mid-flight, and fail then succeed. Bind
`127.0.0.1:0`. Non-blocking accept with an explicit deadline, explicit
`shutdown(Both)`, a shutdown signal in `Drop`, and joined threads.

The resume sequence proptest drives a *sequence* of scripted outcomes through
the real `decide_resume` and `accept_response` against a model of the written
byte range, asserting the written set is exactly `[0, final_len)`. This is what
replaces the dropped Verus resume proof, and unlike a proof it runs against the
shipped code.

The `decide_resume` Kani harness needs `ResumeContext` to be fixed-width plain
data: model the validator as `[u8; 32]` (a digest of the entity tag), never a
`String`, or the harness needs a bounded symbolic string and will time out.

No test may construct the production schedule with real sleeps: five attempts
at a 1-second base sleeps 31 seconds, one retry from the 60-second kill.
`JitterSource` must be deterministic in tests.

Observability: `metrics` is already in the lockfile and `chutoro-core` already
has the house pattern — an optional `metrics` feature, no global recorder,
`metrics-util`'s debugging recorder in dev-dependencies. Mirror it exactly.
Ship at minimum `dataset_download_resume_rejected_total{dataset, reason}` and
`dataset_download_bytes_total{dataset, kind}`; without them, a resume that
silently stops working is undetectable by construction. Labels must be closed
enums — never a URL, never an error string.

Stage B validation: gates green; the truncate-and-resume scenario passes; the
end-to-end scenario runs through `run_recipe`, not by calling the driver
directly.

### Stage C: archives (M12, M13, M14)

**Domain** (`src/archive/`): `format.rs` sniffs magic bytes, not extensions —
`1f 8b` gzip, `42 5a 68` bzip2, `fd 37 7a 58 5a 00` xz — and returns
`ArchiveFormat::None` when nothing matches, which is the common case for bare
`.hdf5` and raw `.bvecs`.

`entry_path.rs` holds `ContainedEntryPath`, constructible only through its
sanitizer. The name says containment, which is the invariant, rather than
"safe", which is an assertion. Four negative obligations: no public
constructor, no `From<Utf8PathBuf>`, no `Default`, and **no `Deserialize`** —
that last is the leak 10.1.3 will otherwise introduce, and if serde is ever
needed it must go via `TryFrom<String>`.

`policy.rs` rejects symlink, hardlink, device, and **sparse** entries outright.
Sparse entries matter because a GNU sparse header declares a huge logical size
while producing few observed bytes, defeating a meter that charges only what it
copies.

`budget.rs` holds `ExpansionBudget` and `ExpansionMeter`, charging the declared
logical size *before* writing and the observed bytes after, with all arithmetic
checked.

**Adapters** (`src/archive/adapters/`): the tar walker drives `entries()`
directly and never calls `Archive::unpack`. `Entry::unpack_in` returns
`Ok(false)` to mean "entry silently skipped for containing `..`", and
`Archive::unpack` discards that boolean — for a pinned dataset a dropped entry
must be a hard error. `tar >= 0.4.46` is the floor, not 0.4.45: the PAX header
scoping fix landed in 0.4.46 and 0.4.45 still carries that parser differential.

Wrap the decompressor — not the compressed reader — in `Read::take` before it
reaches `tar`, so a bomb surfaces as a tar error rather than a full disk. Say
this in a comment so nobody "simplifies" it later. Set `liblzma`'s memory limit
explicitly; it defaults to `u64::MAX`, and the dictionary is allocated up front
so a byte cap alone cannot catch it. Gzip and bzip2 have no equivalent exposure.

The `ArchiveFormat::None` path renames the already-verified staged file into
place with no copy and no extraction.

Extraction writes to a sibling temporary directory named deterministically
(`.extract-<digest>.tmp`), then renames. `rename(2)` on a directory fails when
the destination exists, so **`ENOTEMPTY` on the destination must be treated as
success** — that is the idempotence rule and it is easy to get wrong. Never
resume a half-extracted temporary directory; delete and re-extract.

### Stage D: documentation and final validation (M15, M16)

Three new ADRs, numbered 005, 006, 007, following the template in
`docs/documentation-style-guide.md`:

- ADR-005: a streaming `RangeFetcher` port added *alongside* the byte-oriented
  `Fetcher` rather than replacing it, with the transport-neutral `RangeOutcome`
  seam and the reasoning about 10.1.4.
- ADR-006: the in-repo pin as integrity root because upstream publishes
  nothing; mandatory `Integrity` with a greppable `Unpinned`; chunked digests;
  and the divergence from `pooch` on retry after a digest mismatch.
- ADR-007: archive extraction policy — reject symlinks, hardlinks, and sparse
  entries; never call `Archive::unpack`; the three-axis expansion budget.

Amend `docs/adr-004-bench-dataset-recipe-trait.md`: its "Archive extraction is
deferred… 10.1.1 does not introduce a public `Extractor` port" clause is now
superseded. Do not silently contradict a live ADR.

Update `docs/users-guide.md` §"Preparing benchmark datasets" (feature flags,
the offline story, the fatal-versus-retryable taxonomy),
`docs/developers-guide.md` §"Benchmark dataset recipes" (the new ports, the
feature-flag table, the no-default-features fitness function, scripted-server
conventions), `docs/benchmark-dataset-retrieval.md` §3.2 (planned to delivered),
`docs/contents.md` and `docs/repository-layout.md` (new ADRs and new
subdirectories — the first thing forgotten), and `docs/roadmap.md` (tick
10.1.2, and add a note against 10.3.10 that canonical GloVe ships as `.zip`,
which is outside this item's declared format list).

## Concrete steps

Run everything from the repository root, on branch
`10-1-2-shared-download-primitives`. Capture every gate to a log, because the
terminal truncates long output:

```bash
LOG=/tmp/$ACTION-chutoro-$(git branch --show-current).out
```

Per-stage loop:

```bash
# Red: write the failing test first, then observe the failure.
cargo nextest run -p chutoro-bench-datasets --all-features \
  2>&1 | tee /tmp/red-chutoro-10-1-2-shared-download-primitives.out

# Green: minimal implementation, then the focused test.
cargo nextest run -p chutoro-bench-datasets --all-features \
  2>&1 | tee /tmp/green-chutoro-10-1-2-shared-download-primitives.out

# Gates, sequentially — never in parallel, the build cache depends on it.
make check-fmt 2>&1 | tee /tmp/check-fmt-chutoro-10-1-2-shared-download-primitives.out
make lint      2>&1 | tee /tmp/lint-chutoro-10-1-2-shared-download-primitives.out
make test      2>&1 | tee /tmp/test-chutoro-10-1-2-shared-download-primitives.out

# Architecture fitness: the domain must compile with no infrastructure.
cargo check -p chutoro-bench-datasets --no-default-features --lib
cargo check -p chutoro-bench-datasets --no-default-features --features http
cargo check -p chutoro-bench-datasets --no-default-features --features archives

# Supply chain and docs.
make deny
make markdownlint
make nixie
mbake validate Makefile

coderabbit review --agent
```

Kani and Verus run outside `make test`:

```bash
make verus
cargo kani -p chutoro-bench-datasets --default-unwind 6 \
  --harness verify_expansion_meter_respects_budget
cargo kani -p chutoro-bench-datasets --default-unwind 4 \
  --harness verify_decide_resume_is_total
```

Add the two new harnesses to the `kani` target in the `Makefile`; it names
harnesses explicitly (currently four across two packages), so a harness not
listed there is never run in continuous integration.

Expected transcript shape at a stage boundary:

```plaintext
$ make test
    Starting NNN tests across MM binaries
        PASS [   0.030s] chutoro-bench-datasets integrity::checksum_parse_round_trips
        PASS [   0.412s] chutoro-bench-datasets transfer::resume_after_truncated_body
     Summary [   NN.NNNs] NNN tests run: NNN passed, 0 skipped
```

## Validation and acceptance

Acceptance is behavioural, not structural.

**Integrity.** `Checksum::parse("sha256:" + 64 hex chars)` returns a checksum
whose `Display` reproduces the input. A 63-character digest, a non-hex
character, an unknown algorithm prefix, and a bare digest with no prefix each
return `RecipeError::Integrity` carrying the matching `IntegrityFailure`
variant. `size_of::<RecipeError>()` is still at most 32, asserted both at
compile time and in a test.

**Resume.** With the scripted server configured to send 4 KiB of an 8 KiB body
and then close the connection, the first attempt fails; the staging file holds
4 KiB; the sidecar records offset 4096 and the validator; the second attempt
issues a range request from 4096 with `If-Range`, receives the remainder, and
the assembled 8 KiB matches its pinned digest. With the server configured to
ignore the range and return the whole representation, the staged file is
truncated to zero and rewritten, and the result still verifies. With the server
configured to change its validator between attempts, the partial is discarded
and a fresh transfer starts.

**Reachability.** A recipe implemented in the test suite reaches
`download_verified` through `RecipeContext::transfer()` inside its `fetch`
phase, and the whole thing runs through `run_recipe`. This is the test that
proves the ports are not stranded.

**Extraction.** A `.tar.gz` fixture containing a `../escape` entry causes a
hard error naming the rejected entry, not a silent skip. A fixture containing a
symlink is refused. A fixture whose expansion exceeds a test-only 16 MiB budget
aborts with `ArchiveFailure::BudgetExceeded` naming which cap was hit. A bare
uncompressed file is adopted in place with no copy.

**Fitness.**
`cargo check -p chutoro-bench-datasets --no-default-features --lib` succeeds,
and each single-feature check succeeds, proving no `#[cfg(feature = "http")]`
item references an `archives`-only type.

Quality criteria:

- Tests: `make test` passes with zero failures and zero warnings.
- Lint: `make check-fmt` and `make lint` exit 0. Clippy warnings are denied.
- Proofs: `make verus` exits 0; both Kani harnesses pass under `make kani`.
- Mutation check: deliberately break one production rule — for example, make
  `decide_resume` return `ResumeFrom` when the validator did not match — and
  confirm a test fails with a meaningful message. Restore. A harness that
  survives a deliberate mutation is not testing what it claims.
- Supply chain: `make deny` exits 0.
- Docs: `make markdownlint` and `make nixie` exit 0.
- Network: no test contacts anything outside `127.0.0.1`.

## Idempotence and recovery

Every step is re-runnable. Gates are read-only except `make fmt`, which
rewrites formatting.

If a stage goes wrong, `git restore` the affected files; nothing outside the
repository is mutated. The scripted server binds an ephemeral port and is torn
down in `Drop`, so a failed run leaves no listener. Tests write only to
`tempfile` directories.

If a Verus proof will not close, do **not** insert `assume` — a stray
`assume(false)` proves anything and is a soundness hole. Stop and escalate per
Tolerances.

If a property test fails, the minimal case is written to
`proptest-regressions/`; commit that file so the case is guarded against
regression.

Stage boundaries are the rollback points. Because each stage is its own pull
request, reverting one stage does not disturb the others.

## Artefacts and notes

Evidence captured during planning, retained because it settles design questions
that would otherwise be guesses.

Upstream range support, probed 2026-08-16:

```plaintext
### ANN-Benchmarks SIFT1M (hdf5)
  ranged GET status : 206
  content-range: bytes 100-199/525128288
  accept-ranges: bytes
  etag: "2bf1ef4c0c3e17031bd64980774eea09-63"
  content-encoding: <absent>
```

All three probed hosts (ANN-Benchmarks SIFT1M and GloVe-200, and the in-repo
MNIST mirror) returned the same shape: `206`, a correct `Content-Range`,
`Accept-Ranges: bytes`, a strong entity tag, and no content coding.

Error layout measurements, which drive the boxing decision:

```plaintext
RecipeError                  size=32 align=8   <- budget fully consumed
PortFailure                  size=24           <- largest existing payload
Checksum{alg,[u8;32]}        size=33           <- over budget on its own
{expected,actual,alg}        size=65
```

## Interfaces and dependencies

Dependencies to add to `chutoro-bench-datasets/Cargo.toml`. This list is
exhaustive and pre-authorized; anything else requires escalation.

```toml
[features]
default    = []
http       = ["dep:ureq", "dep:rustix", "cap-std"]
archives   = ["dep:tar", "dep:flate2"]
bz2        = ["archives", "dep:bzip2"]
xz         = ["archives", "dep:liblzma"]
metrics    = ["dep:metrics"]
testing    = ["dep:cap-std"]

[dependencies]
sha2    = "0.10"
ureq    = { version = "3.4", default-features = false,
            features = ["rustls", "platform-verifier"], optional = true }
tar     = { version = "0.4.46", default-features = false, optional = true }
flate2  = { version = "1.1", default-features = false,
            features = ["zlib-rs"], optional = true }
bzip2   = { version = "0.6", optional = true }
liblzma = { version = "0.4", optional = true }
rustix  = { version = "1.1", features = ["fs"], optional = true }
metrics = { version = "0.24", optional = true }
cap-std = { version = "3.4.5", optional = true, features = ["fs_utf8"] }
```

`sha2` is unconditional because integrity is domain, not infrastructure.
`sha2 0.10`, not 0.11, unifies with the existing lockfile entry reached through
`rstest-bdd → i18n-embed → rust-embed`; the `[bans]` rule in `deny.toml`
prevents a second `digest` generation appearing. `flate2` with `zlib-rs`
matches cargo, rustup, and uv. `bzip2 0.6` defaults to the pure-Rust
`libbz2-rs-sys` backend, so no C toolchain. `liblzma` replaces the stale `xz2`
(last released 2022) and is the heaviest item at 5.5 MB of vendored C, hence
its own feature. `cap-std` moves from testing-only to a real optional
dependency.

Signatures below are the contract each stage must produce.

```rust
// src/integrity/digest.rs
pub enum DigestAlgorithm { Sha256 }
pub struct Checksum { /* private */ }
impl Checksum {
    pub fn parse(value: &str) -> Result<Self, RecipeError>;
    pub const fn algorithm(&self) -> DigestAlgorithm;
    pub const fn as_bytes(&self) -> &[u8; 32];
}

// src/integrity/sink.rs — sha2::Digest is NOT object-safe; this is the façade.
pub trait DigestSink: Send {
    fn update(&mut self, chunk: &[u8]);
    fn finish(self: Box<Self>) -> Checksum;
}

// src/newtypes/mod.rs
#[non_exhaustive]
pub enum Integrity {
    Pinned { checksum: Checksum, size_bytes: u64 },
    Signed { checksum: Checksum, size_bytes: u64, signature: SignatureSpec },
    Unpinned { justification: &'static str },
}

// src/error.rs
pub enum FailureClass { Transient, Local, Pin, Hostile, Contract }
impl RecipeError {
    pub fn class(&self) -> FailureClass;
    pub fn is_retryable(&self) -> bool;
}

// src/transfer/ports.rs
#[non_exhaustive]
pub enum RangeOutcome { Full, Partial, NotSatisfiable, Unmodified }

pub trait RangeFetcher: Send + Sync {
    fn open_range(&self, request: &RangeRequest)
        -> Result<RangeResponse<'_>, RecipeError>;
}

pub trait PartialStore: Send + Sync {
    fn acquire(&self, key: &CacheKey)
        -> Result<Box<dyn PartialSlot + '_>, RecipeError>;
    fn sweep(&self, max_age: std::time::Duration) -> Result<u64, RecipeError>;
}

pub trait PartialSlot: Send {
    fn staged_len(&self) -> Result<u64, RecipeError>;
    fn expectation(&self) -> Result<Option<StagedExpectation>, RecipeError>;
    fn reset(&self, len: u64, expectation: &StagedExpectation)
        -> Result<(), RecipeError>;
    fn append(&self, chunk: &[u8]) -> Result<(), RecipeError>;
    fn open_read(&self) -> Result<Box<dyn std::io::Read + '_>, RecipeError>;
    fn commit(self: Box<Self>, destination: &ObjectKey) -> Result<(), RecipeError>;
    fn discard(self: Box<Self>) -> Result<(), RecipeError>;
}

pub trait JitterSource: Send + Sync { fn sample(&self) -> u64; }
pub trait Sleeper: Send + Sync { fn sleep(&self, duration: std::time::Duration); }

// src/transfer/download.rs
pub fn download_verified(
    ports: &TransferPorts<'_>,
    request: &DownloadRequest,
) -> Result<DownloadOutcome, RecipeError>;

// src/context.rs — additive only; `new` keeps its exact signature.
impl<'a> RecipeContext<'a> {
    #[must_use]
    pub const fn with_transfer(self, ports: &'a TransferPorts<'a>) -> Self;
    #[must_use]
    pub const fn transfer(&self) -> Option<&'a TransferPorts<'a>>;
}

// src/archive/ports.rs
pub trait ExtractionSink: Send + Sync {
    fn create_dir(&self, path: &ContainedEntryPath) -> Result<(), RecipeError>;
    fn write_file(
        &self,
        path: &ContainedEntryPath,
        meta: &EntryMetadata,
        contents: &mut dyn std::io::Read,
    ) -> Result<u64, RecipeError>;
    fn finish(self: Box<Self>) -> Result<(), RecipeError>;
}
```

`JitterSource::sample` takes `&self`, not `&mut self`: `RecipeContext` holds
`&dyn` ports, and a `&mut self` method would force `&mut dyn` into the context
and collapse the shared-borrow design. Test doubles use interior mutability.
`ExtractionSink::write_file` sits at exactly four arguments because `&self`
counts toward `too-many-arguments-threshold = 4`; any future per-entry datum
goes into `EntryMetadata`, never into the signature.

## Revision note

2026-08-16: first draft, incorporating a six-expert design review and eight
empirical checks against the working tree and live upstreams. The review
changed the design materially in seven places: HTTP protocol types moved out of
the domain layer so roadmap 10.1.4's object-store adapter need not fabricate
status codes; `RecipeContext` gained a builder accessor so the streaming ports
are reachable from a recipe at all; flat digests became chunked digests after
confirming `sha2 0.10` cannot persist hasher state; the staging sidecar gained
the expectation it was created for, to prevent a pin bump wedging every warm
cache; `Option<Checksum>` became a mandatory `Integrity` field with a greppable
unpinned escape hatch; the two proposed Verus proofs were cut as restatements
and replaced by one proof on the jitter arithmetic plus a proptest state
machine over attempt sequences; and the signature adapter was deferred to
roadmap 10.1.6 after confirming no dataset in §10.3 ships a signature. A
supply-chain stage was added because the repository has no advisory or licence
gate at all, and the work was split into three pull requests after calibrating
against 10.1.1's actual delivery cost.
