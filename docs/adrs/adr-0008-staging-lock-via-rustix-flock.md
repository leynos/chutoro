# Architectural decision record (ADR) 007: staging-file locks via rustix flock

## Status

Accepted - 2026-11-21: staging-file exclusion uses a non-blocking exclusive
`flock` on a capability-opened file descriptor, driven by a bounded, clock-free
acquisition policy.

## Date

2026-11-21

## Context and problem statement

Roadmap item 10.1.2 introduces resumable downloads for benchmark datasets.
Concurrent matrix jobs may prepare the same dataset at the same time, so a
staging file needs cross-process exclusion: only one writer may append to a
partial download, decide its durable length, or commit or discard it.

`cap-std 4.0.3`, the resolved version, has no locking API on
`cap_std::fs_utf8::File`. Its public surface exposes `from_std`, `into_std`,
`sync_all`, `sync_data`, `set_len`, `metadata`, `try_clone`, `set_permissions`,
`open_ambient`, `create_ambient`, `open_ambient_with`, and `options`; there is
no `lock` or `try_lock`. `into_std()` returns a `std::fs::File`, and `std::fs`
is forbidden in `chutoro-bench-datasets` by the Whitaker `no_std_fs_operations`
lint. The crate appears in neither `excluded_crates` nor `excluded_paths` in
the root `dylint.toml`, and this decision adds no exception.

`std::fs::File::lock` stabilized in Rust 1.89, which the workspace
`rust-version` of 1.89.0 admits, but the lint boundary makes it unreachable in
this crate.

Issue #211 recorded the original finding and proposed `rustix::fs::flock`. Two
claims in that ticket are stale and are corrected here: the resolved `cap-std`
is 4.0.3, not 3.4.5, and the resolved `rustix` is 1.1.4, not 1.1.2.
`rustix 1.1.4` does provide the needed signature at `src/fs/fd.rs:317`:

```rust
pub fn flock<Fd: AsFd>(fd: Fd, operation: FlockOperation) -> io::Result<()>
```

`rustix` is already present in `Cargo.lock` through
`cap-primitives 4.0.3 -> cap-std 4.0.3`, so adding the package introduces no
new version. It is not, however, a declared dependency of
`chutoro-bench-datasets`: `cap-std` is optional and currently reachable only
through the `testing` feature and its dev-dependency graph. Using `rustix` in
production code requires a direct, target-scoped optional dependency.

ADR 0005 records the Cranelift development route and ADR 0006 records
sequential trim scoring, so this decision takes the next free serial, 0007;
published ADR numbers are not reused.

## Decision drivers

- All filesystem access in `chutoro-bench-datasets` stays capability-scoped;
  no `std::fs` call and no `no_std_fs_operations` exception.
- A lost or dead lock must not wedge a job. Acquisition is non-blocking,
  bounded by a hard timeout, and observable.
- The staging file survives a process exit, a failed download, and a discarded
  slot; the lockfile has a stable pathname and is never removed or renamed.
- Lock lifetime is tied to the staging operation, not to a process-wide
  singleton or a garbage-collected handle.
- The pure acquisition policy is testable without a clock, filesystem, or
  process.
- The design works on the supported Unix targets with the existing dependency
  set.

## Options considered

### Option A: `std::fs::File::lock` with a Whitaker exception

Rejected. It requires obtaining a `std::fs::File` through `into_std()`, which
discards the capability boundary, and it requires adding
`chutoro_bench_datasets` to the `no_std_fs_operations` exclusions. Both are the
problem this decision exists to avoid.

### Option B: an ambient sidecar `std::fs` lockfile outside the capability root

Rejected. It moves the lock state outside the capability root, so the lock no
longer describes the staged artefact, and it reintroduces `std::fs`.

### Option C: `rustix::fs::flock` on a capability-opened descriptor

Accepted. `cap_std::fs_utf8::Dir` opens the lockfile relative to the capability
root; `cap_std::fs_utf8::File` implements `AsFd`; and `rustix::fs::flock`
requests `NonBlockingLockExclusive`. No `std::fs` path is used, no descriptor
is handed to `std::fs`, and the policy remains independent of the adapter.

### Option D: a lock-free optimistic protocol with atomic rename

Rejected. Commit compares the staged length and digest with the manifest
expectation and then renames. A compare-and-rename protocol would still need an
exclusive writer, and it would be harder to reason about than an advisory lock.

## Decision outcome

`chutoro-bench-datasets` gains a `transfer` module behind the new `http`
feature. `StagingLock::new(root)` holds the capability root path.
`StagingLock::acquire(name, policy)` validates a single relative UTF-8
component ending in `.lock`, opens or creates that file through
`cap_std::fs_utf8::Dir` with write access and no truncation, and attempts a
non-blocking exclusive `flock` on `file.as_fd()`. The open itself sets
`O_NONBLOCK`, so a pre-existing FIFO without a reader cannot block before the
timeout clock starts, and the opened descriptor must be a regular file.

Acquisition follows a pure, clock-free `LockPolicy::next_step` that receives
the attempt outcome, elapsed time, and whether a contention warning has already
been emitted. It returns one of `Acquired`, `Retry { delay, warn }`, or
`TimedOut`. The policy:

- rejects a zero timeout and a poll interval below one millisecond;
- clamps every sleep to the remaining time before the deadline;
- treats `WOULDBLOCK` as contention and `INTR` as an interruption, retrying
  both;
- rejects a late success: a lock granted at or after the deadline is released
  and reported as a timeout;
- emits `warn = true` exactly once, on the first contention.

The adapter sleeps exactly the delay the policy returns. It never blocks in the
kernel, so a holder that dies without unlocking cannot wedge another job beyond
the timeout.

A process-wide `Mutex<HashSet<Utf8PathBuf>>` keyed by the supplied root path
and validated name makes intra-process double acquisition deterministic. The
entry is inserted before the first system call; a duplicate is rejected
immediately with `waited_millis == 0`. The entry is removed on every failure
path and when the guard drops.

`StagingLockGuard` owns the only file descriptor, the path, and the registry
entry. On drop it requests `LOCK_UN`, records a debug event if unlocking fails,
removes the registry entry, and closes the descriptor. It never clones the
descriptor. `cap-std` opens with close-on-exec, so a child process does not
inherit the lock unless the descriptor is explicitly passed.

The four `RecipeError` variants are:

- `StagingLockUnsupportedPlatform` - this target has no `flock` binding.
- `StagingLockInvalidName(Arc<str>)` - the name is not one relative UTF-8
  `.lock` component.
- `StagingLockContended { path: Arc<str>, waited_millis: u64 }` - the lock is
  held, either by this process (zero wait) or by another holder (bounded wait).
- `StagingLockIo { path: Arc<str>, source: std::io::Error }` - opening,
  creating, or locking failed for a reason other than contention or
  interruption.

The enum keeps `size_of::<RecipeError>() <= 32`; the largest payload is the
path and wait pair at 24 bytes, so the existing compile-time assertion holds.

### Platform restriction

The implementation is compiled only for targets where both `cap-std` and
`rustix` expose the needed API. `cap_std::fs_utf8::File` implements `AsFd` only
on non-Windows targets, and `rustix::fs::flock` is absent on `espidf`,
`horizon`, `solaris`, `vita`, and `wasi`. The predicate is:

```text
all(not(windows), not(any(target_os = "espidf", target_os = "horizon",
    target_os = "solaris", target_os = "vita", target_os = "wasi")))
```

On every other target, `acquire` returns `StagingLockUnsupportedPlatform`
before any filesystem operation. The `rustix` dependency is declared under
`[target.'cfg(not(windows))'.dependencies]` as an optional dependency with the
`fs` feature, and the `http` feature enables both `dep:cap-std` and
`dep:rustix`.

### Dependency facts

- `cap-std` is promoted from a `testing`-only optional dependency to a shared
  optional dependency enabled by both `testing` and `http`.
- `rustix` is declared directly with a caret `1.1.4` requirement and the `fs`
  feature. The lockfile already resolves 1.1.4, so the manifest change adds no
  package and no version; `Cargo.lock` changes only in the
  `chutoro-bench-datasets` dependency list.
- No `no_std_fs_operations` exception is added.

## Goals and non-goals

- Goals:
  - Exclude concurrent writers to one staging artefact across processes.
  - Bound acquisition with a hard timeout and a single contention warning.
  - Keep the lock tied to the capability root and the staging slot lifetime.
  - Make the acquisition policy pure and exhaustively testable.
- Non-goals:
  - Cache-index coordination, duplicate-preparation suppression, or the
    10.1.5 lockfile.
  - `PartialSlot` and transfer-port wiring; those are later 10.1.2 stages.
  - NFS and overlay-filesystem guarantees. Advisory locks may be ignored or
    emulated there; the cache root must be a local filesystem.
  - Cross-platform locking on Windows.

## ExecPlan corrections

The draft 10.1.2 ExecPlan is not present on the branch that implements this
decision. It lives on `10-1-2-shared-download-primitives`; the corrections that
must reach it are recorded here so the plan and the code cannot drift:

- The feature is `http = ["dep:rustix", "dep:cap-std"]`, not implicit
  dependency activation.
- `rustix` is target-scoped under `cfg(not(windows))` and carries the `fs`
  feature; the resolved versions are `cap-std 4.0.3` and `rustix 1.1.4`.
- `PartialSlot` owns the staging lock for its whole lifetime. Resume, commit,
  and discard all run while the guard is held; the guard drops after the slot
  operation completes, not when an individual download attempt returns.
- The stale sweep uses the same non-blocking policy and skips any slot whose
  lock it cannot take. It must not wait for a live writer, because the sweep
  runs automatically at the start of every download.
- This ADR is the normative reference for the staging lock.

## Known risks and limitations

- **Advisory locks are filesystem-dependent.** `flock` is unreliable on NFS
  and may be emulated on overlay mounts. The cache root must be a local
  filesystem, and this is documented in the retrieval reference and the
  developer's guide.
- **Per-open-file-description semantics.** `flock` conflicts between separate
  opens, but two descriptors obtained by `dup` or `try_clone` share the same
  lock. The registry covers intra-process double acquisition through the public
  API; it does not cover hand-copied descriptors.
- **Registry aliasing.** The registry key is the supplied root path joined
  with the validated name, not a canonicalized path. Two spellings of the same
  directory (a symlink and its target, or a relative and an absolute path) are
  distinct keys. Callers must pass one stable root spelling.
- **Inherited descriptors.** A fork that keeps the descriptor alive can keep
  the lock held after the parent exits. Close-on-exec does not cover an
  explicit descriptor handoff.
- **A pre-existing non-regular path is rejected.** A FIFO, socket, or device
  at the lockfile path is not a valid lock target. `O_NONBLOCK` stops a
  reader-less FIFO from blocking the open before the acquisition clock starts,
  and descriptor metadata rejects anything that is not a regular file with
  `StagingLockIo`.
- **Timeout is a policy choice.** Too short a timeout turns ordinary
  contention into failures; too long a timeout delays diagnosis. The default
  poll interval is 25 ms and callers choose the timeout explicitly.
- **No lock on unsupported targets.** The API exists but returns
  `StagingLockUnsupportedPlatform`; callers needing cross-platform behaviour
  must provide a target-specific transfer adapter.

## Architectural rationale

The lock is an outbound adapter concern, not a recipe concern. Recipes describe
what to fetch and when a staging slot may be committed; they do not know how
exclusion is implemented. Keeping the policy pure and the adapter thin lets the
domain-side decision be tested over arbitrary traces without a clock or a
filesystem, while the adapter remains small enough to review against the
`flock` contract.

`rustix` is the narrowest available bridge: it borrows the descriptor the
capability already owns, so the capability boundary survives. The alternative,
`into_std()`, silently returns the ambient type the lint exists to keep out.

## Consequences

- Concurrent staging writers to the same root and name are excluded across
  processes and within a process.
- A contended acquisition fails with a typed error after a bounded wait and
  logs one structured warning.
- The lockfile persists across acquisitions and is never removed or renamed.
- The crate gains an `http` feature that enables `cap-std` and `rustix`; the
  `testing` feature and the default feature set are unchanged.
- `Cargo.lock` gains no package version.
- Windows and the excluded `rustix` targets return a typed unsupported error.

## Validation evidence

The `http` feature compiles with `cap-std 4.0.3` and `rustix 1.1.4`. The
library checks under default, `testing`-only, and `http`-only features, and the
crate doctests under all features. `make lint` passes with no
`no_std_fs_operations` exception.

- `tests/staging_lock_policy.rs` covers every terminal `next_step` outcome and
  the retry table, and a `proptest` over random attempt traces asserts that
  total sleep never exceeds the timeout, at most one warning is emitted, and
  the loop terminates with a success only before the deadline.
- `tests/staging_lock_filesystem.rs` covers release and reacquisition, immediate
  in-process rejection, invalid names, lockfile persistence, thread contention,
  and a reader-less FIFO at the lockfile path. The FIFO case runs under a
  watchdog and proves the non-blocking open fails promptly with
  `StagingLockIo`. The suite holds an independent `flock` to prove a bounded
  timeout below the registry, then releases it and reacquires, and asserts one
  `warn` line with `tracing-test`.
- `tests/staging_lock_cross_process.rs` (Linux only) re-runs the test binary in
  a holder role, proves the bounded timeout against a live foreign process, and
  reacquires once the child exits. The child is killed and reaped on every
  failure path.
- `chutoro-compile-contracts/tests/staging_lock_feature.rs` checks a
  feature-enabled fixture compiles and a featureless fixture fails with the
  `StagingLock` unresolved-import diagnostic, using
  `cargo check --offline --locked` into an isolated target directory.

## References

- [Issue #211: cap-std has no file locking](https://github.com/leynos/chutoro/issues/211)
- [Plan PR #199](https://github.com/leynos/chutoro/pull/199)
- [Draft 10.1.2 ExecPlan](https://github.com/leynos/chutoro/blob/10-1-2-shared-download-primitives/docs/execplans/10-1-2-shared-download-primitives.md)
- Roadmap items 10.1.2 (staging) and 10.1.5 (cache lockfile semantics)
