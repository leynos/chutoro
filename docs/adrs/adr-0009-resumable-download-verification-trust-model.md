# Architectural decision record (ADR) 0009: Resumable download verification trust model

## Status

Proposed. This record settles how roadmap 10.1.2 verifies a download that
resumed after an interruption. It replaces the chunked-digest proposal from
issue #210 and the checkpointed-hasher-midstate revision that followed it.
Implementation remains subject to the ExecPlan's gates.

## Date

2026-11-23.

## Context and problem statement

Roadmap 10.1.2 adds resumable transfers to `chutoro-bench-datasets`. A recipe
pins a source URL, a byte length, and a whole-file SHA-256 digest, and the
crate must not publish bytes that fail that pin. A transfer can be interrupted
at any point, including between processes.

The originating ticket, issue #210, observed correctly that `sha2 0.10` cannot
serialize a hasher midstate, and proposed per-chunk staging digests so that a
resume re-hashes at most one partial chunk. Two premises need correction.
First, the lockfile no longer resolves `sha2 0.10`: it resolves `sha2 0.11.0`,
`digest 0.11.3`, and `crypto-common 0.2.2` through
`rstest-bdd -> rust-embed -> rust-embed-utils`, and `sha2::Sha256` implements
`crypto_common::hazmat::SerializableState` there. Second, per-chunk digests
cannot be combined into the pinned whole-file SHA-256, so they do not remove
the need for a final whole-file verification.

The design question is therefore not whether hasher state can be persisted. It
is what a resumed process may trust about the staged prefix, and what
verification must precede publication.

## Decision drivers

- Published bytes must match the pinned whole-file digest, not merely the
  digest of bytes this process once wrote.
- Resume cost must be linear in artefact size and independent of the number of
  resumes.
- The uninterrupted path must pay no read-back at all.
- The design must not add a second trust root beside the pin.
- Sidecars must not hold sensitive cryptographic state.
- The resume path must survive a fresh process, a changed upstream
  representation, and a corrupt or tampered staged prefix.
- The 100 GB-plus datasets in roadmap 10.3 must have a scale path.

## Options considered

| Option                                                                             | Consequence                                                                                                                                                                      | Disposition |
| ---------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| Persist a hasher midstate in the staging sidecar                                   | Restores hash state across processes, but proves only what was hashed in the past; adds a `hazmat` dependency, a sensitive-state sidecar field, and a `0.x` compatibility hazard | Deferred    |
| Store per-chunk digests in the staging sidecar                                     | Bounds a resume re-read to one chunk, but cannot be combined into the pinned whole-file digest and authenticates nothing against the pin                                         | Rejected    |
| Re-hash the complete staged file once, under the advisory lock, before publication | Linear cost bounded by one pass, verifies the bytes actually present, and requires no persisted hash state                                                                       | Adopted     |
| Re-read the whole prefix on every resume, without chunking                         | The ticket's cost model, quadratic in resume count                                                                                                                               | Rejected    |

_Table 1: Verification options for a resumed download._

## Decision outcome

No hash state crosses an interruption. The trust boundary is the process.

An uninterrupted transfer streams every byte through a `DigestSink` as it is
written and verifies the result on completion. It performs no read-back.

A transfer that resumes from a non-zero durable offset discards streaming
continuity. Resume itself re-hashes no prefix: it needs only the sidecar's
`durable_offset`, truncation of the staged file to that offset, and a check of
the recorded validator against the current pin. After the final data fsync the
transfer re-hashes the complete staged file from offset zero through a fresh
read handle. The advisory lock is held from that read through the publishing
rename, so nothing this crate controls can change the file between verification
and publication. Publication happens only on a match.

A full-length mismatch on a resumed transfer earns exactly one clean restart,
which adds at most one further transfer and one further streaming hash. A
mismatch on that restart is fatal.

The read-back budget is therefore: `0` bytes for an uninterrupted transfer, and
`expected_size` bytes after any interruption sequence, independent of the
number of resumes. It is an acceptance property, not only a cost estimate.

Per-chunk staging digests are rejected in every hasher generation. A chunk
digest computed from bytes this process wrote describes the past, cannot be
combined into the pinned whole-file digest, and leaves final verification
reading the whole file. It would add sidecar state and a second, weaker trust
root without removing any read.

### Scale path

Large single-object retries are addressed by publication structure, not by a
staging digest vector. Prepared shards carry per-shard checksum manifests under
roadmap 10.1.3 and 10.3.13, and object-store transfers use multipart uploads
under roadmap 10.1.4 and 10.3.13. A chunked transfer is a set of separately
pinned source shards or a multipart object-store operation. Neither replaces
the pinned whole-file SHA-256: each shard carries its own pin, or the assembled
artefact is still verified against the whole-file digest.

## Verification and acceptance

The planned `proptest` resume state machine injects the verifier's read source
and counts read-back bytes. It asserts `0` for an uninterrupted transfer and
`expected_size` for an interrupted transfer, for any interruption sequence.
This runs against the shipped functions, not a transcription.

Behavioural scenarios must cover:

- An interruption, then publication, with exactly one verification pass.
- At least two interruptions with the downloader recreated between attempts,
  still producing one verification pass and one read-back of `expected_size`
  bytes.
- A one-byte corruption in the staged prefix before a resume: publication is
  refused with `IntegrityFailure`, and the permitted clean restart succeeds.
- An uninterrupted transfer that performs no read-back, asserted through the
  `PartialStore` test double's read counter.

Cost bound at DEEP1B scale (about 512 GB): one read-back takes about 0.3 h on
NVMe or 1.2 h at EBS gp3 baseline throughput, against an 11-hour transfer and
only when a resume occurred. Repeating the prefix read on every resume, as the
ticket's model implies, totals about 2.8 TB, or roughly 1.6 h on NVMe and 6.3 h
on EBS gp3.

## Known risks and limitations

- A resumed transfer pays one full read of the staged artefact. At DEEP1B
  scale that is a material but bounded fraction of the transfer.
- The advisory lock only coordinates writers that take it. An external writer
  that ignores advisory locking, or storage corruption that occurs after the
  re-hash, is outside the guarantee. The decision verifies bytes present at
  publication time; it does not claim durability against arbitrary storage
  faults.
- The single clean restart adds a second transfer and a second streaming
  hash. The alternative is a permanent wedge after a pin change.
- `crypto_common::hazmat::SerializableState` remains deferred, not impossible.
  Revisiting it requires both a material measured read-back cost on supported
  hardware and an ADR accepting a weaker trust rule for bytes recovered from
  the local cache.

## References

  <https://github.com/leynos/chutoro/issues/210>.
  <https://github.com/leynos/chutoro/blob/10-1-2-shared-download-primitives/docs/execplans/10-1-2-shared-download-primitives.md>.
  The original chunked-digest design is commit `cfb4354`. and security-warning
  sections:
  <https://docs.rs/crypto-common/0.2.2/crypto_common/hazmat/trait.SerializableState.html>.
  [§5, §7.13, and §9](../benchmark-dataset-retrieval.md).
