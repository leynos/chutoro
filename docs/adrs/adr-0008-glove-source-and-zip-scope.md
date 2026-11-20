# Architecture Decision Record (ADR) 008: GloVe source and ZIP scope

## Status

Accepted - 2026-11-20: source GloVe from the ANN-Benchmarks HDF5 files built
from the Twitter 27B vectors, and keep `.zip` out of 10.1.2 and 10.3.10.

## Date

2026-11-20

## Context and problem statement

Roadmap 10.1.2 declares archive extraction for `.gz`, `.bz2`, `.xz`, and
`.tar`. Roadmap 10.3.10 requires dimension-selectable GloVe ingestion at 25,
50, 100, and 200 dimensions. Issue #213 observed that no roadmap dataset
actually ships `.bz2` or `.xz`, and that the GloVe download named in the issue,
`glove.6B.zip`, is a ZIP file outside the declared format list. It asked whether
`.zip` should join 10.1.2's scope, join 10.3.10 only, or whether 10.3.10
should consume a source that needs no ZIP support.

The evidence in the issue needs one correction. `glove.6B.zip` is not the
Twitter corpus that 10.3.10 describes. The Stanford GloVe page lists it as the
Wikipedia 2014 + Gigaword 5 bundle: 6B tokens, a 400K vocabulary, and 50, 100,
200, and 300-dimensional vectors. It contains no 25-dimensional member and its
vocabulary is roughly a third of the 1.18M points the roadmap records. The byte
count quoted in the issue is consistent with the published 822 MB download; the
file is simply the wrong corpus. The Twitter bundle is `glove.twitter.27B.zip`:
2B tweets, 27B tokens, a 1.2M vocabulary, and 25, 50, 100, and 200-dimensional
vectors. ANN-Benchmarks builds its GloVe datasets from that bundle, and it
publishes the four required dimensionalities as ready-made HDF5 files.
Consuming Stanford's ZIP files directly would therefore introduce ZIP
extraction to obtain vectors that are already available in a format 10.3.11 and
10.3.12 must read anyway.

## Decision drivers

- 10.3.10 needs all four dimensions, and only the Twitter 27B bundle provides
  them in one corpus.
- ANN-Benchmarks already publishes `glove-{25,50,100,200}-angular.hdf5`, which
  carry the same Twitter 27B vectors with train/test splits and exact
  neighbours.
- Within 10.1.2's declared formats, `.bz2` and `.xz` have no consumer in the
  current dataset backlog. `.gz` is needed by MNIST, Fashion-MNIST, CIFAR-10,
  20 Newsgroups, and the SNAP graph files, and `.tar` by the `.tar.gz` bundles.
  ZIP is not in the declared list at all, and GloVe's ZIP bundle was its only
  candidate consumer.
- Every additional archive decoder is supply-chain surface. The `zip` crate is
  a new dependency with no current consumer once GloVe comes from HDF5.
- 10.3.11 and 10.3.12 need an HDF5 reader regardless, so the GloVe files add
  no new reader class.
- A bare HDF5 download is the common case, not an exception, so "not an
  archive" must be a first-class path rather than a fallback.

## Options considered

### Option A: Add `.zip` to 10.1.2's archive formats

Rejected. It adds a ZIP decoder to the shared download primitives for a source
that 10.3.10 need not use. It also expands 10.1.2's declared surface beyond
what any roadmap dataset demonstrably requires.

### Option B: Add `.zip` to 10.3.10 only

Rejected. It places archive policy in a dataset recipe instead of the shared
extraction boundary, and it still adds a dependency solely to re-derive files
that ANN-Benchmarks publishes.

### Option C: Source the ANN-Benchmarks HDF5 files and exclude `.zip`

Accepted. 10.3.10 consumes the four bare HDF5 downloads directly. The files are
not archives, so 10.1.2 provides `ArchiveFormat::None` with an adopt-in-place
path. `.zip` stays out of both 10.1.2 and 10.3.10.

_Table 1: Options for GloVe acquisition and ZIP scope._

| Option                 | ZIP dependency             | Dimensions covered | Where policy lives               |
| ---------------------- | -------------------------- | ------------------ | -------------------------------- |
| A: ZIP in 10.1.2       | Added to shared primitives | 25-300 by bundle   | Shared extraction boundary       |
| B: ZIP in 10.3.10      | Added to one recipe        | 25-300 by bundle   | Dataset recipe                   |
| C: ANN-Benchmarks HDF5 | None                       | 25, 50, 100, 200   | Shared fetch plus adopt-in-place |

## Decision outcome

10.3.10 fetches four ANN-Benchmarks HDF5 files:

| Dimension | Source URL                                          | Train rows | Test rows |
| --------: | --------------------------------------------------- | ---------: | --------: |
| 25        | `https://ann-benchmarks.com/glove-25-angular.hdf5`  | 1,183,514  | 10,000    |
| 50        | `https://ann-benchmarks.com/glove-50-angular.hdf5`  | 1,183,514  | 10,000    |
| 100       | `https://ann-benchmarks.com/glove-100-angular.hdf5` | 1,183,514  | 10,000    |
| 200       | `https://ann-benchmarks.com/glove-200-angular.hdf5` | 1,183,514  | 10,000    |

_Table 2: Pinned GloVe HDF5 sources and their row counts._

These files are adopted in place. Adoption means the _fetch_ stage is satisfied
by an unarchived download: 10.1.2 writes each file to the local cache, verifies
its digest, and hands the verified bytes onward without extracting them.
`ArchiveFormat::None` is therefore a first-class variant rather than an error
path, because the ANN-Benchmarks HDF5 files and the DEEP1B binary files are
both bare downloads.

Adoption is not publication. The downloaded file is a _source_ artefact. The
`DatasetRecipe` phases still run in order (see
[ADR 004](adr-0004-bench-dataset-recipe-trait.md)): `validate` reads the HDF5
datasets and applies the checks below, `prepare` maps them onto the canonical
prepared-artefact contract from `docs/benchmark-dataset-retrieval.md` §3.1
(`features`, optional `labels`, optional `ground_truth`), and only `publish`
writes the immutable `manifest.json` and its preprocessing hash. Concretely:

- `features` is the file's `train` dataset. The `test` split is the query set
  and travels with `ground_truth` in whatever query field the 10.1.3 schema
  defines; it is not merged into `features`.
- `ground_truth` is the file's `neighbors` and `distances` pair, which is why
  their shapes and ordering are validated.
- `labels` is absent, because GloVe is a pretrained-vector corpus with no class
  labels.
- No preprocessing hash depends on a stochastic transform: the coordinates are
  copied unchanged from the source, so the hash covers the source digest and
  the canonical serialization rather than a fitted projection.

The manifest is therefore produced once, at `publish`, after validation; a
cached source file alone does not satisfy 10.3.10.

`.zip` is excluded from 10.1.2 and 10.3.10. `.bz2` and `.xz` remain in 10.1.2
because the roadmap declares them, but each stays behind its own off-by-default
Cargo feature (`bz2`, `xz`), so neither decoder is on the default build path or
in the default dependency graph.

The four HDF5 files derive from the Stanford Twitter 27B vectors, which the
GloVe project releases under the Open Data Commons Public Domain Dedication and
License v1.0 (`PDDL-1.0`). ANN-Benchmarks packages them with `train_test_split`
using `test_size=10000` and `random_state=1`, writes the `angular` distance,
and does not normalize the generator output or otherwise project the
coordinates.

Each published file stores exactly four datasets and exactly one attribute.
Verified by bounded range read against all four files on 2026-11-23:

| Dataset     | Shape          | Dtype     |
| ----------- | -------------- | --------- |
| `train`     | `(1183514, D)` | `float32` |
| `test`      | `(10000, D)`   | `float32` |
| `neighbors` | `(10000, 100)` | `int32`   |
| `distances` | `(10000, 100)` | `float32` |

_Table 3: Datasets in each published GloVe HDF5 file; `D` is the dimension in
the filename._

The only root attribute is `distance=angular`. The upstream generator in
`ann_benchmarks/datasets.py` also writes `type`, `dimension`, and `point_type`,
but the files published for these four datasets do not contain them. The reader
and validation requirements must therefore match the artefacts that exist:
derive the dimension from `train`'s shape and the filename rather than from an
attribute, check numeric dtypes directly from the datasets, and treat only
`distance` as a required attribute. Requiring `type`, `dimension`, or
`point_type` would reject every file this decision selects.

The files carry no tokens or vocabulary; 10.3.10 therefore needs no tokenizer
and no projection step.

Checksum pins are first-party. On first adoption chutoro computes a SHA-256
digest for each exact downloaded artefact and records it in the repository's
checksum manifest; every later fetch verifies that digest. The MD5 values
associated with the ANN-Benchmarks reproducibility artefact on Zenodo (DOI
10.5281/zenodo.4607761) are not used: they are third-party and MD5 is not
collision-resistant.

10.3.10 requires the following checks before publication:

- `train` has 1,183,514 rows and `test` has 10,000 rows.
- `train` and `test` both have exactly `D` columns, where `D` is the dimension
  named by the file, and their widths agree.
- `train` and `test` hold floating-point coordinates.
- Every coordinate is finite.
- Every row has non-zero Euclidean magnitude, so angular distance is defined.
- `neighbors` is integer-typed with shape `(10000, 100)` and indices in
  `[0, 1183514)`.
- `distances` is floating-point-typed with shape `(10000, 100)`, is finite and
  non-negative, and is non-decreasing along each row.
- The sole required attribute is `distance=angular`. `type`, `dimension`, and
  `point_type` are absent from the published files and are therefore neither
  required nor derived from an attribute.

10.3.10 also depends on a read-only ANN-Benchmarks HDF5 reader shared with
10.3.11 and 10.3.12. A pure-Rust reader is preferred so the crate does not
require a system `libhdf5`; the bounded sub-item under 10.3.10 pins the exact
crate and version.

A feature-gated ZIP extractor is reopened only if the ANN-Benchmarks files
become unusable or a required dimension is withdrawn. That work would add an
off-by-default `zip` Cargo feature using the `zip` crate, preserving the
default dependency graph, and would be tracked as a roadmap item rather than
absorbed silently here.

## Architectural rationale

The decision keeps acquisition policy at the shared boundary. A dataset recipe
states which pinned artefact it needs and which checks make that artefact
acceptable; it does not own a decoder for a container it did not choose. The
ANN-Benchmarks HDF5 files are already the interchange format for the other two
ANN datasets, so sourcing GloVe the same way removes a dependency rather than
adding one.

Treating a bare file as `ArchiveFormat::None` also matches the evidence. The
ANN-Benchmarks and DEEP1B sources are not archives, so modelling "no archive"
as an exceptional case would make the common path the awkward one.

## Known risks and limitations

- **ANN-Benchmarks is no longer actively maintained.** Its README states that
  the project is no longer actively maintained. The four GloVe HDF5 files are
  static, but a future disappearance would force the ZIP path back.
- **The HDF5 reader is shared, not free.** 10.3.10, 10.3.11, and 10.3.12 all
  need it, so its quality gates the whole ANN dataset group. A pure-Rust reader
  reduces build friction but must be validated against these files.
- **The licence is a public-domain dedication, not an SPDX-reviewed permissive
  licence.** `PDDL-1.0` is acceptable for this use, but downstream
  redistributors must re-check it.
- **First-party pins need a trusted first fetch.** The SHA-256 manifest is only
  as trustworthy as the environment that first computed it; the manifest must
  be committed and reviewed like any other source.

## Consequences

- 10.1.2 must provide `ArchiveFormat::None` with adopt-in-place behaviour.
- 10.1.2 and 10.3.10 carry no `.zip` dependency.
- 10.3.10 needs no tokenizer or projection and validates measurable HDF5
  invariants instead.
- 10.3.11 and 10.3.12 reuse the HDF5 reader introduced under 10.3.10.
- A feature-gated ZIP extractor is reopened only under the trigger recorded
  above, so ZIP work cannot be added silently to either roadmap item.

## References

- [Issue #213](https://github.com/leynos/chutoro/issues/213)
- [Stanford GloVe project page](https://nlp.stanford.edu/projects/glove/)
- [ANN-Benchmarks README](https://github.com/erikbern/ann-benchmarks/blob/main/README.md)
- [ANN-Benchmarks dataset generators](https://github.com/erikbern/ann-benchmarks/blob/main/ann_benchmarks/datasets.py)
- [ANN-Benchmarks reproducibility artefact](https://doi.org/10.5281/zenodo.4607761)
- [PDDL 1.0](https://opendatacommons.org/licenses/pddl/1.0/)
