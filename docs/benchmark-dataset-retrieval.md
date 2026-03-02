# Benchmark Dataset Retrieval Assessment

## Scope

This document assesses each dataset listed in the roadmap benchmark suite and
estimates:

- effort to automate download and preparation for benchmark use
- cached/prepared dataset size
- approximate retrieval cost from object storage (AWS S3, Scaleway Object
  Storage, DigitalOcean Spaces)

Pricing and source checks were refreshed on 2026-03-02.

## Process

For each dataset, use this repeatable pipeline:

1. Fetch from canonical source using a pinned URL and checksum validation.
2. Normalize to one benchmark-ready artifact format (`.hdf5`, `.parquet`, or
   `.npz`) with metadata (`dataset`, version, schema, preprocessing hash).
3. Run validation checks (shape, dtype, label/ground-truth integrity, and basic
   distribution sanity checks).
4. Upload to object storage under an immutable key
   (`bench-datasets/<name>/<version>/...`) and publish a manifest file.
5. Retrieve by manifest in benchmark jobs, with local cache fallback.

## Cost Model

Retrieval cost is modeled as:

`cost ~= egress_cost + request_cost`

Assumptions:

- These are retrieval-only estimates (not storage-at-rest).
- Numbers below are marginal egress costs (assume included transfer quotas are
  already exhausted).
- AWS request charge for `GET` is included as a note; it is negligible for the
  object counts expected here.

Provider pricing inputs used:

- AWS S3 Standard (US East): data transfer out to internet `$0.09/GB`
  (first 10 TB), `GET` `$0.0004` per 1,000 requests.
- Scaleway Object Storage: `Request: Included`; `Egress: 75GB free every month
  then EUR0.01/GB`.
- DigitalOcean Spaces: 1,024 GiB outbound included per subscription; outbound
  overage `$0.01/GiB`; no separate standard per-request retrieval fee listed on
  Spaces pricing docs.

Included-transfer notes:

- AWS lists a 100 GB/month free transfer allowance to internet on the S3
  pricing page.
- Scaleway includes 75 GB/month egress.
- DigitalOcean Spaces includes 1,024 GiB/month outbound transfer.

## Dataset Assessments

Effort scale:

- `XS`: fully scriptable, <0.5 day
- `S`: scriptable, ~1 day
- `M`: scriptable with non-trivial preprocessing, 1-3 days
- `L`: partially gated and/or heavier pipeline, 3-5 days
- `XL`: very large-scale operational pipeline, ~1-2 weeks+

| Dataset              | Automatability and prep effort                                                                                               | Cached/prepared size (GiB)                 | AWS S3 retrieval (USD) | Scaleway retrieval (EUR) | DigitalOcean retrieval (USD) |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------- | -----------------------------------------: | ---------------------: | -----------------------: | ---------------------------: |
| `make_blobs`         | Fully automatable synthetic generation. No external download. `XS`.                                                          | 0.000                                      | 0.000                  | 0.000                    | 0.000                        |
| MNIST digits         | Fully automatable (prefer ANN HDF5 packaging). `S`.                                                                          | 0.212                                      | 0.019                  | 0.002                    | 0.002                        |
| Fashion-MNIST        | Fully automatable with direct IDX links and checksums. `S`.                                                                  | 0.212                                      | 0.019                  | 0.002                    | 0.002                        |
| CIFAR-10 / CIFAR-100 | Fully automatable downloads; medium prep to produce fixed embeddings. `M`.                                                   | 0.228 (both)                               | 0.021                  | 0.002                    | 0.002                        |
| 20 Newsgroups        | Fully automatable via `fetch_20newsgroups`; medium prep for text vectorization. `M`.                                         | 0.027 (384d embeddings)                    | 0.002                  | 0.000                    | 0.000                        |
| RCV1-v2              | Fully automatable via `fetch_rcv1`; larger sparse-to-dense prep. `M-L`.                                                      | 1.151 (384d embeddings)                    | 0.104                  | 0.012                    | 0.012                        |
| SNAP com-Amazon      | Fully automatable download; medium graph-to-vector prep. `M`.                                                                | 0.160 (128d embeddings)                    | 0.014                  | 0.002                    | 0.002                        |
| SNAP com-DBLP        | Fully automatable download; medium graph-to-vector prep. `M`.                                                                | 0.151 (128d embeddings)                    | 0.014                  | 0.002                    | 0.002                        |
| PBMC 68k (10x)       | Source page is lead-form gated; pipeline is only partly automatable end-to-end. `L`.                                         | 0.013 (50d PCA + labels)                   | 0.001                  | 0.000                    | 0.000                        |
| GloVe word vectors   | Fully automatable download; moderate normalize/select-dimension prep. `M`.                                                   | 0.896 (GloVe-200 HDF5)                     | 0.081                  | 0.009                    | 0.009                        |
| SIFT1M               | Fully automatable via ANN-Benchmarks HDF5. `S`.                                                                              | 0.489                                      | 0.044                  | 0.005                    | 0.005                        |
| GIST1M               | Fully automatable via ANN-Benchmarks HDF5. `S-M`.                                                                            | 3.600                                      | 0.324                  | 0.036                    | 0.036                        |
| DEEP1B / BigANN      | Automatable but operationally heavy (multi-hundred-GB artifacts, resumable download, sharded upload, long validation). `XL`. | 119.2-357.6 (1B vectors, uint8 to float32) | 10.729-32.187          | 1.192-3.576              | 1.192-3.576                  |

## Notes on Size Estimates

- MNIST, Fashion-MNIST, SIFT1M, GIST1M, GloVe-200 use published ANN-Benchmarks
  HDF5 sizes.
- CIFAR assumes a one-time fixed image embedding step and caching both
  CIFAR-10 and CIFAR-100 in 512d float32 vectors.
- 20 Newsgroups and RCV1 assume caching dense 384d float32 embeddings for
  repeatable ANN/clustering runs.
- SNAP datasets assume caching 128d float32 node embeddings plus labels.
- PBMC assumes caching 50d PCA embeddings plus cell-type labels.
- DEEP1B/BigANN range is computed from datatype and dimensionality in
  Big-ANN specs: `1B * dims * bytes-per-dim`.

## Key Risks and Recommendations

- PBMC 68k is the least automation-friendly source because the 10x dataset page
  is form-gated; keep a mirrored internal artifact and treat it as controlled
  input.
- DEEP1B/BigANN should use chunked transfers and manifest-level checksums, then
  sharded object keys to avoid very large single-object retries.
- For recurring benchmark jobs, retrieval cost is dominated by egress. Request
  charges are usually negligible compared with transfer.

## Sources

- Roadmap dataset list and benchmark tiers: repository-local
  `docs/roadmap.md` section "Benchmark dataset suite"
- AWS S3 pricing: <https://aws.amazon.com/s3/pricing/>
- Scaleway storage pricing: <https://www.scaleway.com/en/pricing/storage/>
- DigitalOcean Spaces pricing:
  <https://docs.digitalocean.com/products/spaces/details/pricing/>
- scikit-learn `make_blobs`:
  <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html>
- scikit-learn `fetch_20newsgroups`:
  <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html>
- scikit-learn `fetch_rcv1`:
  <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_rcv1.html>
- CIFAR-10/100 dataset page: <https://www.cs.toronto.edu/~kriz/cifar.html>
- Fashion-MNIST repository:
  <https://github.com/zalandoresearch/fashion-mnist>
- SNAP com-Amazon: <https://snap.stanford.edu/data/com-Amazon.html>
- SNAP com-DBLP: <https://snap.stanford.edu/data/com-DBLP.html>
- GloVe vectors: <https://nlp.stanford.edu/projects/glove/>
- ANN-Benchmarks dataset table: <https://github.com/erikbern/ann-benchmarks>
- Big-ANN benchmark dataset specs:
  <https://big-ann-benchmarks.com/neurips21.html>
- PBMC 68k dataset landing page (lead-form gate):
  <https://www.10xgenomics.com/datasets/fresh-68-k-pbm-cs-donor-a-1-standard-1-1-0>
