# Changelog

## Unreleased

### Changed

- `DenseMatrixProvider::try_from_parquet_path` now requires a capability-scoped
  `cap_std::fs::Dir` plus a path relative to that directory. This keeps Parquet
  loading aligned with the rest of Chutoro's capability-oriented filesystem
  model. See [migration notes](migration-notes.md#dense-parquet-loading) for
  the minimal caller update.
