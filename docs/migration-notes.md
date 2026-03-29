# Migration notes

## Dense Parquet loading

`DenseMatrixProvider::try_from_parquet_path` now takes a `cap_std::fs::Dir`
together with a path relative to that directory:

```rust
use cap_std::{ambient_authority, fs::Dir};
use chutoro_providers_dense::DenseMatrixProvider;

let dir = Dir::open_ambient_dir("data", ambient_authority())?;
let provider = DenseMatrixProvider::try_from_parquet_path(
    "demo",
    &dir,
    "features.parquet",
    "features",
)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Callers that previously passed a single filesystem path should:

1. Open the parent directory as a capability-scoped `Dir`.
2. Pass the file name, or another path relative to that `Dir`, to
   `try_from_parquet_path`.

The smallest migration from the old call shape:

```rust
use cap_std::{ambient_authority, fs::Dir};
use std::path::Path;

let parquet_path = Path::new("data/features.parquet");
let parent = parquet_path.parent().unwrap_or_else(|| Path::new("."));
let file_name = parquet_path
    .file_name()
    .expect("Parquet path should include a file name");
let dir = Dir::open_ambient_dir(parent, ambient_authority())?;

let provider = DenseMatrixProvider::try_from_parquet_path(
    "demo",
    &dir,
    file_name,
    "features",
)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```
