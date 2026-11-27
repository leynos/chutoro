# Session refresh example

This is the complete worked example linked from the users' guide's "Refreshing
labels" section. The marker below names the fence for the shared tested-example
loader: `chutoro-core/build.rs` extracts it and the integration test and
behavioural scenario compile and run the extracted text, so this file is the
source of truth for the program.

Every fenced block in this file must carry a marker; the loader rejects an
unmarked fence.

<!-- tested-example: session-refresh -->
```rust
use std::sync::Arc;

use chutoro_core::{
    ChutoroBuilder, ChutoroError, DataSource, DataSourceError, MetricDescriptor,
};

struct Numbers(Vec<f32>);

impl DataSource for Numbers {
    fn len(&self) -> usize { self.0.len() }
    fn name(&self) -> &'static str { "numbers" }
    fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
        let a = self.0.get(i).ok_or(DataSourceError::OutOfBounds { index: i })?;
        let b = self.0.get(j).ok_or(DataSourceError::OutOfBounds { index: j })?;
        Ok((a - b).abs())
    }
    fn metric_descriptor(&self) -> MetricDescriptor {
        MetricDescriptor::new("numbers:abs")
    }
}

fn main() -> Result<(), ChutoroError> {
    let source = Arc::new(Numbers(vec![0.0, 1.0, 2.0, 3.0]));
    let mut session = ChutoroBuilder::new()
        .with_min_cluster_size(2)
        .build_session(source)?;

    session.append(&[0, 1, 2, 3])?;
    session.refresh()?;

    let labels = session.labels();
    assert_eq!(labels.len(), 4);
    assert_eq!(session.snapshot_version(), 1);
    Ok(())
}
```
