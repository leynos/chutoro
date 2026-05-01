//! Compile-fail fixture verifying session sources must be thread-safe.

use std::{
    cell::RefCell,
    rc::Rc,
    sync::Arc,
};

use chutoro_core::{ChutoroBuilder, DataSource, DataSourceError, MetricDescriptor};

struct NonThreadSafeSource {
    values: Rc<RefCell<Vec<f32>>>,
}

impl DataSource for NonThreadSafeSource {
    fn len(&self) -> usize {
        self.values.borrow().len()
    }

    fn name(&self) -> &str {
        "non-thread-safe"
    }

    fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
        let values = self.values.borrow();
        let left = values.get(i).ok_or(DataSourceError::OutOfBounds { index: i })?;
        let right = values.get(j).ok_or(DataSourceError::OutOfBounds { index: j })?;
        Ok((left - right).abs())
    }

    fn metric_descriptor(&self) -> MetricDescriptor {
        MetricDescriptor::new("abs")
    }
}

fn main() {
    let source = Arc::new(NonThreadSafeSource {
        values: Rc::new(RefCell::new(vec![0.0, 1.0])),
    });

    let _ = ChutoroBuilder::new().build_session(source);
}
