//! Compile-pass contract for the public `PointPair` batch-distance API.

use chutoro_core::{DataSource, DataSourceError, PointPair};

/// Constant construction proves `new`, `left`, and `right` are `const fn`.
const PAIR: PointPair = PointPair::new(3, 4);
const LEFT: usize = PAIR.left();
const RIGHT: usize = PAIR.right();

struct Dummy;

impl DataSource for Dummy {
    fn len(&self) -> usize { 2 }

    fn name(&self) -> &str { "dummy" }

    fn distance(&self, left: usize, right: usize) -> Result<f32, DataSourceError> {
        if left >= self.len() {
            return Err(DataSourceError::OutOfBounds { index: left });
        }
        if right >= self.len() {
            return Err(DataSourceError::OutOfBounds { index: right });
        }
        Ok((left as f32 - right as f32).abs())
    }
}

fn main() {
    assert_eq!((LEFT, RIGHT), (3, 4));

    let pair = PointPair::new(0, 1);
    assert_eq!(pair.left(), 0);
    assert_eq!(pair.right(), 1);

    let converted = PointPair::from((1, 0));
    let mut out = [0.0_f32; 2];
    Dummy
        .distance_batch(&[pair, converted], &mut out)
        .expect("public batch API must compile");
}
