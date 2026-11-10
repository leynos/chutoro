//! Downstream consumer fixture that imports the opt-in dense test builders.

use chutoro_providers_dense::testing::{
    FixtureError, build_array, feature_field, write_parquet_with_field,
};

fn main() -> Result<(), FixtureError> {
    let array = build_array(&[[1.0, 2.0]])?;
    let field = feature_field(2, false, false)?;
    let parquet = write_parquet_with_field(field, array)?;
    assert!(!parquet.is_empty());
    Ok(())
}
