//! Compile-time regression test for the shared Arrow and Parquet type family.

/// Verifies Arrow and Parquet expose one mutually compatible type family.
#[test]
fn arrow_parquet_types_share_one_family() {
    let cases = trybuild::TestCases::new();
    cases.pass("tests/trybuild/arrow_parquet_alignment.rs");
}
