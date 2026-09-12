//! Compile-time checks for the public `ClusteringResult` API surface.

#[test]
fn clustering_result_panicking_constructor_is_internal() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/trybuild/clustering_result_from_assignments_private.rs");
}
