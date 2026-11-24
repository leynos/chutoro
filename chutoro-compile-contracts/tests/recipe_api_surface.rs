//! Compile-time contract checks for the public `DatasetRecipe` API surface.

/// Verifies recipes cannot skip required lifecycle phases at compile time.
#[test]
fn dataset_recipe_phase_order() {
    let cases = trybuild::TestCases::new();
    cases.pass("tests/trybuild/dataset_recipe_phase_order.rs");
    cases.compile_fail("tests/trybuild/dataset_recipe_skip_phase.rs");
    cases.pass("tests/trybuild/opaque_key_api.rs");
    cases.compile_fail("tests/trybuild/opaque_key_path_api_removed.rs");
    cases.pass("tests/trybuild/partial_state_api.rs");
    cases.compile_fail("tests/trybuild/partial_state_orphaned_cache_key_removed.rs");
}
