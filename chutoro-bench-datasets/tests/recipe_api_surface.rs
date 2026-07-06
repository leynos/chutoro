//! Compile-time contract checks for the public `DatasetRecipe` API surface.

#[test]
fn dataset_recipe_phase_order() {
    let cases = trybuild::TestCases::new();
    cases.pass("tests/trybuild/dataset_recipe_phase_order.rs");
    cases.compile_fail("tests/trybuild/dataset_recipe_skip_phase.rs");
}
