//! Compile-time API contracts for Chutoro workspace packages.

/// Every Rust example in the user-facing documentation, run as a doctest.
///
/// Rustdoc compiles and runs each `rust` fence in these documents against the
/// workspace crates, so a published example cannot drift from the API it
/// teaches. The `documentation_examples_tests` integration test checks the other
/// fences and requires this list to match the documents it loads.
#[cfg(doctest)]
pub mod documentation {
    #[doc = include_str!("../../README.md")]
    pub struct Readme;

    #[doc = include_str!("../../docs/users-guide.md")]
    pub struct UsersGuide;

    #[doc = include_str!("../../docs/v0-1-0-migration-guide.md")]
    pub struct MigrationGuide;
}
