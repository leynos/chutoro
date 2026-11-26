//! Runs the users' guide's published `append -> refresh -> labels` example.
//!
//! The example text is not copied here. The shared tested-example loader in
//! `build_support/users_guide_examples.rs` extracts the fence marked
//! `session-refresh` from `docs/users-guide.md` during the build, and
//! `build.rs` writes it to `OUT_DIR` for inclusion below. Editing the guide is
//! therefore what changes the program this test compiles and runs.

#![cfg(feature = "cpu")]

include!(concat!(env!("OUT_DIR"), "/users_guide_session_refresh.rs"));

#[test]
fn the_users_guide_refresh_example_runs() {
    published_example::run().expect("the published example must run");
}
