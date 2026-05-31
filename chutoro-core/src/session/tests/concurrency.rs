//! Concurrency-focused tests for [`super::ClusteringSession`].
//!
//! These tests document the read-only sharing expectations around sessions
//! wrapped in `RwLock`. They complement append semantics tests by checking that
//! shared readers observe stable point counts and snapshot versions after the
//! writer has finished mutating the session.

use std::sync::Arc;

use rstest::rstest;

use super::common::{make_session, session_builder};
use crate::ChutoroBuilder;

#[rstest]
fn concurrent_readers_observe_consistent_point_count(session_builder: ChutoroBuilder) {
    // Append two points on the writer thread, then share the session
    // read-only across multiple concurrent threads.
    let (mut session, _) = make_session(session_builder, 4);
    session.append(&[0, 1]).expect("append must succeed");
    let shared = Arc::new(std::sync::RwLock::new(session));

    let handles: Vec<_> = (0..8)
        .map(|_| {
            let shared = Arc::clone(&shared);
            std::thread::spawn(move || {
                let guard = shared.read().expect("read lock must not be poisoned");
                (guard.point_count(), guard.snapshot_version())
            })
        })
        .collect();

    for handle in handles {
        let (point_count, snapshot_version) = handle.join().expect("reader thread must not panic");
        assert_eq!(
            point_count, 2,
            "all readers must observe the same point_count"
        );
        assert_eq!(snapshot_version, 0, "snapshot_version must remain 0");
    }
}

#[rstest]
fn snapshot_version_is_immutable_under_concurrent_readers(session_builder: ChutoroBuilder) {
    // snapshot_version must read as 0 under all concurrent readers;
    // append does not publish a label snapshot.
    let (mut session, _) = make_session(session_builder, 4);
    session.append(&[0, 1, 2]).expect("append must succeed");
    let shared = Arc::new(std::sync::RwLock::new(session));

    let handles: Vec<_> = (0..16)
        .map(|_| {
            let shared = Arc::clone(&shared);
            std::thread::spawn(move || {
                shared
                    .read()
                    .expect("read lock must not be poisoned")
                    .snapshot_version()
            })
        })
        .collect();

    for handle in handles {
        let version = handle.join().expect("reader thread must not panic");
        assert_eq!(
            version, 0,
            "snapshot_version must be immutable across all readers"
        );
    }
}
