//! Downstream consumer fixture that imports the `http`-gated staging lock.

use chutoro_bench_datasets::{LockPolicy, StagingLock};

fn main() {
    let lock = StagingLock::new("target/staging-lock-fixture");
    let policy = LockPolicy::new(std::time::Duration::from_secs(1));

    assert_eq!(lock.root().as_str(), "target/staging-lock-fixture");
    assert!(policy.is_ok());
}
