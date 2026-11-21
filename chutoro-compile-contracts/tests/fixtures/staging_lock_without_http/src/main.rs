//! Downstream consumer fixture that must not see the `http`-gated staging lock.

use chutoro_bench_datasets::StagingLock;

fn main() {
    let _lock = StagingLock::new("target/staging-lock-fixture");
}
