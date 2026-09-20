//! Compile-fail fixture verifying the CPU pipeline remains private.

use chutoro_core::run_cpu_pipeline;

fn main() {
    let _ = run_cpu_pipeline;
}
