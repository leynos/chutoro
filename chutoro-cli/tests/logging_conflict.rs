//! How the CLI's logging module behaves when the `tracing` dispatcher is
//! already owned by a caller.
//!
//! `chutoro_cli` is also a library, so an embedding application may install its
//! own subscriber before calling [`init_logging`]. Both global slots stay
//! separate in that case, and the process-wide nature of logging means the
//! scenario needs a process of its own: this module holds a single test.
//!
//! [`init_logging`]: chutoro_cli::logging::init_logging

use tracing_log::log;

/// Logger used only to probe whether the `log` slot is still free.
struct ProbeLogger;

impl log::Log for ProbeLogger {
    fn enabled(&self, _metadata: &log::Metadata<'_>) -> bool {
        false
    }

    fn log(&self, _record: &log::Record<'_>) {}

    fn flush(&self) {}
}

/// The dispatcher and the `log` bridge are separate registrations, so a caller
/// that installed its own subscriber still gains `log` forwarding: the bridge
/// must be attempted even though dispatcher installation reports a conflict.
#[test]
fn a_taken_dispatcher_still_installs_the_log_bridge() {
    // Occupy the dispatcher slot the way an embedding caller would.
    tracing::subscriber::set_global_default(tracing_subscriber::registry())
        .expect("the test process must own the global dispatcher");

    chutoro_cli::logging::init_logging().expect("a taken dispatcher is reported, not returned");

    // The probe can only claim the slot when the bridge left it free.
    assert!(
        log::set_boxed_logger(Box::new(ProbeLogger)).is_err(),
        "the log bridge must claim the global log slot even when the dispatcher is taken"
    );
}
