//! Logging initialization for the chutoro CLI.
//!
//! Installs a global `tracing` subscriber with optional JSON formatting and
//! bridges the `log` facade so crates using either API emit structured events.

use mockable::{DefaultEnv, Env};
use std::{
    env,
    sync::{Mutex, OnceLock},
};
use thiserror::Error;
use tracing_log::LogTracer;
use tracing_subscriber::{
    EnvFilter, Layer, fmt::format::FmtSpan, layer::SubscriberExt, util::SubscriberInitExt,
};

/// Environment variable selecting human-readable or JSON log output.
const LOG_FORMAT_ENV: &str = "CHUTORO_LOG_FORMAT";
/// Environment variable selecting tracing directives.
const RUST_LOG_ENV: &str = "RUST_LOG";

/// Marker indicating that global logging initialization has completed.
static INITIALIZED: OnceLock<()> = OnceLock::new();
/// Mutex serializing logging initialization attempts.
static INIT_GUARD: OnceLock<Mutex<()>> = OnceLock::new();

/// Errors raised while initializing structured logging.
#[derive(Debug, Error)]
pub enum LoggingError {
    /// Environment variable contained invalid UTF-8 data.
    #[error("environment variable `{name}` contained invalid UTF-8: {source}")]
    InvalidUnicode {
        /// Name of the offending environment variable.
        name: &'static str,
        /// Underlying parse failure.
        #[source]
        source: env::VarError,
    },
    /// Unsupported log format requested via `CHUTORO_LOG_FORMAT`.
    #[error("unsupported log format `{provided}`; expected `human` or `json`")]
    UnsupportedFormat {
        /// Raw value supplied by the user.
        provided: String,
    },
    /// Failed to install the global tracing subscriber.
    #[error("failed to install tracing subscriber: {source}")]
    InstallFailed {
        /// Error raised by `tracing_subscriber`.
        #[source]
        source: tracing_subscriber::util::TryInitError,
    },
}

/// Install global structured logging if it has not already been configured.
///
/// The log format defaults to human-readable output, but can be switched to
/// JSON by setting `CHUTORO_LOG_FORMAT=json`. Diagnostics are emitted to
/// `stderr` so CLI payloads on `stdout` remain parseable. The log level is
/// controlled via `RUST_LOG`.
///
/// # Errors
/// Returns [`LoggingError`] if the environment variable contains invalid
/// Unicode or the requested format is unsupported. Subscriber installation
/// failures (for example, when another global logger is already registered)
/// are reported to `stderr` but do not cause this function to return an error.
pub fn init_logging() -> Result<(), LoggingError> {
    // Recover from a poisoned lock: initialization state is tracked by the
    // `INITIALIZED` cell, so a poisoned guard cannot corrupt it.
    let guard = INIT_GUARD
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    if INITIALIZED.get().is_some() {
        return Ok(());
    }

    match install_subscriber() {
        Ok(()) => {
            // The guard serializes initialization, so a duplicate marker
            // update would reveal an internal synchronization error.
            mark_initialized();
        }
        Err(LoggingError::InstallFailed { source }) => {
            report_logging_conflict(&source);
            mark_initialized();
        }
        Err(err) => {
            drop(guard);
            return Err(err);
        }
    }

    Ok(())
}

/// Record that logging initialization has completed.
fn mark_initialized() {
    let was_already_initialized = INITIALIZED.set(()).is_err();
    debug_assert!(
        !was_already_initialized,
        "the initialization guard permits only one marker update",
    );
}

/// Build and install the configured tracing subscriber.
fn install_subscriber() -> Result<(), LoggingError> {
    install_subscriber_with_env(&DefaultEnv)
}

/// Install logging using an injected environment reader for test isolation.
fn install_subscriber_with_env(env: &dyn Env) -> Result<(), LoggingError> {
    let use_json = log_format_from_env(env)?;

    let env_filter = env_filter_from_env(env);

    let base_fmt_layer = tracing_subscriber::fmt::layer()
        .with_span_events(FmtSpan::FULL)
        .with_writer(std::io::stderr);

    let fmt_layer = if use_json {
        base_fmt_layer.json().with_span_list(true).boxed()
    } else {
        base_fmt_layer.boxed()
    };

    // Installing the log bridge is best-effort. If another logger already owns
    // the global slot, crates emitting via the `log` facade will continue using
    // that logger; tracing-native spans and events are unaffected.
    drop(LogTracer::init());

    tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer)
        .try_init()
        .map_err(|source| LoggingError::InstallFailed { source })
}

/// Read the requested output format through an injected environment reader.
fn log_format_from_env(env: &dyn Env) -> Result<bool, LoggingError> {
    match env.raw(LOG_FORMAT_ENV) {
        Ok(raw) => Ok(parse_log_format(&raw)?),
        Err(env::VarError::NotPresent) => Ok(false),
        Err(err @ env::VarError::NotUnicode(_)) => Err(LoggingError::InvalidUnicode {
            name: LOG_FORMAT_ENV,
            source: err,
        }),
    }
}

/// Build a tracing filter from an injected reader, defaulting to `info`.
fn env_filter_from_env(env: &dyn Env) -> EnvFilter {
    env.raw(RUST_LOG_ENV)
        .ok()
        .and_then(|directives| EnvFilter::try_new(directives).ok())
        .unwrap_or_else(|| EnvFilter::new("info"))
}
/// Emit a pre-initialization diagnostic to stderr when structured logging
/// initialization collides with an existing subscriber.
///
/// Clippy's `print_stderr` lint is denied workspace-wide; suppress it narrowly
/// for this pre-init diagnostic because structured logging is not yet available
/// and stderr is the only output channel.
///
/// # Examples
/// ```ignore
/// use tracing_subscriber::util::TryInitError;
///
/// fn on_conflict(err: TryInitError) {
///     report_logging_conflict(&err);
/// }
/// ```
#[expect(
    clippy::print_stderr,
    reason = "pre-initialization diagnostic; structured logging unavailable; stderr is sole output channel"
)]
fn report_logging_conflict(source: &tracing_subscriber::util::TryInitError) {
    eprintln!("structured logging already configured elsewhere: {source}");
}

/// Parse the configured log format, returning whether JSON is enabled.
fn parse_log_format(raw: &str) -> Result<bool, LoggingError> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "human" => Ok(false),
        "json" => Ok(true),
        other => Err(LoggingError::UnsupportedFormat {
            provided: other.to_owned(),
        }),
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for logging initialization.

    use super::*;

    use mockable::MockEnv;
    use rstest::rstest;

    #[rstest]
    #[case("human", false)]
    #[case("HUMAN", false)]
    #[case(" json ", true)]
    fn parse_log_format_accepts_supported_values(#[case] raw: &str, #[case] expected: bool) {
        let format = parse_log_format(raw).expect("format must parse");
        assert_eq!(format, expected);
    }

    #[test]
    fn parse_log_format_rejects_unknown_values() {
        let err = parse_log_format("xml").expect_err("xml is not supported");
        match err {
            LoggingError::UnsupportedFormat { provided } => assert_eq!(provided, "xml"),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[rstest]
    #[case(Ok("human".to_owned()), false)]
    #[case(Ok("json".to_owned()), true)]
    #[case(Err(env::VarError::NotPresent), false)]
    fn log_format_uses_injected_environment(
        #[case] value: Result<String, env::VarError>,
        #[case] expected: bool,
    ) {
        let mut env = MockEnv::new();
        env.expect_raw().returning(move |key| {
            assert_eq!(key, LOG_FORMAT_ENV);
            value.clone()
        });

        let result = log_format_from_env(&env).expect("format must resolve");
        assert_eq!(result, expected);
    }

    #[test]
    fn log_format_rejects_invalid_injected_value() {
        let mut env = MockEnv::new();
        env.expect_raw().returning(|key| {
            assert_eq!(key, LOG_FORMAT_ENV);
            Ok("xml".to_owned())
        });

        let error = log_format_from_env(&env).expect_err("invalid format must fail");
        assert!(matches!(error, LoggingError::UnsupportedFormat { .. }));
    }

    #[test]
    fn env_filter_uses_injected_rust_log() {
        let mut env = MockEnv::new();
        env.expect_raw().returning(|key| {
            assert_eq!(key, RUST_LOG_ENV);
            Ok("debug".to_owned())
        });

        assert_eq!(env_filter_from_env(&env).to_string(), "debug");
    }

    #[test]
    fn env_filter_defaults_when_rust_log_is_unavailable() {
        let mut env = MockEnv::new();
        env.expect_raw().returning(|key| {
            assert_eq!(key, RUST_LOG_ENV);
            Err(env::VarError::NotPresent)
        });

        assert_eq!(env_filter_from_env(&env).to_string(), "info");
    }

    #[test]
    fn env_filter_defaults_when_rust_log_is_invalid() {
        let mut env = MockEnv::new();
        env.expect_raw().returning(|key| {
            assert_eq!(key, RUST_LOG_ENV);
            Ok("chutoro=not-a-level".to_owned())
        });

        assert_eq!(env_filter_from_env(&env).to_string(), "info");
    }

    #[test]
    fn init_logging_is_idempotent() {
        init_logging().expect("logging must initialize");
        init_logging().expect("subsequent calls must be no-ops");
    }
}
