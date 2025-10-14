//! Logging initialisation for the chutoro CLI.
//!
//! Installs a global `tracing` subscriber with optional JSON formatting and
//! bridges the `log` facade so crates using either API emit structured events.

use std::{env, sync::OnceLock};

use thiserror::Error;
use tracing_log::LogTracer;
use tracing_subscriber::{
    EnvFilter, Layer, fmt::format::FmtSpan, layer::SubscriberExt, util::SubscriberInitExt,
};

const LOG_FORMAT_ENV: &str = "CHUTORO_LOG_FORMAT";

static INITIALISED: OnceLock<()> = OnceLock::new();

/// Errors raised while initialising structured logging.
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
/// Unicode, the requested format is unsupported, or the subscriber cannot be
/// installed.
pub fn init_logging() -> Result<(), LoggingError> {
    if INITIALISED.get().is_some() {
        return Ok(());
    }

    match install_subscriber() {
        Ok(()) => {}
        Err(LoggingError::InstallFailed { source }) => {
            eprintln!("structured logging already configured elsewhere: {source}");
        }
        Err(err) => return Err(err),
    }
    let _ = INITIALISED.set(());
    Ok(())
}

fn install_subscriber() -> Result<(), LoggingError> {
    let use_json = match env::var(LOG_FORMAT_ENV) {
        Ok(raw) => parse_log_format(&raw)?,
        Err(env::VarError::NotPresent) => false,
        Err(err @ env::VarError::NotUnicode(_)) => Err(LoggingError::InvalidUnicode {
            name: LOG_FORMAT_ENV,
            source: err,
        })?,
    };

    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_span_events(FmtSpan::FULL)
        .with_writer(std::io::stderr);

    let fmt_layer = if use_json {
        fmt_layer
            .json()
            .with_current_span(true)
            .with_span_list(true)
            .boxed()
    } else {
        fmt_layer.boxed()
    };

    // Installing the log bridge is best-effort; if another logger already owns
    // the global slot we keep the existing configuration.
    let _ = LogTracer::init();

    tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer)
        .try_init()
        .map_err(|source| LoggingError::InstallFailed { source })
}

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
    use super::*;

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

    #[test]
    fn init_logging_is_idempotent() {
        init_logging().expect("logging must initialise");
        init_logging().expect("subsequent calls must be no-ops");
    }
}
