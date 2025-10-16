//! CLI entry point for executing the chutoro walking skeleton.
//!
//! Parses command-line arguments with clap, executes the walking skeleton
//! clustering pipeline, renders the summary to stdout, and maps errors to
//! appropriate exit codes. Logging is initialised eagerly so subsequent
//! operations can emit structured diagnostics via `tracing`.

use std::io::{self, BufWriter, Write};
use std::process::ExitCode;

use anyhow::{Context, Result};
use clap::Parser;

use chutoro_cli::{
    cli::{Cli, CliError, render_summary, run_cli},
    logging::{self, LoggingError},
};
use chutoro_core::{ChutoroError, ChutoroErrorCode};
use tracing::error;

/// Parse CLI arguments, execute the command, render the summary, and flush the
/// output stream.
fn try_main() -> Result<()> {
    let cli = Cli::parse();
    let summary = run_cli(cli).context("failed to execute command")?;
    let stdout = io::stdout();
    let mut writer = BufWriter::new(stdout.lock());
    render_summary(&summary, &mut writer).context("failed to render summary")?;
    writer.flush().context("failed to flush output")?;
    Ok(())
}

fn main() -> ExitCode {
    if let Err(err) = logging::init_logging() {
        report_logging_init_error(&err);
        return ExitCode::FAILURE;
    }

    match try_main() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            if let Some(cli_error) = err.downcast_ref::<CliError>() {
                log_cli_error(cli_error);
            } else {
                error!(error = %err, "command execution failed");
            }
            ExitCode::FAILURE
        }
    }
}

#[expect(
    clippy::print_stderr,
    reason = "Emit one-off diagnostic before tracing is initialised"
)]
fn report_logging_init_error(err: &LoggingError) {
    eprintln!("failed to initialize logging: {err}");
}

fn log_cli_error(err: &CliError) {
    match err {
        CliError::Core(chutoro) => log_core_error(chutoro),
        _ => error!(error = %err, "command execution failed"),
    }
}

fn log_core_error(err: &ChutoroError) {
    match err.data_source_code() {
        Some(data_source_code) => error!(
            error = %err,
            code = %ChutoroErrorCode::DataSourceFailure,
            data_source_code = %data_source_code,
            "command execution failed"
        ),
        None => error!(
            error = %err,
            code = %err.code(),
            "command execution failed"
        ),
    }
}
