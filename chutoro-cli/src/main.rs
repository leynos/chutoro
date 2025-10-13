//! CLI entry point for executing the chutoro walking skeleton.
//!
//! Parses command-line arguments with clap, executes the walking skeleton
//! clustering pipeline, renders the summary to stdout, and maps errors to
//! appropriate exit codes. Logging is initialised eagerly so subsequent
//! operations can emit structured diagnostics via `tracing`.

use std::io::{self, BufWriter, Write};
use std::process::ExitCode;

use clap::Parser;
use thiserror::Error;

use chutoro_cli::{
    cli::{Cli, CliError, render_summary, run_cli},
    logging::{self, LoggingError},
};
use tracing::error;

/// Parse CLI arguments, execute the command, render the summary, and flush the
/// output stream.
fn try_main() -> Result<(), MainError> {
    logging::init_logging()?;
    let cli = Cli::parse();
    let summary = run_cli(cli)?;
    let stdout = io::stdout();
    let mut writer = BufWriter::new(stdout.lock());
    render_summary(&summary, &mut writer)?;
    writer.flush()?;
    Ok(())
}

fn main() -> ExitCode {
    match try_main() {
        Ok(()) => ExitCode::SUCCESS,
        Err(MainError::Logging(err)) => {
            eprintln!("failed to initialise logging: {err}");
            ExitCode::FAILURE
        }
        Err(err) => {
            error!(error = %err, "command execution failed");
            ExitCode::FAILURE
        }
    }
}

#[derive(Debug, Error)]
enum MainError {
    #[error(transparent)]
    Logging(#[from] LoggingError),
    #[error(transparent)]
    Cli(#[from] CliError),
    #[error("failed to write output: {0}")]
    Output(#[from] io::Error),
}
