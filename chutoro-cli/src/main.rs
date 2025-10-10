//! CLI entry point for executing the chutoro walking skeleton.
//!
//! Parses command-line arguments with clap, executes the walking skeleton
//! clustering pipeline, renders the summary to stdout, and maps errors to
//! appropriate exit codes.

use std::io::{self, BufWriter, Write};
use std::process::ExitCode;

use clap::Parser;
use thiserror::Error;

use chutoro_cli::cli::{Cli, CliError, render_summary, run_cli};

/// Parse CLI arguments, execute the command, render the summary, and flush the
/// output stream.
fn try_main() -> Result<(), MainError> {
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
        Err(err) => {
            eprintln!("{err}");
            ExitCode::FAILURE
        }
    }
}

#[derive(Debug, Error)]
enum MainError {
    #[error(transparent)]
    Cli(#[from] CliError),
    #[error("failed to write output: {0}")]
    Output(#[from] io::Error),
}
