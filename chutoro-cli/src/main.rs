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

    if let Err(err) = try_main() {
        let (code, data_source_code) = err
            .chain()
            .find_map(|cause| {
                // Downcast each cause so context layers do not obscure `CliError`
                // instances that carry structured codes.
                let cause: &(dyn std::error::Error + 'static) = cause;
                cause
                    .downcast_ref::<CliError>()
                    .and_then(|cli_error| match cli_error {
                        CliError::Core(core) => Some((Some(core.code()), core.data_source_code())),
                        _ => None,
                    })
            })
            .unwrap_or((None, None));

        error!(
            error = %err,
            code = ?code.map(|c| c.as_str()),
            data_source_code = ?data_source_code.map(|c| c.as_str()),
            "command execution failed"
        );
        return ExitCode::FAILURE;
    }

    ExitCode::SUCCESS
}

#[expect(
    clippy::print_stderr,
    reason = "Emit one-off diagnostic before tracing is initialised"
)]
fn report_logging_init_error(err: &LoggingError) {
    eprintln!("failed to initialise logging: {err}");
}
