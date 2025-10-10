//! CLI entry point for executing the chutoro walking skeleton.

use std::fmt;
use std::io::{self, BufWriter, Write};
use std::process::ExitCode;

use clap::Parser;

use chutoro_cli::cli::{Cli, CliError, render_summary, run_cli};

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
            ExitCode::from(1)
        }
    }
}

#[derive(Debug)]
enum MainError {
    Cli(CliError),
    Output(io::Error),
}

impl From<CliError> for MainError {
    fn from(error: CliError) -> Self {
        Self::Cli(error)
    }
}

impl From<io::Error> for MainError {
    fn from(error: io::Error) -> Self {
        Self::Output(error)
    }
}

impl fmt::Display for MainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cli(error) => write!(f, "{error}"),
            Self::Output(error) => write!(f, "failed to write output: {error}"),
        }
    }
}

impl std::error::Error for MainError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Cli(error) => Some(error),
            Self::Output(error) => Some(error),
        }
    }
}
