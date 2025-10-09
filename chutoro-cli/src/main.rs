use std::io::{self, BufWriter, Write};
use std::process::ExitCode;

use clap::Parser;

use chutoro_cli::cli::{Cli, render_summary, run_cli};

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run_cli(cli) {
        Ok(summary) => {
            let mut stdout = BufWriter::new(io::stdout());
            match render_summary(&summary, &mut stdout) {
                Ok(()) => ExitCode::SUCCESS,
                Err(err) => {
                    let mut stderr = BufWriter::new(io::stderr());
                    let _ = writeln!(stderr, "failed to write output: {err}");
                    ExitCode::from(1)
                }
            }
        }
        Err(err) => {
            let mut stderr = BufWriter::new(io::stderr());
            let _ = writeln!(stderr, "{err}");
            ExitCode::from(1)
        }
    }
}
