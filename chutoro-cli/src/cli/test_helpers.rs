//! Small helpers shared across CLI tests.
//!
//! The CLI unit tests build temporary input files and assert error handling
//! behaviour. These helpers keep the test cases concise and consistent.

use std::fs::File;
use std::io::{self, Write};
use std::path::PathBuf;

use tempfile::TempDir;

use super::super::commands::run_command;
use super::super::{Cli, CliError, RunCommand, run_cli};

pub(super) fn temp_dir() -> TempDir {
    match TempDir::new() {
        Ok(dir) => dir,
        Err(err) => panic!("failed to create temp dir: {err}"),
    }
}

pub(super) fn create_text_file(dir: &TempDir, name: &str, contents: &str) -> io::Result<PathBuf> {
    let path = dir.path().join(name);
    let mut file = File::create(&path)?;
    file.write_all(contents.as_bytes())?;
    Ok(path)
}

pub(super) fn run_cli_expecting_error(cli: Cli, panic_msg: &str) -> CliError {
    match run_cli(cli) {
        Ok(_) => panic!("{panic_msg}"),
        Err(err) => err,
    }
}

pub(super) fn run_command_expecting_error(cmd: RunCommand, panic_msg: &str) -> CliError {
    match run_command(cmd) {
        Ok(_) => panic!("{panic_msg}"),
        Err(err) => err,
    }
}
