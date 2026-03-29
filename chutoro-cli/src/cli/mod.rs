//! Command-line interface orchestration for the chutoro CPU pipeline.
//!
//! The CLI currently offers a minimal `run` command that loads either a Parquet
//! dense matrix or a line-based UTF-8 text corpus and executes the CPU
//! clustering pipeline.

mod commands;
mod output;

pub use commands::{
    Cli, CliError, Command, ParquetArgs, RunCommand, RunSource, TextArgs, TextMetric, run_cli,
};
pub use output::{ExecutionSummary, render_summary};

#[cfg(test)]
mod tests;
