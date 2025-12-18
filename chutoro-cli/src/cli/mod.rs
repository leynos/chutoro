//! Command-line interface orchestration for the chutoro CPU pipeline.
//!
//! The CLI currently offers a minimal `run` command that loads either a Parquet
//! dense matrix or a line-based UTF-8 text corpus and executes the CPU
//! clustering pipeline.

mod commands;

pub use commands::{
    Cli, CliError, Command, ExecutionSummary, ParquetArgs, RunCommand, RunSource, TextArgs,
    TextMetric, render_summary, run_cli,
};

#[cfg(test)]
mod tests;
