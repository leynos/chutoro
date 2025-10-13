//! Command implementations and argument parsing for the chutoro CLI.

use std::fs::File;
use std::io::{self, BufReader, Write};
use std::path::{Path, PathBuf};

use chutoro_core::{Chutoro, ChutoroBuilder, ChutoroError, ClusteringResult, DataSource};
use chutoro_providers_dense::{DenseMatrixProvider, DenseMatrixProviderError};
use chutoro_providers_text::{TextProvider, TextProviderError};
use clap::{Args, Parser, Subcommand, ValueEnum};
use thiserror::Error;
use tracing::{Span, field, info, instrument};

const DEFAULT_MIN_CLUSTER_SIZE: usize = 5;

/// Top-level CLI options parsed by [`clap`].
#[derive(Debug, Parser, Clone)]
#[command(name = "chutoro", about = "Execute the chutoro clustering pipeline.")]
pub struct Cli {
    /// Command to execute.
    #[command(subcommand)]
    pub command: Command,
}

/// Supported CLI commands.
#[derive(Debug, Subcommand, Clone)]
pub enum Command {
    /// Execute the walking skeleton pipeline.
    Run(RunCommand),
}

/// Options accepted by the `run` command.
#[derive(Debug, Args, Clone)]
pub struct RunCommand {
    /// Minimum number of items per cluster.
    #[arg(
        long = "min-cluster-size",
        default_value_t = DEFAULT_MIN_CLUSTER_SIZE,
        value_parser = clap::value_parser!(usize),
    )]
    pub min_cluster_size: usize,

    /// Data source configuration.
    #[command(subcommand)]
    pub source: RunSource,
}

/// Input data sources supported by the walking skeleton.
#[derive(Debug, Subcommand, Clone)]
pub enum RunSource {
    /// Execute against a Parquet file containing a `FixedSizeList<Float32, D>` column.
    Parquet(ParquetArgs),
    /// Execute against a UTF-8 text corpus, one string per line.
    Text(TextArgs),
}

/// Parquet ingestion arguments.
#[derive(Debug, Args, Clone)]
pub struct ParquetArgs {
    /// Path to the Parquet file containing feature vectors.
    pub path: PathBuf,

    /// Column containing `FixedSizeList<Float32, D>` rows.
    #[arg(long)]
    pub column: String,

    /// Override name for the data source (defaults to the file name).
    #[arg(long)]
    pub name: Option<String>,
}

/// Text ingestion arguments.
#[derive(Debug, Args, Clone)]
pub struct TextArgs {
    /// Path to a UTF-8 text file with one string per line.
    pub path: PathBuf,

    /// Distance metric to use when comparing lines.
    #[arg(long, value_enum)]
    pub metric: TextMetric,

    /// Override name for the data source (defaults to the file name).
    #[arg(long)]
    pub name: Option<String>,
}

/// Supported text metrics.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum TextMetric {
    /// Compute Levenshtein edit distance between lines.
    Levenshtein,
}

/// Errors surfaced while executing CLI commands.
#[derive(Debug, Error)]
pub enum CliError {
    /// File I/O failed while loading an input source.
    #[error("failed to open `{path}`: {source}")]
    Io {
        /// Path that triggered the failure.
        path: PathBuf,
        /// Underlying operating system error.
        #[source]
        source: io::Error,
    },
    /// Dense matrix ingestion failed.
    #[error(transparent)]
    Dense(#[from] DenseMatrixProviderError),
    /// Text ingestion failed.
    #[error(transparent)]
    Text(#[from] TextProviderError),
    /// Core orchestration failed.
    #[error(transparent)]
    Core(#[from] ChutoroError),
}

/// Summarises the outcome of executing a CLI command.
#[derive(Debug, Clone)]
pub struct ExecutionSummary {
    /// Name reported by the data source implementation.
    pub data_source: String,
    /// Cluster assignments produced by the walking skeleton.
    pub result: ClusteringResult,
}

/// Executes the CLI command represented by `cli`.
///
/// # Errors
/// Returns [`CliError`] when parsing or execution fails.
///
/// # Examples
/// ```
/// # use std::error::Error;
/// # use chutoro_cli::cli::{Cli, Command, RunCommand, RunSource, TextArgs, TextMetric, run_cli};
/// # use tempfile::NamedTempFile;
/// #
/// # fn main() -> Result<(), Box<dyn Error>> {
/// let file = NamedTempFile::new()?;
/// std::fs::write(file.path(), "alpha\nbeta\n")?;
/// let cli = Cli {
///     command: Command::Run(RunCommand {
///         min_cluster_size: 1,
///         source: RunSource::Text(TextArgs {
///             path: file.path().to_path_buf(),
///             metric: TextMetric::Levenshtein,
///             name: None,
///         }),
///     }),
/// };
/// let summary = run_cli(cli)?;
/// assert_eq!(summary.result.assignments().len(), 2);
/// # Ok(())
/// # }
/// ```
#[instrument(
    name = "cli.run",
    err,
    skip(cli),
    fields(command = field::Empty),
)]
pub fn run_cli(cli: Cli) -> Result<ExecutionSummary, CliError> {
    match cli.command {
        Command::Run(run) => {
            Span::current().record("command", field::display("run"));
            run_command(run)
        }
    }
}

#[instrument(
    name = "cli.execute",
    err,
    skip(command),
    fields(min_cluster_size = field::Empty, source = field::Empty),
)]
pub(super) fn run_command(command: RunCommand) -> Result<ExecutionSummary, CliError> {
    let chutoro = ChutoroBuilder::new()
        .with_min_cluster_size(command.min_cluster_size)
        .build()?;

    let span = Span::current();
    span.record("min_cluster_size", field::display(command.min_cluster_size));

    let summary = match command.source {
        RunSource::Parquet(args) => {
            span.record("source", field::display("parquet"));
            run_parquet(&chutoro, args)?
        }
        RunSource::Text(args) => {
            span.record("source", field::display("text"));
            run_text(&chutoro, args)?
        }
    };

    info!(
        data_source = summary.data_source.as_str(),
        clusters = summary.result.cluster_count(),
        "command completed"
    );
    Ok(summary)
}

#[instrument(
    name = "cli.run_parquet",
    err,
    skip(chutoro, args),
    fields(path = field::Empty, column = field::Empty, override_name = field::Empty),
)]
pub(super) fn run_parquet(
    chutoro: &Chutoro,
    args: ParquetArgs,
) -> Result<ExecutionSummary, CliError> {
    let ParquetArgs { path, column, name } = args;
    let span = Span::current();
    span.record("path", field::display(path.display()));
    span.record("column", field::display(&column));
    span.record(
        "override_name",
        field::display(name.as_deref().unwrap_or("<derived>")),
    );
    let chosen_name = derive_data_source_name(&path, name.as_deref());
    let provider = DenseMatrixProvider::try_from_parquet_path(chosen_name, &path, &column)?;
    let result = chutoro.run(&provider)?;
    info!(
        data_source = provider.name(),
        clusters = result.cluster_count(),
        "parquet execution completed"
    );
    Ok(ExecutionSummary {
        data_source: provider.name().to_owned(),
        result,
    })
}

#[instrument(
    name = "cli.run_text",
    err,
    skip(chutoro, args),
    fields(path = field::Empty, metric = field::Empty, override_name = field::Empty),
)]
pub(super) fn run_text(chutoro: &Chutoro, args: TextArgs) -> Result<ExecutionSummary, CliError> {
    let TextArgs { path, metric, name } = args;
    let span = Span::current();
    span.record("path", field::display(path.display()));
    let metric_label = match metric {
        TextMetric::Levenshtein => "levenshtein",
    };
    span.record("metric", field::display(metric_label));
    span.record(
        "override_name",
        field::display(name.as_deref().unwrap_or("<derived>")),
    );
    let chosen_name = derive_data_source_name(&path, name.as_deref());
    let reader = open_text_reader(&path)?;
    let provider = match metric {
        TextMetric::Levenshtein => TextProvider::try_from_reader(chosen_name, reader)?,
    };
    let result = chutoro.run(&provider)?;
    info!(
        data_source = provider.name(),
        clusters = result.cluster_count(),
        "text execution completed"
    );
    Ok(ExecutionSummary {
        data_source: provider.name().to_owned(),
        result,
    })
}

#[instrument(name = "cli.open_text_reader", err, fields(path = field::Empty))]
pub(super) fn open_text_reader(path: &Path) -> Result<BufReader<File>, CliError> {
    Span::current().record("path", field::display(path.display()));
    let file = File::open(path).map_err(|source| CliError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(BufReader::new(file))
}

pub(super) fn derive_data_source_name(path: &Path, override_name: Option<&str>) -> String {
    if let Some(name) = override_name {
        return name.to_owned();
    }

    path.file_stem()
        .and_then(|value| value.to_str())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| "data_source".to_owned())
}

/// Renders `summary` to `writer` in a human-readable text format.
///
/// # Errors
/// Returns [`io::Error`] if writing to the supplied writer fails.
///
/// # Examples
/// ```
/// # use std::error::Error;
/// # use std::io::Cursor;
/// # use chutoro_cli::cli::{ExecutionSummary, render_summary};
/// # use chutoro_core::{ClusteringResult, ClusterId};
/// #
/// # fn main() -> Result<(), Box<dyn Error>> {
/// let summary = ExecutionSummary {
///     data_source: "demo".into(),
///     result: ClusteringResult::from_assignments(vec![
///         ClusterId::new(0),
///         ClusterId::new(1),
///     ]),
/// };
/// let mut buffer = Cursor::new(Vec::new());
/// render_summary(&summary, &mut buffer)?;
/// assert_eq!(buffer.into_inner().len(), 54);
/// # Ok(())
/// # }
/// ```
pub fn render_summary(summary: &ExecutionSummary, mut writer: impl Write) -> io::Result<()> {
    writeln!(writer, "data source: {}", summary.data_source)?;
    writeln!(writer, "clusters: {}", summary.result.cluster_count())?;
    for (index, cluster) in summary.result.assignments().iter().enumerate() {
        writeln!(writer, "{index}\t{}", cluster.get())?;
    }
    Ok(())
}
