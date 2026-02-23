//! Command implementations and argument parsing for the chutoro CLI.

use std::fs::File;
use std::io::{self, BufReader, Write};
use std::path::{Path, PathBuf};

use chutoro_core::{Chutoro, ChutoroBuilder, ChutoroError, ClusteringResult, DataSource};
use chutoro_providers_dense::{DenseMatrixProvider, DenseMatrixProviderError};
use chutoro_providers_text::{TextProvider, TextProviderError};
use clap::{Args, Parser, Subcommand, ValueEnum};
use thiserror::Error;
use tracing::{info, instrument};

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
    /// Execute the clustering pipeline.
    Run(RunCommand),
}

impl Command {
    fn name(&self) -> &'static str {
        match self {
            Command::Run(_) => "run",
        }
    }
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

    /// Maximum estimated memory (in bytes) allowed for the pipeline.
    ///
    /// Supports human-readable suffixes: K, M, G, T (case-insensitive).
    /// Example: `--max-bytes 2G` or `--max-bytes 2147483648`.
    #[arg(long = "max-bytes", value_parser = parse_byte_size)]
    pub max_bytes: Option<u64>,

    /// Data source configuration.
    #[command(subcommand)]
    pub source: RunSource,
}

/// Input data sources supported by the CLI.
#[derive(Debug, Subcommand, Clone)]
pub enum RunSource {
    /// Execute against a Parquet file containing a `FixedSizeList<Float32, D>` column.
    Parquet(ParquetArgs),
    /// Execute against a UTF-8 text corpus, one string per line.
    Text(TextArgs),
}

impl RunSource {
    fn kind(&self) -> &'static str {
        match self {
            RunSource::Parquet(_) => "parquet",
            RunSource::Text(_) => "text",
        }
    }
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

impl TextMetric {
    fn label(self) -> &'static str {
        match self {
            TextMetric::Levenshtein => "levenshtein",
        }
    }
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
    /// Cluster assignments produced by the clustering pipeline.
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
///         max_bytes: None,
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
#[instrument(name = "cli.run", err, skip(cli), fields(command = %cli.command.name()))]
pub fn run_cli(cli: Cli) -> Result<ExecutionSummary, CliError> {
    match cli.command {
        Command::Run(run) => run_command(run),
    }
}

#[instrument(
    name = "cli.execute",
    err,
    skip(command),
    fields(
        min_cluster_size = command.min_cluster_size,
        source = %command.source.kind()
    ),
)]
pub(super) fn run_command(command: RunCommand) -> Result<ExecutionSummary, CliError> {
    let mut builder = ChutoroBuilder::new().with_min_cluster_size(command.min_cluster_size);
    if let Some(bytes) = command.max_bytes {
        builder = builder.with_max_bytes(bytes);
    }
    let chutoro = builder.build()?;

    let summary = match command.source {
        RunSource::Parquet(args) => run_parquet(&chutoro, args)?,
        RunSource::Text(args) => run_text(&chutoro, args)?,
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
    fields(
        path = %path_label(&args.path),
        column = %args.column,
        override_name = %args.name.as_deref().unwrap_or("<derived>")
    ),
)]
pub(super) fn run_parquet(
    chutoro: &Chutoro,
    args: ParquetArgs,
) -> Result<ExecutionSummary, CliError> {
    let ParquetArgs { path, column, name } = args;
    let chosen_name = derive_data_source_name(&path, name.as_deref());
    let provider = DenseMatrixProvider::try_from_parquet_path(chosen_name, &path, &column)?;
    execute_with_provider(chutoro, provider)
}

#[instrument(
    name = "cli.run_text",
    err,
    skip(chutoro, args),
    fields(
        path = %path_label(&args.path),
        metric = args.metric.label(),
        override_name = %args.name.as_deref().unwrap_or("<derived>")
    ),
)]
pub(super) fn run_text(chutoro: &Chutoro, args: TextArgs) -> Result<ExecutionSummary, CliError> {
    let TextArgs { path, metric, name } = args;
    let chosen_name = derive_data_source_name(&path, name.as_deref());
    let reader = open_text_reader(&path)?;
    let provider = match metric {
        TextMetric::Levenshtein => TextProvider::try_from_reader(chosen_name, reader)?,
    };
    execute_with_provider(chutoro, provider)
}

#[instrument(
    name = "cli.open_text_reader",
    err,
    skip(path),
    fields(path = %path_label(path))
)]
pub(super) fn open_text_reader(path: &Path) -> Result<BufReader<File>, CliError> {
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

/// Parses a human-readable byte size such as `"512M"` or `"2G"` into a `u64`.
///
/// Recognised suffixes (case-insensitive): `K`/`KB`/`KiB`, `M`/`MB`/`MiB`,
/// `G`/`GB`/`GiB`, `T`/`TB`/`TiB`.  Plain integers are treated as bytes.
pub(super) fn parse_byte_size(s: &str) -> Result<u64, String> {
    let s = s.trim();
    if s.is_empty() {
        return Err("byte size must not be empty".to_owned());
    }

    // Split into leading digits and trailing suffix.
    let split = s.find(|ch: char| !ch.is_ascii_digit()).unwrap_or(s.len());
    let (num_part, suffix) = s.split_at(split);

    let base: u64 = num_part
        .parse()
        .map_err(|err| format!("invalid byte size `{num_part}`: {err}"))?;

    let multiplier = match suffix.trim().to_ascii_lowercase().as_str() {
        "" => 1_u64,
        "k" | "kb" | "kib" => 1024,
        "m" | "mb" | "mib" => 1024 * 1024,
        "g" | "gb" | "gib" => 1024 * 1024 * 1024,
        "t" | "tb" | "tib" => 1024_u64 * 1024 * 1024 * 1024,
        other => return Err(format!("unknown size suffix: `{other}`")),
    };

    base.checked_mul(multiplier)
        .ok_or_else(|| "byte size overflows u64".to_owned())
}

/// Produce a redacted label for a path that avoids leaking absolute directories.
fn path_label(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "<unknown>".to_owned())
}

fn execute_with_provider<D>(chutoro: &Chutoro, provider: D) -> Result<ExecutionSummary, CliError>
where
    D: DataSource + Sync,
{
    let result = chutoro.run(&provider)?;
    Ok(ExecutionSummary {
        data_source: provider.name().to_owned(),
        result,
    })
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
/// assert_eq!(buffer.into_inner().len(), 38);
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
