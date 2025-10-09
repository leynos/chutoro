//! Command-line interface orchestration for the chutoro walking skeleton.
//!
//! The CLI currently offers a minimal `run` command that loads either a Parquet
//! dense matrix or a line-based UTF-8 text corpus and executes the skeleton
//! clustering pipeline.

use std::fs::File;
use std::io::{self, BufReader, Write};
use std::path::{Path, PathBuf};

use chutoro_core::{Chutoro, ChutoroBuilder, ChutoroError, ClusteringResult, DataSource};
use chutoro_providers_dense::{DenseMatrixProvider, DenseMatrixProviderError};
use chutoro_providers_text::{TextProvider, TextProviderError};
use clap::{Args, Parser, Subcommand, ValueEnum};
use thiserror::Error;

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
pub fn run_cli(cli: Cli) -> Result<ExecutionSummary, CliError> {
    match cli.command {
        Command::Run(run) => run_command(run),
    }
}

fn run_command(command: RunCommand) -> Result<ExecutionSummary, CliError> {
    let chutoro = ChutoroBuilder::new()
        .with_min_cluster_size(command.min_cluster_size)
        .build()?;

    match command.source {
        RunSource::Parquet(args) => run_parquet(&chutoro, args),
        RunSource::Text(args) => run_text(&chutoro, args),
    }
}

fn run_parquet(chutoro: &Chutoro, args: ParquetArgs) -> Result<ExecutionSummary, CliError> {
    let ParquetArgs { path, column, name } = args;
    let chosen_name = derive_data_source_name(&path, name.as_deref());
    let provider = DenseMatrixProvider::try_from_parquet_path(chosen_name.clone(), &path, &column)?;
    let result = chutoro.run(&provider)?;
    Ok(ExecutionSummary {
        data_source: provider.name().to_owned(),
        result,
    })
}

fn run_text(chutoro: &Chutoro, args: TextArgs) -> Result<ExecutionSummary, CliError> {
    let TextArgs { path, metric, name } = args;
    let chosen_name = derive_data_source_name(&path, name.as_deref());
    let reader = open_text_reader(&path)?;
    let provider = match metric {
        TextMetric::Levenshtein => TextProvider::try_from_reader(chosen_name, reader)?,
    };
    let result = chutoro.run(&provider)?;
    Ok(ExecutionSummary {
        data_source: provider.name().to_owned(),
        result,
    })
}

fn open_text_reader(path: &Path) -> Result<BufReader<File>, CliError> {
    let file = File::open(path).map_err(|source| CliError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(BufReader::new(file))
}

fn derive_data_source_name(path: &Path, override_name: Option<&str>) -> String {
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

#[cfg(test)]
mod tests {
    use super::*;

    use std::path::Path;
    use std::sync::Arc;

    use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::arrow_writer::ArrowWriter;
    use rstest::rstest;
    use tempfile::TempDir;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[rstest]
    #[case::override_name("/tmp/source.parquet", Some("override"), "override")]
    #[case::stem_with_extension("/tmp/source.parquet", None, "source")]
    #[case::stem_without_extension("/tmp/source", None, "source")]
    #[case::missing_stem("", None, "data_source")]
    fn derive_data_source_name_selects_expected_name(
        #[case] raw_path: &str,
        #[case] override_name: Option<&'static str>,
        #[case] expected: &str,
    ) {
        let path = Path::new(raw_path);
        let name = derive_data_source_name(path, override_name);
        assert_eq!(name, expected);
    }

    #[rstest]
    #[case(1, vec![0, 1, 2])]
    #[case(2, vec![0, 0, 1])]
    fn run_text_success(#[case] min_cluster_size: usize, #[case] expected: Vec<u64>) -> TestResult {
        let dir = temp_dir();
        let path = create_text_file(&dir, "lines.txt", "alpha\nbeta\ngamma\n")?;
        let cli = Cli {
            command: Command::Run(RunCommand {
                min_cluster_size,
                source: RunSource::Text(TextArgs {
                    path,
                    metric: TextMetric::Levenshtein,
                    name: None,
                }),
            }),
        };
        let summary = run_cli(cli)?;
        let assignments: Vec<u64> = summary
            .result
            .assignments()
            .iter()
            .map(|id| id.get())
            .collect();
        assert_eq!(assignments, expected);
        Ok(())
    }

    #[rstest]
    fn run_text_rejects_insufficient_items() -> TestResult {
        let dir = temp_dir();
        let path = create_text_file(&dir, "lines.txt", "alpha\nbeta\n")?;
        let cli = Cli {
            command: Command::Run(RunCommand {
                min_cluster_size: 3,
                source: RunSource::Text(TextArgs {
                    path,
                    metric: TextMetric::Levenshtein,
                    name: None,
                }),
            }),
        };
        let err = run_cli_expecting_error(cli, "run must fail for insufficient items");
        assert!(matches!(
            err,
            CliError::Core(ChutoroError::InsufficientItems { .. })
        ));
        Ok(())
    }

    #[rstest]
    fn run_text_rejects_empty_files() -> TestResult {
        let dir = temp_dir();
        let path = create_text_file(&dir, "empty.txt", "")?;
        let cli = Cli {
            command: Command::Run(RunCommand {
                min_cluster_size: 1,
                source: RunSource::Text(TextArgs {
                    path,
                    metric: TextMetric::Levenshtein,
                    name: None,
                }),
            }),
        };
        let err = run_cli_expecting_error(cli, "empty input must fail");
        assert!(matches!(err, CliError::Text(TextProviderError::EmptyInput)));
        Ok(())
    }

    #[rstest]
    fn run_parquet_success() -> TestResult {
        let dir = temp_dir();
        let path = create_parquet_file(&dir, "vectors.parquet")?;
        let cli = Cli {
            command: Command::Run(RunCommand {
                min_cluster_size: 2,
                source: RunSource::Parquet(ParquetArgs {
                    path,
                    column: "features".into(),
                    name: Some("parquet".into()),
                }),
            }),
        };
        let summary = run_cli(cli)?;
        let assignments: Vec<u64> = summary
            .result
            .assignments()
            .iter()
            .map(|id| id.get())
            .collect();
        assert_eq!(assignments, vec![0, 0, 1, 1]);
        Ok(())
    }

    #[rstest]
    fn run_parquet_rejects_missing_column() -> TestResult {
        let dir = temp_dir();
        let path = create_parquet_file(&dir, "vectors.parquet")?;
        let cli = Cli {
            command: Command::Run(RunCommand {
                min_cluster_size: 1,
                source: RunSource::Parquet(ParquetArgs {
                    path,
                    column: "unknown".into(),
                    name: None,
                }),
            }),
        };
        let err = run_cli_expecting_error(cli, "unknown column must fail");
        assert!(matches!(
            err,
            CliError::Dense(DenseMatrixProviderError::ColumnNotFound { .. })
        ));
        Ok(())
    }

    #[rstest]
    fn run_command_rejects_zero_min_cluster_size() -> TestResult {
        let dir = temp_dir();
        let path = create_text_file(&dir, "lines.txt", "alpha\nbeta\ngamma\n")?;
        let err = run_command_expecting_error(
            RunCommand {
                min_cluster_size: 0,
                source: RunSource::Text(TextArgs {
                    path,
                    metric: TextMetric::Levenshtein,
                    name: None,
                }),
            },
            "zero min-cluster-size must fail",
        );
        assert!(matches!(
            err,
            CliError::Core(ChutoroError::InvalidMinClusterSize { .. })
        ));
        Ok(())
    }

    #[rstest]
    fn render_summary_outputs_assignments() -> TestResult {
        let summary = ExecutionSummary {
            data_source: "demo".into(),
            result: ClusteringResult::from_assignments(vec![
                chutoro_core::ClusterId::new(0),
                chutoro_core::ClusterId::new(1),
            ]),
        };
        let mut buffer = Vec::new();
        render_summary(&summary, &mut buffer)?;
        let text = String::from_utf8(buffer)?;
        assert!(text.contains("data source: demo"));
        assert!(text.contains("clusters: 2"));
        assert!(text.contains("0\t0"));
        assert!(text.contains("1\t1"));
        Ok(())
    }

    #[rstest]
    fn clap_rejects_unknown_metric() {
        let args = [
            "chutoro",
            "run",
            "text",
            "data.txt",
            "--metric",
            "unsupported",
        ];
        let result = Cli::try_parse_from(args);
        assert!(result.is_err());
    }

    fn temp_dir() -> TempDir {
        match TempDir::new() {
            Ok(dir) => dir,
            Err(err) => panic!("failed to create temp dir: {err}"),
        }
    }

    fn create_text_file(dir: &TempDir, name: &str, contents: &str) -> io::Result<PathBuf> {
        let path = dir.path().join(name);
        let mut file = File::create(&path)?;
        file.write_all(contents.as_bytes())?;
        Ok(path)
    }

    fn create_parquet_file(
        dir: &TempDir,
        name: &str,
    ) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let path = dir.path().join(name);
        let schema = build_schema();
        let batch = build_record_batch(schema.clone());
        let file = File::create(&path)?;
        let mut writer = ArrowWriter::try_new(file, schema, None)?;
        writer.write(&batch)?;
        writer.close()?;
        Ok(path)
    }

    /// Run CLI and expect an error, panicking with the given message if successful.
    fn run_cli_expecting_error(cli: Cli, panic_msg: &str) -> CliError {
        match run_cli(cli) {
            Ok(_) => panic!("{}", panic_msg),
            Err(err) => err,
        }
    }

    /// Run command and expect an error, panicking with the given message if successful.
    fn run_command_expecting_error(cmd: RunCommand, panic_msg: &str) -> CliError {
        match run_command(cmd) {
            Ok(_) => panic!("{}", panic_msg),
            Err(err) => err,
        }
    }

    fn build_schema() -> Arc<Schema> {
        let item_field = Arc::new(Field::new("item", DataType::Float32, false));
        let list_type = DataType::FixedSizeList(item_field.clone(), 2);
        Arc::new(Schema::new(vec![Field::new("features", list_type, false)]))
    }

    fn build_record_batch(schema: Arc<Schema>) -> RecordBatch {
        let values = Float32Array::from(vec![0.0_f32, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
        let item_field = Arc::new(Field::new("item", DataType::Float32, false));
        let list = FixedSizeListArray::new(item_field, 2, Arc::new(values) as ArrayRef, None);
        match RecordBatch::try_new(schema, vec![Arc::new(list) as ArrayRef]) {
            Ok(batch) => batch,
            Err(err) => panic!("failed to construct record batch: {err}"),
        }
    }
}
