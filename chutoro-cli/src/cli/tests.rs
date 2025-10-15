//! Unit tests for the CLI commands and data ingestion helpers.

use super::commands::{derive_data_source_name, run_command};
use super::{
    Cli, CliError, Command, ExecutionSummary, ParquetArgs, RunCommand, RunSource, TextArgs,
    TextMetric, render_summary, run_cli,
};

use std::fs::File;
use std::io::{self, Write};
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use chutoro_core::{ChutoroError, ClusteringResult};
use clap::Parser;
use parquet::arrow::arrow_writer::ArrowWriter;
use rstest::rstest;
use tempfile::TempDir;
use tracing::Level;
use tracing_subscriber::layer::SubscriberExt;

use chutoro_test_support::tracing::RecordingLayer;

use chutoro_providers_dense::DenseMatrixProviderError;
use chutoro_providers_text::TextProviderError;

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

#[rstest]
fn run_command_emits_tracing_fields() -> TestResult {
    let dir = temp_dir();
    let path = create_text_file(&dir, "lines.txt", "alpha\nbeta\ngamma\n")?;
    let layer = RecordingLayer::default();
    let subscriber = tracing_subscriber::registry().with(layer.clone());

    let command = RunCommand {
        min_cluster_size: 2,
        source: RunSource::Text(TextArgs {
            path: path.clone(),
            metric: TextMetric::Levenshtein,
            name: None,
        }),
    };

    let summary = tracing::subscriber::with_default(subscriber, || run_command(command))?;
    assert_eq!(summary.data_source, "lines");

    let spans = layer.spans();
    let execute = spans
        .iter()
        .find(|span| span.name == "cli.execute")
        .expect("cli.execute span must exist");
    assert_eq!(
        execute.fields.get("min_cluster_size"),
        Some(&"2".to_owned())
    );
    assert_eq!(execute.fields.get("source"), Some(&"text".to_owned()));

    let text_span = spans
        .iter()
        .find(|span| span.name == "cli.run_text")
        .expect("cli.run_text span must exist");
    assert!(
        text_span
            .fields
            .get("path")
            .is_some_and(|value| value.ends_with("lines.txt"))
    );
    assert_eq!(
        text_span.fields.get("metric"),
        Some(&"levenshtein".to_owned())
    );
    assert_eq!(
        text_span.fields.get("override_name"),
        Some(&"<derived>".to_owned())
    );

    let events = layer.events();
    assert!(events.iter().any(|event| {
        event.level == Level::INFO
            && event
                .fields
                .get("message")
                .is_some_and(|value| value == "command completed")
            && event
                .fields
                .get("data_source")
                .is_some_and(|value| value == "lines")
    }));
    Ok(())
}

#[rstest]
fn open_text_reader_records_path_on_error() -> TestResult {
    let dir = temp_dir();
    let missing_path = dir.path().join("missing.txt");
    let layer = RecordingLayer::default();
    let subscriber = tracing_subscriber::registry().with(layer.clone());

    let command = RunCommand {
        min_cluster_size: 1,
        source: RunSource::Text(TextArgs {
            path: missing_path.clone(),
            metric: TextMetric::Levenshtein,
            name: None,
        }),
    };

    let err = tracing::subscriber::with_default(subscriber, || run_command(command))
        .expect_err("missing file must fail");
    assert!(matches!(err, CliError::Io { .. }));

    let spans = layer.spans();
    let reader_span = spans
        .iter()
        .find(|span| span.name == "cli.open_text_reader")
        .expect("reader span must exist");
    assert!(
        reader_span
            .fields
            .get("path")
            .is_some_and(|value| value.ends_with("missing.txt"))
    );

    let run_span = spans
        .iter()
        .find(|span| span.name == "cli.run_text")
        .expect("run_text span must exist");
    assert_eq!(
        run_span.fields.get("override_name"),
        Some(&"<derived>".to_owned())
    );
    Ok(())
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

fn create_parquet_file(dir: &TempDir, name: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
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
