//! Unit tests for the CLI commands and data ingestion helpers.

use super::commands::{derive_data_source_name, run_command};
use super::{
    Cli, CliError, Command, ExecutionSummary, ParquetArgs, RunCommand, RunSource, TextArgs,
    TextMetric, render_summary, run_cli,
};

use std::path::Path;

use chutoro_core::{ChutoroError, ClusteringResult};
use clap::Parser;
use rstest::rstest;
use tracing::Level;
use tracing_subscriber::layer::SubscriberExt;

use chutoro_test_support::tracing::RecordingLayer;

use chutoro_providers_dense::DenseMatrixProviderError;
use chutoro_providers_text::TextProviderError;

#[path = "test_fixtures.rs"]
mod test_fixtures;
use test_fixtures::create_parquet_file;

#[path = "test_helpers.rs"]
mod test_helpers;
use test_helpers::{
    create_text_file, run_cli_expecting_error, run_command_expecting_error, temp_dir,
};

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Runs the text pipeline once with the provided input file and minimum
/// cluster size.
///
/// Returns the [`ExecutionSummary`] produced by the CLI runner.
fn run_text_once(path: &Path, min_cluster_size: usize) -> Result<ExecutionSummary, CliError> {
    let cli = Cli {
        command: Command::Run(RunCommand {
            min_cluster_size,
            source: RunSource::Text(TextArgs {
                path: path.to_path_buf(),
                metric: TextMetric::Levenshtein,
                name: None,
            }),
        }),
    };
    run_cli(cli)
}

/// Asserts that a text run produced a clustering with the expected number of
/// assignments, and returns the observed cluster count.
///
/// This keeps the tests robust by checking invariants that should hold across
/// implementations without relying on exact label ids.
fn assert_text_result_summary(
    summary: &ExecutionSummary,
    expected_items: usize,
) -> Result<usize, Box<dyn std::error::Error>> {
    assert_eq!(summary.result.assignments().len(), expected_items);
    let clusters = summary.result.cluster_count();
    assert!(
        clusters >= 1 && clusters <= expected_items,
        "expected 1..={} clusters for a {}-row input",
        expected_items,
        expected_items
    );
    Ok(clusters)
}

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

#[test]
fn run_text_success() -> TestResult {
    let dir = temp_dir();
    let path = create_text_file(&dir, "lines.txt", "alpha\nbeta\ngamma\n")?;

    let summary_min_1 = run_text_once(path.as_path(), 1)?;
    let clusters_min_1 = assert_text_result_summary(&summary_min_1, 3)?;

    let summary_min_2 = run_text_once(path.as_path(), 2)?;
    let clusters_min_2 = assert_text_result_summary(&summary_min_2, 3)?;

    assert!(
        clusters_min_2 <= clusters_min_1,
        "expected min_cluster_size=2 to yield no more clusters than min_cluster_size=1 (got {} vs {})",
        clusters_min_2,
        clusters_min_1
    );
    assert_ne!(
        clusters_min_1, clusters_min_2,
        "expected min_cluster_size to influence cluster structure for this synthetic input"
    );
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
    assert_eq!(summary.result.assignments().len(), 4);
    assert!(
        summary.result.cluster_count() >= 1 && summary.result.cluster_count() <= 4,
        "expected 1..=4 clusters for a 4-row input"
    );
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
            path,
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
            .is_some_and(|value| value == "lines.txt")
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
    // The recording layer captures fields via `Debug` formatting, which may
    // include quotes for string fields depending on how the collector records
    // them (observed under `cfg(coverage)` in CI). Accept both representations
    // to keep this assertion stable across environments.
    let expected_message = "command completed";
    let expected_message_debug = format!("{expected_message:?}");
    let expected_data_source = "lines";
    let expected_data_source_debug = format!("{expected_data_source:?}");
    assert!(events.iter().any(|event| {
        let message = event.fields.get("message").map(String::as_str);
        let data_source = event.fields.get("data_source").map(String::as_str);
        event.level == Level::INFO
            && matches!(
                message,
                Some(value) if value == expected_message || value == expected_message_debug
            )
            && matches!(
                data_source,
                Some(value) if value == expected_data_source || value == expected_data_source_debug
            )
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
            .is_some_and(|value| value == "missing.txt")
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
