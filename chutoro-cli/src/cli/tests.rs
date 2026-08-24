//! Unit tests for the CLI commands and data ingestion helpers.

use super::commands::{derive_data_source_name, run_command};
use super::{
    Cli, CliError, Command, ExecutionSummary, ParquetArgs, RunCommand, RunSource, render_summary,
    run_cli,
};

use std::io;
use std::path::Path;

use chutoro_core::{ChutoroError, ClusteringResult};
use clap::Parser;
use rstest::rstest;
use tempfile::TempDir;
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
use test_helpers::{create_text_file, expect_err, temp_dir, text_cli, text_run_command};

/// Runs the text pipeline once with the provided input file and minimum
/// cluster size.
///
/// Returns the [`ExecutionSummary`] produced by the CLI runner.
fn run_text_once(path: &Path, min_cluster_size: usize) -> Result<ExecutionSummary, CliError> {
    run_cli(text_cli(path.to_path_buf(), min_cluster_size, None))
}

/// Observed assignment and cluster counts for a completed run.
///
/// Reported as a pure query so the calling test owns every assertion.
fn summary_shape(summary: &ExecutionSummary) -> (usize, usize) {
    (
        summary.result.assignments().len(),
        summary.result.cluster_count(),
    )
}

/// Asserts that a text run produced a clustering with the expected number of
/// assignments, and evaluates to the observed cluster count.
///
/// This keeps the tests robust by checking invariants that should hold across
/// implementations without relying on exact label ids. It is a macro so a
/// failure reports the calling test's line.
macro_rules! assert_run_summary {
    ($summary:expr, $expected_items:expr $(,)?) => {{
        let expected_items = $expected_items;
        let (assignments, clusters) = summary_shape(&$summary);
        assert_eq!(assignments, expected_items);
        assert!(
            (1..=expected_items).contains(&clusters),
            "expected 1..={expected_items} clusters for a {expected_items}-row input",
        );
        clusters
    }};
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

#[rstest]
fn run_text_success(#[from(temp_dir)] temp_dir_result: io::Result<TempDir>) {
    let dir = temp_dir_result.expect("temp dir should be created");
    let path = create_text_file(&dir, "lines.txt", "alpha\nbeta\ngamma\n")
        .expect("text fixture must be written");

    let summary_min_1 = run_text_once(path.as_path(), 1).expect("run must succeed");
    let clusters_min_1 = assert_run_summary!(summary_min_1, 3);

    let summary_min_2 = run_text_once(path.as_path(), 2).expect("run must succeed");
    let clusters_min_2 = assert_run_summary!(summary_min_2, 3);

    assert!(
        clusters_min_2 <= clusters_min_1,
        "expected min_cluster_size=2 to yield no more clusters than min_cluster_size=1 (got {clusters_min_2} vs {clusters_min_1})",
    );
    assert_ne!(
        clusters_min_1, clusters_min_2,
        "expected min_cluster_size to influence cluster structure for this synthetic input"
    );
}

#[rstest]
fn run_text_rejects_insufficient_items(#[from(temp_dir)] temp_dir_result: io::Result<TempDir>) {
    let dir = temp_dir_result.expect("temp dir should be created");
    let path =
        create_text_file(&dir, "lines.txt", "alpha\nbeta\n").expect("text fixture must be written");
    let err = expect_err!(
        run_cli(text_cli(path, 3, None)),
        "run must fail for insufficient items"
    );
    assert!(matches!(
        err,
        CliError::Core(ChutoroError::InsufficientItems { .. })
    ));
}

#[rstest]
fn run_text_rejects_empty_files(#[from(temp_dir)] temp_dir_result: io::Result<TempDir>) {
    let dir = temp_dir_result.expect("temp dir should be created");
    let path = create_text_file(&dir, "empty.txt", "").expect("text fixture must be written");
    let err = expect_err!(run_cli(text_cli(path, 1, None)), "empty input must fail");
    assert!(matches!(err, CliError::Text(TextProviderError::EmptyInput)));
}

#[rstest]
fn run_parquet_success(#[from(temp_dir)] temp_dir_result: io::Result<TempDir>) {
    let dir = temp_dir_result.expect("temp dir should be created");
    let path = create_parquet_file(&dir, "vectors.parquet").expect("parquet fixture must be built");
    let cli = Cli {
        command: Command::Run(RunCommand {
            min_cluster_size: 2,
            max_bytes: None,
            source: RunSource::Parquet(ParquetArgs {
                path,
                column: "features".into(),
                name: Some("parquet".into()),
            }),
        }),
    };
    let summary = run_cli(cli).expect("parquet run must succeed");
    let _ = assert_run_summary!(summary, 4);
}

#[rstest]
fn run_parquet_rejects_missing_column(#[from(temp_dir)] temp_dir_result: io::Result<TempDir>) {
    let dir = temp_dir_result.expect("temp dir should be created");
    let path = create_parquet_file(&dir, "vectors.parquet").expect("parquet fixture must be built");
    let cli = Cli {
        command: Command::Run(RunCommand {
            min_cluster_size: 1,
            max_bytes: None,
            source: RunSource::Parquet(ParquetArgs {
                path,
                column: "unknown".into(),
                name: None,
            }),
        }),
    };
    let err = expect_err!(run_cli(cli), "unknown column must fail");
    assert!(matches!(
        err,
        CliError::Dense(DenseMatrixProviderError::ColumnNotFound { .. })
    ));
}

#[rstest]
fn run_command_rejects_zero_min_cluster_size(
    #[from(temp_dir)] temp_dir_result: io::Result<TempDir>,
) {
    let dir = temp_dir_result.expect("temp dir should be created");
    let path = create_text_file(&dir, "lines.txt", "alpha\nbeta\ngamma\n")
        .expect("text fixture must be written");
    let err = expect_err!(
        run_command(text_run_command(path, 0, None)),
        "zero min-cluster-size must fail"
    );
    assert!(matches!(
        err,
        CliError::Core(ChutoroError::InvalidMinClusterSize { .. })
    ));
}

#[rstest]
fn render_summary_outputs_assignments() {
    let summary = ExecutionSummary {
        data_source: "demo".into(),
        result: ClusteringResult::try_from_assignments(vec![
            chutoro_core::ClusterId::new(0),
            chutoro_core::ClusterId::new(1),
        ])?,
    };
    let mut buffer = Vec::new();
    render_summary(&summary, &mut buffer).expect("rendering must succeed");
    let text = String::from_utf8(buffer).expect("rendered summary must be UTF-8");
    assert!(text.contains("data source: demo"));
    assert!(text.contains("clusters: 2"));
    assert!(text.contains("0\t0"));
    assert!(text.contains("1\t1"));
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
fn run_command_emits_tracing_fields(#[from(temp_dir)] temp_dir_result: io::Result<TempDir>) {
    let dir = temp_dir_result.expect("temp dir should be created");
    let path = create_text_file(&dir, "lines.txt", "alpha\nbeta\ngamma\n")
        .expect("text fixture must be written");
    let layer = RecordingLayer::default();
    let subscriber = tracing_subscriber::registry().with(layer.clone());

    let command = text_run_command(path, 2, None);

    let summary = tracing::subscriber::with_default(subscriber, || run_command(command))
        .expect("run must succeed");
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
}

#[rstest]
fn open_text_reader_records_path_on_error(#[from(temp_dir)] temp_dir_result: io::Result<TempDir>) {
    let dir = temp_dir_result.expect("temp dir should be created");
    let missing_path = dir.path().join("missing.txt");
    let layer = RecordingLayer::default();
    let subscriber = tracing_subscriber::registry().with(layer.clone());

    let command = text_run_command(missing_path, 1, None);

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
}

#[path = "test_memory_guard.rs"]
mod test_memory_guard;
