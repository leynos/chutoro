//! Tests for the `--max-bytes` memory guard and `parse_byte_size` parser.

use super::super::commands::{parse_byte_size, run_command};
use super::super::{Cli, CliError, Command, RunCommand, RunSource, TextArgs, TextMetric};

use chutoro_core::ChutoroError;
use clap::Parser;
use rstest::rstest;

use super::test_helpers::{create_text_file, run_command_expecting_error, temp_dir};

type TestResult = Result<(), Box<dyn std::error::Error>>;

// -- parse_byte_size: happy paths -------------------------------------------

#[rstest]
#[case::plain_bytes("1024", 1024)]
#[case::zero("0", 0)]
#[case::suffix_k_lower("100k", 100 * 1024)]
#[case::suffix_k_upper("100K", 100 * 1024)]
#[case::suffix_kb("100KB", 100 * 1024)]
#[case::suffix_kib("100KiB", 100 * 1024)]
#[case::suffix_m_lower("512m", 512 * 1024 * 1024)]
#[case::suffix_m_upper("512M", 512 * 1024 * 1024)]
#[case::suffix_mb("512MB", 512 * 1024 * 1024)]
#[case::suffix_mib("512MiB", 512 * 1024 * 1024)]
#[case::suffix_g_lower("2g", 2 * 1024 * 1024 * 1024)]
#[case::suffix_g_upper("2G", 2 * 1024 * 1024 * 1024)]
#[case::suffix_gb("2GB", 2 * 1024 * 1024 * 1024)]
#[case::suffix_gib("2GiB", 2 * 1024 * 1024 * 1024)]
#[case::suffix_t("1T", 1024_u64 * 1024 * 1024 * 1024)]
#[case::suffix_tb("1TB", 1024_u64 * 1024 * 1024 * 1024)]
#[case::suffix_tib("1TiB", 1024_u64 * 1024 * 1024 * 1024)]
fn parse_byte_size_accepts_valid_input(#[case] input: &str, #[case] expected: u64) {
    assert_eq!(
        parse_byte_size(input).expect("valid input must parse"),
        expected
    );
}

// -- parse_byte_size: unhappy paths -----------------------------------------

#[rstest]
#[case::empty("")]
#[case::only_suffix("M")]
#[case::unknown_suffix("100X")]
#[case::negative("-100")]
#[case::decimal("1.5G")]
fn parse_byte_size_rejects_invalid_input(#[case] input: &str) {
    assert!(
        parse_byte_size(input).is_err(),
        "expected `{input}` to be rejected"
    );
}

#[rstest]
fn parse_byte_size_rejects_overflow() {
    // 18446744073709551615 TiB would overflow u64.
    assert!(parse_byte_size("18446744073709551615T").is_err());
}

// -- CLI memory guard: integration ------------------------------------------

#[rstest]
fn run_command_rejects_when_max_bytes_exceeded() -> TestResult {
    let dir = temp_dir();
    let path = create_text_file(&dir, "lines.txt", "alpha\nbeta\ngamma\n")?;

    // A limit of 100 bytes is far too small for any real pipeline run.
    let err = run_command_expecting_error(
        RunCommand {
            min_cluster_size: 1,
            max_bytes: Some(100),
            source: RunSource::Text(TextArgs {
                path,
                metric: TextMetric::Levenshtein,
                name: None,
            }),
        },
        "100-byte limit must be exceeded",
    );
    assert!(
        matches!(
            err,
            CliError::Core(ChutoroError::MemoryLimitExceeded { .. })
        ),
        "expected MemoryLimitExceeded, got {err:?}"
    );
    Ok(())
}

#[rstest]
fn run_command_succeeds_when_max_bytes_sufficient() -> TestResult {
    let dir = temp_dir();
    let path = create_text_file(&dir, "lines.txt", "alpha\nbeta\ngamma\n")?;

    // 1 GiB should be more than enough for 3 items.
    let summary = run_command(RunCommand {
        min_cluster_size: 1,
        max_bytes: Some(1_073_741_824),
        source: RunSource::Text(TextArgs {
            path,
            metric: TextMetric::Levenshtein,
            name: None,
        }),
    })?;
    assert_eq!(summary.result.assignments().len(), 3);
    Ok(())
}

#[rstest]
fn run_command_zero_max_bytes_rejects_any_dataset() -> TestResult {
    let dir = temp_dir();
    let path = create_text_file(&dir, "lines.txt", "alpha\nbeta\n")?;
    let err = run_command_expecting_error(
        RunCommand {
            min_cluster_size: 1,
            max_bytes: Some(0),
            source: RunSource::Text(TextArgs {
                path,
                metric: TextMetric::Levenshtein,
                name: None,
            }),
        },
        "zero max_bytes must reject any dataset",
    );
    assert!(matches!(
        err,
        CliError::Core(ChutoroError::MemoryLimitExceeded { .. })
    ));
    Ok(())
}

#[rstest]
fn clap_parses_max_bytes_flag() {
    let args = [
        "chutoro",
        "run",
        "--max-bytes",
        "2G",
        "text",
        "data.txt",
        "--metric",
        "levenshtein",
    ];
    let cli = Cli::try_parse_from(args).expect("valid args must parse");
    match cli.command {
        Command::Run(cmd) => {
            assert_eq!(cmd.max_bytes, Some(2 * 1024 * 1024 * 1024));
        }
    }
}

#[rstest]
fn clap_omits_max_bytes_when_absent() {
    let args = [
        "chutoro",
        "run",
        "text",
        "data.txt",
        "--metric",
        "levenshtein",
    ];
    let cli = Cli::try_parse_from(args).expect("valid args must parse");
    match cli.command {
        Command::Run(cmd) => {
            assert_eq!(cmd.max_bytes, None);
        }
    }
}
