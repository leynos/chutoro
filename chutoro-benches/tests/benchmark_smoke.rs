//! Executable smoke checks for Criterion benchmark discovery paths.

use std::{error::Error, io, process::Command, str};

const HNSW_EXACT_BENCH: &str = "hnsw_build/n=100,M=8,ef=16";
const MST_EXACT_BENCH: &str = "parallel_kruskal/n=100";

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

fn cargo_bench_output(bench: &str, criterion_args: &[&str]) -> TestResult<String> {
    let mut command = Command::new(env!("CARGO"));
    command.args(["bench", "-p", "chutoro-benches", "--bench", bench, "--"]);
    command.args(criterion_args);
    command.env("CHUTORO_BENCH_HNSW_MEMORY_PROFILE", "0");
    command.env("CHUTORO_BENCH_HNSW_RECALL_REPORT", "0");
    command.env("CHUTORO_BENCH_HNSW_CLUSTER_QUALITY_REPORT", "0");

    let output = command.output()?;
    let stdout = str::from_utf8(&output.stdout)?;
    let stderr = str::from_utf8(&output.stderr)?;
    let combined_output = format!("{stdout}{stderr}");

    if !output.status.success() {
        return Err(io::Error::other(format!(
            "cargo bench for {bench} failed:\n{combined_output}"
        ))
        .into());
    }

    Ok(combined_output)
}

fn assert_contains(haystack: &str, needle: &str) {
    assert!(
        haystack.contains(needle),
        "expected output to contain {needle:?}, got:\n{haystack}"
    );
}

fn assert_absent(haystack: &str, needle: &str) {
    assert!(
        !haystack.contains(needle),
        "expected output not to contain {needle:?}, got:\n{haystack}"
    );
}

fn assert_list_discovery(
    bench: &str,
    expected_entries: &[String],
    absent_benchmark: &str,
) -> TestResult {
    let output = cargo_bench_output(bench, &["--list"])?;
    for entry in expected_entries {
        assert_contains(&output, entry);
    }
    assert_absent(&output, absent_benchmark);

    Ok(())
}

#[test]
fn benchmark_binaries_cover_discovery_and_exact_smoke_paths() -> TestResult {
    assert_list_discovery(
        "hnsw",
        &[
            format!("{HNSW_EXACT_BENCH}: benchmark"),
            "hnsw_build_with_edges/n=5000,M=24,ef=48: benchmark".to_owned(),
        ],
        &format!("Benchmarking {HNSW_EXACT_BENCH}"),
    )?;

    assert_list_discovery(
        "mst",
        &[
            format!("{MST_EXACT_BENCH}: benchmark"),
            "parallel_kruskal/n=1000: benchmark".to_owned(),
        ],
        &format!("Benchmarking {MST_EXACT_BENCH}"),
    )?;

    let hnsw_exact_output = cargo_bench_output("hnsw", &[HNSW_EXACT_BENCH, "--exact"])?;
    assert_contains(
        &hnsw_exact_output,
        &format!("Benchmarking {HNSW_EXACT_BENCH}"),
    );
    assert_contains(&hnsw_exact_output, "time:");

    let mst_exact_output = cargo_bench_output("mst", &[MST_EXACT_BENCH, "--exact"])?;
    assert_contains(
        &mst_exact_output,
        &format!("Benchmarking {MST_EXACT_BENCH}"),
    );
    assert_contains(&mst_exact_output, "time:");

    Ok(())
}
