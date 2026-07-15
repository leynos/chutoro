//! Executable smoke checks for Criterion benchmark discovery paths.

use std::{error::Error, io, process::Command, str};

use cap_std::{ambient_authority, fs_utf8::Dir};
use chutoro_benches::neighbour_scoring::{REPORT_DIR_NAME, report_path};

const HNSW_EXACT_BENCH: &str = "hnsw_build/n=100,M=8,ef=16";
const MST_EXACT_BENCH: &str = "parallel_kruskal/n=100";
const LANE_REPORT: &str = "neighbour_scoring_lane_utilisation.csv";

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

fn cargo_bench_output(bench: &str, criterion_args: &[&str]) -> TestResult<String> {
    let mut command = Command::new(env!("CARGO"));
    command.args(["bench", "-p", "chutoro-benches", "--bench", bench, "--"]);
    command.args(criterion_args);
    command.env("CHUTORO_BENCH_HNSW_MEMORY_PROFILE", "0");
    command.env("CHUTORO_BENCH_HNSW_RECALL_REPORT", "0");
    command.env("CHUTORO_BENCH_HNSW_CLUSTER_QUALITY_REPORT", "0");
    command.env("CHUTORO_BENCH_NEIGHBOUR_PROFILE", "0");

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

    assert_list_discovery(
        "neighbour_scoring",
        &[
            "neighbour_scoring/realistic/dim_32_candidates_8: benchmark".to_owned(),
            "neighbour_scoring/diagnostic/dim_768_candidates_1024: benchmark".to_owned(),
        ],
        "Benchmarking neighbour_scoring/realistic/dim_32_candidates_8",
    )?;
    let lane_target = report_path(LANE_REPORT);
    let report_parent =
        Dir::open_ambient_dir(lane_target.report_parent_dir(), ambient_authority())?;
    let report = report_parent
        .open_dir(REPORT_DIR_NAME)?
        .read_to_string(lane_target.filename())?;
    assert_contains(
        &report,
        "bucket_kind,candidate_count,padded_lanes,wasted_lanes,lane_utilisation_basis_points\n",
    );
    assert_contains(&report, "realistic,8,16,8,5000\n");

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
