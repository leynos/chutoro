//! Behavioural tests for the benchmark regression gate binary.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use rstest::{fixture, rstest};

#[derive(Debug, Clone, Copy)]
struct GateCase {
    event: &'static str,
    policy: Option<&'static str>,
    expected_mode: &'static str,
    expected_should_compare: bool,
}

#[rstest]
#[case::pr_default(GateCase {
    event: "pull_request",
    policy: None,
    expected_mode: "discovery_only",
    expected_should_compare: false,
})]
#[case::scheduled_default(GateCase {
    event: "schedule",
    policy: None,
    expected_mode: "baseline_compare",
    expected_should_compare: true,
})]
#[case::workflow_dispatch_default(GateCase {
    event: "workflow_dispatch",
    policy: None,
    expected_mode: "baseline_compare",
    expected_should_compare: true,
})]
#[case::pr_always_baseline(GateCase {
    event: "pull_request",
    policy: Some("always-baseline"),
    expected_mode: "baseline_compare",
    expected_should_compare: true,
})]
#[case::schedule_disabled(GateCase {
    event: "schedule",
    policy: Some("disabled"),
    expected_mode: "disabled",
    expected_should_compare: false,
})]
#[case::invalid_policy_falls_back_to_default(GateCase {
    event: "pull_request",
    policy: Some("invalid-policy"),
    expected_mode: "discovery_only",
    expected_should_compare: false,
})]
fn benchmark_gate_binary_outputs_expected_mode(
    gate_runner: std::io::Result<GateRunner>,
    #[case] case: GateCase,
) {
    let runner = gate_runner.expect("gate runner must resolve the gate binary");
    let output = runner
        .run(case.event, case.policy)
        .expect("benchmark_regression_gate binary must run");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("expected success, got failure: {stderr}");
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let Some(mode) = parse_value(&stdout, "mode") else {
        panic!("missing mode output: {stdout}");
    };
    let should_compare = match parse_value(&stdout, "should_compare") {
        Some("true") => true,
        Some("false") => false,
        Some(other) => panic!("unexpected should_compare value: {other}"),
        None => panic!("missing should_compare output: {stdout}"),
    };

    assert_eq!(mode, case.expected_mode);
    assert_eq!(should_compare, case.expected_should_compare);
}

#[rstest]
#[case("schedule", "schedule")]
#[case("workflow_dispatch", "workflow_dispatch")]
#[case("pull_request_target", "pull_request")]
#[case("push", "other")]
fn benchmark_gate_binary_reports_event(
    gate_runner: std::io::Result<GateRunner>,
    #[case] event: &str,
    #[case] expected_event: &str,
) {
    let runner = gate_runner.expect("gate runner must resolve the gate binary");
    let output = runner
        .run(event, None)
        .expect("benchmark_regression_gate binary must run");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("expected success, got failure: {stderr}");
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let printed_event = parse_value(&stdout, "event").unwrap_or("<missing>");
    assert_eq!(printed_event, expected_event);
}

/// A default invocation must identify the gate binary by its primary key.
///
/// This case uses no policy override, so it holds the identity of the
/// executable and its output contract independently of the policy table. A
/// test harness run in place of the gate would exit successfully without
/// printing `mode=`.
#[rstest]
fn benchmark_gate_binary_writes_its_primary_output(gate_runner: std::io::Result<GateRunner>) {
    let runner = gate_runner.expect("gate runner must resolve the gate binary");
    let output = runner
        .run("pull_request", None)
        .expect("benchmark_regression_gate binary must run");
    assert!(
        output.status.success(),
        "expected success, got failure: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        parse_value(&stdout, "mode").is_some(),
        "a default invocation must print mode=: {stdout}"
    );
}

#[fixture]
fn gate_runner() -> std::io::Result<GateRunner> {
    GateRunner::new()
}

struct GateRunner {
    binary_path: PathBuf,
}

impl GateRunner {
    /// Point at the gate binary Cargo built for this integration test.
    ///
    /// Cargo guarantees `CARGO_BIN_EXE_<name>` at compile time for a binary
    /// in the same package, whatever the target or build-directory layout.
    /// The path is checked here so a build that somehow embedded the wrong
    /// artefact fails before the gate's own assertions can misread it.
    ///
    /// # Errors
    ///
    /// Returns [`std::io::ErrorKind::NotFound`] when the compile-time path
    /// does not name an existing file.
    fn new() -> std::io::Result<Self> {
        let binary_path = PathBuf::from(env!("CARGO_BIN_EXE_benchmark_regression_gate"));
        if !binary_path.is_file() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "benchmark_regression_gate binary is not a file: {}",
                    binary_path.display()
                ),
            ));
        }
        Ok(Self { binary_path })
    }

    fn run(&self, event: &str, policy: Option<&str>) -> std::io::Result<Output> {
        let mut command = Command::new(&self.binary_path);
        command
            .env("GITHUB_EVENT_NAME", event)
            .env_remove("GITHUB_OUTPUT")
            .env_remove("CHUTORO_BENCH_CI_POLICY");

        if let Some(value) = policy {
            command.env("CHUTORO_BENCH_CI_POLICY", value);
        }

        let output = command.output()?;
        reject_test_harness(&self.binary_path, &output)?;
        Ok(output)
    }
}

/// Reject a successful process that was a test harness rather than the gate.
///
/// A libtest harness run in place of the gate prints `running 0 tests` and
/// exits successfully, so a caller would otherwise read a successful process
/// with empty output. The gate never prints that line: every successful
/// invocation writes at least one `key=value` line to stdout, and a rejected
/// input writes its diagnosis to stderr. Checking the pair keeps a legitimate
/// failure path from being mistaken for a harness.
///
/// # Errors
///
/// Returns [`std::io::ErrorKind::InvalidData`] when the output carries the
/// harness marker and no gate key.
fn reject_test_harness(binary_path: &Path, output: &Output) -> std::io::Result<()> {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let printed_key_value = stdout.lines().any(|line| {
        line.split_once('=')
            .is_some_and(|(key, _)| !key.trim().is_empty())
    });
    let ran_harness = stdout.contains("running 0 tests") || stderr.contains("running 0 tests");
    if printed_key_value || !ran_harness {
        return Ok(());
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!(
            concat!(
                "a test harness ran instead of the gate binary at ",
                "{}: stdout={:?}, stderr={:?}"
            ),
            binary_path.display(),
            stdout,
            stderr
        ),
    ))
}

fn parse_value<'a>(stdout: &'a str, key: &str) -> Option<&'a str> {
    let prefix = format!("{key}=");
    for line in stdout.lines() {
        if let Some(value) = line.strip_prefix(&prefix) {
            return Some(value.trim());
        }
    }

    None
}
