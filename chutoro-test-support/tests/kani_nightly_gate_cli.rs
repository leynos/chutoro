//! Behavioural tests for the nightly Kani gate binary.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use rstest::{fixture, rstest};

const SECONDS_PER_DAY: u64 = 86_400;

#[derive(Debug, Clone, Copy)]
struct GateCase {
    commit_epoch: u64,
    now_epoch: u64,
    force: Option<&'static str>,
    expected_run: bool,
}

#[rstest]
#[case::same_timestamp(GateCase {
    commit_epoch: SECONDS_PER_DAY,
    now_epoch: SECONDS_PER_DAY,
    force: None,
    expected_run: true,
})]
#[case::within_window(GateCase {
    commit_epoch: SECONDS_PER_DAY,
    now_epoch: SECONDS_PER_DAY * 2,
    force: None,
    expected_run: true,
})]
#[case::stale_commit(GateCase {
    commit_epoch: SECONDS_PER_DAY - 1,
    now_epoch: SECONDS_PER_DAY * 2 + 1,
    force: None,
    expected_run: false,
})]
#[case::small_future_skew(GateCase {
    commit_epoch: SECONDS_PER_DAY + 10,
    now_epoch: SECONDS_PER_DAY + 9,
    force: None,
    expected_run: false,
})]
#[case::force_override(GateCase {
    commit_epoch: SECONDS_PER_DAY + 10,
    now_epoch: SECONDS_PER_DAY + 9,
    force: Some("true"),
    expected_run: true,
})]
fn kani_gate_binary_outputs_decision(
    gate_runner: std::io::Result<GateRunner>,
    #[case] case: GateCase,
) {
    let runner = gate_runner.expect("gate runner must resolve the gate binary");
    let output = runner
        .run(case.commit_epoch, case.now_epoch, case.force)
        .expect("kani_nightly_gate binary must run");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("expected success, got failure: {stderr}");
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let Some(should_run) = parse_should_run(&stdout) else {
        panic!("missing should_run output: {stdout}");
    };

    assert_eq!(should_run, case.expected_run);
}

#[rstest]
#[case::invalid_force("maybe")]
#[case::invalid_force_whitespace("true-ish")]
fn kani_gate_binary_rejects_invalid_force(
    gate_runner: std::io::Result<GateRunner>,
    #[case] force_value: &str,
) {
    let runner = gate_runner.expect("gate runner must resolve the gate binary");
    let output = runner
        .run(SECONDS_PER_DAY, SECONDS_PER_DAY, Some(force_value))
        .expect("kani_nightly_gate binary must run");
    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        panic!("expected failure, got success: {stdout}");
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("invalid boolean"));
}

/// A default invocation must identify the gate binary by its primary key.
///
/// The case sets both timestamps to the same value and passes no override, so
/// it holds the identity of the executable and its output contract
/// independently of the policy table. A test harness run in place of the gate
/// would exit successfully without printing `should_run=`.
#[rstest]
fn kani_gate_binary_writes_its_primary_output(gate_runner: std::io::Result<GateRunner>) {
    let runner = gate_runner.expect("gate runner must resolve the gate binary");
    let output = runner
        .run(SECONDS_PER_DAY, SECONDS_PER_DAY, None)
        .expect("kani_nightly_gate binary must run");
    assert!(
        output.status.success(),
        "expected success, got failure: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        parse_should_run(&stdout).is_some(),
        "a default invocation must print should_run=: {stdout}"
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
        let binary_path = PathBuf::from(env!("CARGO_BIN_EXE_kani_nightly_gate"));
        if !binary_path.is_file() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "kani_nightly_gate binary is not a file: {}",
                    binary_path.display()
                ),
            ));
        }
        Ok(Self { binary_path })
    }

    fn run(
        &self,
        commit_epoch: u64,
        now_epoch: u64,
        force: Option<&str>,
    ) -> std::io::Result<Output> {
        let mut command = Command::new(&self.binary_path);
        command
            .env("CHUTORO_KANI_COMMIT_EPOCH", commit_epoch.to_string())
            .env("CHUTORO_KANI_NOW_EPOCH", now_epoch.to_string())
            .env_remove("GITHUB_OUTPUT");

        match force {
            Some(value) => {
                command.env("CHUTORO_KANI_FORCE", value);
            }
            None => {
                command.env_remove("CHUTORO_KANI_FORCE");
            }
        }

        let output = command.output()?;
        reject_test_harness(&self.binary_path, &output.stdout, &output.stderr)?;
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
fn reject_test_harness(
    binary_path: &Path,
    stdout_bytes: &[u8],
    stderr_bytes: &[u8],
) -> std::io::Result<()> {
    let stdout = String::from_utf8_lossy(stdout_bytes);
    let stderr = String::from_utf8_lossy(stderr_bytes);
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

#[rstest]
#[case::stdout_harness(b"running 0 tests\n", b"", false)]
#[case::stderr_harness(b"", b"running 0 tests\n", false)]
#[case::gate_key_wins(b"should_run=true\n", b"running 0 tests\n", true)]
#[case::gate_key_without_harness(b"should_run=true\n", b"", true)]
fn kani_gate_rejects_a_substituted_harness(
    #[case] stdout: &[u8],
    #[case] stderr: &[u8],
    #[case] is_gate_output: bool,
) {
    let result = reject_test_harness(Path::new("kani_nightly_gate"), stdout, stderr);
    if is_gate_output {
        result.expect("gate output must be accepted");
        return;
    }

    let error = result.expect_err("harness output must be rejected");
    let message = error.to_string();
    assert!(
        message.contains("a test harness ran instead of the gate binary"),
        "unexpected diagnostic: {message}"
    );
    assert!(
        message.contains("kani_nightly_gate"),
        "the diagnostic must name the binary: {message}"
    );
}

fn parse_should_run(stdout: &str) -> Option<bool> {
    for line in stdout.lines() {
        if let Some(value) = line.strip_prefix("should_run=") {
            return match value.trim() {
                "true" => Some(true),
                "false" => Some(false),
                _ => None,
            };
        }
    }

    None
}
