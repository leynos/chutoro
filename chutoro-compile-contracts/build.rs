//! Detect whether compile-contract tests are built with nightly Rust.

use std::{
    ffi::{OsStr, OsString},
    io,
    str::Utf8Error,
};

#[cfg(not(test))]
use std::{error::Error, fmt, io::Write, process::Command};

use mockable::{DefaultEnv, Env};

/// Emits Cargo configuration for the compiler driving compile-contract tests.
#[cfg(not(test))]
fn main() -> Result<(), Box<dyn Error>> {
    emit_cargo_directive("cargo:rerun-if-changed=build.rs")?;
    emit_cargo_directive("cargo:rerun-if-env-changed=RUSTC")?;
    emit_cargo_directive("cargo:rustc-check-cfg=cfg(nightly)")?;

    if detect_compiler_channel(&DefaultEnv, &ProcessCompilerVersionProbe)?
        == CompilerChannel::Nightly
    {
        emit_cargo_directive("cargo:rustc-cfg=nightly")?;
    }
    Ok(())
}

/// Writes one build-script directive for Cargo to consume.
#[cfg(not(test))]
fn emit_cargo_directive(directive: &str) -> io::Result<()> {
    writeln!(io::stdout().lock(), "{directive}")
}

/// Distinguishes nightly compilers from other successful compiler probes.
#[derive(Debug, PartialEq, Eq)]
enum CompilerChannel {
    /// The compiler identifies its release channel as nightly.
    Nightly,
    /// The compiler completed the probe but did not identify as nightly.
    Other,
}

/// Describes output captured from one successful compiler process launch.
#[derive(Clone, Debug)]
struct CompilerVersionOutput {
    /// Whether the compiler process exited successfully.
    succeeded: bool,
    /// The exit status code, when the operating system supplies one.
    status_code: Option<i32>,
    /// Bytes written by the compiler to standard output.
    stdout: Vec<u8>,
}

/// Runs the one compiler-version command required by this build script.
trait CompilerVersionProbe {
    /// Runs `<rustc> --version` and returns its process outcome.
    fn probe_version(&self, rustc: &OsStr) -> io::Result<CompilerVersionOutput>;
}

/// Executes the compiler-version probe with the host operating system.
#[cfg(not(test))]
struct ProcessCompilerVersionProbe;

#[cfg(not(test))]
impl CompilerVersionProbe for ProcessCompilerVersionProbe {
    /// Runs the configured compiler with its version flag.
    fn probe_version(&self, rustc: &OsStr) -> io::Result<CompilerVersionOutput> {
        let output = Command::new(rustc).arg("--version").output()?;
        Ok(CompilerVersionOutput {
            succeeded: output.status.success(),
            status_code: output.status.code(),
            stdout: output.stdout,
        })
    }
}

/// Explains why the compiler-version probe could not determine a channel.
#[derive(Debug)]
enum CompilerProbeError {
    /// The operating system could not launch the configured compiler.
    Launch(io::Error),
    /// The compiler ran but returned an unsuccessful status.
    UnsuccessfulExit {
        /// Status code supplied by the operating system, if any.
        status_code: Option<i32>,
    },
    /// The compiler wrote standard output that was not valid UTF-8.
    InvalidOutput(Utf8Error),
}

#[cfg(not(test))]
impl fmt::Display for CompilerProbeError {
    /// Formats the probe failure so Cargo reports the underlying cause.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Launch(error) => write!(formatter, "could not launch Rust compiler: {error}"),
            Self::UnsuccessfulExit { status_code } => write!(
                formatter,
                "Rust compiler exited unsuccessfully with status code {status_code:?}"
            ),
            Self::InvalidOutput(error) => {
                write!(
                    formatter,
                    "Rust compiler version output was not UTF-8: {error}"
                )
            }
        }
    }
}

#[cfg(not(test))]
impl Error for CompilerProbeError {
    /// Returns the operating-system or decoding error that caused this failure.
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Launch(error) => Some(error),
            Self::UnsuccessfulExit { .. } => None,
            Self::InvalidOutput(error) => Some(error),
        }
    }
}

/// Resolves the compiler command from the environment, preserving Cargo's default.
fn compiler_command(env: &dyn Env) -> OsString {
    env.os_string("RUSTC")
        .unwrap_or_else(|| OsString::from("rustc"))
}

/// Detects the configured compiler channel with a separately injectable probe.
fn detect_compiler_channel(
    env: &dyn Env,
    probe: &dyn CompilerVersionProbe,
) -> Result<CompilerChannel, CompilerProbeError> {
    let output = probe
        .probe_version(&compiler_command(env))
        .map_err(CompilerProbeError::Launch)?;
    if !output.succeeded {
        return Err(CompilerProbeError::UnsuccessfulExit {
            status_code: output.status_code,
        });
    }

    let version = std::str::from_utf8(&output.stdout).map_err(CompilerProbeError::InvalidOutput)?;
    Ok(if version.contains("nightly") {
        CompilerChannel::Nightly
    } else {
        CompilerChannel::Other
    })
}

#[cfg(test)]
mod tests {
    //! Unit tests for the compiler-channel probe's success and failure boundary.

    use super::*;

    /// Supplies a deterministic compiler outcome without launching a process.
    #[derive(Debug)]
    struct StubCompilerVersionProbe {
        /// Outcome returned whenever the version probe is requested.
        result: StubProbeResult,
    }

    /// Represents the two outcomes needed by the test probe.
    #[derive(Debug)]
    enum StubProbeResult {
        /// A compiler process outcome supplied to the detection logic.
        Output(CompilerVersionOutput),
        /// An operating-system launch failure synthesized for the test.
        LaunchFailure,
    }

    impl CompilerVersionProbe for StubCompilerVersionProbe {
        /// Returns the configured outcome without inspecting the compiler path.
        fn probe_version(&self, _: &OsStr) -> io::Result<CompilerVersionOutput> {
            match &self.result {
                StubProbeResult::Output(output) => Ok(output.clone()),
                StubProbeResult::LaunchFailure => Err(io::Error::other("compiler unavailable")),
            }
        }
    }

    /// Builds a successful process outcome with the supplied compiler output.
    fn successful_output(stdout: Vec<u8>) -> CompilerVersionOutput {
        CompilerVersionOutput {
            succeeded: true,
            status_code: Some(0),
            stdout,
        }
    }

    /// Detects nightly output without invoking the configured compiler.
    #[test]
    fn nightly_output_selects_nightly_channel() {
        let probe = StubCompilerVersionProbe {
            result: StubProbeResult::Output(successful_output(
                b"rustc 1.92.0-nightly (abc 2026-09-20)".to_vec(),
            )),
        };

        assert_eq!(
            detect_compiler_channel(&DefaultEnv, &probe).expect("nightly output must decode"),
            CompilerChannel::Nightly
        );
    }

    /// Detects stable output without invoking the configured compiler.
    #[test]
    fn stable_output_selects_other_channel() {
        let probe = StubCompilerVersionProbe {
            result: StubProbeResult::Output(successful_output(b"rustc 1.91.0".to_vec())),
        };

        assert_eq!(
            detect_compiler_channel(&DefaultEnv, &probe).expect("stable output must decode"),
            CompilerChannel::Other
        );
    }

    /// Preserves an operating-system launch failure for the build-script caller.
    #[test]
    fn launch_failure_returns_error() {
        let probe = StubCompilerVersionProbe {
            result: StubProbeResult::LaunchFailure,
        };

        let probe_error = detect_compiler_channel(&DefaultEnv, &probe)
            .expect_err("an unavailable compiler must remain a launch failure");
        match probe_error {
            CompilerProbeError::Launch(launch_error) => {
                assert_eq!(launch_error.kind(), io::ErrorKind::Other);
            }
            other => panic!("expected launch failure, got {other:?}"),
        }
    }

    /// Rejects a compiler that exits unsuccessfully instead of treating it as stable.
    #[test]
    fn unsuccessful_exit_returns_error() {
        let probe = StubCompilerVersionProbe {
            result: StubProbeResult::Output(CompilerVersionOutput {
                succeeded: false,
                status_code: Some(1),
                stdout: Vec::new(),
            }),
        };

        assert!(matches!(
            detect_compiler_channel(&DefaultEnv, &probe),
            Err(CompilerProbeError::UnsuccessfulExit {
                status_code: Some(1)
            })
        ));
    }

    /// Rejects non-UTF-8 compiler output instead of treating it as stable.
    #[test]
    fn invalid_utf8_output_returns_error() {
        let probe = StubCompilerVersionProbe {
            result: StubProbeResult::Output(successful_output(vec![0xFF])),
        };

        let probe_error = detect_compiler_channel(&DefaultEnv, &probe)
            .expect_err("non-UTF-8 output must not select a compiler channel");
        match probe_error {
            CompilerProbeError::InvalidOutput(utf8_error) => {
                assert_eq!(utf8_error.valid_up_to(), 0);
            }
            other => panic!("expected invalid output, got {other:?}"),
        }
    }
}
