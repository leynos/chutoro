//! Run a validated Kani command, streaming its output while capturing it.
//!
//! Capturing and streaming happen together so a workflow log still shows the
//! verifier's progress while the checker keeps the exact text it must parse. A
//! closed reader is not an error: a log piped into a pager that exits early
//! must not turn a completed verification into a failure.

use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Command, ExitStatus, Stdio};
use std::thread;

use thiserror::Error;

use super::runner::Delegated;

/// A reason the delegated process could not be run.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RunError {
    /// The process could not be started.
    #[error("cannot start {program}: {detail}")]
    Spawn {
        /// The program that failed to start.
        program: String,
        /// The underlying failure.
        detail: String,
    },
    /// The process's streams could not be captured.
    #[error("cannot capture the delegated process's output: {detail}")]
    Capture {
        /// The underlying failure.
        detail: String,
    },
    /// A reader thread failed.
    #[error("the delegated process's {stream} reader failed")]
    Reader {
        /// The stream whose reader failed.
        stream: &'static str,
    },
}

/// What the delegated process did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunOutcome {
    /// The process's exit status.
    pub status: ExitStatus,
    /// Everything the process wrote to standard output.
    pub stdout: String,
}

/// Run a validated delegated command, streaming its output while capturing it.
///
/// Standard output is forwarded to this process's standard output and
/// captured for parsing; standard error is forwarded to standard error. A
/// closed reader is not an error: a log piped into a pager that exits early
/// must not turn a completed verification into a failure.
///
/// # Errors
///
/// Returns [`RunError`] when the process cannot be started or its streams
/// cannot be read.
pub fn run(delegated: &Delegated) -> Result<RunOutcome, RunError> {
    let mut child = Command::new(&delegated.program)
        .args(&delegated.args)
        .envs(delegated.env.iter().map(|(name, value)| (name, value)))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| RunError::Spawn {
            program: delegated.program.clone(),
            detail: error.to_string(),
        })?;
    let stdout = child.stdout.take().ok_or_else(|| RunError::Capture {
        detail: "standard output was not captured".to_owned(),
    })?;
    let stderr = child.stderr.take().ok_or_else(|| RunError::Capture {
        detail: "standard error was not captured".to_owned(),
    })?;
    let captured = thread::spawn(move || pump(stdout, true));
    let forwarded = thread::spawn(move || pump(stderr, false).map(|_| ()));
    let status = child.wait().map_err(|error| RunError::Capture {
        detail: error.to_string(),
    })?;
    let stdout_text = join(captured, "standard output")?;
    forwarded
        .join()
        .map_err(|_| RunError::Reader {
            stream: "standard error",
        })?
        .map_err(|error| RunError::Capture {
            detail: error.to_string(),
        })?;
    Ok(RunOutcome {
        status,
        stdout: stdout_text,
    })
}

/// Join a reader thread, naming the stream it served.
fn join(
    handle: thread::JoinHandle<std::io::Result<String>>,
    stream: &'static str,
) -> Result<String, RunError> {
    handle
        .join()
        .map_err(|_| RunError::Reader { stream })?
        .map_err(|error| RunError::Capture {
            detail: error.to_string(),
        })
}

/// Forward `reader` to one of this process's streams, capturing standard output.
fn pump(source: impl Read, is_stdout: bool) -> std::io::Result<String> {
    let mut reader = BufReader::new(source);
    let mut buffer = Vec::new();
    let mut captured = String::new();
    loop {
        buffer.clear();
        if reader.read_until(b'\n', &mut buffer)? == 0 {
            return Ok(captured);
        }
        let line = String::from_utf8_lossy(&buffer);
        if is_stdout {
            captured.push_str(&line);
            write_line(&mut std::io::stdout().lock(), &line);
        } else {
            write_line(&mut std::io::stderr().lock(), &line);
        }
    }
}

/// Write one forwarded line, ignoring a reader that has already gone away.
fn write_line(sink: &mut impl Write, line: &str) {
    if sink.write_all(line.as_bytes()).is_err() {
        // The consumer closed the stream; the capture still has the run.
    }
    if sink.flush().is_err() {
        // As above: a closed stream must not invalidate a completed run.
    }
}
