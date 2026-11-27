//! A reader stub that fails every operation, shared by the text provider tests.
//!
//! The unit tests exercise the private `TextProvider::read_lines`, while the
//! integration tests drive the public `TextProvider::try_from_reader`. Both
//! need the same failure injection, and neither may cross the other's test
//! boundary, so the stub is included by `#[path]` rather than published from
//! the crate or added to `chutoro-test-support`.

use std::io::{self, BufRead, Read};

/// Reader whose every operation reports the same injected I/O failure.
pub(super) struct FailingReader;

impl Read for FailingReader {
    /// Reports the injected I/O failure instead of filling the buffer.
    fn read(&mut self, _buf: &mut [u8]) -> io::Result<usize> {
        Err(io::Error::other("boom"))
    }
}

impl BufRead for FailingReader {
    /// Reports the injected I/O failure instead of exposing buffered bytes.
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        Err(io::Error::other("boom"))
    }

    /// Accepts the consumed byte count without changing any state.
    fn consume(&mut self, _amt: usize) {}

    /// Reports the injected I/O failure instead of appending a line.
    ///
    /// `TextProvider::read_lines` calls this method directly, so the override
    /// keeps the failure on the path under test rather than relying on the
    /// default implementation's `fill_buf` call.
    fn read_line(&mut self, _buf: &mut String) -> io::Result<usize> {
        Err(io::Error::other("boom"))
    }
}
