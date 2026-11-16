//! Capability-scoped report-file writing shared by benchmark report writers.
//!
//! Benchmark report destinations are supplied by the environment and may name
//! directories that do not exist yet. This module owns the single
//! implementation of creating those directories and writing the file, so the
//! recall, clustering-quality, and memory-profile writers agree on UTF-8 path
//! handling and on error behaviour.

use std::io;

use camino::{Utf8Path, Utf8PathBuf};
use cap_std::{ambient_authority, fs_utf8::Dir};

/// Writes `contents` to `report_path`, creating missing parent directories.
///
/// The report path is user-configured, so it is opened through ambient
/// authority. Every other operation stays relative to the opened directory.
///
/// # Errors
///
/// Returns an I/O error when the report path does not name a file, a parent
/// directory cannot be created, or the write fails.
pub(crate) fn write_report_file(report_path: &Utf8Path, contents: &str) -> io::Result<Utf8PathBuf> {
    let file_name = report_path.file_name().ok_or_else(invalid_report_path)?;
    let report_dir = ensure_directory(report_parent(report_path))?;
    report_dir.write(file_name, contents)?;
    Ok(report_path.to_owned())
}

/// Returns the parent directory, treating a bare file name as the current dir.
fn report_parent(report_path: &Utf8Path) -> &Utf8Path {
    report_path
        .parent()
        .map_or_else(|| Utf8Path::new("."), non_empty_path)
}

/// Normalises an empty path component to the current directory.
fn non_empty_path(path: &Utf8Path) -> &Utf8Path {
    if path.as_str().is_empty() {
        Utf8Path::new(".")
    } else {
        path
    }
}

/// Opens `path`, creating it and its ancestors when they are missing.
fn ensure_directory(path: &Utf8Path) -> io::Result<Dir> {
    let normalized = non_empty_path(path);
    match Dir::open_ambient_dir(normalized, ambient_authority()) {
        Ok(dir) => Ok(dir),
        Err(err) if err.kind() == io::ErrorKind::NotFound => create_directory(normalized),
        Err(err) => Err(err),
    }
}

/// Creates one missing directory after ensuring its parent exists.
fn create_directory(path: &Utf8Path) -> io::Result<Dir> {
    let name = path.file_name().ok_or_else(invalid_report_path)?;
    let parent = path
        .parent()
        .map_or_else(|| Utf8Path::new("."), non_empty_path);
    let parent_dir = ensure_directory(parent)?;
    match parent_dir.create_dir(name) {
        Ok(()) => parent_dir.open_dir(name),
        Err(err) if err.kind() == io::ErrorKind::AlreadyExists => parent_dir.open_dir(name),
        Err(err) => Err(err),
    }
}

/// Builds the error used when a report path does not name a file.
fn invalid_report_path() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, "report path must name a file")
}

#[cfg(test)]
mod tests {
    //! Unit tests for capability-scoped report-file writing.

    use camino::Utf8Path;
    use cap_std::{ambient_authority, fs_utf8::Dir};

    use super::write_report_file;

    #[test]
    fn write_report_file_creates_missing_parent_directories() {
        let temp_dir = tempfile::tempdir().expect("tempdir must be created");
        let temp_path = Utf8Path::from_path(temp_dir.path()).expect("tempdir path must be UTF-8");
        let report_path = temp_path.join("nested/reports/rows.csv");

        let written_path =
            write_report_file(&report_path, "point_count\n1\n").expect("report must be written");

        assert_eq!(written_path, report_path);
        let report_dir =
            Dir::open_ambient_dir(temp_path, ambient_authority()).expect("tempdir must open");
        let contents = report_dir
            .read_to_string("nested/reports/rows.csv")
            .expect("report must be readable");
        assert_eq!(contents, "point_count\n1\n");
    }

    #[test]
    fn write_report_file_rejects_a_path_without_a_file_name() {
        let err = write_report_file(Utf8Path::new(""), "data").expect_err("empty path must fail");

        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }
}
