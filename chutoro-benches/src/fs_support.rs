//! Capability-scoped filesystem helpers for benchmark data and reports.
//!
//! Paths are validated before use and then resolved by descending from a fixed
//! capability root: `.` for relative paths and `/` for absolute paths. The
//! helpers reject parent-directory traversal so callers cannot escape the
//! chosen root while still allowing benchmark fixtures to use either
//! repository-relative or absolute scratch paths.

use std::ffi::OsStr;
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};

use cap_std::{
    ambient_authority,
    fs::{Dir, OpenOptions},
};

pub(crate) fn read(path: &Path) -> io::Result<Vec<u8>> {
    let (dir, file_name) = open_parent_dir(path, false)?;
    let mut file = dir.open(file_name)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    Ok(bytes)
}

pub(crate) fn read_proc_status() -> io::Result<String> {
    let dir = Dir::open_ambient_dir("/proc/self", ambient_authority())?;
    dir.read_to_string("status")
}

#[cfg(test)]
pub(crate) fn read_to_string(path: &Path) -> io::Result<String> {
    let (dir, file_name) = open_parent_dir(path, false)?;
    dir.read_to_string(file_name)
}

#[cfg(test)]
pub(crate) fn remove_dir_all(path: &Path) -> io::Result<()> {
    let (dir, name) = open_parent_dir(path, false)?;
    dir.remove_dir_all(name)
}

pub(crate) fn remove_file(path: &Path) -> io::Result<()> {
    let (dir, name) = open_parent_dir(path, false)?;
    dir.remove_file(name)
}

pub(crate) fn rename(from: &Path, to: &Path) -> io::Result<()> {
    let (from_dir, from_name) = open_parent_dir(from, false)?;
    let (to_dir, to_name) = open_parent_dir(to, true)?;
    from_dir.rename(from_name, &to_dir, to_name)
}

pub(crate) fn try_exists(path: &Path) -> io::Result<bool> {
    validate_scoped_path(path)?;
    let Some(file_name) = path.file_name() else {
        return Err(invalid_path(path));
    };
    let Some(dir) = open_existing_parent_dir(path)? else {
        return Ok(false);
    };
    dir.try_exists(file_name)
}

pub(crate) fn write(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let (dir, file_name) = open_parent_dir(path, true)?;
    let mut options = OpenOptions::new();
    options.write(true).create(true).truncate(true);
    let mut file = dir.open_with(file_name, &options)?;
    file.write_all(bytes)
}

pub(crate) fn write_string(path: &Path, output: &str) -> io::Result<PathBuf> {
    write(path, output.as_bytes())?;
    Ok(path.to_path_buf())
}

fn invalid_path(path: &Path) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        format!(
            "path must name a file or directory entry: {}",
            path.display()
        ),
    )
}

fn invalid_scoped_path(path: &Path) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        format!(
            "path must not contain parent-directory traversal components: {}",
            path.display()
        ),
    )
}

fn validate_scoped_path(path: &Path) -> io::Result<()> {
    if path
        .components()
        .any(|component| matches!(component, Component::ParentDir | Component::Prefix(_)))
    {
        return Err(invalid_scoped_path(path));
    }
    Ok(())
}

fn split_scope(path: &Path) -> io::Result<(Dir, PathBuf)> {
    validate_scoped_path(path)?;
    let root_path = if path.is_absolute() {
        Path::new("/")
    } else {
        Path::new(".")
    };
    let root_dir = Dir::open_ambient_dir(root_path, ambient_authority())?;
    let mut relative = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => relative.push(part),
            Component::CurDir
            | Component::RootDir
            | Component::ParentDir
            | Component::Prefix(_) => {}
        }
    }
    Ok((root_dir, relative))
}

fn open_scoped_dir(root_dir: Dir, relative_path: &Path, create_dir: bool) -> io::Result<Dir> {
    if relative_path.as_os_str().is_empty() {
        return Ok(root_dir);
    }

    if create_dir {
        root_dir.create_dir_all(relative_path)?;
    }

    root_dir.open_dir(relative_path)
}

fn open_existing_parent_dir(path: &Path) -> io::Result<Option<Dir>> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let (root_dir, relative_parent) = split_scope(parent)?;
    if relative_parent.as_os_str().is_empty() {
        return Ok(Some(root_dir));
    }

    match root_dir.open_dir(&relative_parent) {
        Ok(dir) => Ok(Some(dir)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

fn open_parent_dir(path: &Path, create_parent: bool) -> io::Result<(Dir, &OsStr)> {
    validate_scoped_path(path)?;
    let Some(file_name) = path.file_name() else {
        return Err(invalid_path(path));
    };
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let (root_dir, relative_parent) = split_scope(parent)?;
    let dir = open_scoped_dir(root_dir, &relative_parent, create_parent)?;
    Ok((dir, file_name))
}
