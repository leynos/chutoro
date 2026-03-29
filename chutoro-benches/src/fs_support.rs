//! Capability-scoped filesystem helpers for benchmark data and reports.

use std::ffi::OsStr;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

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
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let Some(file_name) = path.file_name() else {
        return Err(invalid_path(path));
    };

    match Dir::open_ambient_dir(parent, ambient_authority()) {
        Ok(dir) => dir.try_exists(file_name),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error),
    }
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

fn open_parent_dir(path: &Path, create_parent: bool) -> io::Result<(Dir, &OsStr)> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    if create_parent {
        Dir::create_ambient_dir_all(parent, ambient_authority())?;
    }

    let dir = Dir::open_ambient_dir(parent, ambient_authority())?;
    let Some(file_name) = path.file_name() else {
        return Err(invalid_path(path));
    };
    Ok((dir, file_name))
}
