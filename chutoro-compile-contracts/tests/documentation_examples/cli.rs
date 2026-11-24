//! Runs documented `chutoro-cli` invocations against their documented inputs.
//!
//! A documented command is parsed by the real `clap` definition and executed
//! through the same `run_cli` and `render_summary` path as the binary, inside
//! a temporary directory holding the documented input under the file name the
//! command uses. Process-level behaviour, such as quiet `stderr`, is covered by
//! `chutoro-cli`'s own `logging_startup` tests.

use std::{io::Write, sync::Arc};

use anyhow::{Context, Result, ensure};
use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use camino::{Utf8Path, Utf8PathBuf};
use cap_std::{ambient_authority, fs_utf8::Dir};
use chutoro_cli::cli::{Cli, render_summary, run_cli};
use clap::Parser;
use parquet::arrow::ArrowWriter;
use tempfile::TempDir;

/// The program name every documented invocation starts with.
const PROGRAM: &str = "chutoro-cli";

/// A scratch directory standing in for the reader's working directory.
pub struct Workspace {
    directory: TempDir,
}

impl Workspace {
    /// Create an empty workspace.
    pub fn new() -> Result<Self> {
        Ok(Self {
            directory: tempfile::tempdir().context("create a documentation workspace")?,
        })
    }

    fn root(&self) -> Result<&Utf8Path> {
        Utf8Path::from_path(self.directory.path())
            .context("the temporary directory should be UTF-8")
    }

    fn dir(&self) -> Result<Dir> {
        Dir::open_ambient_dir(self.root()?, ambient_authority())
            .context("open the documentation workspace")
    }

    /// Write `contents` to `name` inside the workspace.
    pub fn write_text(&self, name: &str, contents: &str) -> Result<()> {
        self.dir()?
            .write(name, contents)
            .with_context(|| format!("write {name}"))
    }

    /// Write a Parquet file whose `column` holds two well-separated groups of
    /// six 2-D points as `FixedSizeList<Float32, 2>` rows.
    pub fn write_parquet_points(&self, name: &str, column: &str) -> Result<()> {
        let values = vec![
            0.0_f32, 0.0, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4, 0.5,
            -0.5, // near the origin
            50.0, 50.0, 50.1, 49.9, 50.2, 49.8, 50.3, 49.7, 50.4, 49.6, 50.5,
            49.5, // near (50, 50)
        ];
        let item = Arc::new(Field::new("item", DataType::Float32, false));
        let list_type = DataType::FixedSizeList(Arc::clone(&item), 2);
        let schema = Arc::new(Schema::new(vec![Field::new(column, list_type, false)]));
        let rows =
            FixedSizeListArray::try_new(item, 2, Arc::new(Float32Array::from(values)), None)?;
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(rows) as ArrayRef])?;

        let file = self
            .dir()?
            .create(name)
            .with_context(|| format!("create {name}"))?;
        let mut writer = ArrowWriter::try_new(file, schema, None)?;
        writer.write(&batch)?;
        writer.close()?;
        Ok(())
    }

    /// Map an argument naming a workspace file to that file's full path.
    ///
    /// Other arguments pass through unchanged, so options and their values
    /// keep their documented spelling.
    fn resolve(&self, argument: &str) -> Result<String> {
        if self.dir()?.is_file(argument) {
            let path: Utf8PathBuf = self.root()?.join(argument);
            return Ok(path.into_string());
        }
        Ok(argument.to_owned())
    }
}

/// Split a documented command into its arguments, checking the program name.
///
/// Documented commands avoid shell quoting, so whitespace splitting reproduces
/// what a shell would pass. Leading `NAME=value` environment assignments are
/// returned separately.
pub fn split_command(command: &str) -> Result<(Vec<&str>, Vec<&str>)> {
    let line = command.trim();
    ensure!(
        !line.contains('\n'),
        "a documented command should be one line: {line}"
    );
    ensure!(
        !line.contains(['"', '\'', '\\', '$', '|', ';', '&']),
        "a documented command should need no shell quoting: {line}"
    );
    let mut words = line.split_whitespace().peekable();
    let mut environment = Vec::new();
    while let Some(word) = words.next_if(|word| word.contains('=')) {
        environment.push(word);
    }
    let program = words
        .next()
        .context("a documented command should name a program")?;
    ensure!(
        program == PROGRAM,
        "expected `{PROGRAM}`, found `{program}`"
    );
    Ok((environment, words.collect()))
}

/// Parse a documented command with the CLI's own argument definition.
pub fn parse(command: &str) -> Result<Cli> {
    let (_, arguments) = split_command(command)?;
    Cli::try_parse_from(std::iter::once(PROGRAM).chain(arguments))
        .with_context(|| format!("the documented command should parse: {command}"))
}

/// Run a documented command in `workspace` and return its rendered summary.
///
/// An argument naming a file in the workspace is rewritten to that file's
/// full path, keeping its file name so the derived data-source name matches
/// the documentation.
pub fn run(workspace: &Workspace, command: &str) -> Result<String> {
    let (_, arguments) = split_command(command)?;
    let resolved = arguments
        .iter()
        .map(|argument| workspace.resolve(argument))
        .collect::<Result<Vec<_>>>()?;
    let cli = Cli::try_parse_from(std::iter::once(PROGRAM.to_owned()).chain(resolved))
        .with_context(|| format!("the documented command should parse: {command}"))?;
    let summary = run_cli(cli)
        .with_context(|| format!("the documented command should succeed: {command}"))?;
    let mut rendered = Vec::new();
    render_summary(&summary, &mut rendered)?;
    rendered.flush()?;
    String::from_utf8(rendered).context("the summary should be UTF-8")
}
