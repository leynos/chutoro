//! Executable contracts for every example in the user-facing documentation.
//!
//! Rust fences run as doctests through `src/lib.rs`; this target checks that
//! every fence is registered, that Rust fences are doctested verbatim, and
//! that each TOML, shell, and text fence means what it says against the
//! current workspace.

mod documentation_examples;
#[path = "documentation_examples/cli.rs"]
mod documented_cli;

use std::collections::BTreeSet;

use anyhow::{Context, Result, ensure};
use documentation_examples::{
    DOCUMENTS, DocumentedExample, documented_example, load_documented_examples,
    read_repository_file,
};
use documented_cli::Workspace;

/// Every documented example, by its stable `tested-example` identifier.
///
/// Adding a fence to a user-facing document means adding its identifier here
/// and, for anything other than Rust, a contract below.
const EXPECTED_EXAMPLE_IDS: &[&str] = &[
    "guide-cli-install",
    "guide-cli-parquet-run",
    "guide-cli-text-corpus",
    "guide-cli-text-output",
    "guide-cli-text-run",
    "guide-core-dependency",
    "guide-cpu-hnsw",
    "guide-dense-provider",
    "guide-distance-batch",
    "guide-distance-helpers",
    "guide-error-codes",
    "guide-first-run",
    "guide-memory-limit",
    "guide-min-cluster-size",
    "guide-mst-hierarchy",
    "guide-quality-score",
    "guide-session",
    "guide-text-provider",
    "migration-cli-verbose",
    "migration-dependency",
    "migration-point-pair-construction",
    "migration-point-pair-override",
    "migration-run-pipeline",
    "migration-try-from-assignments",
    "readme-cli-corpus",
    "readme-cli-install",
    "readme-cli-output",
    "readme-cli-run",
    "readme-library-dependency",
    "readme-library-usage",
];

/// Crates a reader may install or depend on from crates.io.
const PUBLISHED_CRATES: &[&str] = &[
    "chutoro-cli",
    "chutoro-core",
    "chutoro-providers-dense",
    "chutoro-providers-text",
];

/// The version every published crate shares, inherited from the workspace.
const WORKSPACE_VERSION: &str = env!("CARGO_PKG_VERSION");

#[test]
fn documented_examples_match_the_registry() -> Result<()> {
    let found: BTreeSet<String> = load_documented_examples()?
        .into_iter()
        .map(|example| example.id)
        .collect();
    let expected: BTreeSet<String> = EXPECTED_EXAMPLE_IDS
        .iter()
        .map(|id| (*id).to_owned())
        .collect();
    ensure!(
        found == expected,
        "unregistered examples: {:?}; registered but missing: {:?}",
        found.difference(&expected).collect::<Vec<_>>(),
        expected.difference(&found).collect::<Vec<_>>()
    );
    Ok(())
}

/// Every loaded document must also be compiled as a doctest, and vice versa,
/// or its Rust fences would be checked for markers but never run.
#[test]
fn every_documented_file_is_doctested() -> Result<()> {
    let library = read_repository_file("chutoro-compile-contracts/src/lib.rs")?;
    let included: BTreeSet<&str> = library
        .split("include_str!(\"../../")
        .skip(1)
        .filter_map(|rest| rest.split_once("\")").map(|(path, _)| path))
        .collect();
    let loaded: BTreeSet<&str> = DOCUMENTS.iter().copied().collect();
    ensure!(
        included == loaded,
        "doctested {included:?} but loaded {loaded:?}"
    );
    Ok(())
}

/// Rust fences must run exactly as shown.
///
/// Rustdoc's `ignore`, `no_run`, and similar attributes would skip or weaken
/// the run, and hidden `# ` lines render literally on GitHub and crates.io, so
/// readers would copy code that differs from what was tested.
#[test]
fn rust_examples_run_exactly_as_shown() -> Result<()> {
    for example in load_documented_examples()?
        .iter()
        .filter(|example| example.language.starts_with("rust"))
    {
        ensure!(
            example.language == "rust",
            "{} should be a plain `rust` fence",
            example.id
        );
        let hidden = example
            .body
            .lines()
            .find(|line| is_hidden_doctest_line(line));
        ensure!(
            hidden.is_none(),
            "{} hides a line from readers: {hidden:?}",
            example.id
        );
    }
    Ok(())
}

fn is_hidden_doctest_line(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed == "#" || trimmed.starts_with("# ")
}

#[test]
fn examples_use_known_fence_languages() -> Result<()> {
    for example in load_documented_examples()? {
        ensure!(
            ["rust", "toml", "sh", "text"].contains(&example.language.as_str()),
            "{} uses an unexpected fence language `{}`",
            example.id,
            example.language
        );
    }
    Ok(())
}

/// Dependency snippets must name published crates at the released version.
///
/// A plain `0.1.0` requirement does not match a `0.1.0-beta1` prerelease, so
/// a stale version here would fail to resolve for every reader.
#[test]
fn dependency_snippets_name_published_crates_at_the_workspace_version() -> Result<()> {
    let snippets: Vec<DocumentedExample> = load_documented_examples()?
        .into_iter()
        .filter(|example| example.language == "toml")
        .collect();
    ensure!(
        !snippets.is_empty(),
        "the documentation should show a dependency snippet"
    );
    for snippet in snippets {
        let manifest: toml::Table = toml::from_str(&snippet.body)
            .with_context(|| format!("{} should parse", snippet.id))?;
        let dependencies = manifest
            .get("dependencies")
            .and_then(toml::Value::as_table)
            .with_context(|| format!("{} should declare [dependencies]", snippet.id))?;
        for (name, requirement) in dependencies {
            ensure!(
                PUBLISHED_CRATES.contains(&name.as_str()),
                "{} names unpublished `{name}`",
                snippet.id
            );
            ensure!(
                requirement.as_str() == Some(WORKSPACE_VERSION),
                "{} should require `{name} = \"{WORKSPACE_VERSION}\"`, found {requirement}",
                snippet.id
            );
        }
    }
    Ok(())
}

/// The install commands must fetch the released version of a crate that
/// publishes the `chutoro-cli` binary.
#[test]
fn install_commands_fetch_the_released_cli() -> Result<()> {
    let manifest: toml::Table = toml::from_str(&read_repository_file("chutoro-cli/Cargo.toml")?)?;
    let package = manifest
        .get("package")
        .and_then(toml::Value::as_table)
        .context("chutoro-cli has [package]")?;
    ensure!(package.get("name").and_then(toml::Value::as_str) == Some("chutoro-cli"));
    ensure!(
        package.get("publish").is_none(),
        "chutoro-cli must stay publishable"
    );
    ensure!(
        manifest.get("bin").is_none(),
        "the binary must keep its default name, chutoro-cli"
    );

    let expected = format!("cargo install chutoro-cli@{WORKSPACE_VERSION}\n");
    for id in ["readme-cli-install", "guide-cli-install"] {
        ensure!(
            documented_example(id)?.body == expected,
            "{id} should read `{}`",
            expected.trim_end()
        );
    }
    Ok(())
}

/// Each documented text run must print exactly the documented output.
#[test]
fn documented_text_runs_print_the_documented_output() -> Result<()> {
    for prefix in ["readme-cli", "guide-cli-text"] {
        let corpus = documented_example(&format!("{prefix}-corpus"))?;
        let command = documented_example(&format!("{prefix}-run"))?;
        let output = documented_example(&format!("{prefix}-output"))?;

        let file_name = documented_cli::split_command(&command.body)?
            .1
            .last()
            .copied()
            .context("the text command should end with its input file")?
            .to_owned();
        let workspace = Workspace::new()?;
        workspace.write_text(&file_name, &corpus.body)?;

        let printed = documented_cli::run(&workspace, &command.body)?;
        ensure!(
            printed == output.body,
            "{prefix} printed:\n{printed}\nbut documents:\n{}",
            output.body
        );
    }
    Ok(())
}

/// The documented Parquet run must succeed against a matching file.
#[test]
fn documented_parquet_run_clusters_its_column() -> Result<()> {
    let command = documented_example("guide-cli-parquet-run")?;
    let workspace = Workspace::new()?;
    workspace.write_parquet_points("embeddings.parquet", "embedding")?;

    let printed = documented_cli::run(&workspace, &command.body)?;
    ensure!(
        printed.starts_with("data source: embeddings\nclusters: 2\n"),
        "unexpected summary:\n{printed}"
    );
    Ok(())
}

/// The verbose-diagnostics example must set `RUST_LOG` on a valid command.
#[test]
fn verbose_example_sets_rust_log_on_a_valid_command() -> Result<()> {
    let command = documented_example("migration-cli-verbose")?;
    let (environment, _) = documented_cli::split_command(&command.body)?;
    ensure!(
        environment == ["RUST_LOG=info"],
        "unexpected environment {environment:?}"
    );
    documented_cli::parse(&command.body)?;
    Ok(())
}
