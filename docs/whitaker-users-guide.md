# Whitaker user's guide

Whitaker is a collection of opinionated Dylint lints for Rust. This guide
explains how to integrate the lints into a project and configure them.

For contributors who want to develop new lints or work on Whitaker itself, see
the [Developer's Guide](developers-guide.md).

## Quick setup

### Prerequisites

Install `cargo-dylint` and `dylint-link`:

```sh
cargo install cargo-dylint dylint-link
```

### Standalone installation (recommended)

The simplest way to use Whitaker is via the standalone installer, which handles
setup automatically:

```sh
cargo install whitaker-installer
whitaker-installer
whitaker --all
```

This:

1. Installs `cargo-dylint` and `dylint-link` if not present
2. Clones the Whitaker repository to a platform-specific data directory
3. Builds the lint libraries
4. Creates `whitaker` and `whitaker-ls` wrapper scripts. `whitaker` invokes
   `cargo dylint` with the correct `DYLINT_LIBRARY_PATH`, and `whitaker-ls`
   lists installed Whitaker suite libraries
5. Ensures the pinned Rust toolchain and components are installed via rustup

After installation, run `whitaker --all` in any Rust project to lint it. Use
`whitaker-ls` to list the installed Whitaker suite libraries.

**Options:**

- `--skip-deps` — Skip `cargo-dylint`/`dylint-link` installation check
- `--skip-wrapper` — Skip wrapper script generation (prints
  `DYLINT_LIBRARY_PATH` instructions instead)
- `--no-update` — Don't update existing repository clone

### Adding Whitaker to a project

Add the following to the workspace `Cargo.toml`:

```toml
[workspace.metadata.dylint]
[[workspace.metadata.dylint.libraries]]
git = "https://github.com/leynos/whitaker"
pattern = "whitaker_suite"
```

Then run the lints:

```sh
cargo dylint --all
```

### Version pinning

For reproducible builds, pin to a specific release tag or commit:

```toml
[workspace.metadata.dylint]
[[workspace.metadata.dylint.libraries]]
git = "https://github.com/leynos/whitaker"
pattern = "whitaker_suite"
tag = "v0.1.0"
```

Or pin to a specific commit:

```toml
[workspace.metadata.dylint]
[[workspace.metadata.dylint.libraries]]
git = "https://github.com/leynos/whitaker"
pattern = "whitaker_suite"
rev = "abc123def456"
```

### Selecting individual lints

To load specific lints instead of the full suite, specify each lint explicitly:

```toml
[workspace.metadata.dylint]
[[workspace.metadata.dylint.libraries]]
git = "https://github.com/leynos/whitaker"
pattern = "crates/module_max_lines"

[[workspace.metadata.dylint.libraries]]
git = "https://github.com/leynos/whitaker"
pattern = "crates/no_expect_outside_tests"
```

### Standard vs experimental lints

Whitaker lints are divided into two categories:

- **Standard lints** are stable, well-tested, and included in the default suite.
  They are recommended for general use and have predictable behaviour.
- **Experimental lints** are newer or more aggressive checks that may produce
  false positives or undergo breaking changes between releases. They must be
  explicitly enabled.

The default `whitaker_suite` pattern includes only standard lints. Experimental
lints can be added individually or enabled via the `--experimental` flag when
using the standalone installer.

### Enabling experimental lints

#### Via standalone installer

```sh
whitaker-installer --experimental
```

#### Via Cargo.toml

Add experimental lints alongside the suite:

```toml
[workspace.metadata.dylint]
[[workspace.metadata.dylint.libraries]]
git = "https://github.com/leynos/whitaker"
pattern = "whitaker_suite"

[[workspace.metadata.dylint.libraries]]
git = "https://github.com/leynos/whitaker"
pattern = "crates/bumpy_road_function"
```

## Lint configuration

Configure lint behaviour in `dylint.toml` at the workspace root:

```toml
# Diagnostic language (default: en-GB)
locale = "cy"

# Module size threshold (default: 400)
[module_max_lines]
max_lines = 500

# Conditional branch limit (default: 2)
[conditional_max_n_branches]
max_branches = 3

# Custom test attributes
[no_expect_outside_tests]
additional_test_attributes = ["my_framework::test", "async_std::test"]

# Allow panics in main
[no_unwrap_or_else_panic]
allow_in_main = true
```

## Localized diagnostics

Whitaker supports multiple languages for diagnostic messages. Set the locale
via the `DYLINT_LOCALE` environment variable or in `dylint.toml`:

```toml
locale = "cy"
```

Available locales:

- `en-GB` (default) - English
- `cy` - Welsh (Cymraeg)
- `gd` - Scottish Gaelic (Gàidhlig)

______________________________________________________________________

## Available lints

### `conditional_max_n_branches`

Limits the complexity of conditional predicates by enforcing a maximum number
of boolean branches.

**Configuration:**

```toml
[conditional_max_n_branches]
max_branches = 2
```

The default threshold is 2 branches. A predicate like `a && b && c` has three
branches and would trigger the lint.

**How to fix:** Extract complex conditions into helper functions:

```rust
// Before: Too many branches
if condition_a && condition_b && condition_c {
    // action
}

// After: Extract to helper function
fn should_proceed() -> bool {
    condition_a && condition_b && condition_c
}

if should_proceed() {
    // action
}
```

______________________________________________________________________

### `function_attrs_follow_docs`

Ensures doc comments appear before all other outer attributes on functions.

**How to fix:** Move doc comments to appear before other attributes:

```rust
// Wrong
#[inline]
/// This function does something.
fn example() {}

// Correct
/// This function does something.
#[inline]
fn example() {}
```

______________________________________________________________________

### `module_max_lines`

Warns when modules exceed a configurable line count threshold.

**Configuration:**

```toml
[module_max_lines]
max_lines = 400
```

**How to fix:** Split large modules into smaller, focused submodules.

______________________________________________________________________

### `module_must_have_inner_docs`

Enforces that every module begins with an inner documentation comment (`//!`).

**How to fix:**

```rust
mod my_module {
    //! Explain the module's purpose here.
    pub fn value() {}
}
```

______________________________________________________________________

### `no_expect_outside_tests`

Forbids calling `.expect()` on `Option` or `Result` outside test contexts.

**Configuration:**

```toml
[no_expect_outside_tests]
additional_test_attributes = ["my_framework::test", "async_std::test"]
```

**How to fix:** Use proper error handling (`?`, `map_err`) or move the code to
a test context.

______________________________________________________________________

### `no_std_fs_operations`

Enforces capability-based filesystem access by forbidding direct use of
`std::fs` operations.

**Configuration:**

```toml
[no_std_fs_operations]
excluded_crates = ["my_cli_entrypoint", "my_test_utilities"]
```

The `excluded_crates` option allows specified crates to use `std::fs`
operations without triggering diagnostics. This is useful for:

- CLI entry points where ambient filesystem access is the intended boundary
- Test support utilities that manage fixtures with ambient access
- Build scripts or code generators that require direct filesystem operations

> **Note:** Use Rust crate names (underscores), not Cargo package names
> (hyphens). For example, use `my_cli_app` rather than `my-cli-app`.

**How to fix:** Replace `std::fs` with `cap_std`:

```rust
// Before
use std::fs;
fn read_config() -> std::io::Result<String> {
    fs::read_to_string("config.toml")
}

// After
use cap_std::fs::Dir;
use camino::Utf8Path;
fn read_config(config_dir: &Dir, path: &Utf8Path) -> std::io::Result<String> {
    config_dir.read_to_string(path)
}
```

______________________________________________________________________

### `no_unwrap_or_else_panic`

Denies panicking `unwrap_or_else` fallbacks on `Option`/`Result`, including
tests. Doctest runs remain exempt.

**Configuration:**

```toml
[no_unwrap_or_else_panic]
allow_in_main = true
```

**What is allowed:**

- Panicking `unwrap_or_else` fallbacks inside doctests
- Panicking `unwrap_or_else` fallbacks inside `main` when
  `allow_in_main = true`
- Non-panicking `unwrap_or_else` fallbacks

**What is denied:**

- `unwrap_or_else(|| panic!(..))`
- `unwrap_or_else(|| value.unwrap())`

**How to fix:** Propagate errors with `?` or use `.expect()` with a clear
message if a panic is truly intended. In tests, replace
`unwrap_or_else(|| panic!("msg"))` with `.expect("msg")` for clarity and
brevity.
