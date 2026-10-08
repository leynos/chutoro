//! Fixture text for the build-standard contract: configurations, toolchain files,
//! workflow steps and `make -n` output, each shaped to draw one verdict from a reader.

/// A toolchain file pinning a nightly channel.
pub const NIGHTLY: &str = "[toolchain]\nchannel = \"nightly-2026-05-28\"\n";
/// A toolchain file pinning a stable channel.
pub const STABLE: &str = "[toolchain]\nchannel = \"1.94.0\"\n";
/// A compliant nightly configuration: the frontend flag in every source and
/// mold in the Linux table alone.
pub const NIGHTLY_OK: &str = concat!(
    "[build]\nrustflags = [\"-Zthreads=8\"]\n",
    "[target.'cfg(target_os = \"linux\")']\n",
    "rustflags = [\"-Zthreads=8\", \"-Clink-arg=-fuse-ld=mold\"]\n"
);
/// The same, with the linker flag spelled as the `-C` pair Cargo also accepts.
pub const NIGHTLY_SPELLED_APART: &str = concat!(
    "[build]\nrustflags = [\"-Zthreads=8\"]\n",
    "[target.'cfg(target_os = \"linux\")']\n",
    "rustflags = [\"-Zthreads=8\", \"-C\", \"link-arg=-fuse-ld=mold\"]\n"
);
/// A compliant stable configuration: mold alone, in the Linux table.
pub const STABLE_OK: &str = concat!(
    "[target.'cfg(target_os = \"linux\")']\n",
    "rustflags = [\"-Clink-arg=-fuse-ld=mold\"]\n"
);
/// A nightly configuration whose `[build]` source lost the frontend flag, so it
/// is missing it and also differs from the Linux source.
pub const BUILD_LOSES_THREADS: &str = concat!(
    "[build]\nrustflags = [\"-Dwarnings\"]\n",
    "[target.'cfg(target_os = \"linux\")']\n",
    "rustflags = [\"-Zthreads=8\", \"-Clink-arg=-fuse-ld=mold\"]\n"
);
/// A nightly configuration whose Linux table lost mold.
pub const LINUX_LOSES_LINKER: &str = concat!(
    "[build]\nrustflags = [\"-Zthreads=8\"]\n",
    "[target.'cfg(target_os = \"linux\")']\n",
    "rustflags = [\"-Zthreads=8\"]\n"
);
/// A nightly configuration that names mold in `[build]`, beyond Linux.
pub const LINKER_IN_BUILD: &str = concat!(
    "[build]\nrustflags = [\"-Zthreads=8\", \"-Clink-arg=-fuse-ld=mold\"]\n",
    "[target.'cfg(target_os = \"linux\")']\n",
    "rustflags = [\"-Zthreads=8\", \"-Clink-arg=-fuse-ld=mold\"]\n"
);
/// A nightly configuration with no `[build]` source for the other hosts.
pub const NO_BUILD_SOURCE: &str = concat!(
    "[target.'cfg(target_os = \"linux\")']\n",
    "rustflags = [\"-Zthreads=8\", \"-Clink-arg=-fuse-ld=mold\"]\n"
);
/// A stable configuration that names the nightly-only frontend flag.
pub const STABLE_WITH_THREADS: &str = concat!(
    "[target.'cfg(target_os = \"linux\")']\n",
    "rustflags = [\"-Zthreads=8\", \"-Clink-arg=-fuse-ld=mold\"]\n"
);
/// A compliant configuration whose table headers and entries carry comments,
/// with a hash inside a quoted value.
pub const COMMENTED_OK: &str = concat!(
    "[build] # every host\nrustflags = [\"-Zthreads=8\"] # the frontend\n",
    "[target.'cfg(target_os = \"linux\")'] # Linux\n",
    "rustflags = [\"-Zthreads=8\", \"-Clink-arg=-fuse-ld=mold\"]\n",
    "note = \"a # inside a string\"\n"
);
/// A compliant configuration with a sibling key that only starts like `rustflags`.
pub const SIBLING_KEY_OK: &str = concat!(
    "[build]\nrustflags = [\"-Zthreads=8\"]\nrustflags-extra = [\"-Dwarnings\"]\n",
    "[target.'cfg(target_os = \"linux\")']\n",
    "rustflags = [\"-Zthreads=8\", \"-Clink-arg=-fuse-ld=mold\"]\n"
);
/// A nightly configuration whose Linux table names one triple, not every Linux
/// target: mold would reach x86-64 alone.
pub const TRIPLE_ONLY: &str = concat!(
    "[build]\nrustflags = [\"-Zthreads=8\"]\n",
    "[target.x86_64-unknown-linux-gnu]\nlinker = \"clang\"\n",
    "rustflags = [\"-Zthreads=8\", \"-Clink-arg=-fuse-ld=mold\"]\n"
);
/// A `rustflags` array spread over several lines, which the reader refuses.
pub const SPREAD_ARRAY: &str = "[build]\nrustflags = [\n  \"-Zthreads=8\",\n]\n";
/// A toolchain file that names no channel.
pub const NO_CHANNEL: &str = "[toolchain]\ncomponents = [\"clippy\"]\n";
/// A toolchain file that names two channels.
pub const TWO_CHANNELS: &str = "[toolchain]\nchannel = \"stable\"\nchannel = \"nightly\"\n";
/// A toolchain file naming a channel the standard does not know.
pub const UNKNOWN_CHANNEL: &str = "[toolchain]\nchannel = \"weekly\"\n";
/// A workflow step that passes the input, quoted.
pub const STEP_INSTALLS: &str = concat!(
    "    steps:\n      - name: Setup Rust\n",
    "        uses: org/shared-actions/.github/actions/setup-rust@0123456789abcdef0123456789abcdef01234567\n",
    "        with:\n          install-mold: 'true'\n"
);
/// The same, with the bare value.
pub const STEP_INSTALLS_BARE: &str = concat!(
    "    steps:\n      - uses: org/shared-actions/.github/actions/setup-rust@0123456789abcdef0123456789abcdef01234567\n",
    "        with:\n          install-mold: true\n"
);
/// A step with no input at all.
pub const STEP_MISSING_INPUT: &str = concat!(
    "    steps:\n      - name: Setup Rust\n",
    "        uses: org/shared-actions/.github/actions/setup-rust@0123456789abcdef0123456789abcdef01234567\n"
);
/// A step that turns the input off.
pub const STEP_INPUT_OFF: &str = concat!(
    "    steps:\n      - name: Setup Rust\n",
    "        uses: org/shared-actions/.github/actions/setup-rust@0123456789abcdef0123456789abcdef01234567\n",
    "        with:\n          install-mold: 'false'\n"
);
/// A step without the input, followed by a step that has one for another
/// action.
pub const STEP_BEFORE_A_SIBLING_THAT_INSTALLS: &str = concat!(
    "    steps:\n      - name: Setup Rust\n",
    "        uses: org/shared-actions/.github/actions/setup-rust@0123456789abcdef0123456789abcdef01234567\n",
    "      - name: Other\n        uses: org/other@abc\n        with:\n          install-mold: 'true'\n"
);
/// A comment that names the action, and no step.
pub const COMMENT_NAMING_THE_ACTION: &str =
    "    steps:\n      # setup-rust@abc installs it\n      - run: make\n";
/// A coverage step that assigns `RUSTFLAGS` without a standard flag.
pub const COVERAGE_OK: &str = concat!(
    "    steps:\n      - name: Cover\n",
    "        uses: org/shared-actions/.github/actions/generate-coverage@0123456789abcdef0123456789abcdef01234567\n",
    "        env:\n          RUSTFLAGS: -D warnings\n"
);
/// A coverage step with no assignment.
pub const COVERAGE_UNASSIGNED: &str = concat!(
    "    steps:\n      - name: Cover\n",
    "        uses: org/shared-actions/.github/actions/generate-coverage@0123456789abcdef0123456789abcdef01234567\n"
);
/// A coverage step that takes the frontend flag.
pub const COVERAGE_WITH_THREADS: &str = concat!(
    "    steps:\n      - name: Cover\n",
    "        uses: org/shared-actions/.github/actions/generate-coverage@0123456789abcdef0123456789abcdef01234567\n",
    "        env:\n          RUSTFLAGS: -D warnings -Zthreads=8\n"
);
/// A coverage step that takes mold.
pub const COVERAGE_WITH_LINKER: &str = concat!(
    "    steps:\n      - name: Cover\n",
    "        uses: org/shared-actions/.github/actions/generate-coverage@0123456789abcdef0123456789abcdef01234567\n",
    "        env:\n          RUSTFLAGS: -Clink-arg=-fuse-ld=mold\n"
);
/// A coverage step whose assignment belongs to the next step.
pub const COVERAGE_BORROWING_A_SIBLING: &str = concat!(
    "    steps:\n      - name: Cover\n",
    "        uses: org/shared-actions/.github/actions/generate-coverage@0123456789abcdef0123456789abcdef01234567\n",
    "      - name: Other\n        env:\n          RUSTFLAGS: -D warnings\n"
);

/// A coverage step that assigns an empty warning policy.
pub const COVERAGE_EMPTY_POLICY: &str = concat!(
    "steps:\n  - name: coverage\n    uses: org/generate-coverage@abc\n",
    "    env:\n      RUSTFLAGS: \"\"\n"
);
/// A coverage step that assigns a different policy from `-D warnings`.
pub const COVERAGE_OTHER_POLICY: &str = concat!(
    "steps:\n  - name: coverage\n    uses: org/generate-coverage@abc\n",
    "    env:\n      RUSTFLAGS: -W unused\n"
);

/// A coverage step whose policy only starts like `-D warnings`.
pub const COVERAGE_LOOKALIKE_POLICY: &str = concat!(
    "steps:\n  - name: coverage\n    uses: org/generate-coverage@abc\n",
    "    env:\n      RUSTFLAGS: -D warnings-extra\n"
);
/// A coverage step whose only mention of `-D warnings` is an inline comment.
pub const COVERAGE_COMMENTED_POLICY: &str = concat!(
    "steps:\n  - name: coverage\n    uses: org/generate-coverage@abc\n",
    "    env:\n      RUSTFLAGS: # -D warnings\n"
);
/// A coverage step that denies warnings and carries an inline comment about the standard flags.
pub const COVERAGE_DENYING_WITH_COMMENT: &str = concat!(
    "steps:\n  - name: coverage\n    uses: org/generate-coverage@abc\n",
    "    env:\n      RUSTFLAGS: -D warnings # not -Zthreads=8, not mold\n"
);
