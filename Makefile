.PHONY: help all clean test test-doc build release typecheck lint lint-clippy lint-whitaker fmt check-fmt markdownlint nixie spelling install-kani install-verus kani kani-check kani-full verus bench test-workflow-contracts interrogate

# `lint-clippy` and `lint-whitaker` share Cargo's package cache and the
# compiler-admission pool. Running them under `make -j` makes two builds
# compete for the same slots, so this gate stays sequential.
.NOTPARALLEL: lint

export PATH := $(HOME)/.cargo/bin:$(HOME)/.local/bin:$(HOME)/.bun/bin:$(PATH)

APP ?= chutoro-cli
CARGO ?= cargo
# Optional development-speed route. The standard `build`, `test`, `typecheck`
# and `lint` recipes keep the repository's stable pin and the build standard;
# `dev-build` and `dev-test` select the pinned nightly Cranelift fragment
# instead. Cargo must see that fragment only through the explicit `--config`
# argument in `DEV_CARGO`, so nothing activates it by accident.
DEV_FAST_TOOLCHAIN := $(strip $(file <tools/dev-fast/TOOLCHAIN))
DEV_FAST_CONFIG := tools/dev-fast/config.toml
DEV_CARGO = $(CARGO) +$(DEV_FAST_TOOLCHAIN) --config $(DEV_FAST_CONFIG)
# `$(file <…)` yields nothing for a missing file and strips to nothing for an
# empty one, and `cargo +` with nothing after it falls back to the default
# toolchain. The route would then compile with a toolchain the pin never
# names and report success, which is the one outcome the pin exists to
# prevent. The check lives here rather than in a file-wide `$(error)` for the
# same reason the host checks do: the standard gates read this variable too,
# and a broken development pin must not fail them.
DEV_FAST_TOOLCHAIN_CHECK = $(if $(DEV_FAST_TOOLCHAIN),,$(error tools/dev-fast/TOOLCHAIN is missing or empty, so the development route cannot pin a toolchain; restore the pinned nightly before running dev-build, dev-test, or install-dev-fast))
# An assigned RUSTFLAGS replaces every `rustflags` table, including the
# fragment's, so both development routes compose the fragment's flags into an
# inherited value rather than replacing it. Passing RUSTFLAGS to `dev-build`
# at all is what turns the substitution on: with it unset Cargo reads the
# fragment's own table, but with it set - even to an empty string - Cargo uses
# the assignment and silently drops the mold flag and both documentation
# denials. `$${RUSTFLAGS:+$$RUSTFLAGS }` therefore restates the fragment's
# flags exactly as `$(STANDARD_RUSTFLAGS)` restates `.cargo/config.toml`'s.
#
# The host probe is evaluated once, with its exit status. Defaulting the pair
# to `$(shell uname -s)` under `?=` re-ran the probe on every reference, so one
# recipe could pass the emptiness check and then select its flags from a second
# expansion that failed, dropping the linker flag while the target reported
# success. `:=` captures one answer for the whole run.
# The exit status is appended by the shell itself rather than read back from
# `.SHELLSTATUS`. That variable reports whichever `$(shell)` call ran last, so
# a read back is correct only while no other call intervenes between the probe
# and the read — an ordering this file satisfies today only because
# `BUILD_HOST_OS`'s call comes further down. Appending the status inside the
# probe binds it to its own answer, so the check cannot be broken by adding a
# shell call above or below.
DEV_FAST_HOST_PROBE := $(shell { uname -s; echo "DEV_FAST_PROBE_STATUS=$$?"; })
DEV_FAST_HOST_PROBE_OS := $(filter-out DEV_FAST_PROBE_STATUS=%,$(DEV_FAST_HOST_PROBE))
DEV_FAST_HOST_PROBE_STATUS := $(patsubst DEV_FAST_PROBE_STATUS=%,%,$(filter DEV_FAST_PROBE_STATUS=%,$(DEV_FAST_HOST_PROBE)))
DEV_FAST_HOST_OS ?= $(DEV_FAST_HOST_PROBE_OS)
# A caller-supplied value is checked for emptiness but not against the probe,
# so `DEV_FAST_HOST_OS=Linux` still works on a host where `uname` cannot run.
DEV_FAST_HOST_OS_IS_PROBED := $(filter file,$(origin DEV_FAST_HOST_OS))
# A failed probe yields an empty string, and an empty string is not Linux, so
# every `$(if $(filter Linux,...))` below would quietly select the non-Linux
# branch. A probe that prints a value and still exits non-zero is just as
# misleading: its answer would select the Linux branch while the command that
# produced it reported failure. The checks live in the development recipes
# rather than in a file-wide `$(error)`, because parse-time validation would
# also fail the standard gates, which do not use this route at all.
DEV_FAST_HOST_EMPTY_CHECK = $(if $(strip $(DEV_FAST_HOST_OS)),,$(error Cannot determine the development host OS: `uname -s` returned nothing. Set DEV_FAST_HOST_OS explicitly))
DEV_FAST_HOST_STATUS_CHECK = $(if $(DEV_FAST_HOST_OS_IS_PROBED),$(if $(filter 0,$(DEV_FAST_HOST_PROBE_STATUS)),,$(error Cannot determine the development host OS: `uname -s` exited $(DEV_FAST_HOST_PROBE_STATUS). Set DEV_FAST_HOST_OS explicitly)),)
DEV_FAST_HOST_CHECK = $(DEV_FAST_HOST_EMPTY_CHECK)$(DEV_FAST_HOST_STATUS_CHECK)
# Every development recipe needs both answers, so one name carries the pair.
DEV_FAST_CHECK = $(DEV_FAST_TOOLCHAIN_CHECK)$(DEV_FAST_HOST_CHECK)
DEV_FRAGMENT_RUSTFLAGS = $(strip -Dmissing_docs \
	-Dmissing_crate_level_docs \
	$(if $(filter Linux,$(DEV_FAST_HOST_OS)),-Clink-arg=-fuse-ld=mold))
DEV_TEST_RUSTFLAGS = $(strip -D warnings $(DEV_FRAGMENT_RUSTFLAGS))
# The poisoned-lock regression panics while a graph write guard is alive, then
# inspects the poison through `catch_unwind`. That needs a backend that can
# unwind a panic; the opt-in Cranelift route aborts the process instead
# ("failed to initiate panic, error 5"). `dev-test` therefore excludes exactly
# this case, and the standard `test` gate - which stays on the stable pin -
# deliberately does not, so the regression still runs at the merge gate.
POISONED_LOCK_TEST = hnsw::cpu::test_helpers::tests::graph_helpers_report_poisoning_without_reconfiguring
DEV_TEST_FILTER = not kind(bench) & not (package(chutoro-core) & test(=$(POISONED_LOCK_TEST)))
WHITAKER ?= whitaker
BUILD_JOBS ?=
CLIPPY_FLAGS ?= --all-targets --all-features -- -D warnings
RUSTDOC_FLAGS ?= --cfg docsrs -D warnings
MDLINT ?= markdownlint-cli2
# `make fmt` and `make check-fmt` call mdtablefix directly. `--git` selects the
# Markdown files Git tracks and `--include-untracked` adds the untracked files
# Git does not ignore, so a new document is formatted before it is staged.
# Both modes need mdtablefix 0.6.0 or later; CI pins the version at the
# install-mdtablefix step.
MDTABLEFIX ?= mdtablefix
MDTABLEFIX_SELECT = --git --include-untracked
MDTABLEFIX_RULES = --wrap --renumber --breaks --ellipsis --fences
NEXTEST_PROFILE ?= $(if $(CI),ci,default)
NIXIE ?= nixie
UV ?= uv
UV_ENV = UV_CACHE_DIR=.uv-cache UV_TOOL_DIR=.uv-tools

# The CV-005 CodeScene contracts live in shared-actions and run from a full
# commit, so a fix is a pin bump. `.github/cv005.toml` holds this repository's
# only parameters.
CV005_CONTRACTS_REF ?= 3754876bc4e94dc65c83af3def37828934f8f65e
CV005_CONTRACTS = $(UV_ENV) $(UV) tool run --python 3.13 \
	--from 'git+https://github.com/leynos/shared-actions@$(CV005_CONTRACTS_REF)\#subdirectory=packages/cv005-contracts' \
	cv005-contracts

TYPOS_CONFIG_BUILDER_VERSION ?= v0.1.3
TYPOS_CONFIG_BUILDER = $(UV_ENV) $(UV) tool run --python 3.14 --from \
	"git+https://github.com/leynos/typos-config-builder.git@$(TYPOS_CONFIG_BUILDER_VERSION)" \
	typos-config-builder

# Prover acquisition and execution belong to the shared rust-prover-tools
# package, pinned to a reviewed commit. It owns the bounded HTTPS downloader,
# the archive digests, extraction, and the install layout, so this repository
# keeps only the pins under `tools/`, the workflow wiring, and the runner
# parameters. `uv tool run` keeps a Rust contributor from needing a project
# virtual environment, and `UV_ENV` keeps uv's state beside the other helper
# tooling.
PROVER_TOOLS_SOURCE ?= git+https://github.com/leynos/rust-prover-tools@592558f7183b554fc749696246984c5d7300d029
PROVER_TOOLS_PYTHON ?= 3.14
PROVER_TOOLS ?= $(UV_ENV) $(UV) tool run --python $(PROVER_TOOLS_PYTHON) \
	--from $(PROVER_TOOLS_SOURCE) prover-tools

KANI_INSTALL_FLAGS ?=
VERUS_INSTALL_FLAGS ?=
# The legacy single-file override still names one proof; otherwise both edge
# harvest proofs run, in this order. Verus accepts one input filename per
# invocation, so the package verifies them one at a time and stops at the
# first failure.
VERUS_PROOF_FILES ?= $(if $(VERUS_PROOF_FILE),$(VERUS_PROOF_FILE),\
	verus/edge_harvest_proofs.rs verus/session_core_distance.rs)
VERUS_PROOF_ARGS = $(foreach proof,$(VERUS_PROOF_FILES),--proof-file "$(proof)")

# `prover-tools kani install` runs `cargo kani setup`, which unpacks the
# verified bundle and its pinned nightly toolchain under KANI_HOME at
# `kani-<version>/toolchain`.
KANI_HOME ?= $(HOME)/.kani
KANI_VERSION_FILE ?= tools/kani/VERSION
override KANI_VERSION := $(strip $(file <$(KANI_VERSION_FILE)))
KANI_VERSION_PARTS := $(subst ., ,$(KANI_VERSION))
# Remove the valid SemVer characters so the parse-time checks below can reject
# malformed file contents before deriving the toolchain library path.
KANI_VERSION_REMAINDER := $(KANI_VERSION)
KANI_VERSION_REMAINDER := $(subst 0,,$(KANI_VERSION_REMAINDER))
KANI_VERSION_REMAINDER := $(subst 1,,$(KANI_VERSION_REMAINDER))
KANI_VERSION_REMAINDER := $(subst 2,,$(KANI_VERSION_REMAINDER))
KANI_VERSION_REMAINDER := $(subst 3,,$(KANI_VERSION_REMAINDER))
KANI_VERSION_REMAINDER := $(subst 4,,$(KANI_VERSION_REMAINDER))
KANI_VERSION_REMAINDER := $(subst 5,,$(KANI_VERSION_REMAINDER))
KANI_VERSION_REMAINDER := $(subst 6,,$(KANI_VERSION_REMAINDER))
KANI_VERSION_REMAINDER := $(subst 7,,$(KANI_VERSION_REMAINDER))
KANI_VERSION_REMAINDER := $(subst 8,,$(KANI_VERSION_REMAINDER))
KANI_VERSION_REMAINDER := $(subst 9,,$(KANI_VERSION_REMAINDER))
KANI_VERSION_REMAINDER := $(subst .,,$(KANI_VERSION_REMAINDER))
ifneq ($(words $(KANI_VERSION_PARTS)),3)
_KANI_VERSION_CHECK := $(error KANI_VERSION must be MAJOR.MINOR.PATCH)
endif
ifneq ($(strip $(KANI_VERSION_REMAINDER)),)
_KANI_VERSION_CHECK := $(error KANI_VERSION must be MAJOR.MINOR.PATCH)
endif
KANI_LIB_PATH ?= $(KANI_HOME)/kani-$(KANI_VERSION)/toolchain/lib
KANI_ENV ?= LD_LIBRARY_PATH="$(KANI_LIB_PATH):$(LD_LIBRARY_PATH)"

# The development build standard (concordat rule `rust-build-defaults`):
# the mold linker on Linux. The pin is stable, so the nightly-only parallel
# frontend flag is not used. An assigned RUSTFLAGS replaces every
# `rustflags` table in .cargo/config.toml, so each recipe that sets it
# composes this onto any inherited value (CI's setup-rust exports one),
# except coverage, which stays on the platform linker, and Whitaker, which runs
# under the toolchain its installer manages and takes none of the standard.
BUILD_HOST_OS := $(shell uname -s)
STANDARD_LINKER_FLAGS := $(if $(filter Linux,$(BUILD_HOST_OS)),-Clink-arg=-fuse-ld=mold)
STANDARD_RUSTFLAGS := -Dmissing_docs -Dmissing_crate_level_docs $(STANDARD_LINKER_FLAGS)

build: target/debug/$(APP) ## Build debug binary
release: target/release/$(APP) ## Build release binary

all: release spelling test-workflow-contracts ## Default target builds release binary and checks spelling

clean: ## Remove build artefacts
	$(CARGO) clean

test: ## Run tests with warnings treated as errors
	$(CARGO) fetch --locked --target "$${CARGO_BUILD_TARGET:-host-tuple}"
	RUSTFLAGS="$${RUSTFLAGS:+$$RUSTFLAGS }-D warnings $(STANDARD_RUSTFLAGS)" $(CARGO) nextest run --profile $(NEXTEST_PROFILE) --all-targets --all-features $(BUILD_JOBS) -E 'not kind(bench)'
	$(MAKE) test-doc

# `cargo nextest` does not run doctests, so `make test` runs this target as a
# second step and CI invokes it directly. Doctest snippets are compiled as
# external crates, which is what makes a public example a genuine API check.
test-doc: ## Compile and run workspace doctests
	RUSTFLAGS="$${RUSTFLAGS:+$$RUSTFLAGS }-D warnings $(STANDARD_RUSTFLAGS)" $(CARGO) test --doc --workspace --all-features $(BUILD_JOBS)

target/%/$(APP): ## Build binary in debug or release mode
	$(if $(findstring release,$(@)),RUSTFLAGS="$${RUSTFLAGS:+$$RUSTFLAGS }-Dmissing_docs -Dmissing_crate_level_docs",RUSTFLAGS="$${RUSTFLAGS:+$$RUSTFLAGS }$(STANDARD_RUSTFLAGS)") $(CARGO) build $(BUILD_JOBS) $(if $(findstring release,$(@)),--release) --bin $(APP)

# Interrogate measures docstring presence over every Python file the repository
# owns. Add a new Python directory to INTERROGATE_TARGETS. No `--python` pin: a
# managed interpreter would be downloaded, and a PyO3 build could link it.
INTERROGATE_VERSION ?= 1.7.0
INTERROGATE_TARGETS ?= scripts tests
# Level 2 prints a MISSED row per undocumented definition, which the
# `interrogate` target fails on: the rounded percentage alone passes a lone gap.
INTERROGATE_FLAGS ?= -vv --fail-under 100
INTERROGATE = $(UV) tool run \
	--from 'interrogate==$(INTERROGATE_VERSION)' interrogate

# Interrogate rounds its percentage to one decimal, so `--fail-under 100` alone
# passes a lone gap in a large repository; fail on any MISSED row as well.
interrogate: ## Enforce complete Python docstring coverage
	@report="$$($(INTERROGATE) $(INTERROGATE_FLAGS) $(INTERROGATE_TARGETS))"; \
	  status=$$?; \
	  printf '%s\n' "$$report"; \
	  test "$$status" -eq 0 || exit "$$status"; \
	  if printf '%s\n' "$$report" | grep -q '| *MISSED *|'; then \
	    printf '%s\n' 'interrogate: undocumented definitions remain; see the MISSED rows above' >&2; \
	    exit 1; \
	  fi

# The explicit development-speed names carry their own recipes rather than
# depending on `build` and `test`: those standard targets deliberately stay on
# the stable pin, so an alias would never select the fragment. They declare
# themselves phony here rather than in the file-wide list, so this opt-in
# feature owns its own declarations.
.PHONY: dev-build dev-test install-dev-fast

dev-build: ## Build the debug binary with the selected development backend
	$(DEV_FAST_CHECK)RUSTFLAGS="$${RUSTFLAGS:+$$RUSTFLAGS }$(DEV_FRAGMENT_RUSTFLAGS)" $(DEV_CARGO) build $(BUILD_JOBS) --bin $(APP)

dev-test: ## Run the tests with the selected development backend
	$(DEV_FAST_CHECK)RUSTFLAGS="$${RUSTFLAGS:+$$RUSTFLAGS }$(DEV_TEST_RUSTFLAGS)" $(DEV_CARGO) nextest run --config $(DEV_FAST_CONFIG) --profile $(NEXTEST_PROFILE) --all-targets --all-features $(BUILD_JOBS) -E '$(DEV_TEST_FILTER)'

install-dev-fast: ## Install pinned nightly components and the verified mold binary
	$(DEV_FAST_CHECK)rustup toolchain install $(DEV_FAST_TOOLCHAIN) --profile minimal --component rustfmt --component clippy --component rust-analyzer --component rustc-codegen-cranelift
ifeq ($(DEV_FAST_HOST_OS),Linux)
	scripts/install-mold.sh
else
	@echo 'mold is Linux-only; the platform linker remains selected'
endif

lint: interrogate lint-clippy lint-whitaker ## Run Clippy and the Whitaker Dylint suite with warnings denied

lint-clippy: ## Run rustdoc and Clippy with warnings denied
	RUSTDOCFLAGS="$(RUSTDOC_FLAGS)" RUSTFLAGS="$${RUSTFLAGS:+$$RUSTFLAGS }$(STANDARD_RUSTFLAGS)" $(CARGO) doc --workspace --no-deps
	RUSTFLAGS="$${RUSTFLAGS:+$$RUSTFLAGS }$(STANDARD_RUSTFLAGS)" $(CARGO) clippy $(CLIPPY_FLAGS)

# Whitaker runs under the toolchain its installer manages, not the repository's
# pin, so it takes none of the development standard (no parallel frontend, no
# `mold`, no backend). The literal `RUSTFLAGS` is the boundary: it displaces the
# `rustflags` tables and nothing else in the configuration reaches it. It also
# stays free of the project's documentation lints, which would fail the driver
# build. `RUSTFLAGS` is assigned to deny warnings.
lint-whitaker: ## Run the Whitaker Dylint suite with warnings denied
	RUSTFLAGS="-D warnings" $(WHITAKER) --all -- --all-targets --all-features

typecheck: ## Type-check all workspace targets and features
	RUSTFLAGS="$${RUSTFLAGS:+$$RUSTFLAGS }$(STANDARD_RUSTFLAGS)" $(CARGO) check --workspace --all-targets --all-features $(BUILD_JOBS)

fmt: ## Format Rust and Markdown sources
	$(CARGO) fmt --all
	$(MDTABLEFIX) --in-place $(MDTABLEFIX_SELECT) $(MDTABLEFIX_RULES)
	@unset FORCE_COLOR; $(MDLINT) --fix "**/*.md"

check-fmt: ## Verify formatting
	$(CARGO) fmt --all -- --check
	$(MDTABLEFIX) --check $(MDTABLEFIX_SELECT) $(MDTABLEFIX_RULES)

markdownlint: spelling ## Lint Markdown files and enforce spelling
	find . -type f -name '*.md' -not -path './target/*' -not -path './.verus/*' \
		-not -path './.uv-cache/*' -not -path './.uv-tools/*' -print0 | \
		xargs -0 $(MDLINT)

spelling: ## Enforce en-GB-oxendict spelling
	$(TYPOS_CONFIG_BUILDER) gate --repository .

nixie: ## Validate Mermaid diagrams
	find . -type f -name '*.md' -not -path './target/*' -not -path './.verus/*' \
		-not -path './.uv-cache/*' -not -path './.uv-tools/*' -print0 | \
		xargs -0 $(NIXIE) --no-sandbox

install-kani: ## Install the pinned Kani verifier
	@$(PROVER_TOOLS) kani install $(KANI_INSTALL_FLAGS) || \
		{ status=$$?; printf 'install-kani failed (exit %s)\n' "$$status" >&2; exit "$$status"; }

install-verus: ## Install the pinned Verus verifier
	@$(PROVER_TOOLS) verus install $(VERUS_INSTALL_FLAGS) || \
		{ status=$$?; printf 'install-verus failed (exit %s)\n' "$$status" >&2; exit "$$status"; }

kani-check: ## Check the installed Kani verifier against its pin
	@$(PROVER_TOOLS) kani check-version --kani-command "$(CARGO) kani" || \
		{ status=$$?; printf 'kani-check failed (exit %s)\n' "$$status" >&2; exit "$$status"; }

kani: ## Run Kani practical harnesses
	$(KANI_ENV) $(CARGO) kani -p chutoro-core --default-unwind 4 --harness verify_bidirectional_links_smoke_2_nodes_1_layer
	$(KANI_ENV) $(CARGO) kani -p chutoro-core --default-unwind 4 --harness verify_bidirectional_links_reconciliation_2_nodes_1_layer
	$(KANI_ENV) $(CARGO) kani -p chutoro-core --default-unwind 12 --harness verify_mst_structural_correctness_4_nodes
	$(KANI_ENV) $(CARGO) kani -p chutoro-core --default-unwind 10 --harness verify_mst_minimality_3_nodes
	$(KANI_ENV) $(CARGO) kani -p chutoro-providers-dense --default-unwind 4 --harness verify_dense_simd_dispatch_selection_respects_support_masks
	$(KANI_ENV) $(CARGO) kani -p chutoro-providers-dense --default-unwind 18 --harness verify_dense_simd_tail_padding_lane_bounds

kani-full: ## Run all Kani formal verification harnesses
	$(KANI_ENV) $(CARGO) kani -p chutoro-core --default-unwind 10
	$(KANI_ENV) $(CARGO) kani -p chutoro-providers-dense --default-unwind 18

verus: ## Run Verus proofs for edge harvest primitives
	@$(PROVER_TOOLS) verus run $(VERUS_PROOF_ARGS) || \
		{ status=$$?; printf 'verus proofs failed (exit %s)\n' "$$status" >&2; exit "$$status"; }

bench: ## Run Criterion benchmarks
	$(CARGO) bench -p chutoro-benches

# `--doctest-modules` is what makes the documented examples load-bearing.
# The support modules carry their examples as doctests, but nothing ran
# them, so `_fraction_nanoseconds` had shipped an example that does not
# produce what it claims. An example nobody executes is a comment that
# looks like evidence.
test-workflow-contracts: ## Validate the CI workflow contracts
	$(CV005_CONTRACTS) check --repository .
	uv run --with 'pytest>=8' --with 'pyyaml>=6' --with 'pathspec>=0.12' \
		--with 'hypothesis>=6' pytest tests/workflow_contracts \
		--doctest-modules -q

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS=":"; printf "Available targets:\n"} {printf "  %-20s %s\n", $$1, $$2}'
