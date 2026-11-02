.PHONY: help all clean test build release typecheck lint lint-clippy lint-whitaker fmt check-fmt markdownlint nixie spelling install-kani install-verus kani kani-check kani-full verus bench test-workflow-contracts

export PATH := $(HOME)/.cargo/bin:$(HOME)/.bun/bin:$(PATH)

APP ?= chutoro-cli
CARGO ?= cargo
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
$(error KANI_VERSION must be MAJOR.MINOR.PATCH)
endif
ifneq ($(strip $(KANI_VERSION_REMAINDER)),)
$(error KANI_VERSION must be MAJOR.MINOR.PATCH)
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
	$(CARGO) fetch --locked && RUSTFLAGS="$${RUSTFLAGS:+$$RUSTFLAGS }-D warnings $(STANDARD_RUSTFLAGS)" $(CARGO) nextest run --profile $(NEXTEST_PROFILE) --all-targets --all-features $(BUILD_JOBS) -E 'not kind(bench)'

target/%/$(APP): ## Build binary in debug or release mode
	$(if $(findstring release,$(@)),RUSTFLAGS="$${RUSTFLAGS:+$$RUSTFLAGS }-Dmissing_docs -Dmissing_crate_level_docs",RUSTFLAGS="$${RUSTFLAGS:+$$RUSTFLAGS }$(STANDARD_RUSTFLAGS)") $(CARGO) build $(BUILD_JOBS) $(if $(findstring release,$(@)),--release) --bin $(APP)

lint: lint-clippy lint-whitaker ## Run Clippy and the Whitaker Dylint suite with warnings denied

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
