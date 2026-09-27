.PHONY: help all clean test build release dev-build dev-test install-dev-fast typecheck lint lint-clippy lint-whitaker fmt check-fmt markdownlint nixie spelling kani kani-full verus bench test-workflow-contracts
.NOTPARALLEL: lint

export PATH := $(HOME)/.cargo/bin:$(HOME)/.local/bin:$(HOME)/.bun/bin:$(PATH)

APP ?= chutoro-cli
CARGO ?= cargo
DEV_FAST_TOOLCHAIN := $(strip $(file <tools/dev-fast/TOOLCHAIN))
DEV_FAST_CONFIG := tools/dev-fast/config.toml
DEV_CARGO = $(CARGO) +$(DEV_FAST_TOOLCHAIN) --config $(DEV_FAST_CONFIG)
# RUSTFLAGS overrides Cargo configuration rather than extending it. Preserve
# the Linux linker selection when the test gate adds its warning policy.
DEV_FAST_HOST_OS ?= $(shell uname -s)
DEV_TEST_RUSTFLAGS = $(strip -D warnings -Dmissing_docs \
	-Dmissing_crate_level_docs \
	$(if $(filter Linux,$(DEV_FAST_HOST_OS)),-Clink-arg=-fuse-ld=mold))
# Exact compiler diagnostics belong to the repository's stable toolchain.
STABLE_VERIFY_RUSTFLAGS = -D warnings -Dmissing_docs -Dmissing_crate_level_docs
NEXTEST_ARGS ?=
STABLE_POISONED_LOCK_TEST = hnsw::cpu::test_helpers::tests::graph_helpers_report_poisoning_without_reconfiguring
NEXTEST_FILTER = not kind(bench) & not (package(chutoro-core) & binary(result_api_surface)) & not (package(chutoro-core) & binary(session_api_surface)) & not (package(chutoro-providers-dense) & test(portable_simd_without_feature_is_rejected)) & not (package(chutoro-core) & test(=$(STABLE_POISONED_LOCK_TEST)))
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
TYPOS_CONFIG_BUILDER_VERSION ?= v0.1.1
TYPOS_CONFIG_BUILDER = $(UV_ENV) $(UV) tool run --python 3.14 --from \
	"git+https://github.com/leynos/typos-config-builder.git@$(TYPOS_CONFIG_BUILDER_VERSION)" \
	typos-config-builder
VERUS_BIN ?= verus
# Mirrors scripts/install-kani.sh, which unpacks the verifier bundle under
# KANI_HOME and links the pinned nightly toolchain into it.
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

build: ## Build debug binary
	$(DEV_CARGO) build $(BUILD_JOBS) --bin $(APP)
release: target/release/$(APP) ## Build release binary

all: release spelling ## Default target builds release binary and checks spelling

clean: ## Remove build artefacts
	$(CARGO) clean

test: ## Run tests with warnings treated as errors
	RUSTFLAGS="$(DEV_TEST_RUSTFLAGS)" $(DEV_CARGO) nextest run --config $(DEV_FAST_CONFIG) --profile $(NEXTEST_PROFILE) --all-targets --all-features $(NEXTEST_ARGS) $(BUILD_JOBS) -E '$(NEXTEST_FILTER)'
	RUSTFLAGS="$(STABLE_VERIFY_RUSTFLAGS)" env -u RUSTUP_TOOLCHAIN $(CARGO) test -p chutoro-core --all-features --test result_api_surface --test session_api_surface $(BUILD_JOBS)
	RUSTFLAGS="$(STABLE_VERIFY_RUSTFLAGS)" env -u RUSTUP_TOOLCHAIN $(CARGO) test -p chutoro-providers-dense --all-features --test portable_simd_gating $(BUILD_JOBS) -- --exact portable_simd_without_feature_is_rejected
	# Cranelift aborts on this lock-poison panic; verify under pinned stable LLVM.
	RUSTFLAGS="$(STABLE_VERIFY_RUSTFLAGS)" env -u RUSTUP_TOOLCHAIN $(CARGO) test -p chutoro-core --all-features --lib $(BUILD_JOBS) -- --exact $(STABLE_POISONED_LOCK_TEST)

target/%/$(APP): ## Build binary in debug or release mode
	$(if $(findstring release,$(@)),$(CARGO),$(DEV_CARGO)) build $(BUILD_JOBS) $(if $(findstring release,$(@)),--release) --bin $(APP)

# The explicit names use the same selected route as the standard debug gates.
dev-build: build ## Build the debug binary with the selected development backend

dev-test: test ## Run the tests with the selected development backend

install-dev-fast: ## Install pinned nightly components and the verified mold binary
	rustup toolchain install $(DEV_FAST_TOOLCHAIN) --profile minimal --component rustfmt --component clippy --component rust-analyzer --component rustc-codegen-cranelift-preview
ifeq ($(DEV_FAST_HOST_OS),Linux)
	scripts/install-mold.sh
else
	@echo 'mold is Linux-only; the platform linker remains selected'
endif

lint: lint-clippy lint-whitaker ## Run Clippy and the Whitaker Dylint suite with warnings denied

lint-clippy: ## Run rustdoc and Clippy with warnings denied
	RUSTDOCFLAGS="$(RUSTDOC_FLAGS)" $(DEV_CARGO) doc --workspace --no-deps
	$(DEV_CARGO) clippy $(CLIPPY_FLAGS)

lint-whitaker: ## Run the Whitaker Dylint suite with warnings denied
	RUSTFLAGS="-D warnings" $(WHITAKER) --all -- --all-targets --all-features

typecheck: ## Type-check all workspace targets and features
	$(DEV_CARGO) check --workspace --all-targets --all-features $(BUILD_JOBS)

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
	VERUS_BIN="$(VERUS_BIN)" scripts/run-verus.sh

bench: ## Run Criterion benchmarks
	$(CARGO) bench -p chutoro-benches

# `--doctest-modules` is what makes the documented examples load-bearing.
# The support modules carry their examples as doctests, but nothing ran
# them, so `_fraction_nanoseconds` had shipped an example that does not
# produce what it claims. An example nobody executes is a comment that
# looks like evidence.
test-workflow-contracts: ## Validate the CI workflow contracts
	uv run --with 'pytest>=8' --with 'pyyaml>=6' --with 'pathspec>=0.12' \
		--with 'hypothesis>=6' pytest tests/workflow_contracts \
		--doctest-modules -q

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS=":"; printf "Available targets:\n"} {printf "  %-20s %s\n", $$1, $$2}'
