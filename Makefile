.PHONY: help all clean test build release typecheck lint fmt check-fmt markdownlint nixie kani kani-full verus bench test-workflow-contracts

export PATH := $(HOME)/.cargo/bin:$(HOME)/.bun/bin:$(PATH)

APP ?= chutoro-cli
CARGO ?= cargo
BUILD_JOBS ?=
CLIPPY_FLAGS ?= --all-targets --all-features -- -D warnings
RUSTDOC_FLAGS ?= --cfg docsrs -D warnings
MDLINT ?= markdownlint-cli2
NEXTEST_PROFILE ?= $(if $(CI),ci,default)
NIXIE ?= nixie
VERUS_BIN ?= verus
KANI_VERSION ?= $(shell $(CARGO) kani -V | awk '{print $$2}')
KANI_LIB_PATH ?= $(HOME)/.kani/kani-$(KANI_VERSION)/toolchain/lib
KANI_ENV ?= LD_LIBRARY_PATH="$(KANI_LIB_PATH):$(LD_LIBRARY_PATH)"

build: target/debug/$(APP) ## Build debug binary
release: target/release/$(APP) ## Build release binary

all: release ## Default target builds release binary

clean: ## Remove build artifacts
	$(CARGO) clean

typecheck: ## Typecheck workspace targets
	$(CARGO) check --workspace --all-targets --all-features $(BUILD_JOBS)

test: ## Run tests with warnings treated as errors
	RUSTFLAGS="-D warnings" $(CARGO) nextest run --profile $(NEXTEST_PROFILE) --all-targets --all-features $(BUILD_JOBS)

target/%/$(APP): ## Build binary in debug or release mode
	$(CARGO) build $(BUILD_JOBS) $(if $(findstring release,$(@)),--release) --bin $(APP)

lint: ## Run Clippy with warnings denied
	RUSTDOCFLAGS="$(RUSTDOC_FLAGS)" $(CARGO) doc --workspace --no-deps
	$(CARGO) clippy $(CLIPPY_FLAGS)

fmt: ## Format Rust and Markdown sources
	$(CARGO) fmt --all
	mdformat-all

check-fmt: ## Verify formatting
	$(CARGO) fmt --all -- --check

markdownlint: ## Lint Markdown files
	find . -type f -name '*.md' -not -path './target/*' -not -path './.verus/*' -print0 | \
		xargs -0 $(MDLINT)

nixie: ## Validate Mermaid diagrams
	find . -type f -name '*.md' -not -path './target/*' -not -path './.verus/*' -print0 | \
		xargs -0 $(NIXIE) --no-sandbox

kani: ## Run Kani practical harnesses
	$(KANI_ENV) $(CARGO) kani -p chutoro-core --default-unwind 4 --harness verify_bidirectional_links_smoke_2_nodes_1_layer
	$(KANI_ENV) $(CARGO) kani -p chutoro-core --default-unwind 4 --harness verify_bidirectional_links_reconciliation_2_nodes_1_layer
	$(KANI_ENV) $(CARGO) kani -p chutoro-providers-dense --default-unwind 4 --harness verify_dense_simd_dispatch_selection_respects_support_masks
	$(KANI_ENV) $(CARGO) kani -p chutoro-providers-dense --default-unwind 18 --harness verify_dense_simd_tail_padding_lane_bounds

kani-full: ## Run all Kani formal verification harnesses
	$(KANI_ENV) $(CARGO) kani -p chutoro-core --default-unwind 10
	$(KANI_ENV) $(CARGO) kani -p chutoro-providers-dense --default-unwind 18

verus: ## Run Verus proofs for edge harvest primitives
	VERUS_BIN="$(VERUS_BIN)" scripts/run-verus.sh

bench: ## Run Criterion benchmarks
	$(CARGO) bench -p chutoro-benches

test-workflow-contracts: ## Validate the mutation-testing caller contract
	uv run --with 'pytest>=8' --with 'pyyaml>=6' pytest tests/workflow_contracts -q

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS=":"; printf "Available targets:\n"} {printf "  %-20s %s\n", $$1, $$2}'
