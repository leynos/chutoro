.PHONY: help all clean test build release lint fmt check-fmt markdownlint nixie kani kani-full verus

APP ?= chutoro-cli
CARGO ?= cargo
BUILD_JOBS ?=
CLIPPY_FLAGS ?= --all-targets --all-features -- -D warnings
RUSTDOC_FLAGS ?= --cfg docsrs -D warnings
MDLINT ?= markdownlint
NEXTEST_PROFILE ?= $(if $(CI),ci,default)
NIXIE ?= nixie
VERUS_BIN ?= verus

build: target/debug/$(APP) ## Build debug binary
release: target/release/$(APP) ## Build release binary

all: release ## Default target builds release binary

clean: ## Remove build artifacts
	$(CARGO) clean

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
	find . -type f -name '*.md' -not -path './target/*' -print0 | xargs -0 $(MDLINT)

nixie: ## Validate Mermaid diagrams
	find . -type f -name '*.md' -not -path './target/*' -print0 | xargs -0 $(NIXIE) --no-sandbox

kani: ## Run Kani practical harnesses (smoke + 2-node reconciliation)
	$(CARGO) kani -p chutoro-core --default-unwind 4 --harness verify_bidirectional_links_smoke_2_nodes_1_layer
	$(CARGO) kani -p chutoro-core --default-unwind 4 --harness verify_bidirectional_links_reconciliation_2_nodes_1_layer

kani-full: ## Run all Kani formal verification harnesses
	$(CARGO) kani -p chutoro-core --default-unwind 10

verus: ## Run Verus proofs for edge harvest primitives
	VERUS_BIN="$(VERUS_BIN)" scripts/run-verus.sh

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS=":"; printf "Available targets:\n"} {printf "  %-20s %s\n", $$1, $$2}'
