.PHONY: help all clean test build release typecheck lint lint-clippy lint-whitaker fmt check-fmt markdownlint nixie spelling spelling-config spelling-phrase-check spelling-helper-test kani kani-full verus bench test-workflow-contracts

export PATH := $(HOME)/.cargo/bin:$(HOME)/.bun/bin:$(PATH)

APP ?= chutoro-cli
CARGO ?= cargo
WHITAKER ?= whitaker
BUILD_JOBS ?=
CLIPPY_FLAGS ?= --all-targets --all-features -- -D warnings
RUSTDOC_FLAGS ?= --cfg docsrs -D warnings
MDLINT ?= markdownlint-cli2
NEXTEST_PROFILE ?= $(if $(CI),ci,default)
NIXIE ?= nixie
UV ?= uv
UV_ENV = UV_CACHE_DIR=.uv-cache UV_TOOL_DIR=.uv-tools
RUFF_VERSION ?= 0.15.12
PATHSPEC_VERSION ?= 1.1.1
TYPOS_VERSION ?= 1.48.0
TYPOS_CONFIG_BUILDER_COMMIT := d6da92f02240a79a945c835f69bdd08a888da1d0
TYPOS_CONFIG_BUILDER_SOURCE := git+https://github.com/leynos/typos-config-builder.git@$(TYPOS_CONFIG_BUILDER_COMMIT)
TYPOS_CONFIG_BUILDER := $(UV_ENV) $(UV) tool run --python 3.14 \
	--from "$(TYPOS_CONFIG_BUILDER_SOURCE)" typos-config-builder
VERUS_BIN ?= verus
KANI_VERSION ?= $(shell $(CARGO) kani -V | awk '{print $$2}')
KANI_LIB_PATH ?= $(HOME)/.kani/kani-$(KANI_VERSION)/toolchain/lib
KANI_ENV ?= LD_LIBRARY_PATH="$(KANI_LIB_PATH):$(LD_LIBRARY_PATH)"
SPELLING_PY_SRCS := \
	scripts/typos_rollout_check.py scripts/tests/test_typos_rollout_check.py
SPELLING_PY_TESTS := scripts/tests/test_typos_rollout_check.py
SPELLING_COVERAGE_ARGS := --cov=typos_rollout_check --cov-fail-under=90
SPELLING_HELPER_PYTEST = PYTHONPATH=scripts $(UV_ENV) $(UV) run --no-project \
	--python 3.14 --with pathspec==$(PATHSPEC_VERSION) --with pytest==9.0.2 \
	--with pytest-cov==7.0.0 python -m pytest

build: target/debug/$(APP) ## Build debug binary
release: target/release/$(APP) ## Build release binary

all: release spelling ## Default target builds release binary and checks spelling

clean: ## Remove build artefacts
	$(CARGO) clean

test: ## Run tests with warnings treated as errors
	RUSTFLAGS="-D warnings" $(CARGO) nextest run --profile $(NEXTEST_PROFILE) --all-targets --all-features $(BUILD_JOBS) -E 'not kind(bench)'

target/%/$(APP): ## Build binary in debug or release mode
	$(CARGO) build $(BUILD_JOBS) $(if $(findstring release,$(@)),--release) --bin $(APP)

lint: lint-clippy lint-whitaker ## Run Clippy and the Whitaker Dylint suite with warnings denied

lint-clippy: ## Run rustdoc and Clippy with warnings denied
	RUSTDOCFLAGS="$(RUSTDOC_FLAGS)" $(CARGO) doc --workspace --no-deps
	$(CARGO) clippy $(CLIPPY_FLAGS)

lint-whitaker: ## Run the Whitaker Dylint suite with warnings denied
	RUSTFLAGS="-D warnings" $(WHITAKER) --all -- --all-targets --all-features

typecheck: ## Type-check all workspace targets and features
	$(CARGO) check --workspace --all-targets --all-features $(BUILD_JOBS)

fmt: ## Format Rust and Markdown sources
	$(CARGO) fmt --all
	mdformat-all

check-fmt: ## Verify formatting
	$(CARGO) fmt --all -- --check

markdownlint: spelling ## Lint Markdown files and enforce spelling
	find . -type f -name '*.md' -not -path './target/*' -not -path './.verus/*' \
		-not -path './.uv-cache/*' -not -path './.uv-tools/*' -print0 | \
		xargs -0 $(MDLINT)

spelling: spelling-phrase-check ## Enforce en-GB-oxendict spelling in Markdown prose
	@git ls-files -z '*.md' | \
		xargs -0 -r env $(UV_ENV) $(UV) tool run typos@$(TYPOS_VERSION) \
		--config typos.toml --force-exclude

spelling-phrase-check: spelling-config ## Reject prohibited spelling phrases
	@PYTHONPATH=scripts $(UV_ENV) $(UV) run --no-project --python 3.14 scripts/typos_rollout_check.py --repository .

spelling-config: spelling-helper-test ## Generate and verify the spelling configuration
	@git ls-files --error-unmatch typos.toml >/dev/null
	@$(TYPOS_CONFIG_BUILDER) --repository . --check

spelling-helper-test: ## Validate the shared spelling-policy integration
	@$(UV_ENV) $(UV) tool run ruff@$(RUFF_VERSION) format --isolated --target-version py314 --check $(SPELLING_PY_SRCS)
	@$(UV_ENV) $(UV) tool run ruff@$(RUFF_VERSION) check --isolated --target-version py314 $(SPELLING_PY_SRCS)
	@$(SPELLING_HELPER_PYTEST) $(SPELLING_PY_TESTS) -c /dev/null --rootdir=. -p no:cacheprovider $(SPELLING_COVERAGE_ARGS)

nixie: ## Validate Mermaid diagrams
	find . -type f -name '*.md' -not -path './target/*' -not -path './.verus/*' \
		-not -path './.uv-cache/*' -not -path './.uv-tools/*' -print0 | \
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
