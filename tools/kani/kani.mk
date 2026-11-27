# Kani verification wiring.
#
# The harness manifest classifies every proof; `kani-suite-build` produces the
# checker that validates a run against it; and the two tiers are the recipes
# the pull-request and nightly workflows execute. Keeping all of it in one
# included fragment gives the prover surface a single owner while the root
# Makefile keeps its own targets together.
#
# Included by the root Makefile after the prover-tools variables it uses.

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
# The verifier's shared library path, applied to the delegated `cargo kani`
# process rather than to the checker. The checker itself needs none of the
# verifier's toolchain, and the leading `NAME=VALUE` word is how the Makefile
# hands the assignment to the child without exporting it into the recipe.
KANI_ENV ?= LD_LIBRARY_PATH="$(KANI_LIB_PATH):$(LD_LIBRARY_PATH)"

# Every Kani invocation runs through this checker, which requires the exact
# successful harness multiset the manifest lists for the package and tier.
# It is built once per invocation of `make`, rather than through `cargo run` on
# every recipe line: a proof run should not pay six times for a binary the
# first line could have built, and the extra builds would spend the job's
# 30-minute budget on the queue. The seam is command-line only, so an
# environment value cannot repoint the gate; the contract tests substitute a
# recorder there.
KANI_MANIFEST ?= tools/kani/harnesses.toml
# The checker is copied to a directory this file names, from the path Cargo
# reports it wrote. That path is not assumed: a `build.target` in a user-level
# Cargo configuration, or a `CARGO_BUILD_TARGET` in the environment, moves the
# artefact under `<target-dir>/<triple>/debug/`, where an assumed
# `debug/kani_suite` is either missing or stale. Cargo's JSON build messages
# name the executable it produced, and `json-render-diagnostics` keeps the
# compiler diagnostics readable on standard error.
#
# `CARGO_TARGET_DIR` is honoured, because a caller that redirects the build
# directory would otherwise leave the proofs executing a path Cargo never
# wrote; the default is the target directory every other target uses.
CARGO_TARGET_DIR ?= target
override KANI_SUITE_TARGET_DIR := $(CARGO_TARGET_DIR)
override KANI_SUITE_DIR := $(KANI_SUITE_TARGET_DIR)/kani-suite
# Quoted because the directory comes from the environment and may contain a
# space or a shell metacharacter; the assignment reaches the shell verbatim.
override KANI_SUITE_BINARY := "$(KANI_SUITE_DIR)/kani_suite"
KANI_SUITE_PREREQ :=
ifeq ($(origin KANI_SUITE),command line)
else
override KANI_SUITE := $(KANI_SUITE_BINARY)
KANI_SUITE_PREREQ := kani-suite-build
endif

# The build names the host triple explicitly, so its output path is the one
# Cargo documents rather than one this file assumes: a `build.target` in a
# user-level Cargo configuration, or a `CARGO_BUILD_TARGET` in the environment,
# otherwise moves the artefact under `<target-dir>/<triple>/debug/`. The probe
# runs here rather than at parse time, because a `$(shell ...)` in the variable
# table would start a Cargo process for every target in this file, including
# the ones whose contracts count Cargo invocations. The checker has no
# target-specific code, so the host build is the correct one.
#
# Publication is a rename within the checker's own directory. Copying over the
# destination would fail with ETXTBSY while another `make` is running the
# checker there, and a reader could see a partial file; a rename replaces the
# directory entry atomically and leaves a running process holding the old
# inode. A failed copy therefore leaves the previous checker in place.
.PHONY: kani-suite-build
kani-suite-build: ## Build the Kani suite checker
	@set -eu; \
	host=$$($(CARGO) -vV | sed -n 's/^host: //p'); \
	test -n "$$host" || { \
		printf 'kani-suite-build could not read the host triple\n' >&2; \
		exit 1; \
	}; \
	CARGO_TARGET_DIR="$(KANI_SUITE_TARGET_DIR)" \
		$(CARGO) build --package chutoro-test-support --bin kani_suite \
		--target-dir "$(KANI_SUITE_TARGET_DIR)" --target "$$host"; \
	mkdir -p "$(KANI_SUITE_DIR)"; \
	cp "$(KANI_SUITE_TARGET_DIR)/$$host/debug/kani_suite" \
		"$(KANI_SUITE_DIR)/kani_suite.$$$$"; \
	mv -f "$(KANI_SUITE_DIR)/kani_suite.$$$$" $(KANI_SUITE_BINARY)

kani: $(KANI_SUITE_PREREQ) ## Run Kani practical harnesses
	$(KANI_SUITE) --manifest "$(KANI_MANIFEST)" --package chutoro-core --scope fast -- \
		$(KANI_ENV) $(CARGO) kani -p chutoro-core --lib --default-unwind 4 --exact \
		--harness hnsw::kani_proofs::bidirectional::verify_bidirectional_links_smoke_2_nodes_1_layer
	$(KANI_SUITE) --manifest "$(KANI_MANIFEST)" --package chutoro-core --scope fast -- \
		$(KANI_ENV) $(CARGO) kani -p chutoro-core --lib --default-unwind 4 --exact \
		--harness hnsw::kani_proofs::bidirectional::verify_bidirectional_links_reconciliation_2_nodes_1_layer
	$(KANI_SUITE) --manifest "$(KANI_MANIFEST)" --package chutoro-core --scope fast -- \
		$(KANI_ENV) $(CARGO) kani -p chutoro-core --lib --default-unwind 9 --exact \
		--harness hnsw::kani_proofs::connectivity::verify_base_layer_healing_links_isolated_node_2_nodes
	$(KANI_SUITE) --manifest "$(KANI_MANIFEST)" --package chutoro-core --scope fast -- \
		$(KANI_ENV) $(CARGO) kani -p chutoro-core --lib --default-unwind 12 --exact \
		--harness mst::kani_harness::kani_proofs::verify_mst_structural_correctness_4_nodes
	$(KANI_SUITE) --manifest "$(KANI_MANIFEST)" --package chutoro-core --scope fast -- \
		$(KANI_ENV) $(CARGO) kani -p chutoro-core --lib --default-unwind 10 --exact \
		--harness mst::kani_harness::kani_proofs::verify_mst_minimality_3_nodes
	$(KANI_SUITE) --manifest "$(KANI_MANIFEST)" --package chutoro-providers-dense --scope fast -- \
		$(KANI_ENV) $(CARGO) kani -p chutoro-providers-dense --lib --default-unwind 4 --exact \
		--harness simd::kani_proofs::verify_dense_simd_dispatch_selection_respects_support_masks
	$(KANI_SUITE) --manifest "$(KANI_MANIFEST)" --package chutoro-providers-dense --scope fast -- \
		$(KANI_ENV) $(CARGO) kani -p chutoro-providers-dense --lib --default-unwind 18 --exact \
		--harness simd::kani_proofs::verify_dense_simd_tail_padding_lane_bounds

kani-full: $(KANI_SUITE_PREREQ) ## Run all Kani formal verification harnesses
	$(KANI_SUITE) --manifest "$(KANI_MANIFEST)" --package chutoro-core --scope full -- \
		$(KANI_ENV) $(CARGO) kani -p chutoro-core --lib --default-unwind 10
	$(KANI_SUITE) --manifest "$(KANI_MANIFEST)" --package chutoro-providers-dense --scope full -- \
		$(KANI_ENV) $(CARGO) kani -p chutoro-providers-dense --lib --default-unwind 18
