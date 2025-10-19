# A framework for invariant-based correctness verification in the `chutoro` library

## Executive summary

This document presents a comprehensive design for a property-based testing
(PBT) framework for the `chutoro` library, leveraging the `proptest` and
`test-strategy` crates. The initial implementation targets three
performance-critical and algorithmically complex components: the CPU HNSW graph
implementation, the Candidate Edge Harvest algorithm, and the Parallel
Kruskal's MST implementation. This framework is designed to move beyond
traditional example-based unit testing towards a more rigorous, invariant-based
verification model.

This document details the core invariants that define correctness for each
algorithm. It proposes sophisticated data generation strategies engineered
to uncover subtle edge-case bugs and provides a complete architectural
blueprint for a two-tiered Continuous Integration (CI) system using GitHub
Actions. This CI architecture balances the need for rapid developer feedback
on pull requests with the necessity for deep, exhaustive verification
performed on a weekly basis. The ultimate objective of this initiative is to
establish a rigorous, automated correctness backstop for the `chutoro`
library. This will empower developers to perform aggressive optimizations,
conduct large-scale refactoring, and add new features with a high degree of
confidence in the system's stability and correctness.

## Section 1: Foundational principles of property-based testing with `proptest`

This section establishes the conceptual groundwork for the property-based
testing framework. A shared understanding of these principles is essential for
the engineering team to effectively implement, maintain, and derive maximum
value from the proposed verification suite. The focus is on the philosophical
shift from testing specific examples to verifying universal properties, and the
powerful tooling provided by the Rust ecosystem to facilitate this shift.

### 1.1. The paradigm shift: From example-based to invariant-based verification

Traditional unit testing, while valuable, is fundamentally limited by the
developer's imagination. A test case such as `assert_eq!(my_func(2), 4)`
verifies that the system works for a single, hand-picked input. This approach
can effectively catch simple regressions but is notoriously poor at discovering
"unknown unknowns"---bugs that occur under complex, unforeseen, or esoteric
conditions. The developer is burdened with the task of manually identifying all
relevant edge cases (zero, negative numbers, empty lists, very large inputs), a
task that is difficult for simple functions and nearly impossible for complex
algorithms like HNSW or parallel MST.

Property-based testing (PBT) inverts this model. Instead of asserting the
output for a specific input, the developer specifies a universal truth, or an
*invariant*, that must hold true for *all* valid inputs. The PBT framework then
takes on the burden of searching for a counterexample. It does this by
generating hundreds, thousands, or even millions of random inputs and checking
if the invariant holds for each one. The developer's role shifts from picking
inputs to defining properties.

The primary entry point for this in the `proptest` crate is the `proptest!`
macro. A property is defined as a function or closure that accepts randomly
generated arguments. The macro executes this function repeatedly. A failure
occurs if the function panics, returns an `Err` variant, or if a `prop_assert!`
macro within it evaluates to false. This mechanism transforms the test from a
passive check into an active search for bugs.

This paradigm shift has a profound impact beyond bug discovery. The collection
of properties for a given module becomes a form of executable documentation. A
property like
`fn search_is_at_least_as_good_as_brute_force(index: Hnsw, query: Vector)` is a
precise, machine-verifiable specification of the HNSW index's contract. Unlike
prose comments, which can become outdated, this specification is continuously
validated against the implementation by the CI system. It serves as a durable,
reliable guide for future developers seeking to understand the algorithm's
guaranteed behaviours.

### 1.2. The `proptest` ecosystem: Core primitives and the `Strategy` trait

The central abstraction in the `proptest` ecosystem is the `Strategy` trait. A
value of type `impl Strategy<Value = T>` is an object that encapsulates the
logic for both generating random values of type `T` and "shrinking" a failing
value of type `T` to a minimal counterexample. The framework provides a rich
set of primitive strategies for common types. For instance, `any::<u32>()`
generates arbitrary `u32` values, `0..100` generates integers within a specific
range, and `prop::collection::vec(any::<u8>(), 0..100)` generates vectors of
bytes with a length between 0 and 100.

The true power of `proptest` lies in the ability to compose these primitive
strategies into more complex ones. The `Strategy` trait provides a suite of
combinator methods, such as `.map()`, `.prop_map()`, `.and_then()`, and
`.prop_flat_map()`, that allow developers to build sophisticated data
generators from simple building blocks. For example, one could start with a
strategy that generates a tuple of `(width, height)` and use `.map()` to
transform it into a strategy that generates a `Rectangle` struct.

Furthermore, `proptest` offers strategies for controlling the distribution of
generated data. The `prop_oneof!` macro and the `Union` strategy allow for the
creation of a new strategy that picks from several alternatives. This is not
merely a convenience; it is a critical tool for effective testing. The efficacy
of a PBT suite is causally linked to the quality and complexity of its data
generation strategies. A simple strategy that only generates uniform,
"well-behaved" data is unlikely to uncover deep bugs. In contrast, a
sophisticated, composite strategy designed to produce pathological or "spiky"
data---for instance, a graph strategy that 95% of the time generates a random
graph but 5% of the time generates a graph with many nodes of degree zero---has
a much higher probability of discovering subtle flaws. The engineering
investment in designing these complex strategies has a direct and significant
return in the form of bugs found and system robustness gained. This approach
embodies a form of white-box testing, where knowledge of the algorithm's
potential weaknesses is used to guide the random data generator towards the
most problematic regions of the input space.

### 1.3. Accelerating strategy creation with `test-strategy`

While `proptest`'s combinators are powerful, manually implementing the
`Strategy` trait for every custom struct and enum in a large codebase like
`chutoro` can be tedious and boilerplate-intensive. The `test-strategy` crate
provides a high-level solution to this problem in the form of a procedural
macro, `#`.

By annotating a custom data structure with this macro, developers can instruct
`test-strategy` to automatically generate a corresponding `proptest::Strategy`
implementation. The macro inspects the fields of the struct or the variants of
the enum and combines the strategies for each component part into a strategy
for the whole type. For example, given the following struct:

```rust
#
struct MyStruct {
    #[strategy(1..=100u32)]
    val: u32,
    name: String,
}

```

The `test-strategy` macro will generate a strategy that produces `MyStruct`
instances where `val` is an integer between 1 and 100, and `name` is an
arbitrary `String`. This dramatically reduces the amount of manual effort
required to make the library's domain models testable, allowing engineers to
focus on defining properties rather than writing data generators.

### 1.4. The power of shrinking: Debugging with minimal counterexamples

Perhaps the single most important feature of `proptest` for debugging complex
systems is *shrinking*. When a property fails, it often does so on a large,
complex, randomly generated input---for example, a 10,000-node graph with a
specific topological feature. Attempting to debug a failure with such an input
is often intractable.

Shrinking is the process by which `proptest` takes the initial failing input
and systematically simplifies it, searching for a smaller, simpler input that
still causes the property to fail. It might try making numbers smaller,
removing elements from a collection, or picking an earlier variant of an enum.
This process continues iteratively until it finds a "minimal"
counterexample---the simplest possible input that reproduces the bug.

The result is that instead of a bug report like "Test failed on a 10,000-node
graph (seed: 12345)," the developer receives a report like "Test failed on a
3-node graph where two nodes are disconnected from the third." This minimal
counterexample is vastly easier to reason about and allows the developer to
quickly pinpoint the logical flaw in the code. The `ProptestConfig` struct
allows for the configuration of test execution parameters, such as the number
of `cases` to run. While increasing the number of cases improves the *coverage*
of the input space, it is the shrinking mechanism that makes that coverage
*practical* from a debugging and maintenance perspective. Without effective
shrinking, property-based testing in complex domains would be far less viable.

## Section 2: Invariant-based testing for the CPU HNSW graph

This section provides a detailed testing plan for the HNSW (Hierarchical
Navigable Small World) graph implementation. HNSW is a probabilistic data
structure with a set of complex structural invariants that are critical to its
performance and correctness. The goal of this PBT suite is to rigorously verify
these invariants under a wide range of data distributions and operational
sequences.

### 2.1. Characterising the HNSW input space

The primary inputs for constructing an HNSW index are a set of high-dimensional
vectors, a distance metric (e.g., Euclidean, Cosine), and a set of construction
parameters (e.g., `M`, the maximum number of neighbours per layer, and
`ef_construction`, the size of the dynamic candidate list during insertion).
The PBT strategies must be designed to explore this input space thoroughly.

A `proptest` strategy will be developed to generate the input `Vec<Vec<f32>>`.
Crucially, this strategy must not be limited to generating uniformly random
vectors. Real-world data is often highly structured, and ANN algorithms like
HNSW can exhibit performance or correctness degradation on such data.
Therefore, the vector generation strategy will be a composite one, using
`prop_oneof!`, designed to produce varied distributions:

- **Uniformly Random Vectors:** Points scattered randomly within a hypercube.
    This serves as a baseline case.

- **Clustered Vectors:** A strategy that first generates `k` random centroid
    vectors and then generates points with a Gaussian or normal distribution
    around each centroid. This tests the algorithm's ability to handle dense
    regions in the vector space.

- **Low-Dimensional Manifold Vectors:** A strategy that generates points that
    lie on or near a lower-dimensional subspace (e.g., a plane or sphere)
    embedded within the high-dimensional space. This can challenge the
    layer-assignment heuristic in HNSW.

- **Vectors with Duplicates:** A strategy that explicitly introduces multiple
    copies of the same vector to test for correct handling of duplicate points.

Explicitly generating these "pathological" datasets proactively tests the
algorithm's robustness at its known weak points. That proactive strategy is
far more effective than waiting for failures to emerge from specific user data.

For the HNSW construction parameters, a struct like `HnswConfig` can be
annotated with `#` from `test-strategy`. This will allow `proptest` to
automatically explore how the algorithm behaves with different values for `M`,
`ef_construction`, and other tuning parameters, potentially uncovering bugs
that only manifest under specific configurations.

### 2.2. Core structural invariants of a valid HNSW graph

A correctly constructed HNSW graph must adhere to several structural invariants
at all times. The PBT suite will include functions to check these invariants,
which will be called within properties after construction or mutation
operations.

- **Invariant 1: Layer Consistency:** For any node `u` that exists at a layer
    `L > 0`, that same node `u` must also exist at all lower layers `l` where
    `$0 \le l < L$`. A violation of this invariant would break the navigational
    structure of the graph.

- **Invariant 2: Neighbourhood size constraints:** The number of neighbours for
    any node at a layer `L > 0` must be less than or equal to the parameter
    `M`. The number of neighbours at the base layer (`L = 0`) must be less than
    or equal to `$2 \times M$`. These constraints are fundamental to the "Small
    World" property and keep the graph sparse.

- **Invariant 3: Entry Point Reachability:** Every node in the graph must be
    reachable from the designated global entry point by traversing the graph's
    edges (at any layer). If a node becomes disconnected, it is no longer
    searchable, violating the core purpose of the index. This can be verified
    using a graph traversal algorithm like Breadth-First Search (BFS) or
    Depth-First Search (DFS) starting from the entry point.

- **Invariant 4: Bidirectional Links (if applicable):** Many HNSW
    implementations rely on symmetric edges for efficient traversal. If the
    `chutoro` implementation assumes that an edge `$(u, v)$` implies the
    existence of an edge `$(v, u)$`, this property must be rigorously verified
    across the entire graph.

### 2.3. Property-based tests for HNSW

With the input strategies and invariant checkers in place, the following
properties will be implemented.

#### 2.3.1. Property 1: Search correctness (oracle-based testing)

This is the most critical property for the HNSW component. It directly verifies
that the index fulfils its primary function: finding approximate nearest
neighbours efficiently.

- **Description:** The property will first generate a set of vectors and HNSW
    construction parameters. It will then build an HNSW index from these
    vectors. Subsequently, it will generate a random query vector and a value
    for `k`. The test will find the `k` nearest neighbours to the query vector
    using two different methods:

    1. The `chutoro` HNSW index's search function.

    2. A simple, trusted, brute-force linear scan that computes the distance
        to every point in the dataset and sorts the results. This linear scan
        acts as the "oracle."

- **Assertion:** The property will assert that the set of neighbours returned
    by the HNSW search is identical to the set returned by the brute-force
    oracle. For an ANN algorithm, this can be relaxed to assert a minimum
    recall percentage (e.g., `recall >= 0.99`), especially for larger datasets
    where perfect recall may not be guaranteed.

- **Justification:** This property provides an end-to-end correctness check.
    It validates not only the graph structure but also the search algorithm
    that navigates it. Furthermore, this test serves a dual purpose. By timing
    both the HNSW search and the brute-force search within the property, the
    test can log the performance speedup. Aggregating this data across a CI run
    provides a statistical distribution of the algorithm's performance on
    varied data. A sudden drop in the average speedup ratio can signal a
    performance regression, transforming the PBT suite into a lightweight
    performance monitoring tool and providing early warnings without the
    overhead of a dedicated benchmarking framework for every pull request.

#### 2.3.2. Property 2: Structural integrity after operations

This property tests the robustness of the graph's mutation logic, which is a
common source of subtle bugs like dangling pointers or inconsistent state.

- **Description:** This will be a stateful property test. The `proptest`
    strategy will generate an initial set of vectors and a sequence of
    subsequent operations. The `Operation` enum will include variants like
    `Add(Vector)`, `Delete(NodeId)`, and potentially `Reconfigure(HnswConfig)`.

- **Execution Flow:**

    1. Build an initial HNSW index from the starting vectors.

    2. Verify that all core structural invariants from section 2.2 hold.

    3. Iterate through the generated sequence of operations, applying each one
        to the index.

    4. After *every* operation, re-verify that all structural invariants still
        hold.

- **Justification:** This property targets bugs that arise from the complex
    state transitions involved in adding or removing nodes from the
    multi-layered graph structure. A deletion, for example, must correctly
    update neighbour lists across multiple layers without violating degree
    constraints or creating disconnected components.

#### 2.3.3. Property 3: Idempotency of insertion

This property verifies that the insertion logic correctly handles duplicate
data and avoids unintended side effects.

- **Description:** The strategy will generate a set of vectors and select one
    vector to be the "duplicate." It will then construct two HNSW indices. The
    first index will be built from the original set of vectors. The second
    index will be built from the same set but with the "duplicate" vector added
    multiple times. The property will assert that the final graph structures of
    both indices are identical.

- **Justification:** A robust system should produce the same result
    regardless of how many times the same piece of data is inserted. This test
    ensures that repeated insertions do not corrupt the graph state, for
    example by creating duplicate nodes or exceeding degree constraints.

## Section 3: Verifying the candidate edge harvest algorithm

This section details the testing strategy for the Candidate Edge Harvest
algorithm. This algorithm is likely a heuristic used during graph construction
(e.g., for NN-Descent or a similar neighbour-graph refinement process). Testing
heuristics with PBT presents a different challenge than testing algorithms with
a single, well-defined correct answer. The focus shifts from verifying a
specific output to ensuring the output adheres to a set of desirable properties
and behavioural bounds.

### 3.1. Characterising the input and output space

- **Input:** The primary input is a graph, likely represented as an adjacency
    list, where each node has a set of "candidate" neighbours. This graph is the
    result of a prior, possibly noisy, neighbour-finding step.

- **Output:** The output is a refined graph where the harvesting heuristic
    has been applied. This typically involves pruning some edges and adding
    others to improve the graph's quality (e.g., for subsequent search
    operations).

The `proptest` strategy for this component must generate a wide variety of
input graphs to test the heuristic's behaviour under different topological
conditions. A composite strategy using `prop_oneof!` will be designed to
generate graphs with distinct characteristics:

- **Random Graphs:** Erdos-Renyi style graphs where edges exist with a
    uniform probability.

- **Scale-Free Graphs:** Graphs exhibiting a power-law degree distribution,
    with a few high-degree "hub" nodes and many low-degree nodes. These are
    known to be challenging for many graph algorithms.

- **Grid or Lattice Graphs:** Highly structured graphs with uniform local
    connectivity.

- **Graphs with Disconnected Components:** To test how the heuristic behaves
    when the input is not fully connected.

By systematically generating these varied topologies, it becomes possible to
establish connections between the input structure and the heuristic's behaviour.
If, for example, a property consistently fails only on scale-free graphs, it
provides a powerful clue that the heuristic's logic is flawed in its handling
of high-degree nodes. This targeted diagnostic information is far more valuable
than a single, isolated failure on a random input and can guide developers
directly to the logical flaw.

### 3.2. Defining heuristic invariants and properties

Since there is no single "correct" output, the properties for the harvesting
algorithm will verify its behavioural characteristics.

- **Property 1: Determinism:** For a given input graph and a fixed random
    seed, the output of the harvesting algorithm must be identical across
    multiple runs. Many heuristics involve choices, and it is critical to
    ensure these choices are made deterministically (e.g., by sorting
    candidates before selection) rather than being subject to uncontrolled
    sources of randomness like `HashMap` iteration order. This property ensures
    the algorithm is repeatable and debuggable.

- **Property 2: Adherence to Degree Constraints:** The heuristic is often
    responsible for managing the graph's density. This property will assert
    that the degree of any node in the output graph is within a specified,
    acceptable range (e.g., `prop_assert!(graph.degree(node) <= MAX_DEGREE)`).
    This verifies that the edge pruning logic is working correctly and not
    creating overly dense nodes that would harm search performance.

- **Property 3: Connectivity Preservation (or Bounded Destruction):** A good
    refinement heuristic should not catastrophically degrade the graph's
    connectivity. This property will check that if the input graph is
    connected, the output graph is also connected. If the heuristic is
    intentionally aggressive, the property can be relaxed to assert that the
    number of connected components does not increase by more than a small,
    bounded amount.

- **Property 4: Reverse nearest neighbour (RNN) inclusion:** A common
    heuristic in neighbour graph construction is based on the idea of
    strengthening symmetric connections. The property can verify that the
    heuristic is working as intended by asserting that the proportion of
    symmetric (RNN) edges in the output graph is significantly higher than in
    the input graph. This provides a quantitative measure of the heuristic's
    effectiveness.

These properties do more than just pass or fail; they provide quantitative
measurements of the output graph's characteristics (degree distribution,
connectivity, symmetry). By logging these metrics during PBT runs, the test
suite becomes an analysis tool. Engineers can observe how the heuristic behaves
across thousands of generated graph topologies. This data can then be used to
make informed, data-driven decisions when tuning the heuristic's internal
parameters for optimal real-world performance.

## Section 4: Ensuring correctness in the parallel Kruskal's MST implementation

This section addresses the verification of the parallel implementation of
Kruskal's algorithm for finding a Minimum Spanning Tree (MST). Testing a
parallel algorithm requires a dual focus: ensuring the final result is correct
according to the mathematical definition of an MST, and ensuring the
implementation is free from concurrency bugs like race conditions and
non-determinism.

### 4.1. Defining the invariants of a minimum spanning tree (MST)

Any graph claiming to be an MST of a given input graph must satisfy a set of
strict mathematical invariants. These form the basis of the property tests.

- **Invariant 1: Acyclicity:** The output graph must be a forest (or a tree
    if the input is connected), meaning it must not contain any cycles. This
    can be verified with a cycle detection algorithm (e.g., using a Disjoint
    Set Union data structure).

- **Invariant 2: Connectivity:** If the input graph is connected, the output
    graph must span all vertices and must also be connected. This ensures it is
    a *spanning* tree.

- **Invariant 3: Minimality (The Cut Property):** This is the defining
    property of an MST. For any "cut" that partitions the graph's vertices into
    two disjoint sets, if an edge `$(u, v)$` from the MST crosses that cut, its
    weight must be less than or equal to the weight of any other edge that also
    crosses the cut. While difficult to check directly, this property is
    implicitly verified by comparing the total weight of the tree against an
    oracle.

- **Invariant 4: Edge Count:** For a connected input graph with `$V$`
    vertices, the resulting MST must contain exactly `$V - 1$` edges.

### 4.2. Strategies for generating pathological graphs

The correctness and stability of a parallel algorithm are most effectively
tested by feeding it inputs designed to stress its concurrency logic. The
`proptest` strategy for generating the input weighted, undirected graph will
therefore be the most critical component of this test suite. It will be a
composite strategy using `prop_oneof!` to generate graphs specifically designed
to probe for weaknesses.

- **Graphs with Unique Edge Weights:** This is the simplest case, where the
    MST is unique and the greedy choices made by Kruskal's algorithm are
    unambiguous. This serves as a baseline correctness check.

- **Graphs with Many Identical Edge Weights:** This is the most important and
    challenging case. When multiple edges have the same weight, there can be
    multiple valid MSTs with the same total weight. A parallel implementation
    must handle the ambiguity of which edge to select next in a correct and
    deterministic manner. This scenario is designed to create contention and
    flush out race conditions in the logic that merges components or selects
    edges in parallel.

- **Disconnected Graphs:** The algorithm should correctly produce a minimum
    spanning *forest*, one MST for each connected component of the input graph.

- **Dense vs. Sparse Graphs:** The test suite will generate graphs at both
    extremes of edge density to ensure the algorithm's performance and
    correctness are not dependent on the graph's sparsity.

### 4.3. Property-based tests for parallel Kruskal's

The following properties will be implemented to verify the parallel Kruskal's
implementation against the invariants and pathological inputs.

#### 4.3.1. Property 1: Equivalence with a sequential oracle

This is the primary correctness check for the algorithm.

- **Description:** For any generated input graph, the test will compute the
    MST using two methods:

    1. The `chutoro` parallel Kruskal's implementation.

    2. A simple, trusted, sequential implementation of Kruskal's algorithm,
        which can be implemented directly in the test suite. This sequential
        version serves as the oracle.

- **Assertion:** The property will assert that the *total weight* of the MST
    produced by the parallel implementation is identical to the total weight of
    the MST produced by the sequential oracle. It is crucial to compare total
    weights rather than the set of edges, as multiple valid MSTs with the same
    weight can exist if the graph contains duplicate edge weights.

- **Justification:** This property provides a strong guarantee of correctness
    against a known-good implementation. This test also implicitly enforces a
    deterministic tie-breaking rule. The sequential oracle, by sorting edges,
    has a stable, implicit rule for which edge to pick first among those with
    equal weight. By comparing results, the PBT suite pressures the parallel
    implementation to conform to a similarly deterministic behaviour, making the
    algorithm's output fully predictable even in ambiguous cases.

#### 4.3.2. Property 2: Structural invariant verification

This property provides a secondary correctness check that does not rely on the
oracle.

- **Description:** For any MST or minimum spanning forest produced by the
    parallel algorithm, this test will directly verify the structural
    invariants defined in section 4.1. It will run a cycle detection algorithm,
    check for connectivity on the appropriate vertex sets, and verify that the
    number of edges is correct (`$V - C$`, where `$C$` is the number of
    connected components).

- **Justification:** This test ensures that the output is a structurally
    valid spanning tree/forest, guarding against bugs that might lead to a
    graph with the correct weight but an invalid structure (e.g., containing a
    cycle). It provides a safety net in the unlikely event that the sequential
    oracle itself has a bug.

#### 4.3.3. Property 3: Concurrency safety

This property is specifically designed to detect non-determinism arising from
race conditions.

- **Description:** Within a single `proptest!` execution, this property will
    run the parallel Kruskal's algorithm on the *same input graph* multiple
    times in a loop (e.g., 5-10 times). It will then assert that the total
    weight of the resulting MST is identical in every single run.

- **Justification:** If a data race or other concurrency bug exists,
    different OS thread interleavings across the multiple runs could lead to
    different (and likely incorrect) choices when processing edges of equal
    weight. This would result in MSTs with different total weights, causing the
    property to fail. This test acts as a "fuzzer for thread interleavings."
    While it cannot prove the absence of race conditions, `proptest`'s ability
    to generate a vast number of different input graphs dramatically increases
    the probability of hitting a rare thread scheduling sequence that exposes a
    race. When run under a thread sanitizer like `TSan`, this property becomes
    an exceptionally powerful tool for detecting and diagnosing data races in
    the underlying concurrent data structures.

### Table 4.1: Summary of core properties and invariants

| **Component**          | **Property Name**         | **Invariant(s) Protected**                          | **Test Method**                           |
| ---------------------- | ------------------------- | --------------------------------------------------- | ----------------------------------------- |
| HNSW                   | Search Correctness        | Returns the true (or near-true) k-NN set            | Oracle (Brute-Force Linear Scan)          |
| HNSW                   | Structural Integrity      | Layer Consistency, Neighbourhood Size, Reachability | Stateful Invariant Checks After Mutations |
| HNSW                   | Idempotency of Insertion  | State is unaffected by duplicate insertions         | Comparison of graph states                |
| Candidate Edge Harvest | Determinism               | Repeatable output for a given input                 | Multiple runs and comparison              |
| Candidate Edge Harvest | Degree Constraints        | Node degrees remain within specified bounds         | Direct graph traversal and degree check   |
| Candidate Edge Harvest | Connectivity Preservation | Graph connectivity is not catastrophically degraded | Comparison of connected components        |
| Parallel Kruskal's MST | Equivalence with Oracle   | MST Minimality (Correct Total Weight)               | Oracle (Sequential Kruskal's)             |
| Parallel Kruskal's MST | Structural Integrity      | Acyclicity, Connectivity, Edge Count                | Direct Invariant Checks                   |
| Parallel Kruskal's MST | Concurrency Safety        | Determinism, Freedom from Race Conditions           | Repeated Execution & Comparison           |

## Section 5: A two-tiered CI architecture with GitHub Actions

To effectively integrate this PBT framework into the development lifecycle, a
two-tiered CI architecture will be implemented using GitHub Actions. This
architecture is designed to balance two competing needs: the need for rapid,
low-friction feedback for developers during the pull request process, and the
need for deep, exhaustive verification to catch rare and subtle bugs.

### 5.1. The lightweight PR suite: Rapid feedback

The primary goal of this suite is to provide essential correctness checks on
every pull request without imposing a significant delay on the development
workflow.

- **Trigger:** This workflow will be triggered on every push to a pull
    request targeting the main branch, using the `on: [pull_request]` event.

- **Configuration:**

  - **Test Cases:** The `ProptestConfig` will be set to a relatively low
        number of test cases (e.g., `cases: 250`). This provides a reasonable
        level of coverage to catch common regressions and simple bugs quickly.

  - **Timeout:** A hard timeout will be enforced on the job using
        `timeout-minutes: 10` in the GitHub Actions workflow configuration.
        This prevents a misbehaving or non-terminating test from blocking the
        entire CI queue.

- **Optimization via selective execution:** To minimize CI costs and feedback
    time, the workflow will be structured into separate jobs for each major
    component (HNSW, Kruskal, etc.). Path-based filtering will be used to
    ensure that only the relevant test suite is run. For example, a change to
    files within `src/hnsw/` will only trigger the HNSW PBT job. This prevents
    redundant test runs and provides more targeted feedback.

### Table 5.1: Comparison of CI suite configurations

| **Parameter**          | **PR Suite (Lightweight)**     | **Weekly Suite (Exhaustive)**           |
| ---------------------- | ------------------------------ | --------------------------------------- |
| **Trigger**            | `on: pull_request`             | `on: schedule` (e.g., weekly)           |
| **`Proptest` Cases**   | `250` (default)                | `25,000` (via env var `PROGTEST_CASES`) |
| **`Proptest` Forking** | `false`                        | `true`                                  |
| **Job Timeout**        | `10 minutes`                   | `120 minutes`                           |
| **Execution Scope**    | Path-filtered (selective jobs) | Full repository test suite              |
| **Failure Action**     | Block PR merge                 | Alert team & upload failure artifact    |

### 5.2. The weekly deep-coverage analysis: Exhaustive verification

The goal of this suite is to perform a much more intensive search for bugs that
are too computationally expensive to run on every PR. This is where the true
power of PBT to find deep, rare bugs is realized.

- **Trigger:** This workflow will run on a fixed schedule, for example, every
    Sunday at 2:00 AM UTC, using the `on: schedule: - cron: '0 2 * * 0'` syntax.

- **Configuration:**

  - **Test Cases:** The number of test cases will be significantly
        increased. This will be managed via an environment variable,
        `PROGTEST_CASES=25000`, which the test runner code will be configured
        to read. This allows the same test code to be used in both suites, with
        behaviour configured at the CI level.

  - **Forking:** `proptest` will be configured with `fork: true`. This runs
        each set of test cases for a property in a separate process. It
        provides better isolation, preventing a crash or memory corruption bug
        in one test from bringing down the entire test suite.

  - **Timeout:** The job timeout will be extended significantly, for
        example, to `timeout-minutes: 120`, to accommodate the much longer run
        time.

- **Failure Management:** Failures in the weekly run require a robust
    reporting mechanism.

  - **Artifacts:** When a `proptest` test fails, it reports the seed that
        can be used to reproduce the failure. The CI workflow will be
        configured to capture this seed from the test logs and upload it as a
        build artifact. This allows a developer to download the seed and
        reproduce the exact failing test case locally without needing to re-run
        the entire 25,000-case suite.

  - **Notifications:** The workflow will be configured to send an automated
        notification (e.g., to a designated Slack channel or mailing list) upon
        failure, ensuring that the team is promptly alerted to any regressions
        discovered by the deep analysis.

### 5.3. YAML configuration blueprints

The following provides skeleton YAML configurations for the two GitHub Actions
workflows.

#### 5.3.1. `pr-checks.yml`

```yaml
name: PR Checks

on:
  pull_request:
    branches: [ main ]

jobs:
  test-hnsw:
    if: |
      github.event.pull_request.draft == false &&
      (contains(github.event.pull_request.labels.*.name, 'run-ci'))
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v3
      - name: Setup Rust toolchain
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
          override: true
      - name: Run HNSW PBT Suite
        run: cargo test --package chutoro --test hnsw_pbt -- --nocapture

  test-kruskal:
    if: |
      github.event.pull_request.draft == false &&
      (contains(github.event.pull_request.labels.*.name, 'run-ci'))
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v3
      - name: Setup Rust toolchain
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
          override: true
      - name: Run Kruskal PBT Suite
        run: cargo test --package chutoro --test kruskal_pbt -- --nocapture

```

*(Note: The above uses labels for triggering as a robust alternative to path
filtering, which can be complex with workspace dependencies.)*

#### 5.3.2. `weekly-deep-test.yml`

```yaml
name: Weekly Deep Coverage Analysis

on:
  schedule:
    - cron: '0 2 * * 0' # Every Sunday at 2 AM UTC
  workflow_dispatch: # Allows manual triggering

jobs:
  deep-test:
    runs-on: ubuntu-latest
    timeout-minutes: 120
    steps:
      - uses: actions/checkout@v3
      - name: Setup Rust toolchain
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
          override: true
      - name: Run Full PBT Suite (Deep)
        env:
          PROGTEST_CASES: "25000"
          RUST_BACKTRACE: "1"
        run: |
          cargo test --workspace -- --nocapture || \
            (echo "PROGTEST_FAILURE_SEED=$(grep -o 'proptest-seed: [0-9]*' test_output.log \
              | cut -d' ' -f2)" >> $GITHUB_ENV && exit 1 \
            )
        continue-on-error: true
      - name: Upload Failure Seed
        if: failure()
        uses: actions/upload-artifact@v3
        with:
          name: proptest-failure-seed
          path: |
            **/proptest-regressions/*
      - name: Notify on Failure
        if: failure()
        run: |
          # Add notification script here (e.g., curl to Slack webhook)
          echo "Weekly deep test failed. See artifacts for failure seed."

```

## Section 6: Strategic implementation and best practices

The successful adoption of this PBT framework requires more than just technical
implementation; it requires a strategic rollout plan, a commitment to best
practices, and a clear understanding of how to leverage the framework for
debugging and design improvement.

### 6.1. An incremental rollout plan

Adopting a new and powerful testing paradigm like PBT can have a learning
curve. To ensure successful adoption and build team expertise incrementally, a
phased rollout is recommended. This approach is fundamentally a risk mitigation
strategy. By starting with the most straightforward component, the team can
build confidence and familiarity with the tools and concepts before tackling
the more nuanced and complex testing challenges.

- **Phase 1: Foundations & Parallel Kruskal's.** This is the ideal starting
    point. Kruskal's algorithm has a clear, binary definition of correctness
    and a simple oracle (a sequential implementation). This allows the team to
    focus on learning the `proptest` API, writing basic strategies, and
    understanding the test-failure-debug cycle in a controlled environment.

- **Phase 2: HNSW Integration.** With the fundamentals established, the team
    can move on to the HNSW component. This will introduce the challenges of
    testing a probabilistic data structure, requiring the implementation of
    more complex properties like approximate recall and stateful invariant
    checking. The strategies for generating varied vector distributions will
    also be a key learning objective.

- **Phase 3: Heuristic Verification & CI Integration.** The final
    implementation phase involves tackling the Candidate Edge Harvest
    algorithm. This will require the team to fully embrace the concept of
    testing for behavioural properties rather than a single correct answer. Once
    all three test suites are implemented and stable, they will be integrated
    into the two-tiered CI pipeline described in Section 5.

### 6.2. Writing effective and maintainable properties

- **Focus:** Each `proptest!` block should test a single, well-defined
    property or invariant. Avoid creating monolithic tests that check multiple
    unrelated things.

- **Composability:** Data generation logic should be encapsulated in
    standalone functions that return an `impl Strategy`. These functions should
    be placed in a dedicated, shared module (e.g., `tests/strategies.rs`) to
    promote reuse across different test suites.

- **Clarity:** Use the `prop_assert!` macros with descriptive failure
    messages (e.g.,
    `prop_assert!(is_acyclic, "The generated MST contains a cycle")`). This
    provides immediate context when a test fails.

### 6.3. Debugging `proptest` failures

When a test fails, `proptest` provides a detailed report. The key piece of
information is the failure seed. To debug, a developer can set the
`PROGTEST_SOURCE_FILE` environment variable to point to the failing test file
and run the test again. `proptest` will automatically pick up the failure from
the regression directory and deterministically reproduce it. The most important
step is to analyse the *shrunken* counterexample provided in the report. This
minimal input is the key to understanding the root cause of the bug, as
described in Section 1.4.

### 6.4. Performance profiling of the test suite

The test suite itself is code and can have performance bottlenecks,
particularly in complex data generation strategies. If the weekly CI run time
becomes unacceptably long, tools like `cargo-flamegraph` should be used to
profile the test execution. This can identify slow strategies that may need
optimization.

### 6.5. Future work: Extending the framework

The PBT framework established here for the core algorithms can and should be
extended to other parts of the `chutoro` library. Potential future targets
include:

- **Serialization/Deserialization:** Properties can be written to assert that
    for any generated `Hnsw` graph, `deserialize(serialize(graph)) == graph`.
    This is a classic "round-trip" property test.

- **Data Ingestion Pipelines:** Test that the data processing pipeline is
    robust to malformed or unexpected input formats.

- **Other Graph Algorithms:** Apply the same principles of invariant-based
    testing to any other graph algorithms within the library.

Ultimately, this PBT suite should be promoted as a key quality feature of the
`chutoro` project, providing a strong, verifiable guarantee of correctness to
its users.

The process of building this framework will also yield a significant, positive
side effect on the library's design. As engineers attempt to write `proptest`
strategies for the library's data structures, they will inevitably find that
APIs which are stateful, highly coupled, or have complex setup requirements are
difficult to test. For instance, if creating a data structure for a test
requires five separate method calls, it signals a design smell. This difficulty
provides strong, direct feedback that the API could be improved. The effort to
make the code more testable---for example, by creating a simple `::new()` or
`::from_vectors()` constructor---often leads to a cleaner, more modular, and
more ergonomic public API for the library's end-users. In this way, the PBT
suite acts as an evolutionary pressure, driving the entire codebase towards a
higher standard of quality and design.
