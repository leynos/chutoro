Feature: Minimum spanning forest edge preparation
  Candidate edge i joins nodes (i mod N) and ((i + 1) mod N), has distance
  i + 1, and has sequence i, where N is the node count. Each scenario then
  changes the listed edges before the forest is built.

  Scenario: Mutual-reachability weights drive forest selection
    Given a graph with 6 nodes and 9 candidate edges
    And node 3 has core distance 10 and every other node has core distance 0
    When I build the minimum spanning forest
    Then the forest has 5 edges
    And the forest excludes the edge between nodes 3 and 4

  Scenario: Duplicate candidate edges are collapsed
    Given a graph with 4 nodes and 8 candidate edges
    And the last 4 candidate edges are exact copies of the first 4
    When I build the minimum spanning forest
    Then the forest has 3 edges
    And every forest edge has source less than target

  Scenario: Self-loops are discarded without failing
    Given a graph with 5 nodes and 7 candidate edges
    And the last 2 candidate edges are self-loops
    When I build the minimum spanning forest
    Then the forest has 4 edges

  Scenario: The lowest-index invalid edge is reported
    Given a graph with 4 nodes and 6 candidate edges
    And candidate edge 1 references node 9
    And candidate edge 5 has a non-finite weight
    When I build the minimum spanning forest
    Then forest construction fails with error code "INVALID_NODE_ID"

  Scenario: An edge that is both out of range and non-finite reports its node
    Given a graph with 4 nodes and 3 candidate edges
    And candidate edge 2 references node 9
    And candidate edge 2 has a non-finite weight
    When I build the minimum spanning forest
    Then forest construction fails with error code "INVALID_NODE_ID"

  Scenario: A disconnected graph yields a forest, not a tree
    Given a graph with 6 nodes and 4 candidate edges
    And the candidate edges span only nodes 0 to 3
    When I build the minimum spanning forest
    Then the forest reports 3 components
    And the forest is not a tree
