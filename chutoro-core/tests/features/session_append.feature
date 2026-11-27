Feature: Clustering session append
  Scenario: Appending valid source indices
    Given an empty clustering session
    When I append source indices "0,1,2"
    Then the session contains 3 points
    And the snapshot version is 0

  Scenario: Duplicate index rejection
    Given an empty clustering session
    When I append source indices "0,0"
    Then the append is rejected
    And the session contains 1 points
    And the snapshot version is 0

  Scenario: Out-of-bounds index rejection
    Given an empty clustering session
    When I append source indices "3"
    Then the append is rejected
    And the session contains 0 points
    And the snapshot version is 0

  Scenario: Empty index list no-op
    Given an empty clustering session
    When I append source indices ""
    Then the append succeeds
    And the session contains 0 points
    And the snapshot version is 0

  Scenario: Snapshot version immutability across multiple appends
    Given an empty clustering session
    When I append source indices "0"
    And I append source indices "1,2"
    Then the append succeeds
    And the session contains 3 points
    And the snapshot version is 0

  Scenario: Recomputing core distances after append
    Given an empty clustering session
    When I append source indices "0,1,2"
    And I recompute core distances
    Then the append succeeds
    And source index 0 has core distance 1
    And the snapshot version is 0

  Scenario: Refreshing a populated session publishes labels for every point
    Given a clustering session over 4 points with min cluster size 2
    When I append source indices "0,1,2,3"
    And I refresh the session
    Then the label snapshot has length 4
    And the snapshot version is 1

  Scenario: Refreshing an empty session bumps the version, empty labels
    When I refresh the session
    Then the label snapshot is empty
    And the snapshot version is 1

  Scenario: The published users' guide example completes
    When I run the published users' guide refresh example
    Then the published example succeeds
