Feature: Dataset recipe port contracts
  Scenario: In-memory fetcher returns declared sources
    Given the in-memory fetcher has two sources
    When the recipe fetches the declared sources
    Then the fetched bytes match the declared sources

  Scenario: Filesystem fetcher returns declared sources
    Given the filesystem fetcher has two sources
    When the recipe fetches the declared sources
    Then the fetched bytes match the declared sources
