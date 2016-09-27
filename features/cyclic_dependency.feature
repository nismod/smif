Feature: Cyclic dependencies link two sector models in both directions
  Cyclic dependencies allow the exploration of interactions between sectors but
  are not yet implemented

  Scenario: A cyclic dependency is defined raising a not-yet-implemented error
    Given that nismod is initialised with an energy_supply sector
      And that nismod is initialised with a water_supply sector
      And both of the sectors cover the same region
      And are run for the same time-period
      And a dependency for energy_supply is water
      And a dependency for water_supply is electricity
    When the simulation is performed
    Then a cyclic-dependency error is raised
