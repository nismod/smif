Feature: Direct dependencies link two sector models in one direction
  Dependencies allow the exploration of interactions between sectors

  Scenario: The energy_supply sector consumes available water from the water_supply sector model
    Given that nismod is initialised with an energy_supply sector
      And nismod is initialised with a water_supply sector
      And both of the sectors cover the same region
      And are run for the same time-period
    When the simulation is performed
      And raininess is 1
    Then the energy_supply sector uses all the water
      And the energy_supply sector produces 5 electricity
      And the water_supply sector raises a shortage of 10
