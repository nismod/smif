Feature: A researcher can choose which sector model they are developing

  Scenario:
    Given that the energy_supply sector model has been checked out of github
      And is located in the `../develop` folder
    When the command `smif develop install` is typed on the command line
    Then the model in `../develop` is installed in develop mode

  Scenario: A researcher should be able to test that their integration with smif works
    Given that the energy_supply sector model has been checked out of github
      And is located in the `../develop` folder
    When the command `smif develop test` is typed on the command line
    Then the smif diagnostics are run against the develop version of the energy_supply sector model
