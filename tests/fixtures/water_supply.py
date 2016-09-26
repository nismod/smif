#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""First approximation of a simulation model to be used in the framework

"""

import subprocess

from smif.abstract import SectorModel, State


class ExampleWaterSupplySimulation:
    """An example simulation model used for testing purposes

    """
    def __init__(self, raininess):
        self.raininess = raininess
        self.water = None
        self.cost = None

    def simulate(self):
        self.water = self.raininess
        self.cost = 1
        return {
            "water": self.water,
            "cost": self.cost
        }


class ExampleWaterSupplySimulationAsset(ExampleWaterSupplySimulation):
    """An example simulation model which includes assets

    Parameters
    ==========
    raininess : int

    number_of_treatment_plants : int

    """
    def __init__(self, raininess, number_of_treatment_plants):
        """Overrides the basic example class to include treatment plants

        """
        self.number_of_treatment_plants = number_of_treatment_plants
        self.water = None
        self.cost = None
        super().__init__(raininess)

    def simulate(self):
        """Runs the water supply model

        Only 1 unit of water is produced per treatment plant,
        no matter how rainy.

        Each treatment plant costs 1.0 unit.
        """
        self.water = max(self.number_of_treatment_plants, self.raininess)
        self.cost = 1.0 * self.number_of_treatment_plants
        return {
            "water": self.water,
            "cost": self.cost
        }


class WaterSupplyPython(SectorModel):
    """A concrete instance of the water supply model wrapper for testing

    """

    def initialise(self, data):
        self.model = ExampleWaterSupplySimulation(data['raininess'])
        self.results = None
        self.run_successful = None

    def optimise(self, method, decision_vars, objective_function):
        pass

    def simulate(self):
        self.results = self.model.simulate()
        self.run_successful = True

    def model_executable(self):
        pass


class WaterSupplyPythonAssets(SectorModel):
    """A concrete instance of the water supply wrapper for testing with assets

    Track the state of the model
    self.model.number_of_treatment_plants

    Methods to add & remove treatment plants
    self.model.add_a_treatment_plant(new_capacity)
    self.model.remove_a_treatment_plant(capacity_to_remove)

    """

    def initialise(self, data, assets):
        self.model = ExampleWaterSupplySimulationAsset(data['raininess'],
                                                       data['plants'])
        self.results = None
        self.run_successful = None

        state_parameter_map = {'treatment plant':
                               self.model.number_of_treatment_plants
                               }

        self.state = State('oxford', 2010,
                           'water_supply',
                           state_parameter_map)
        self.state.initialise_from_tuples(assets)

    def get_state(self):
        return self.model.number_of_treatment_plants

    def optimise(self, method, decision_vars, objective_function):
        pass

    def decision_vars(self):
        return self.model.number_of_treatment_plants

    def objective_function(self):
        return self.model.cost

    def simulate(self):
        self.results = self.model.simulate()
        self.run_successful = True

    def model_executable(self):
        pass


class WaterSupplyExecutable(SectorModel):
    """A concrete instance of the water supply which wraps a command line model

    """

    def initialise(self, data):
        self.model = self.model_executable
        self.data = data
        self.results = None
        self.run_successful = None
        self.model_executable = './tests/fixtures/water_supply_exec.py'

    def optimise(self, method, decision_vars, objective_function):
        pass

    def simulate(self):
        executable = self.model_executable
        raininess = self.data['raininess']
        argument = "--raininess={}".format(str(raininess))
        output = subprocess.check_output([executable, argument])
        self.results = process_results(output)
        self.run_successful = True


def process_results(output):
    """Utility function which decodes stdout text from the water supply model

    """
    results = {}
    raw_results = output.decode('utf-8').split('\n')
    for result in raw_results[0:2]:
        values = result.split(',')
        if len(values) == 2:
            results[str(values[0])] = float(values[1])
    return results
