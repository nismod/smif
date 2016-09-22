#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""First approximation of a simulation model to be used in the framework

"""

import subprocess
from smif.abstract import SectorModel


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
    """
    """
    def __init__(self, raininess, number_of_treatment_plants):
        """Overrides the basic example class to include treatment plants
        """
        self.raininess = raininess
        self.number_of_treatment_plants = number_of_treatment_plants
        self.water = None
        self.cost = None

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

    def add_a_treatment_plant(self):
        self.number_of_treatment_plants += 1

    def remove_a_treatment_plant(self):
        if self.number_of_treatment_plants > 0:
            self.number_of_treatment_plants -= 1


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

    """

    def initialise(self, data):
        self.model = ExampleWaterSupplySimulationAsset(data['raininess'],
                                                       data['plants'])
        self.results = None
        self.run_successful = None

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
        self.results = self.process_results(output)
        self.run_successful = True

    def process_results(self, output):
        results = {}
        raw_results = output.decode('utf-8').split('\n')
        for result in raw_results[0:2]:
            values = result.split(',')
            if len(values) == 2:
                results[str(values[0])] = float(values[1])
        return results
