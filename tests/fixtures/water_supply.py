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
        try:
            self.results = self.model.simulate()
            self.run_successful = True
        except:
            self.results = None
            self.run_successful = False

    def model_executable(self):
        pass


class WaterSupplyExecutable(SectorModel):
    """A concrete instance of the water supply model which wraps a command line model

    """

    def initialise(self, data):
        self.model = self.model_executable()
        self.data = data
        self.results = None
        self.run_successful = None

    def optimise(self, method, decision_vars, objective_function):
        pass

    def simulate(self):
        # try:
        executable = self.model_executable()
        raininess = self.data['raininess']
        argument = "--raininess={}".format(str(raininess))
        output = subprocess.check_output([executable, argument])
        self.results = self.process_results(output)
        self.run_successful = True
        # except:
        #     self.results = None
        #     self.run_successful = False

    def process_results(self, output):
        results = {}
        raw_results = output.decode('utf-8').split('\n')
        for result in raw_results[0:2]:
            values = result.split(',')
            if len(values) == 2:
                results[str(values[0])] = float(values[1])
        return results

    def model_executable(self):
        return './tests/fixtures/water_supply_exec.py'
