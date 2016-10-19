#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""First approximation of a simulation models to be used in the framework

Simulation Models
-----------------

There are a bunch of very simple simulation models to demonstrate different
functionality.

`ExampleWaterSupplySimulation`: simulation only, no assets

`ExampleWaterSupplySimulationAsset`: simulation of water with outputs also
a function of assets

`ExampleWaterSupplySimulationReservoir`: simulation with outputs a function
of a reservoir level which should be persistent over years


Wrappers Around the Models
--------------------------

Around each of the above simulation models, subclassed wrappers based on
:class:`SectorModel` are used to present a consistent API

`WaterSupplyPython` - wraps `ExampleWaterSupplySimulation`

`WaterSupplyPythonAssets` - wraps `ExampleWaterSupplySimulationAsset`

`WaterSupplyExecutable` - wraps `water_supply_exec.py`

"""

import math
import subprocess

from smif.abstract import AbstractModelWrapper


def one_input():
    inputs = {'decision variables': ['water treatment capacity'],
              'parameters': ['raininess'],
              'water treatment capacity': {'bounds': (0, 20),
                                           'index': 0,
                                           'init': 10
                                           },
              'raininess': {'bounds': (0, 5),
                            'index': 0,
                            'value': 3
                            }
              }

    return inputs


def two_inputs():
    inputs = {'decision variables': ['water treatment capacity',
                                     'reservoir pumpiness'],
              'parameters': ['raininess'],
              'water treatment capacity': {'bounds': (0, 20),
                                           'index': 1,
                                           'init': 10
                                           },
              'reservoir pumpiness': {'bounds': (0, 100),
                                      'index': 0,
                                      'init': 24.583
                                      },
              'raininess': {'bounds': (0, 5),
                            'index': 0,
                            'value': 3
                            }
              }
    return inputs


def raininess_oracle(timestep):
    """Mimics an external data source for raininess

    Arguments
    =========
    timestep : int
        Requires a year between 2010 and 2050

    Returns
    =======
    raininess : int

    """
    msg = "timestep {} is outside of the range [2010, 2050]".format(timestep)
    assert timestep in [x for x in range(2010, 2051, 1)], msg

    raininess = math.floor((timestep - 2000) / 10)

    return raininess


class WaterModelWrapExec(AbstractModelWrapper):

    def simulate(self, static_inputs, decision_variables):
        """Runs the executable version of the water supply model

        Arguments
        =========
        static_inputs : x-by-1 :class:`numpy.ndarray`
            x_0 is raininess
            x_1 is capacity of water treatment plants
        """
        model_executable = './tests/fixtures/water_supply_exec.py'
        argument = "--raininess={}".format(str(static_inputs))
        output = subprocess.check_output([model_executable, argument])
        results = process_results(output)
        return results

    def extract_obj(self, results):
        return results


def process_results(output):
    """Utility function which decodes stdout text from the water supply model

    Returns
    =======
    results : dict
        A dictionary where keys are the results e.g. `cost` and `water`

    """
    results = {}
    raw_results = output.decode('utf-8').split('\n')
    for result in raw_results[0:2]:
        values = result.split(',')
        if len(values) == 2:
            results[str(values[0])] = float(values[1])
    return results


class WaterSupplySimulationWrapper(AbstractModelWrapper):
    """Provides an interface for :class:`ExampleWaterSupplySimulation`
    """

    def simulate(self, static_inputs, decision_variables):
        """

        Arguments
        =========
        static_inputs : x-by-1 :class:`numpy.ndarray`
            x_0 is raininess
            x_1 is capacity of water treatment plants
        """
        raininess = static_inputs
        instance = self.model(raininess)
        results = instance.simulate()
        return results

    def extract_obj(self, results):
        return results['cost']


class ExampleWaterSupplySimulation:
    """An example simulation model used for testing purposes

    Parameters
    ==========
    raininess : int
        The amount of rain produced in each simulation
    """
    def __init__(self, raininess):
        self.raininess = raininess
        self.water = None
        self.cost = None

    def simulate(self):
        """Run the model

        Returns
        =======
        dict
        """
        self.water = self.raininess
        self.cost = 1
        return {
            "water": self.water,
            "cost": self.cost
        }


class WaterSupplySimulationAssetWrapper(AbstractModelWrapper):
    """Provides an interface for :class:`ExampleWaterSupplyAssetSimulation
    """

    def simulate(self, static_inputs, decision_variables):
        """

        Arguments
        =========
        static_inputs : x-by-1 :class:`numpy.ndarray`
            x_0 is raininess
            x_1 is capacity of water treatment plants
        """
        raininess = static_inputs
        capacity = decision_variables
        instance = self.model(raininess, capacity)
        results = instance.simulate()
        return results

    def extract_obj(self, results):
        return results['cost']


class ExampleWaterSupplySimulationAsset(ExampleWaterSupplySimulation):
    """An example simulation model which includes assets

    Parameters
    ==========
    raininess : int
        The amount of rain produced in each simulation
    number_of_treatment_plants : int
        The amount of water is a function of the number of treatment plants and
        the amount of raininess

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
        print("There are {} plants".format(self.number_of_treatment_plants))
        print("It is {} rainy".format(self.raininess))
        water = min(self.number_of_treatment_plants, self.raininess)
        cost = 1.264 * self.number_of_treatment_plants
        print("The system costs £{}".format(cost))
        return {
            "water": water,
            "cost": cost
        }


class ExampleWaterSupplySimulationReservoir(ExampleWaterSupplySimulation):
    """This simulation model has a state which is a non-asset variables

    The reservoir level is a function of the previous reservoir level,
    raininess and a fixed demand for water.

    Parameters
    ==========
    raininess : int
    reservoir_level : int

    Returns
    =======
    dict

    """
    fixed_demand = 1

    def __init__(self, raininess, reservoir_level):
        super().__init__(raininess)
        if reservoir_level < 0:
            raise ValueError("Reservoir level cannot be negative")
        self._reservoir_level = reservoir_level

    def simulate(self):
        """Run the model

        Note
        ====
        This simulate method has mixed the state transition (computing the new
        reservoir level) with the simulation of the model.

        """
        # Work out available water from raininess and initial reservoir level
        self.water = self.raininess + self._reservoir_level
        # Compute the reservoir level at the end of the year
        self._reservoir_level = self.water - self.fixed_demand
        self.cost = 1 + (0.1 * self._reservoir_level)
        return {'water': self.water,
                'cost': self.cost,
                'reservoir level': self._reservoir_level
                }
