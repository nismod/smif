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
from smif.abstract import SectorModel, State


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


class WaterSupplyPython(SectorModel):
    """A concrete instance of the water supply model wrapper for testing

    Inherits :class:`SectorModel` to wrap the example simulation tool.

    """

    def initialise(self, data):
        """Set up the model

        Parameters
        ==========
        data : dict
            A dictionary of which one key 'raininess' must contain the amount
            of rain
        """
        self.inputs['raininess'] = data['raininess']

        self.model = ExampleWaterSupplySimulation(data['raininess'])
        self.results = None
        self.run_successful = None

    def optimise(self, method, decision_vars, objective_function):
        pass

    def simulate(self):
        """Runs the model and stores the results in the results parameter

        """
        self.results = self.model.simulate()
        self.run_successful = True

    def model_executable(self):
        pass


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
        self.water = min(self.number_of_treatment_plants, self.raininess)
        self.cost = 1.0 * self.number_of_treatment_plants
        return {
            "water": self.water,
            "cost": self.cost
        }


class WaterSupplyPythonAssets(SectorModel):
    """A concrete instance of the water supply wrapper for testing with assets

    Inherits :class:`SectorModel` to wrap the example simulation tool including
    asset management.

    The __state__ of the model is tracked in the asset parameter
    `number_of_treatment_plants`.

    """
    def initialise(self, data, assets):
        """Initialises the model
        """
        self.model = ExampleWaterSupplySimulationAsset(data['raininess'],
                                                       data['plants'])
        self.results = None
        self.run_successful = None

        treatment_plants = self.model.number_of_treatment_plants
        state_parameter_map = {'treatment plant': treatment_plants}

        self.state = State('oxford', 2010,
                           'water_supply',
                           state_parameter_map)
        self.state.initialise_from_tuples(assets)

    def optimise(self, method, decision_vars, objective_function):
        pass

    def decision_vars(self):
        return self.model.number_of_treatment_plants

    def objective_function(self):
        return self.model.cost

    def simulate(self):
        self.model.number_of_treatment_plants = \
            self.state.current_state['assets']['treatment plant']
        self.results = self.model.simulate()
        self.run_successful = True

    def model_executable(self):
        pass


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


class WaterSupplyExecutable(SectorModel):
    """A concrete instance of the water supply which wraps a command line model

    """

    def initialise(self, data):
        self.model = self.model_executable
        self.data = data
        self.results = None
        self.run_successful = None
        self.model_executable = 'water_supply_exec'

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
