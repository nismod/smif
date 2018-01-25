#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Water Supply model
- WaterSupplySectorModel implements SectorModel
- wraps ExampleWaterSupplySimulationModel
- instantiate a model instance

"""

import logging

import numpy as np
from smif import StateData
from smif.model.sector_model import SectorModel


class WaterSupplySectorModel(SectorModel):
    """Example of a class implementing the SectorModel interface,
    using one of the toy water models below to simulate the water supply
    system.
    """

    def initialise(self, initial_conditions):
        """Set up system here
        """
        pass

    def simulate(self, data):
        """

        Arguments
        =========
        data
            - scenario data, e.g. expected level of rainfall
            - decisions
                - asset build instructions, demand-side interventions to apply
            - state
                - existing system/network (unless implementation means maintaining
                system in sector model)
                - system state, e.g. reservoir level at year start
        """

        self.logger.debug("Initial conditions: %s", self._initial_state)

        state = self._initial_state[0]

        # unpack inputs
        per_capita_water_demand = \
            data.get_parameter('per_capita_water_demand')  # liter/person

        population = data.get_data('population')  # people

        water_demand = (population * per_capita_water_demand) + \
            data.get_data('water_demand')  # liter

        raininess = sum(data.get_data('raininess'))  # megaliters

        reservoir_level = state.data['current_level']['value'] \
            - (1e-6 * water_demand.sum())  # megaliters

        self.logger.debug("Parameters:\n  Population: %s\n  Raininess: %s\n "
                          "Reservoir level: %s\n  ", population.sum(),
                          raininess.sum(), reservoir_level)

        # unpack assets
        number_of_treatment_plants = 2

        self.logger.debug(state)
        self.logger.debug(data.get_parameters())

        # simulate (wrapping toy model)
        instance = ExampleWaterSupplySimulationModel()
        instance.raininess = raininess
        instance.number_of_treatment_plants = number_of_treatment_plants
        instance.reservoir_level = reservoir_level

        water, cost = instance.run()
        data.set_results('water', np.ones((3, 1)) * water / 3)
        data.set_results("cost", np.ones((3, 1)) * cost / 3)
        data.set_results("energy_demand", np.ones((3, 1)) * 3)

        state = StateData('Kielder Water', {
            'current_level': {'value': instance.reservoir_level}
        })

    def extract_obj(self, results):
        return results['cost'].sum()


class ExampleWaterSupplySimulationModel(object):
    """An example simulation model used for testing purposes

    Parameters
    ==========
    raininess : int
        The amount of rain produced in each simulation
    number_of_treatment_plants : int
        The amount of water is a function of the number of treatment plants and
        the amount of raininess
    """
    def __init__(self,
                 raininess=None,
                 number_of_treatment_plants=None,
                 reservoir_level=None):
        self.raininess = raininess
        self.number_of_treatment_plants = number_of_treatment_plants
        self.reservoir_level = reservoir_level

    def run(self):
        """Runs the water supply model

        Only 1 unit of water is produced per treatment plant,
        no matter how rainy.

        Each treatment plant costs 1.0 unit.
        """
        logger = logging.getLogger(__name__)

        logger.debug("There are %s plants", self.number_of_treatment_plants)
        logger.debug("It is %s rainy", self.raininess)

        logger.debug("Reservoir level was %s", self.reservoir_level)
        self.reservoir_level += self.raininess

        water = min(self.number_of_treatment_plants, self.reservoir_level)
        logger.debug("The system produces %s water", water)

        self.reservoir_level -= water
        logger.debug("Reservoir level now %s", self.reservoir_level)

        cost = 1.264 * self.number_of_treatment_plants
        logger.debug("The system costs £%s", cost)

        return water, cost


if __name__ == '__main__':
    """Run core model if this script is run from the command line
    """
    CORE_MODEL = ExampleWaterSupplySimulationModel(1, 1, 2)
    CORE_MODEL.run()
