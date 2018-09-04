#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Water Supply model
- WaterSupplySectorModel implements SectorModel
- wraps ExampleWaterSupplySimulationModel
- instantiate a model instance

"""

import logging

import numpy as np
from smif.model.sector_model import SectorModel


class WaterSupplySectorModel(SectorModel):
    """Example of a class implementing the SectorModel interface,
    using one of the toy water models below to simulate the water supply
    system.
    """
    def simulate(self, data):
        """Simulate water supply

        Arguments
        =========
        data
            - inputs/parameters, implicitly includes:
                - scenario data, e.g. expected level of rainfall
                - data output from other models in workflow
                - parameters set or adjusted for this model run
                - system state data, e.g. reservoir level at year start
            - system state, implicity includes:
                - initial existing system/network
                - decisions, e.g. asset build instructions, demand-side interventions to apply
        """
        # State

        current_interventions = data.get_current_interventions()

        print("Current interventions: {}".format(current_interventions))
        number_of_treatment_plants = 2

        # Inputs
        per_capita_water_demand = data.get_parameter('per_capita_water_demand')  # liter/person
        population = data.get_data('population')  # people

        water_demand = data.get_data('water_demand')  # liter
        final_water_demand = (population * per_capita_water_demand) + water_demand

        raininess = sum(data.get_data('precipitation'))  # milliliters to mega
        reservoir_level = sum(data.get_data('reservoir_level'))  # megaliters

        self.logger.debug(
            "Parameters:\n "
            "Population: %s\n"
            "Raininess: %s\n "
            "Reservoir level: %s\n  "
            "Final demand: %s\n",
            population.sum(),
            raininess.sum(),
            reservoir_level,
            final_water_demand
        )

        # Parameters
        self.logger.debug(data.get_parameters())

        # simulate (wrapping toy model)
        instance = ExampleWaterSupplySimulationModel()
        instance.raininess = raininess
        instance.number_of_treatment_plants = number_of_treatment_plants
        instance.reservoir_level = reservoir_level

        # run
        water, cost = instance.run()

        self.logger.info(
            "Water: %s, Cost: %s, Reservoir: %s", water, cost, instance.reservoir_level)

        # set results
        data.set_results('water', np.ones((3, )) * water / 3)
        data.set_results("cost", np.ones((3, )) * cost / 3)
        data.set_results("energy_demand", np.ones((3, )) * 3)

        # state data output - hack around using national resolution to start
        output = np.zeros((3, ))
        # output[0] = instance.reservoir_level  # will continually increase, need to access
        # t-1 TODO
        data.set_results("reservoir_level", output)

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
