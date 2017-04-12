#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Water Supply model
- WaterSupplySectorModel implements SectorModel
- wraps ExampleWaterSupplySimulationModel
- instantiate a model instance

"""

import logging
from smif import SpaceTimeValue, StateData
from smif.sector_model import SectorModel


class WaterSupplySectorModel(SectorModel):
    """Example of a class implementing the SectorModel interface,
    using one of the toy water models below to simulate the water supply
    system.
    """
    def initialise(self, initial_conditions):
        """Set up system here
        """
        pass

    def simulate(self, decisions, state, data):
        """

        Arguments
        =========
        decisions
            - asset build instructions, demand-side interventions to apply
        state
            - existing system/network (unless implementation means maintaining
              system in sector model)
            - system state, e.g. reservoir level at year start
        data
            - scenario data, e.g. expected level of rainfall
        """

        # unpack inputs
        reservoir_level = state[0].data['current_level']['value']
        raininess = sum([d.value for d in data['raininess']])

        # unpack assets
        number_of_treatment_plants = 2

        self.logger.debug(decisions)
        self.logger.debug(state)
        self.logger.debug(data)

        # simulate (wrapping toy model)
        instance = ExampleWaterSupplySimulationModel()
        instance.raininess = raininess
        instance.number_of_treatment_plants = number_of_treatment_plants
        instance.reservoir_level = reservoir_level

        water, cost = instance.run()
        results = {
            "water": [
                SpaceTimeValue('England', 1, water/3, "Ml"),
                SpaceTimeValue('Scotland', 1, water/3, "Ml"),
                SpaceTimeValue('Wales', 1, water/3, "Ml"),
            ],
            "cost": [
                SpaceTimeValue('England', 1, cost/3, "million £"),
                SpaceTimeValue('Scotland', 1, cost/3, "million £"),
                SpaceTimeValue('Wales', 1, cost/3, "million £"),
            ],
            "energy_demand": [
                SpaceTimeValue('England', 1, 3, "MWh"),
                SpaceTimeValue('Scotland', 1, 3, "MWh"),
                SpaceTimeValue('Wales', 1, 3, "MWh")
            ]
        }
        state = [
            StateData('Kielder Water', {'current_level': {'value': instance.reservoir_level}}),
        ]

        return state, results

    def extract_obj(self, results):
        return results['cost']

    def constraints(self, parameters):
        """

        Notes
        =====
        This constraint below expresses that water supply must be greater than
        or equal to 3.  ``x[0]`` is the decision variable for water treatment
        capacity, while the value ``parameters[0]`` in the min term is the
        value of the raininess parameter.
        """
        constraints = (
            {
                'type': 'ineq',
                'fun': lambda x: min(x[0], parameters[0]) - 3
            }
        )
        return constraints


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
