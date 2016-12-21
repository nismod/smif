#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Water Supply model
- WaterSupplySectorModel implements SectorModel
- wraps ExampleWaterSupplySimulationModel
- instantiate a model instance

"""

import logging
from smif.sector_model import SectorModel

class WaterSupplySectorModel(SectorModel):
    """Example of a class implementing the SectorModel interface,
    using one of the toy water models below to simulate the water supply
    system.
    """

    def simulate(self, decisions, state, data):
        """

        Arguments
        =========
        decisions : :class:`numpy.ndarray`
            x_0 is new capacity of water treatment plants
        state
            unused here:
            - should contain existing capacity of water treatment plants
            - could for example contain reservoir level at year start
        data : dict of dicts
            contains scenario data about expected level of rainfall
        """

        # unpack inputs
        raininess = data['raininess']['value']

        # unpack decision variables
        number_of_treatment_plants = decisions[0, ]

        # simulate (wrapping toy model)
        instance = ExampleWaterSupplySimulationModel(raininess, number_of_treatment_plants)
        results = instance.run()

        return results

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
        constraints = ({'type': 'ineq',
                        'fun': lambda x: min(x[0], parameters[0]) - 3}
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
    def __init__(self, raininess, number_of_treatment_plants):
        self.raininess = raininess
        self.number_of_treatment_plants = number_of_treatment_plants

    def run(self):
        """Runs the water supply model

        Only 1 unit of water is produced per treatment plant,
        no matter how rainy.

        Each treatment plant costs 1.0 unit.
        """
        logger = logging.getLogger(__name__)

        logger.debug("There are %s plants", self.number_of_treatment_plants)
        logger.debug("It is %s rainy", self.raininess)

        water = min(self.number_of_treatment_plants, self.raininess)
        logger.debug("The system produces %s water", water)

        cost = 1.264 * self.number_of_treatment_plants
        logger.debug("The system costs £%s", cost)

        return {
            "water": water,
            "cost": cost
        }

# instantiate model for easy access when imported by smif
model = WaterSupplySectorModel()

if __name__ == '__main__':
    """Run core model if this script is run from the command line
    """
    CORE_MODEL = ExampleWaterSupplySimulationModel(1, 1)
    CORE_MODEL.run()
