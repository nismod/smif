"""Energy demand dummy model
"""
from itertools import product

import numpy as np
from smif.model.sector_model import SectorModel


class EDMWrapper(SectorModel):
    """Energy model
    """
    def simulate(self, data):

        # Get the current timestep

        now = data.current_timestep
        self.logger.info("EDMWrapper received inputs in %s",
                         now)

        # State

        current_interventions = data.get_current_interventions()

        print("Current interventions: {}".format(current_interventions.keys()))

        # Demonstrates how to get the value for a model parameter
        parameter_value = data.get_parameter('smart_meter_savings')
        self.logger.info('Smart meter savings: %s', parameter_value)

        # Demonstrates how to get the value for a model input
        # (defaults to the current time period)
        current_energy_demand = data.get_data('energy_demand')
        self.logger.info("Current energy demand in %s is %s",
                         now, current_energy_demand)

        # Demonstrates how to get the value for a model input from the base
        # timeperiod
        base_energy_demand = data.get_base_timestep_data('energy_demand')
        base_year = data.base_timestep
        self.logger.info("Base year energy demand in %s was %s", base_year,
                         base_energy_demand)

        # Demonstrates how to get the value for a model input from the previous
        # timeperiod
        if now > base_year:
            prev_energy_demand = data.get_previous_timestep_data('energy_demand')
            prev_year = data.previous_timestep
            self.logger.info("Previous energy demand in %s was %s",
                             prev_year, prev_energy_demand)

        # Pretend to call the 'energy model'
        # This code prints out debug logging messages for each input
        # defined in the energy_demand configuration
        for name in self.inputs:
            spec = self.inputs[name]

            for idx in product(*[range(len(coord.ids)) for coord in spec.coords]):
                label_idx = tuple([
                    spec.coords[i].ids[j]
                    for i, j in enumerate(idx)
                ])
                self.logger.info(
                    "Read %s for %s at %s",
                    data.get_data(name)[idx],
                    name,
                    label_idx
                )

        # Write pretend results to data handler
        data.set_results("cost", np.ones((3, )) * 3)
        data.set_results("water_demand", np.ones((3, )) * 3)

        self.logger.info("EDMWrapper produced outputs in %s",
                         now)

    def extract_obj(self, results):
        return 0
