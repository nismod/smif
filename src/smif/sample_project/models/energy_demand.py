"""Energy demand dummy model
"""
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

        state = data.get_state()

        current_interventions = self.get_current_interventions(state)

        print("Current state of {} is {}".format(self.name, state))
        print("Current interventions: {}".format(current_interventions))

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
        for name in self.inputs.names:
            time_intervals = self.inputs[name].get_interval_names()
            regions = self.inputs[name].get_region_names()
            for i, region in enumerate(regions):
                for j, interval in enumerate(time_intervals):
                    self.logger.info(
                        "%s %s %s",
                        interval,
                        region,
                        data.get_data(name)[i, j])

        # Write pretend results to data handler
        data.set_results("cost", np.ones((3, 1)) * 3)
        data.set_results("water_demand", np.ones((3, 1)) * 3)

        self.logger.info("EDMWrapper produced outputs in %s",
                         now)

    def extract_obj(self, results):
        return 0
