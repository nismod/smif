"""Energy demand dummy model
"""
import numpy as np
from smif.model.sector_model import SectorModel


class EDMWrapper(SectorModel):
    """Energy model
    """
    def initialise(self, initial_conditions):
        pass

    def simulate(self, decisions, state, data):
        # receive population, energy_demand at annual/national
        self.logger.info("EDMWrapper received inputs in %s", data['timestep'])
        for name in self.model_inputs.names:
            time_intervals = self.model_inputs[name].get_interval_names()
            regions = self.model_inputs[name].get_region_names()
            dataset = data[name]
            for i, region in enumerate(regions):
                for j, interval in enumerate(time_intervals):
                    self.logger.info(
                        "%s %s %s",
                        interval,
                        region,
                        dataset[i][j])

        # output cost, water_demand at annual/nations
        results = {
            "cost": np.ones((3, 1)) * 3,
            "water_demand": np.ones((3, 1)) * 3
        }

        self.logger.info("EDMWrapper produced outputs in %s", data['timestep'])
        for name in self.model_outputs.names:
            time_intervals = self.model_outputs[name].get_interval_names()
            regions = self.model_outputs[name].get_region_names()
            dataset = results[name]
            for i, region in enumerate(regions):
                for j, interval in enumerate(time_intervals):
                    self.logger.info(
                        "%s %s %s",
                        interval,
                        region,
                        dataset[i][j])

        return {self.name, results}

    def extract_obj(self, results):
        return 0
