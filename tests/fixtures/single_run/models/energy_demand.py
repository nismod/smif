"""Energy demand dummy model
"""
import numpy as np
from smif.sector_model import SectorModel


class EDMWrapper(SectorModel):
    """Energy model
    """
    def initialise(self, initial_conditions):
        pass

    def simulate(self, decisions, state, data):
        results = {
            "cost": np.ones((3, 1)) * 3,
            "water_demand": np.ones((3, 1)) * 3
        }
        return [], results

    def extract_obj(self, results):
        return 0
