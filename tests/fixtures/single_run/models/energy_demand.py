"""Energy demand dummy model
"""
from smif import SpaceTimeValue
from smif.sector_model import SectorModel


class EDMWrapper(SectorModel):
    """Energy model
    """
    def initialise(self, initial_conditions):
        pass

    def simulate(self, decisions, state, data):
        results = {
            "cost": [
                SpaceTimeValue('England', 1, 3, "million £"),
                SpaceTimeValue('Scotland', 1, 3, "million £"),
                SpaceTimeValue('Wales', 1, 3, "million £")
            ],
            "water_demand": [
                SpaceTimeValue('England', 1, 3, "Ml"),
                SpaceTimeValue('Scotland', 1, 3, "Ml"),
                SpaceTimeValue('Wales', 1, 3, "Ml")
            ],
        }
        return [], results

    def extract_obj(self, results):
        return 0
