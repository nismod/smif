"""Energy demand dummy model
"""
from smif.sector_model import SectorModel

class EDMWrapper(SectorModel):
    def simulate(self, decisions, state, data):
        return []

    def extract_obj(self, results):
        return 0
