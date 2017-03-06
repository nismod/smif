"""Handles conversion between the sets of regions used in the `SosModel`
"""


class RegionRegister(object):
    """Holds the sets of regions used by the SectorModels and provides conversion
    between compatible sets of regions.
    """
    def __init__(self):
        self._register = {}

    def register_area_set(self, area_set):
        """Register a set of regions as a source/target for conversion
        """
        pass

    def convert(self, data, from_set_name, to_set_name):
        """Convert a list of data points for a given set of regions
        to another set of regions.
        """
        pass
