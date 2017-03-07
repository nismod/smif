"""Handles conversion between the sets of regions used in the `SosModel`
"""
from collections import namedtuple

from rtree import index
from shapely.geometry import shape


def proportion_of_a_intersecting_b(shape_a, shape_b):
    """Calculate the proportion of shape a that intersects with shape b
    """
    intersection = shape_a.intersection(shape_b)
    return intersection.area / shape_a.area


NamedShape = namedtuple('NamedShape', ['name', 'shape'])


class RegionSet(object):
    """Hold a set of regions, spatially indexed for ease of lookup when
    constructing conversion matrices.

    Parameters
    ----------
    set_name : str
        Name to use as identifier for this set of regions
    fiona_shape_iter: iterable
        Iterable (probably a list or a reader handle) of fiona feature records

    """
    def __init__(self, set_name, fiona_shape_iter):
        self.name = set_name
        self.regions = [
            NamedShape(region['properties']['name'], shape(region['geometry']))
            for region in fiona_shape_iter
        ]

        self._idx = index.Index()
        for pos, region in enumerate(self.regions):
            self._idx.insert(pos, region.shape.bounds)


class RegionRegister(object):
    """Holds the sets of regions used by the SectorModels and provides conversion
    between data values relating to compatible sets of regions.
    """
    def __init__(self):
        self._register = {}

    @property
    def registered_sets(self):
        """Names of registered region sets

        Returns
        -------
        sets: list of str
        """
        return list(self._register.keys())

    def register_region_set(self, region_set):
        """Register a set of regions as a source/target for conversion
        """
        self._register[region_set.name] = region_set

    def convert(self, data, from_set_name, to_set_name):
        """Convert a list of data points for a given set of regions
        to another set of regions.
        """
        pass
