"""Handles conversion between the sets of regions used in the `SosModel`
"""
import logging
from collections import OrderedDict, defaultdict, namedtuple

import numpy as np
from rtree import index
from shapely.geometry import shape

from smif.convert import Register

__author__ = "Will Usher, Tom Russell"
__copyright__ = "Will Usher, Tom Russell"
__license__ = "mit"


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
        Iterable (probably a list or a reader handle)
        of fiona feature records e.g. the 'features' entry of
        a GeoJSON collection

    """
    def __init__(self, set_name, fiona_shape_iter):
        self.name = set_name
        self._regions = [
            NamedShape(region['properties']['name'], shape(region['geometry']))
            for region in fiona_shape_iter
        ]

        self._idx = index.Index()
        for pos, region in enumerate(self._regions):
            self._idx.insert(pos, region.shape.bounds)

    def intersection(self, bounds):
        """Return the subset of regions intersecting with a bounding box
        """
        return [self._regions[pos] for pos in self._idx.intersection(bounds)]

    def __getitem__(self, key):
        return self._regions[key]

    def __len__(self):
        return len(self._regions)


class RegionRegister(Register):
    """Holds the sets of regions used by the SectorModels and provides conversion
    between data values relating to compatible sets of regions.
    """
    def __init__(self):
        self._register = OrderedDict()
        self._conversions = defaultdict(dict)
        self.logger = logging.getLogger(__name__)

    @property
    def names(self):
        """Names of registered region sets

        Returns
        -------
        sets: list of str
        """
        return list(self._register.keys())

    def get_regions_in_set(self, set_name):
        """Return regions for a given set
        """
        if set_name in self._register:
            return self._register[set_name]
        else:
            raise ValueError("Region set {} not registered".format(set_name))

    def get_entry(self, name):
        return self.get_regions_in_set(name)

    def register(self, region_set):
        """Register a set of regions as a source/target for conversion
        """
        already_registered = self.names
        self._register[region_set.name] = region_set
        for other_set_name in already_registered:
            self._generate_coefficients(region_set, self._register[other_set_name])

    def convert(self, data, from_set_name, to_set_name):
        """Convert a list of data points for a given set of regions
        to another set of regions.

        Parameters
        ----------
        data: numpy.ndarray with dimension regions
        from_set_name: str
        to_set_name: str

        """
        from_set = self._register[from_set_name]
        from_set_names = [region.name for region in from_set]
        to_set = self._register[to_set_name]
        to_set_names = [region.name for region in to_set]

        converted = np.zeros(len(to_set))
        coefficents = self._conversions[from_set.name][to_set.name]

        for from_region_name, from_value in zip(from_set_names, data):
            for to_region_name, coef in coefficents[from_region_name]:
                to_region_idx = to_set_names.index(to_region_name)
                converted[to_region_idx] += coef*from_value

        return converted

    def _generate_coefficients(self, set_a, set_b):
        # from a to b
        self._conversions[set_a.name][set_b.name] = self._conversion_coefficients(set_a, set_b)
        # from b to a
        self._conversions[set_b.name][set_a.name] = self._conversion_coefficients(set_b, set_a)

    @staticmethod
    def _conversion_coefficients(from_set, to_set):
        """Return a dict containing the proportions of to_regions intersecting
        with each given from_region::

            {
                from_region.name: [
                    (to_region.name, proportion),
                    # ...
                ],
                # ...
            }

        """
        coefficients = defaultdict(list)

        for to_region in to_set:
            intersecting_from_regions = from_set.intersection(to_region.shape.bounds)

            for from_region in intersecting_from_regions:
                proportion = proportion_of_a_intersecting_b(from_region.shape, to_region.shape)
                coefficient_pair = (to_region.name, proportion)
                coefficients[from_region.name].append(coefficient_pair)

        return coefficients
