"""Handles conversion between the sets of regions used in the `SosModel`
"""
from collections import namedtuple

from rtree import index
from shapely.geometry import mapping, shape
from shapely.validation import explain_validity
from smif.convert.register import NDimensionalRegister, ResolutionSet

__author__ = "Will Usher, Tom Russell"
__copyright__ = "Will Usher, Tom Russell"
__license__ = "mit"


NamedShape = namedtuple('NamedShape', ['name', 'shape'])


class RegionSet(ResolutionSet):
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
        self._regions = []
        self.data = fiona_shape_iter

        self._idx = index.Index()
        for pos, region in enumerate(self._regions):
            self._idx.insert(pos, region.shape.bounds)

    @property
    def data(self):
        return self._regions

    @data.setter
    def data(self, value):
        names = {}
        for region in value:
            name = region['properties']['name']
            if name in names:
                raise AssertionError(
                    "Region set must have uniquely named regions - %s duplicated", name)
            names[name] = True
            self._regions.append(
                NamedShape(
                    name,
                    shape(region['geometry'])
                )
            )

    def get_entry_names(self):
        return [region.name for region in self.data]

    def as_features(self):
        """Get the regions as a list of feature dictionaries

        Returns
        -------
        list
            A list of GeoJSON-style dicts
        """
        return [
            {
                'type': 'Feature',
                'geometry': mapping(region.shape),
                'properties': {
                    'name': region.name
                }
            }
            for region in self._regions
        ]

    def centroids_as_features(self):
        """Get the region centroids as a list of feature dictionaries

        Returns
        -------
        list
            A list of GeoJSON-style dicts, with Point features corresponding to
            region centroids
        """
        return [
            {
                'type': 'Feature',
                'geometry': mapping(region.shape.centroid),
                'properties': {
                    'name': region.name
                }
            }
            for region in self._regions
        ]

    def intersection(self, to_entry):
        """Return the set of regions intersecting with the bounds of `to_entry`
        """
        bounds = to_entry.shape.bounds
        return [x for x in self._idx.intersection(bounds)]

    def get_proportion(self, from_idx, entry_b):
        """Calculate the proportion of shape a that intersects with shape b
        """
        entry_a = self.data[from_idx]
        if self.check_valid_shape(entry_a.shape):
            if self.check_valid_shape(entry_b.shape):
                intersection = entry_a.shape.intersection(entry_b.shape)
                return intersection.area / entry_a.shape.area
            else:
                raise RuntimeError("Shape {} is not valid".format(entry_b.name))
        else:
            raise RuntimeError("Shape {} from {} is not valid".format(entry_a.name, self.name))

    def check_valid_shape(self, shape):
        if not shape.is_valid:
            validity = explain_validity(shape)
            print("Shape is not valid. Explanation: %s", validity)
            return False
        else:
            return True

    @staticmethod
    def get_bounds(entry):
        return entry.shape.bounds

    @property
    def coverage(self):
        return sum([x.shape.area for x in self.data])

    def __getitem__(self, key):
        return self._regions[key]

    def __len__(self):
        return len(self._regions)


class RegionRegister(NDimensionalRegister):
    """Holds the sets of regions used by the SectorModels and provides conversion
    between data values relating to compatible sets of regions.

    Notes
    -----
    The argument ``axis=0`` refers to the dimension of the data array that is
    associated with the regions dimension.
    """
    def __init__(self):
        super().__init__(axis=0)


__REGISTER = RegionRegister()


def get_register():
    """Return single copy of RegionRegister
    """
    return __REGISTER
