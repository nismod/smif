"""In this module, we implement the conversion across space and time

The :class:`SpaceTimeConvertor` is instantiated with data to convert,
and the names of the four source and destination spatio-temporal resolutions.

The :meth:`~SpaceTimeConvertor.convert` method returns a new
:class:`numpy.ndarray` for passing to a sector model.
"""
import logging

from smif.convert.area import get_register as get_region_register
from smif.convert.interval import get_register as get_interval_register
from smif.convert.unit import get_register as get_unit_register

__author__ = "Will Usher, Tom Russell, Roald Schoenmakers"
__copyright__ = "Will Usher, Tom Russell, Roald Schoenmakers"
__license__ = "mit"


class SpaceTimeUnitConvertor(object):
    """Handles the conversion of time and space for a list of values

    Notes
    -----
    Future development requires using a data object which allows multiple views
    upon the values across the three dimensions of time, space and units. This
    will then allow more efficient conversion across any one of these dimensions
    while holding the others constant.  One option could be
    :class:`collections.ChainMap`.

    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.regions = get_region_register()
        self.intervals = get_interval_register()
        self.units = get_unit_register()

    def convert(self, data,
                from_spatial, to_spatial,
                from_temporal, to_temporal,
                from_unit, to_unit):
        """Convert the data from set of regions and intervals to another

        Parameters
        ----------
        data: numpy.ndarray
            An array of values with dimensions regions x intervals
        from_spatial: str
            The name of the spatial resolution of the data
        to_spatial: str
            The name of the required spatial resolution
        from_temporal: str
            The name of the temporal resolution of the data
        to_temporal: str
            The name of the required temporal resolution
        from_unit: str
            The name of the unit of the data
        to_unit: str
            The name of the required unit

        Returns
        -------
        numpy.ndarray
            An array of data with dimensions regions x intervals
        """
        assert from_spatial in self.regions.names, \
            "Cannot convert from spatial resolution {}".format(from_spatial)
        assert to_spatial in self.regions.names, \
            "Cannot convert to spatial resolution {}".format(to_spatial)
        assert from_temporal in self.intervals.names, \
            "Cannot convert from temporal resolution {}".format(from_temporal)
        assert to_temporal in self.intervals.names, \
            "Cannot convert to temporal resolution {}".format(to_temporal)
        assert from_unit in self.units.names, \
            "Cannot convert from unit {}".format(from_unit)
        assert to_unit in self.units.names, \
            "Cannot convert to unit {}".format(to_unit)

        converted = data

        if from_spatial != to_spatial and from_temporal != to_temporal:

            results = self.regions.convert(converted,
                                           from_spatial,
                                           to_spatial)
            converted = self.intervals.convert(results,
                                               from_temporal,
                                               to_temporal)

        elif from_temporal != to_temporal:
            converted = self.intervals.convert(converted,
                                               from_temporal,
                                               to_temporal)

        elif from_spatial != to_spatial:
            converted = self.regions.convert(converted,
                                             from_spatial,
                                             to_spatial)

        converted = self.units.convert(converted,
                                       from_unit,
                                       to_unit)

        return converted
