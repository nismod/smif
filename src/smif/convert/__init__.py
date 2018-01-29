"""In this module, we implement the conversion across space and time

The :class:`SpaceTimeConvertor` is instantiated with data to convert,
and the names of the four source and destination spatio-temporal resolutions.

The :meth:`~SpaceTimeConvertor.convert` method returns a new
:class:`numpy.ndarray` for passing to a sector model.
"""
import logging
import numpy as np

from smif.convert.area import get_register as get_region_register
from smif.convert.interval import get_register as get_interval_register

__author__ = "Will Usher, Tom Russell"
__copyright__ = "Will Usher, Tom Russell"
__license__ = "mit"


class SpaceTimeConvertor(object):
    """Handles the conversion of time and space for a list of values

    Arguments
    ---------
    region_register: :class:`smif.convert.area.RegionRegister`
        A fully populated register of the modelled regions
    interval_register: :class:`smif.convert.interval.TimeIntervalRegister`
        A fully populated register of the modelled intervals

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

    def convert(self, data, from_spatial, to_spatial, from_temporal, to_temporal):
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

        if from_spatial != to_spatial and from_temporal != to_temporal:
            converted = self._convert_regions(
                self._convert_intervals(
                    data,
                    from_temporal,
                    to_temporal
                ),
                from_spatial,
                to_spatial
            )
        elif from_temporal != to_temporal:
            converted = self._convert_intervals(
                data,
                from_temporal,
                to_temporal
            )
        elif from_spatial != to_spatial:
            converted = self._convert_regions(
                data,
                from_spatial,
                to_spatial
            )
        else:
            converted = data

        return converted

    def _convert_regions(self, data, from_spatial, to_spatial):
        """Slice, convert and compose regions
        """
        converted = np.apply_along_axis(self.regions.convert, 0, data,
                                        from_spatial, to_spatial)
        return converted

    def _convert_intervals(self, data, from_temporal, to_temporal):
        """Slice, convert and compose intervals
        """
        converted = np.apply_along_axis(self.intervals.convert, 1, data,
                                        from_temporal, to_temporal)
        return converted
