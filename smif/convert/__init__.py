"""In this module, we implement the conversion across space and time

The :class:`SpaceTimeConvertor` is instantiated with data to convert,
and the names of the four source and destination spatio-temporal resolutions.

The :meth:`~SpaceTimeConvertor.convert` method returns a new
:class:`numpy.ndarray` for passing to a sector model.
"""
import logging

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

    def __init__(self, region_register, interval_register):
        self.logger = logging.getLogger(__name__)
        self.regions = region_register
        self.intervals = interval_register

    def convert(self, data, from_spatial, to_spatial, from_temporal, to_temporal):
        """Convert the data from set of regions and intervals to another

        Parameters
        ----------
        data: numpy.ndarray
            An array of values with dimensions intervals x regions
        from_spatial: str
            The name of the spatial resolution of the data
        to_spatial: str
            The name of the required spatial resolution
        from_temporal: str
            The name of the temporal resolution of the data
        to_temporal: str
            The name of the required temproal resolution

        Returns
        -------
        numpy.ndarray
            An array of data with dimensions regions x intervals
        """
        assert from_spatial in self.regions.region_set_names, \
            "Cannot convert from spatial resolution {}".format(from_spatial)
        assert to_spatial in self.regions.region_set_names, \
            "Cannot convert to spatial resolution {}".format(to_spatial)
        assert from_temporal in self.intervals.interval_set_names, \
            "Cannot convert from temporal resolution {}".format(from_temporal)
        assert to_temporal in self.intervals.interval_set_names, \
            "Cannot convert to temporal resolution {}".format(to_temporal)

        if from_spatial != to_spatial and from_temporal != to_temporal:
            converted = self.regions.convert(
                self.intervals.convert(
                    data,
                    from_temporal,
                    to_temporal
                ),
                from_spatial,
                to_spatial
            )
        elif from_temporal != to_temporal:
            converted = self.intervals.convert(
                data,
                from_temporal,
                to_temporal
            )
        elif from_spatial != to_spatial:
            converted = self.regions.convert(
                data,
                from_spatial,
                to_spatial
            )
        else:
            converted = data

        return converted
