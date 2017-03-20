"""In this module, we implement the conversion across space and time

The :class:`SpaceTimeConvertor` is instantiated with data to convert,
and the names of the four source and destination spatio-temporal resolutions.

The :meth:`~SpaceTimeConvertor.convert` method returns a new list of
:class:`smif.SpaceTimeValue` namedtuples for passing to a sector model.
"""
from smif.convert.area import RegionRegister
from smif.convert.interval import TimeIntervalRegister, TimeSeries


class SpaceTimeConvertor(object):
    """Handles the conversion of time and space for a list of values

    Arguments
    ---------
    data: list
        A list of :class:`smif.SpaceTimeValue`
    from_spatial: str
    to_spatial: str
    from_temporal: str
    to_temporal: str
    region_register: :class:`smif.convert.area.RegionRegister`
    interval_register: :class:`smif.convert.interval.TimeIntervalRegister`

    """

    def __init__(self, data,
                 from_spatial, to_spatial,
                 from_temporal, to_temporal,
                 region_register, interval_register):
        self.data = data
        self.from_spatial = from_spatial
        self.to_spatial = to_spatial
        self.from_temporal = from_temporal
        self.to_temporal = to_temporal

        self.regions = region_register
        self.intervals = interval_register

    def convert(self):
        """
        """
        if self._convert_regions_required():
            data = self.regions.convert(self.data,
                                        self.from_spatial,
                                        self.to_spatial)
        else:
            data = self.data

        if self._convert_regions_required():
            timeseries = TimeSeries(self.data)
            converted_data = self.intervals.convert(timeseries,
                                                    self.from_temporal,
                                                    self.to_temporal)
            data = converted_data
        else:
            data = self.data

        return data

    def _convert_regions_required(self):
        """Returns True if it is necessary to convert over space

        Returns
        -------
        bool
        """
        if self.from_spatial == self.to_spatial:
            return False
        elif self.from_spatial != self.to_spatial:
            return True
        else:
            msg = "Cannot determine if region conversion is required"
            raise ValueError(msg)

    def _convert_intervals_required(self):
        """Returns True if it is necessary to convert over time

        Returns
        -------
        bool
        """
        if self.from_temporal == self.to_temporal:
            return False
        elif self.from_temporal != self.to_temporal:
            return True
        else:
            msg = "Cannot determine if time conversion is required"
            raise ValueError(msg)

    def _convert_space(self):
        """
        """
        pass

    def _convert_time(self):
        """
        """
        pass
