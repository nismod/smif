"""In this module, we implement the conversion across space and time

The :class:`SpaceTimeConvertor` is instantiated with data to convert,
and the names of the four source and destination spatio-temporal resolutions.

The :meth:`~SpaceTimeConvertor.convert` method returns a new list of
:class:`smif.SpaceTimeValue` namedtuples for passing to a sector model.
"""
from collections import OrderedDict
import logging
from smif.convert.interval import TimeSeries
from smif import SpaceTimeValue

__author__ = "Will Usher, Tom Russell"
__copyright__ = "Will Usher, Tom Russell"
__license__ = "mit"


class SpaceTimeConvertor(object):
    """Handles the conversion of time and space for a list of values

    Arguments
    ---------
    data: list
        A list of :class:`smif.SpaceTimeValue`
    from_spatial: str
        The name of the spatial resolution of the data
    to_spatial: str
        The name of the required spatial resolution
    from_temporal: str
        The name of the temporal resolution of the data
    to_temporal: str
        The name of the required temproal resolution
    region_register: :class:`smif.convert.area.RegionRegister`
        A fully populated register of the models' regions
    interval_register: :class:`smif.convert.interval.TimeIntervalRegister`
        A fully populated register of the models' intervals

    Notes
    -----
    Future development requires using a data object which allows multiple views
    upon the values across the three dimensions of time, space and units. This
    will then allow more efficient conversion across any one of these dimensions
    while holding the others constant.  One option could be
    :class:`collections.ChainMap`.

    """

    def __init__(self, data,
                 from_spatial, to_spatial,
                 from_temporal, to_temporal,
                 region_register, interval_register):
        self.logger = logging.getLogger(__name__)

        self._check_uniform_units(data)

        self.data = data
        self.from_spatial = from_spatial
        self.to_spatial = to_spatial
        self.from_temporal = from_temporal
        self.to_temporal = to_temporal

        self.regions = region_register
        self.intervals = interval_register

    @property
    def data_regions(self):
        data_by_regions = self._regionalise_data(self.data)
        return set(data_by_regions.keys())

    @property
    def data_by_region(self):
        return self._regionalise_data(self.data)

    def _check_uniform_units(self, data):
        units = []
        for entry in data:
            units.append(entry.units)
        if len(set(units)) > 1:
            msg = "SpaceTimeConvertor cannot handle multiple units for conversion"
            raise NotImplementedError(msg)

    @staticmethod
    def _regionalise_data(data):
        """

        Parameters
        ----------
        data: list
            A list of :class:`smif.SpaceTimeValue`

        Returns
        -------
        :class:`collections.OrderedDict`
            A dictionary where the key is a region, and the value is a list of
            :class:`smif.SpaceTimeValue`

        """
        data_by_region = OrderedDict()
        for entry in data:
            if entry.region not in data_by_region:
                data_by_region[entry.region] = [entry]
            else:
                data_by_region[entry.region].append(entry)
        return data_by_region

    @staticmethod
    def _intervalise_data(data):
        """

        Parameters
        ----------
        data: list
            A list of :class:`smif.SpaceTimeValue`

        Returns
        -------
        :class:`collections.OrderedDict`
            A dictionary where the key is an interval, and the value is a list of
            :class:`smif.SpaceTimeValue`

        """
        data_by_intervals = OrderedDict()
        for entry in data:
            if entry.interval not in data_by_intervals:
                data_by_intervals[entry.interval] = [entry]
            else:
                data_by_intervals[entry.interval].append(entry)
        return data_by_intervals

    def convert(self):
        """Convert the data according to the parameters passed
        to the SpaceTimeConvertor

        Returns
        -------
        list
            A list of :class:`smif.SpaceTimeValue`
        """
        assert self.from_spatial in self.regions.region_set_names
        assert self.to_spatial in self.regions.region_set_names
        assert self.from_temporal in self.intervals.interval_set_names
        assert self.to_temporal in self.intervals.interval_set_names

        if self._convert_intervals_required() and self._convert_regions_required():
            self.logger.debug("Converting intervals and regions")
            interval_data = self._loop_over_regions(self.data)
            data = self._loop_over_intervals(interval_data)
        elif self._convert_intervals_required():
            self.logger.debug("Converting intervals only")
            data = self._loop_over_regions(self.data)
        elif self._convert_regions_required():
            self.logger.debug("Converting regions only")
            data = self._loop_over_intervals(self.data)
        else:
            self.logger.debug("No conversion required, passing data through")
            data = self.data

        return data

    def _loop_over_regions(self, data=None):
        """
        """
        converted_data = []
        data_by_region = self._regionalise_data(data)
        num_regions = len(set(data_by_region.keys()))
        if num_regions > 1:
            for region, region_data in data_by_region.items():
                converted_data.extend(self._convert_time(region_data))
        elif num_regions == 1:
            converted_data = self._convert_time(data)
        return converted_data

    def _loop_over_intervals(self, data=None):
        """
        """
        converted_data = []
        data_by_intervals = self._intervalise_data(data)
        num_intervals = len(set(data_by_intervals.keys()))
        if num_intervals > 1:
            for interval, interval_data in data_by_intervals.items():
                converted_data.extend(self._convert_space(interval_data))
        elif num_intervals == 1:
            converted_data = self._convert_space(data)
        return converted_data

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

    def _convert_space(self, data):
        """Wraps the call to the area conversion

        Parameters
        ----------
        data: list
            A list of :class:`smif.SpaceTimeValue` for one interval

        Returns
        -------
        data: list
            A list of :class:`smif.SpaceTimeValue` for one interval
        """
        timeseries_data = {}

        interval = data[0].interval
        units = data[0].units
        msg = "Converting regions for interval %s with units %s"
        self.logger.debug(msg, interval, units)

        for entry in data:
            timeseries_data[entry.region] = entry.value

        converted_data = self.regions.convert(timeseries_data,
                                              self.from_spatial,
                                              self.to_spatial)
        return [SpaceTimeValue(region_name, interval, converted_value, units)
                for region_name, converted_value in converted_data.items()]

    def _convert_time(self, data):
        """Wraps the call to the interval conversion

        Parameters
        ----------
        data: list
            A list of :class:`smif.SpaceTimeValue` for one region

        Returns
        -------
        data: list
            A list of :class:`smif.SpaceTimeValue` for one region
        """

        timeseries_data = []

        region = data[0].region
        units = data[0].units

        for entry in data:
            timeseries_data.append({'id': entry.interval,
                                    'value': entry.value})

        timeseries = TimeSeries(timeseries_data)
        converted_data = self.intervals.convert(timeseries,
                                                self.from_temporal,
                                                self.to_temporal)

        return [SpaceTimeValue(region, entry['id'], entry['value'], units)
                for entry in converted_data]
