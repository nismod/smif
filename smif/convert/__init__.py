"""In this module, we implement the conversion across space and time

The :class:`SpaceTimeConvertor` is instantiated with data to convert,
and the names of the four source and destination spatio-temporal resolutions.

The :meth:`~SpaceTimeConvertor.convert` method returns a new list of
:class:`smif.SpaceTimeValue` namedtuples for passing to a sector model.
"""
from smif.convert.interval import TimeSeries
from smif import SpaceTimeValue


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

        self.data_by_region = {}
        self.data_by_units = {}
        self.data_by_intervals = {}

        for entry in data:
            if entry.region not in self.data_by_region:
                self.data_by_region[entry.region] = [entry]
            else:
                self.data_by_region[entry.region].append(entry)

            if entry.units not in self.data_by_units:
                self.data_by_units[entry.units] = [entry]
            else:
                self.data_by_units[entry.units].append(entry)

        self.data_regions = set(self.data_by_region.keys())

        if len(set(self.data_by_units.keys())) > 1:
            msg = "SpaceTimeConvertor cannot handle multiple units for conversion"
            raise NotImplementedError(msg)

    def convert(self):
        """
        """
        data = []

        if self._convert_intervals_required():
            if len(self.data_regions) > 1:
                for region, region_data in self.data_by_region.items():
                    data.extend(self._convert_time(region_data))
            elif len(self.data_regions) == 1:
                data = self._convert_time(self.data)
        else:
            data = self.data

        if self._convert_regions_required():
            data = self._convert_space(data)

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

    def _convert_space(self, data):
        """
        """
        data = self.regions.convert(data,
                                    self.from_spatial,
                                    self.to_spatial)
        return data

    def _convert_time(self, data):
        """
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
