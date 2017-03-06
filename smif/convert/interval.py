"""Handles conversion between the set of time intervals used in the `SosModel`

There are three main classes, which are currently rather intertwined.
:class:`Interval` represents an individual defitions of a period within a year. This is specified using the ISO8601 period syntax and exposes 
methos which use the isodate library to parse this into an internal hourly
representation of the period.

:class:`TimeIntervalRegister` holds the definitions of time-interval sets
specified for the sector models at the :class:`~smif.sos_model.SosModel` 
level.  
This class exposes one public method, 
:py:meth:`~TimeIntervalRegister.add_interval_set` which allows the SosModel
to add an interval definition from a model configuration to the register.

:class:`TimeSeries` is used to encapsulate any data associated with a 
time interval definition set, and handles conversion from the current time
interval resolution to a target time interval definition held in the register.

"""
from isodate import parse_duration
from datetime import datetime, timedelta
import logging
import numpy as np

class Interval(object):
    """A time interval

    """

    def __init__(self, name, start, end, base_year=2010):
        self._name = name
        self._start = parse_duration(start)
        self._end = parse_duration(end)
        self._reference = datetime(base_year, 1, 1, 0)

    def to_hours(self):
        """Return a tuple of the interval in terms of hours

        """
        start = self.convert_to_hours(self._start)
        end = self.convert_to_hours(self._end)

        return (start, end)

    def convert_to_hours(self, duration):
        """
        """
        if isinstance(duration, timedelta):
            hours = duration.days * 24 + duration.seconds // 3600
        else:
            td = duration.totimedelta(self._reference)
            hours = td.days * 24 + td.seconds // 3600
        return hours


    def to_datetime_tuple(self):
        """
        """
        start_time = self._reference + self._start
        end_time = self._reference + self._end
        period_tuple = (start_time, end_time)
        return period_tuple

    def duration(self):
        return timedelta(self._end - self._start)

class TimeSeries(object):
    """A series of values associated with a DatetimeIndex

    Parameters
    ----------
    data: list
        A list of dicts, each entry containing 'name' and 'value' keys
    register: :class:`~smif.convert.interval.TimeIntervalRegister`
        A pointer to the time interval register
    """
    def __init__(self, data, register):
        self.logger = logging.getLogger(__name__)
        values = []
        name_list = []
        for row in data:
            name_list.append(row['name'])
            values.append(row['value'])
        self.names = name_list
        self.values = values
        self._hourly_values = np.zeros(8760, dtype=np.float)
        self._register = register
        self.parse_values_into_hourly_buckets()

    def parse_values_into_hourly_buckets(self):
        """Iterates through the time series and assigns values to hourly buckets

        """
        for name, value in zip(self.names, self.values):
            lower, upper = self.get_hourly_range(name)
            self.logger.debug("lower: %s, upper: %s", lower, upper)

            number_hours_in_range = upper - lower
            self.logger.debug("number_hours: %s", number_hours_in_range)

            apportioned_value = float(value) / (number_hours_in_range)
            self.logger.debug("apportioned_value: %s", apportioned_value)
            self._hourly_values[lower:(upper)] = apportioned_value

    def get_hourly_range(self, name):
        """Returns the upper and lower hours of a particular interval

        Parameters
        ----------
        name: str
            The name of a registered time interval

        Returns
        -------
        tuple
        """
        return self._register.get_interval(name).to_hours()

    def convert(self, to_interval_type):
        """Convert some data to a time_interval type

        Parameters
        ----------
        to_interval_type: str
            The unique identifier of a registered interval type.

        Returns
        -------
        data

        """
        results = []

        target_intervals = self._register.get_intervals_in_set(to_interval_type)
        for name, interval in target_intervals.items():
            self.logger.debug("Resampling to %s", name)
            lower, upper = interval.to_hours()
            self.logger.debug("Range: %s-%s", lower, upper)

            if upper < lower:
                # We are looping around the end of the year
                end_of_year = sum(self._hourly_values[lower:8760])
                start_of_year = sum(self._hourly_values[0:upper])
                total = end_of_year + start_of_year
            else:
                total = sum(self._hourly_values[lower:upper])

            results.append({'name': name,
                            'value': total
                           })
        return results

class TimeIntervalRegister:
    """Holds the set of time-intervals used by the SectorModels

    Parameters
    ----------
    base_year: int, default=2010
        Set the year which is used as a reference by all time interval sets
        and repeated for each future year
    """

    def __init__(self, base_year=2010):
        self._base_year = base_year
        self._register = {}
        self.logger = logging.getLogger(__name__)
        self._id_interval_set = {}

    def get_intervals_in_set(self, set_name):
        return self._register[set_name]

    def get_interval(self, name):
        """Returns the interval definition given the name of an Interval

        Returns
        -------
        :class:`smif.convert.interval.Interval`
        """

        set_name = self._id_interval_set[name]
        interval = self._register[set_name][name]

        return interval

    def add_interval_set(self, intervals, set_name):
        """Add a time-interval definition to the set of intervals types

        Parameters
        ----------
        intervals: list
            Time intervals required as a list of dicts, with required keys
            ``start``, ``end`` and ``name``
        set_name: str
            A unique identifier for the set of time intervals

        """
        self._register[set_name] = {}

        for interval in intervals:

            name = interval['name']

            self._id_interval_set[name] = set_name

            self._register[set_name][name] = Interval(name,
                                                      interval['start'],
                                                      interval['end'])

        self.logger.debug("Added %s to register", set_name)
