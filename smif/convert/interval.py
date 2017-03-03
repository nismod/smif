"""Handles conversion between the set of time intervals used in the `SosModel`

"""
from isodate import parse_duration
from datetime import datetime, timedelta
import logging
from pandas import Series, PeriodIndex, Period, DatetimeIndex, period_range
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
        self._hourly_values = np.array(8760, dtype=np.float)
        self._register = register

    def parse_values_into_hourly_buckets(self):
        """Iterates through the time series and assigns values to hourly buckets

        """

        for name, value in zip(self.names, self.values):
            lower, upper = self.get_hourly_range(name)
            self.logger.debug("lower: %s, upper: %s", lower, upper)

            number_hours_in_range = upper - lower
            self.logger.debug("number_hours: %s", number_hours_in_range)
            self._hourly_values[lower:upper] = value / (number_hours_in_range)

    def get_hourly_range(self, name):
        """Returns the upper and lower indexes of the hour for a particular interval

        Parameters
        ----------
        name: str
            The name of a registered time interval

        Returns
        -------
        tuple
        """
        return self._register.get_interval(name).to_hours()


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

    def get_interval(self, name):

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


    def convert(self, data, to_interval_type):
        """Convert some data to a time_interval type

        Parameters
        ----------
        data: list
            A list of dicts, containing the keys ``name`` and ``value``.
            ``name`` must correspond to a key in the register.
        to_interval_type: str
            The unique identifier of a registered interval type.

        Returns
        -------
        data

        """

        period = []
        values = []
        for row in data:
            # map name to timeperiod
            name = row['name']
            interval_set = self.get_interval_set(name)

            interval = self.register(interval_set)[name]

            period.append(interval)
            values.append(row['value'])

        self.logger.debug(period)
        time_index = DatetimeIndex(period)

        series = Series(data=values, index=time_index)

        destination = list(self._register[to_interval_type].values())
        to_index = DatetimeIndex(destination)
        return series.resample(to_index)
        # series.asfreq(self._register[to_interval_type])

    @staticmethod
    def _check_timeperiods_match(period_from, period_to):
        """Checks that two periods match one another

        Parameters
        ----------
        period_from: list
            A list of time period (datetime) tuples
        period_to: list
            A list of time period (datetime) tuples

        """
        if len(period_from) > len(period_to):
            # We are aggregating, so check that we cover the whole of `period_to`

            begin = period_from[0]
            print(begin)
            end = period_to[0]
            print(end)

            assert begin[0] == end[0]
            assert period_from[-1][1] == period_to[-1][1]

        elif len(period_from) < len(period_to):
            # We are interpolating
            pass
        elif len(period_from) == len(period_to):
            # The time periods could be the same...
            pass



    def convert_iso_period_to_pandas_period(self, iso_interval):
        """
        """
        start = parse_duration(iso_interval['start'])
        start_time = datetime(self.base_year, 1, 1) + start

        end = parse_duration(iso_interval['end'])
        end_time = datetime(self.base_year, 1, 1) + end

        pandas_period = Period(start_time, end_time - start_time)
        return pandas_period

    def convert_datetime_to_pandas_period(self, dt_interval):
        """
        """
        start_time = dt_interval[0]
        end_time = dt_interval[1]
        duration = end_time - start_time

        pandas_period = Period(start_time, duration)
        return pandas_period
