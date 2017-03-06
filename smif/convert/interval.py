"""Handles conversion between the set of time intervals used in the `SosModel`

There are three main classes, which are currently rather intertwined.
:class:`Interval` represents an individual definition of a period
within a year.
This is specified using the ISO8601 period syntax and exposes
methods which use the isodate library to parse this into an internal hourly
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
import logging
from collections import OrderedDict
from datetime import datetime, timedelta

import numpy as np
from isodate import parse_duration


class Interval(object):
    """A time interval

    Parameters
    ----------
    name: str
        The unique name of the Interval
    start: str
        A valid ISO8601 duration definition string denoting the time elapsed from
        the beginning of the year to the beginning of the interval
    end: str
        A valid ISO8601 duration definition string denoting the time elapsed from
        the beginning of the year to the end of the interval
    base_year: int, default=2010
        The reference year used for conversion to a datetime tuple

    """

    def __init__(self, name, start, end, base_year=2010):
        self._name = name
        self._start = start
        self._end = end
        self._baseyear = base_year

    def __repr__(self):
        msg = "Interval('{}', '{}', '{}', base_year={})"
        return msg.format(self._name, self._start, self._end, self._baseyear)

    def __str__(self):
        msg = "Interval '{}' starts at hour {} and ends at hour {}"
        start, end = self.to_hours()
        return msg.format(self._name, start, end)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def to_hours(self):
        """Return a tuple of the interval in terms of hours

        Returns
        -------
        tuple
            The start and end hours of the year of the interval

        """
        start = self._convert_to_hours(self._start)
        end = self._convert_to_hours(self._end)

        return (start, end)

    def _convert_to_hours(self, duration):
        """

        Parameters
        ----------
        duration: str
            A valid ISO8601 duration definition string

        Returns
        -------
        int
            The hour in the year associated with the duration

        """
        reference = datetime(self._baseyear, 1, 1, 0)
        parsed_duration = parse_duration(duration)
        if isinstance(parsed_duration, timedelta):
            hours = parsed_duration.days * 24 + \
                    parsed_duration.seconds // 3600
        else:
            time = parsed_duration.totimedelta(reference)
            hours = time.days * 24 + time.seconds // 3600
        return hours

    def to_datetime_tuple(self):
        """
        """
        reference = datetime(self._baseyear, 1, 1, 0)
        start_time = reference + parse_duration(self._start)
        end_time = reference + parse_duration(self._end)
        period_tuple = (start_time, end_time)
        return period_tuple


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
        self._hourly_values = np.zeros(8760, dtype=np.float64)
        self._register = register
        self._convert_to_hourly_buckets()

    def _convert_to_hourly_buckets(self):
        """Iterates through the time series and assigns values to hourly buckets

        """
        for name, value in zip(self.names, self.values):
            lower, upper = self.get_hourly_range(name)
            self.logger.debug("lower: %s, upper: %s", lower, upper)

            number_hours_in_range = upper - lower
            self.logger.debug("number_hours: %s", number_hours_in_range)

            apportioned_value = float(value) / number_hours_in_range
            self.logger.debug("apportioned_value: %s", apportioned_value)
            self._hourly_values[lower:upper] = apportioned_value

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
        dict
            A dictionary with keys `name` and `value`, where the entries
            for `key` are the name of the target time interval, and the
            values are the resampled timeseries values.

        """
        results = []

        target_intervals = self._register.get_intervals_in_set(to_interval_type)
        for name, interval in target_intervals.items():
            self.logger.debug("Resampling to %s", name)
            lower, upper = interval.to_hours()
            self.logger.debug("Range: %s-%s", lower, upper)

            if upper < lower:
                # The interval loops around the end/start hours of the year
                end_of_year = sum(self._hourly_values[lower:8760])
                start_of_year = sum(self._hourly_values[0:upper])
                total = end_of_year + start_of_year
            else:
                total = sum(self._hourly_values[lower:upper])

            results.append({'name': name,
                            'value': total})
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
        """

        Parameters
        ----------
        set_name: str
            The unique identifying name of the interval definitions

        Returns
        -------
        :class:`collections.OrderedDict`
            Returns a collection of the intervals in the order in which they
            were defined

        """
        return self._register[set_name]

    def get_interval(self, name):
        """Return the interval definition given the unique name

        Parameters
        ----------
        name: str
            The unique name of the interval

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
        self._register[set_name] = OrderedDict()

        for interval in intervals:

            name = interval['name']
            self.logger.debug("Adding interval '%s' to set '%s'", name, set_name)

            self._id_interval_set[name] = set_name

            self._register[set_name][name] = Interval(name,
                                                      interval['start'],
                                                      interval['end'],
                                                      self._base_year)

        self.logger.info("Adding interval set '%s' to register", set_name)
