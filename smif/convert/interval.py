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

Quantities
----------
Quantities are associated with a duration, period or interval.
For example 120 GWh of electricity generated during each week of February.::

        Week 1: 120 GW
        Week 2: 120 GW
        Week 3: 120 GW
        Week 4: 120 GW

Other examples of quantities:

- greenhouse gas emissions
- demands for infrastructure services
- materials use
- counts of cars past a junction
- costs of investments, operation and maintenance

Upscale: Divide
~~~~~~~~~~~~~~~

To convert to a higher temporal resolution, the values need to be apportioned
across the new time scale. In the above example, the 120 GWh of electricity
would be divided over the days of February to produce a daily time series
of generation.  For example::

        1st Feb: 17 GWh
        2nd Feb: 17 GWh
        3rd Feb: 17 GWh
        ...

Downscale: Sum
~~~~~~~~~~~~~~

To resample weekly values to a lower temporal resolution, the values
would need to be accumulated.  A monthly total would be::

        Feb: 480 GWh

Remapping
---------

Remapping quantities, as is required in the conversion from energy
demand (hourly values over a year) to energy supply (hourly values
for one week for each of four seasons) requires additional
averaging operations.  The quantities are averaged over the
many-to-one relationship of hours to time-slices, so that the
seasonal-hourly timeslices in the model approximate the hourly
profiles found across the particular seasons in the year. For example::

        hour 1: 20 GWh
        hour 2: 15 GWh
        hour 3: 10 GWh
        ...
        hour 8592: 16 GWh
        hour 8593: 12 GWh
        hour 8594: 21 GWh
        ...
        hour 8760: 43 GWh

To::

        season 1 hour 1: 20+16+.../4 GWh # Denominator number hours in sample
        season 1 hour 2: 15+12+.../4 GWh
        season 1 hour 3: 10+21+.../4 GWh
        ...

Prices
------

Unlike quantities, prices are associated with a point in time.
For example a spot price of £870/GWh.  An average price
can be associated with a duration, but even then,
we are just assigning a price to any point in time within a
range of times.

Upscale: Fill
~~~~~~~~~~~~~

Given a timeseries of monthly spot prices, converting these
to a daily price can be done by a fill operation.
E.g. copying the monthly price to each day.

From::

        Feb: £870/GWh

To::

        1st Feb: £870/GWh
        2nd Feb: £870/GWh
        ...

Downscale: Average
~~~~~~~~~~~~~~~~~~

On the other hand, going down scale, such as from daily prices
to a monthly price requires use of an averaging function. From::

        1st Feb: £870/GWh
        2nd Feb: £870/GWh
        ...

To::

        Feb: £870/GWh

Development Notes
-----------------

- We could use :py:meth:`numpy.convolve` to compare time intervals as hourly arrays
  before adding them to the set of intervals

"""
import logging
from collections import OrderedDict
from datetime import datetime, timedelta

import numpy as np
from isodate import parse_duration

__author__ = "Will Usher, Tom Russell"
__copyright__ = "Will Usher, Tom Russell"
__license__ = "mit"


"""Used as the reference year for computing time intervals
"""
BASE_YEAR = 2010


class Interval(object):
    """A time interval

    Parameters
    ----------
    id: str
        The unique name of the Interval
    list_of_intervals: str
        A list of tuples of valid ISO8601 duration definition
        string denoting the time elapsed from the beginning
        of the year to the (beginning, end) of the interval
    base_year: int, default=2010
        The reference year used for conversion to a datetime tuple

    Example
    -------

            >>> a = Interval('id', ('PT0H', 'PT1H'))
            >>> a.interval = ('PT1H', 'PT2H')
            >>> repr(a)
            "Interval('id', [('PT0H', 'PT1H'), ('PT1H', 'PT2H')], base_year=2010)"
            >>> str(a)
            "Interval 'id' starts at hour 0 and ends at hour 1"

    """

    def __init__(self, name, list_of_intervals, base_year=BASE_YEAR):
        self._name = name
        self._baseyear = base_year

        if len(list_of_intervals) == 0:
            msg = "Must construct Interval with at least one interval"
            raise ValueError(msg)

        if isinstance(list_of_intervals, list):
            for interval in list_of_intervals:
                assert isinstance(interval, tuple), "Interval must be constructed with tuples"
                if len(interval) != 2:
                    msg = "Interval tuple must take form (<start>, <end>)"
                    raise ValueError(msg)
            self._interval = list_of_intervals
        elif isinstance(list_of_intervals, tuple):
            self._interval = []
            self._interval.append(list_of_intervals)
        else:
            msg = "Interval tuple must take form (<start>, <end>)"
            raise ValueError(msg)

        self._validate()

    def _validate(self):
        for lower, upper in self.to_hours():
            if lower > upper:
                msg = "A time interval must not end before it starts - found %d > %d"
                raise ValueError(msg, lower, upper)

    @property
    def name(self):
        return self._name

    @property
    def start(self):
        """The start hour of the interval(s)

        Returns
        -------
        list
            A list of integers, representing the hour from the beginning of the
            year associated with the start of each of the intervals
        """
        if len(self._interval) == 1:
            return self._interval[0][0]
        else:
            return [x[0] for x in self._interval]

    @property
    def end(self):
        """The end hour of the interval(s)

        Returns
        -------
        list
            A list of integers, representing the hour from the beginning of the
            year associated with the end of each of the intervals
        """
        if len(self._interval) == 1:
            return self._interval[0][1]
        else:
            return [x[1] for x in self._interval]

    @property
    def interval(self):
        """The list of intervals

        Setter appends a tuple or list of intervals to the
        list of intervals
        """
        return sorted(self._interval)

    @interval.setter
    def interval(self, value):
        if isinstance(value, tuple):
            self._interval.append(value)
        elif isinstance(value, list):
            for element in value:
                assert isinstance(element, tuple), "A time interval must be a tuple"
            self._interval.extend(value)
        else:
            msg = "A time interval must add either a single tuple or a list of tuples"
            raise ValueError(msg)

        self._validate()

    @property
    def baseyear(self):
        """The reference year
        """
        return self._baseyear

    def __repr__(self):
        msg = "Interval('{}', {}, base_year={})"
        return msg.format(self.name, self.interval, self.baseyear)

    def __str__(self):
        string = "Interval '{}' maps to:\n".format(self.name)
        for interval in self.to_hours():
            start = interval[0]
            end = interval[1]
            suffix = "  hour {} to hour {}\n".format(start, end)
            string += suffix

        return string

    def __eq__(self, other):
        if (self.name == other.name) \
           and (self.interval == other.interval) \
           and (self.baseyear == other.baseyear):
            return True
        else:
            return False

    def to_hours(self):
        """Return a list of tuples of the intervals in terms of hours

        Returns
        -------
        list
            A list of tuples of the start and end hours of the year
            of the interval

        """
        hours = []
        for start_interval, end_interval in self.interval:
            start = self._convert_to_hours(start_interval)
            end = self._convert_to_hours(end_interval)
            hours.append((start, end))
        return hours

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

    def to_hourly_array(self):
        """Converts a list of intervals to a boolean array of hours

        """
        array = np.zeros(8760, dtype=np.int)
        list_of_tuples = self.to_hours()
        for lower, upper in list_of_tuples:
            array[lower:upper] += 1
        return array


class TimeSeries(object):
    """A series of values associated with an interval definition

    Parameters
    ----------
    data: list
        A list of dicts, each entry containing 'name' and 'value' keys
    """
    def __init__(self, data):
        self.logger = logging.getLogger(__name__)
        values = []
        name_list = []
        for row in data:
            name_list.append(row['id'])
            values.append(row['value'])
        self.names = name_list
        self.values = values
        self._hourly_values = np.zeros(8760, dtype=np.float64)

    @property
    def hourly_values(self):
        """The timeseries resampled to hourly values
        """
        return self._hourly_values


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

    @property
    def interval_set_names(self):
        """A list of the interval set names contained in the register

        Returns
        -------
        list
        """
        return list(self._register.keys())

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
        self._check_interval_in_register(set_name)
        return self._register[set_name]

    def register(self, intervals, set_name):
        """Add a time-interval definition to the set of intervals types

        Detects duplicate references to the same annual-hours by performing a
        convolution of the two one-dimensional arrays of time-intervals.

        Parameters
        ----------
        intervals: list
            Time intervals required as a list of dicts, with required keys
            ``start``, ``end`` and ``name``
        set_name: str
            A unique identifier for the set of time intervals

        """
        if set_name in self._register:
            msg = "An interval set named {} has already been loaded"
            raise ValueError(msg.format(set_name))

        self._register[set_name] = OrderedDict()

        for interval in intervals:

            name = interval['id']
            self.logger.debug("Adding interval '%s' to set '%s'", name, set_name)

            if name in self._register[set_name]:
                interval_tuple = (interval['start'], interval['end'])
                self._register[set_name][name].interval = interval_tuple
            else:
                interval_tuple = (interval['start'], interval['end'])
                self._register[set_name][name] = Interval(name,
                                                          interval_tuple,
                                                          self._base_year)

        self.logger.info("Adding interval set '%s' to register", set_name)
        self._validate_intervals()

    def _check_interval_in_register(self, interval):
        """

        Parameters
        ----------
        interval: str
            The name of the interval to look for

        """
        if interval not in self._register:
            msg = "The interval set '{}' is not in the register"
            raise ValueError(msg.format(interval))

    def convert(self, timeseries, from_interval, to_interval):
        """Convert some data to a time_interval type

        Parameters
        ----------
        timeseries: :class:`~smif.convert.interval.TimeSeries`
            The timeseries to convert from `from_interval` to `to_interval`
        from_interval: str
            The unique identifier of a interval type which matches the
            timeseries
        to_interval: str
            The unique identifier of a registered interval type

        Returns
        -------
        dict
            A dictionary with keys `name` and `value`, where the entries
            for `key` are the name of the target time interval, and the
            values are the resampled timeseries values.

        """
        results = []

        self._check_interval_in_register(from_interval)
        self._convert_to_hourly_buckets(timeseries, from_interval)

        target_intervals = self.get_intervals_in_set(to_interval)
        for name, interval in target_intervals.items():
            self.logger.debug("Resampling to %s", name)
            interval_tuples = interval.to_hours()

            total = 0

            for lower, upper in interval_tuples:
                self.logger.debug("Range: %s-%s", lower, upper)
                total += sum(timeseries.hourly_values[lower:upper])

            results.append({'id': name,
                            'value': total})

        return results

    def _convert_to_hourly_buckets(self, timeseries, interval_set):
        """Iterates through the `timeseries` and assigns values to hourly buckets

        Parameters
        ----------
        timeseries: :class:`~smif.convert.interval.TimeSeries`
            The timeseries to convert to hourly buckets ready for further
            operations
        interval_set: str
            The name of the interval set associated with the timeseries

        """
        for name, value in zip(timeseries.names, timeseries.values):
            list_of_intervals = self._register[interval_set][name].to_hours()
            divisor = len(list_of_intervals)
            for lower, upper in list_of_intervals:
                self.logger.debug("lower: %s, upper: %s", lower, upper)
                number_hours_in_range = upper - lower
                self.logger.debug("number_hours: %s", number_hours_in_range)

                apportioned_value = float(value) / number_hours_in_range
                self.logger.debug("apportioned_value: %s", apportioned_value)
                timeseries.hourly_values[lower:upper] = apportioned_value / divisor

    def _get_hourly_array(self, set_name):
        """
        """
        array = np.zeros(8760, dtype=np.int)
        for interval in self.get_intervals_in_set(set_name).values():
            array += interval.to_hourly_array()
        return array

    def _validate_intervals(self):
        for set_name in self._register.keys():
            array = self._get_hourly_array(set_name)
            duplicate_hours = np.where(array > 1)[0]
            if len(duplicate_hours) == 0:
                self.logger.debug("No duplicate hours in %s", set_name)
            else:
                hour = duplicate_hours[0]
                msg = "Duplicate entry for hour {} in interval set {}."
                self.logger.warning(msg.format(hour, set_name))
                raise ValueError(msg.format(hour, set_name))
