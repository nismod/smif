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
from collections import OrderedDict, defaultdict
from datetime import datetime, timedelta

import numpy as np
from isodate import parse_duration
from smif.convert.register import Register, ResolutionSet

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

        if not list_of_intervals:
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
        """The name (or id) of the interval(s)

        Returns
        -------
        str
            Name or ID
        """
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
        int or list
            An integer or list of integers, representing the hour from the
            beginning of the year associated with the end of each of the
            intervals
        """
        if len(self._interval) == 1:
            return self._interval[0][1]
        else:
            return [x[1] for x in self._interval]

    @property
    def interval(self):
        """The list of intervals

        Setter appends a tuple or list of intervals to the list of intervals
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


class IntervalSet(ResolutionSet):
    """A collection of intervals

    Arguments
    ---------
    name: str
        A unique identifier for the set of time intervals
    data: list
        Time intervals required as a list of dicts, with required keys
        ``start``, ``end`` and ``name``
    """

    def __init__(self, name, data, base_year=2010):
        super().__init__()
        self.name = name
        self._base_year = base_year
        self._data = None
        self.data = data

    @property
    def data(self):
        """Returns the intervals as an ordered dict

        Returns
        -------
        OrderedDict
        """
        return self._data

    @data.setter
    def data(self, interval_data):
        intervals = OrderedDict()

        for interval in interval_data:
            name = interval['id']

            if name in intervals:
                interval_tuple = (interval['start'], interval['end'])
                intervals[name].interval = interval_tuple
            else:
                interval_tuple = (interval['start'], interval['end'])
                intervals[name] = Interval(name,
                                           interval_tuple,
                                           self._base_year)
        self._data = intervals
        self._validate_intervals()

    def _get_hourly_array(self):
        array = np.zeros(8760, dtype=np.int)
        for interval in self._data.values():
            array += interval.to_hourly_array()
        return array

    def _validate_intervals(self):
        array = self._get_hourly_array()
        duplicate_hours = np.where(array > 1)[0]
        if len(duplicate_hours):
            hour = duplicate_hours[0]
            msg = "Duplicate entry for hour {} in interval set {}."
            raise ValueError(msg.format(hour, self.name))

    def get_entry_names(self):
        """Returns the names of the intervals
        """
        return [interval.name for interval in self.data.values()]

    def __getitem___(self, key):
        return self._data[key]

    def __len__(self):
        return len(self._data)


class TimeIntervalRegister(Register):
    """Holds the set of time-intervals used by the SectorModels
    """

    def __init__(self):
        self._register = OrderedDict()
        self._conversions = defaultdict(dict)
        self.logger = logging.getLogger(__name__)

    @property
    def names(self):
        """A list of the interval set names contained in the register

        Returns
        -------
        list
        """
        return list(self._register.keys())

    def get_entry(self, name):
        """Returns the ResolutionSet of `name`

        Arguments
        ---------
        name : str
            The unique identifier of a ResolutionSet in the register

        Returns
        -------
        smif.convert.interval.IntervalSet

        """
        if name not in self._register:
            msg = "Interval set '{}' not registered"
            raise ValueError(msg.format(name))
        return self._register[name]

    def register(self, interval_set):
        """Add a time-interval definition to the set of intervals types

        Detects duplicate references to the same annual-hours by performing a
        convolution of the two one-dimensional arrays of time-intervals.

        Parameters
        ----------
        interval_set : :class:`smif.convert.interval.IntervalSet`
            A collection of intervals
        """
        if interval_set.name in self._register:
            msg = "An interval set named {} has already been loaded"
            raise ValueError(msg.format(interval_set.name))

        self._register[interval_set.name] = interval_set
        self.logger.info("Adding interval set '%s' to register", interval_set.name)

    def convert(self, data, from_interval_set_name, to_interval_set_name):
        """Convert some data to a time_interval type

        Parameters
        ----------
        data: :class:`numpy.ndarray`
            The timeseries to convert from `from_interval` to `to_interval`
        from_interval_set_name: str
            The unique identifier of a interval type which matches the
            timeseries
        to_interval_set_name: str
            The unique identifier of a registered interval type

        Returns
        -------
        :class:`numpy.ndarray`
            An array of the resampled timeseries values.

        """
        from_interval_set = self.get_entry(from_interval_set_name)
        to_interval_set = self.get_entry(to_interval_set_name)

        converted = np.zeros(len(to_interval_set))
        hourly_values = self._convert_to_hourly_buckets(data, from_interval_set)

        for idx, interval in enumerate(to_interval_set.data.values()):
            interval_tuples = interval.to_hours()

            for lower, upper in interval_tuples:
                self.logger.debug("Range: %s-%s", lower, upper)
                converted[idx] += sum(hourly_values[lower:upper])

        return converted

    def _convert_to_hourly_buckets(self, timeseries, interval_set):
        """Iterates through the `timeseries` and assigns values to hourly buckets

        Parameters
        ----------
        timeseries: :class:`numpy.ndarray`
            The timeseries to convert to hourly buckets ready for further
            operations
        interval_set: list of :class:`smif.convert.Interval`
            The interval set associated with the timeseries

        """
        hourly_values = np.zeros(8760)

        for value, interval in zip(timeseries, interval_set.data.values()):
            list_of_intervals = interval.to_hours()
            divisor = len(list_of_intervals)
            for lower, upper in list_of_intervals:
                self.logger.debug("lower: %s, upper: %s", lower, upper)
                number_hours_in_range = upper - lower
                self.logger.debug("number_hours: %s", number_hours_in_range)

                apportioned_value = value / number_hours_in_range
                self.logger.debug("apportioned_value: %s", apportioned_value)
                hourly_values[lower:upper] = apportioned_value / divisor

        return hourly_values


__REGISTER = TimeIntervalRegister()


def get_register():
    """Return single copy of TimeIntervalRegister
    """
    return __REGISTER
