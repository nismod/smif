
from datetime import datetime, timedelta
from isodate import parse_duration
from pytest import fixture
import pytest
import numpy as np

from smif.convert.interval import TimeIntervalRegister, Interval, TimeSeries

@fixture(scope='function')
def months():
    months = [{'name': '1_0', 'start': 'P0M', 'end': 'P1M'},
              {'name': '1_1', 'start': 'P1M', 'end': 'P2M'},
              {'name': '1_2', 'start': 'P2M', 'end': 'P3M'},
              {'name': '1_3', 'start': 'P3M', 'end': 'P4M'},
              {'name': '1_4', 'start': 'P4M', 'end': 'P5M'},
              {'name': '1_5', 'start': 'P5M', 'end': 'P6M'},
              {'name': '1_6', 'start': 'P6M', 'end': 'P7M'},
              {'name': '1_7', 'start': 'P7M', 'end': 'P8M'},
              {'name': '1_8', 'start': 'P8M', 'end': 'P9M'},
              {'name': '1_9', 'start': 'P9M', 'end': 'P10M'},
              {'name': '1_10', 'start': 'P10M', 'end': 'P11M'},
              {'name': '1_11', 'start': 'P11M', 'end': 'P12M'}]
    return months

@fixture(scope='function')
def seasons():
    seasons = [{'name': 'winter', 'start': 'P12M', 'end': 'P2M'},
               {'name': 'spring', 'start': 'P3M', 'end': 'P5M'},
               {'name': 'summer', 'start': 'P6M', 'end': 'P8M'},
               {'name': 'autumn', 'start': 'P9M', 'end': 'P11M'}]
    return seasons

@fixture(scope='function')
def twenty_four_hours():
    twenty_four_hours = \
        [{'name': '1_0', 'start': 'PT0H', 'end': 'PT1H'},
         {'name': '1_1', 'start': 'PT1H', 'end': 'PT2H'},
         {'name': '1_2', 'start': 'PT2H', 'end': 'PT3H'},
         {'name': '1_3', 'start': 'PT3H', 'end': 'PT4H'},
         {'name': '1_4', 'start': 'PT4H', 'end': 'PT5H'},
         {'name': '1_5', 'start': 'PT5H', 'end': 'PT6H'},
         {'name': '1_6', 'start': 'PT6H', 'end': 'PT7H'},
         {'name': '1_7', 'start': 'PT7H', 'end': 'PT8H'},
         {'name': '1_8', 'start': 'PT8H', 'end': 'PT9H'},
         {'name': '1_9', 'start': 'PT9H', 'end': 'PT10H'},
         {'name': '1_10', 'start': 'PT10H', 'end': 'PT11H'},
         {'name': '1_11', 'start': 'PT11H', 'end': 'PT12H'},
         {'name': '1_12', 'start': 'PT12H', 'end': 'PT13H'},
         {'name': '1_13', 'start': 'PT13H', 'end': 'PT14H'},
         {'name': '1_14', 'start': 'PT14H', 'end': 'PT15H'},
         {'name': '1_15', 'start': 'PT15H', 'end': 'PT16H'},
         {'name': '1_16', 'start': 'PT16H', 'end': 'PT17H'},
         {'name': '1_17', 'start': 'PT17H', 'end': 'PT18H'},
         {'name': '1_18', 'start': 'PT18H', 'end': 'PT19H'},
         {'name': '1_19', 'start': 'PT19H', 'end': 'PT20H'},
         {'name': '1_20', 'start': 'PT20H', 'end': 'PT21H'},
         {'name': '1_21', 'start': 'PT21H', 'end': 'PT22H'},
         {'name': '1_22', 'start': 'PT22H', 'end': 'PT23H'},
         {'name': '1_23', 'start': 'PT23H', 'end': 'PT24H'}]

    return twenty_four_hours

@fixture(scope='function')
def one_day():

    one_day = [{'name': 'one_day', 'start': 'P0D', 'end': 'P1D'}]

    return one_day

class TestTimeSeries:

    def test_load_time_series(self, months):

        data = [{'name': '1_0', 'value': 1},
                {'name': '1_1', 'value': 1},
                {'name': '1_2', 'value': 1},
                {'name': '1_3', 'value': 1},
                {'name': '1_4', 'value': 1},
                {'name': '1_5', 'value': 1},
                {'name': '1_6', 'value': 1},
                {'name': '1_7', 'value': 1},
                {'name': '1_8', 'value': 1},
                {'name': '1_9', 'value': 1},
                {'name': '1_10', 'value': 1},
                {'name': '1_11', 'value': 1}]

        register = TimeIntervalRegister(2010)
        register.add_interval_set(months, 'months')

        timeseries = TimeSeries(data, register)
        actual = timeseries.names
        expected = ['1_0', '1_1', '1_2', '1_3', '1_4', '1_5',
                    '1_6', '1_7', '1_8', '1_9', '1_10', '1_11']
        assert actual == expected

        actual = timeseries.values
        expected = [1] * 12
        assert actual == expected

        actual = timeseries._hourly_values
        expected = np.array(8760, dtype=np.float)
        assert actual == expected

        timeseries.parse_values_into_hourly_buckets()
        actual = timeseries._hourly_values
        expected = np.array(8760, dtype=np.float)
        assert actual == expected

class TestInterval:

    def test_load_interval(self):

        interval = Interval('test', 'PT0H', 'PT1H')

        assert interval._name == 'test'
        assert interval._start == timedelta(0)
        assert interval._end == timedelta(0, 3600)
        assert interval._reference == datetime(2010, 1, 1, 0)

    def test_convert_hour_to_hours(self):

        interval = Interval('test', 'PT0H', 'PT1H')
        actual = interval.convert_to_hours(interval._start)
        expected = 0
        assert actual == expected

        actual = interval.convert_to_hours(interval._end)
        expected = 1
        assert actual == expected


    def test_convert_month_to_hours(self):

        interval = Interval('test', 'P1M', 'P2M')
        actual = interval.convert_to_hours(interval._start)
        expected = 744
        assert actual == expected

        actual = interval.convert_to_hours(interval._end)
        expected = 1416
        assert actual == expected

    def test_convert_week_to_hours(self):

        interval = Interval('test', 'P2D', 'P3D')
        actual = interval.convert_to_hours(interval._start)
        expected = 48
        assert actual == expected

        actual = interval.convert_to_hours(interval._end)
        expected = 72
        assert actual == expected

    def test_to_hours_zero(self):

        interval = Interval('test', 'PT0H', 'PT1H')
        actual = interval.to_hours()

        assert actual == (0, 1)

    def test_to_hours_month(self):

        interval = Interval('test', 'P2M', 'P3M')
        actual = interval.to_hours()

        assert actual == (1416, 2160)

class TestIntervalRegister:

    def test_interval_loads(self):
        """Pass a time-interval definition into the register

        """
        data = [{'name': '1_1',
                 'start': 'PT0H',
                 'end': 'PT1H'}]

        register = TimeIntervalRegister()
        register.add_interval_set(data, 'energy_supply_hourly')

        actual = register.register('energy_supply_hourly')
        expected = {'1_1': datetime(2010, 1, 1, 0)}

        assert actual == expected

    def test_months_load(self, months):
        """Pass a monthly time-interval definition into the register

        """
        register = TimeIntervalRegister()
        register.add_interval_set(months, 'months')

        actual = register.register('months')

        expected_names = \
            ['1_0', '1_1', '1_2', '1_3', '1_4', '1_5',
             '1_6', '1_7', '1_8', '1_9', '1_10', '1_11']

        expected = DatetimeIndex([datetime(2010, x, 1) for x in range(1, 13)])

        for name, time in zip(expected_names, expected):
            assert actual[name] == time


class TestIntervalRegisterConversion:

    def test_convert_from_month_to_seasons(self, months, seasons):



        register = TimeIntervalRegister()
        register.add_interval_set(months, 'months')
        register.add_interval_set(seasons, 'seasons')

        actual = register.convert(data, 'seasons')
        expected = [{'name': 'winter', 'value': 3},
                    {'name': 'spring', 'value': 3},
                    {'name': 'summer', 'value': 3},
                    {'name': 'autumn', 'value': 3}]
        assert actual == expected

    def test_convert_from_hour_to_day(self, twenty_four_hours, one_day):

        data = [{'name': '1_0', 'value': 1},
                {'name': '1_1', 'value': 1},
                {'name': '1_2', 'value': 1},
                {'name': '1_3', 'value': 1},
                {'name': '1_4', 'value': 1},
                {'name': '1_5', 'value': 1},
                {'name': '1_6', 'value': 1},
                {'name': '1_7', 'value': 1},
                {'name': '1_8', 'value': 1},
                {'name': '1_9', 'value': 1},
                {'name': '1_10', 'value': 1},
                {'name': '1_11', 'value': 1},
                {'name': '1_12', 'value': 1},
                {'name': '1_13', 'value': 1},
                {'name': '1_14', 'value': 1},
                {'name': '1_15', 'value': 1},
                {'name': '1_16', 'value': 1},
                {'name': '1_17', 'value': 1},
                {'name': '1_18', 'value': 1},
                {'name': '1_19', 'value': 1},
                {'name': '1_20', 'value': 1},
                {'name': '1_21', 'value': 1},
                {'name': '1_22', 'value': 1},
                {'name': '1_23', 'value': 1}]

        register = TimeIntervalRegister()
        register.add_interval_set(twenty_four_hours, 'hourly_day')
        register.add_interval_set(one_day, 'one_day')

        actual = register.convert(data, 'one_day')
        expected = [{'name': 'one_day', 'value': 24}]

        assert actual == expected
