
from collections import OrderedDict
import numpy as np
from numpy.testing import assert_equal
from pytest import approx, fixture, raises
from smif.convert.interval import Interval, TimeIntervalRegister, TimeSeries


@fixture(scope='function')
def remap_months():
    """Remapping four representative months to months across the year

    In this case we have a model which represents the seasons through
    the year using one month for each season. We then map the four
    model seasons 1, 2, 3 & 4 onto the months throughout the year that
    they represent.

    The data will be presented to the model using the four time intervals,
    1, 2, 3 & 4. When converting to hours, the data will be replicated over
    the year.  When converting from hours to the model time intervals,
    data will be averaged and aggregated.

    """
    data = [{'name': '1', 'start': 'P0M', 'end': 'P1M'},
            {'name': '2', 'start': 'P1M', 'end': 'P2M'},
            {'name': '2', 'start': 'P2M', 'end': 'P3M'},
            {'name': '2', 'start': 'P3M', 'end': 'P4M'},
            {'name': '3', 'start': 'P4M', 'end': 'P5M'},
            {'name': '3', 'start': 'P5M', 'end': 'P6M'},
            {'name': '3', 'start': 'P6M', 'end': 'P7M'},
            {'name': '4', 'start': 'P7M', 'end': 'P8M'},
            {'name': '4', 'start': 'P8M', 'end': 'P9M'},
            {'name': '4', 'start': 'P9M', 'end': 'P10M'},
            {'name': '1', 'start': 'P10M', 'end': 'P11M'},
            {'name': '1', 'start': 'P11M', 'end': 'P12M'}]
    return data


@fixture(scope='function')
def data_remap():
    data = [{'name': '1', 'value': 30+31+31},
            {'name': '2', 'value': 28+31+30},
            {'name': '3', 'value': 31+31+30},
            {'name': '4', 'value': 30+31+31}]
    return data


@fixture(scope='function')
def expected_data_remap():
    data = [{'name': '1_0', 'value': 30.666666666},
            {'name': '1_1', 'value': 29.666666666},
            {'name': '1_2', 'value': 29.666666666},
            {'name': '1_3', 'value': 29.666666666},
            {'name': '1_4', 'value': 30.666666666},
            {'name': '1_5', 'value': 30.666666666},
            {'name': '1_6', 'value': 30.666666666},
            {'name': '1_7', 'value': 30.666666666},
            {'name': '1_8', 'value': 30.666666666},
            {'name': '1_9', 'value': 30.666666666},
            {'name': '1_10', 'value': 30.666666666},
            {'name': '1_11', 'value': 30.666666666}]
    return data


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
def monthly_data():
    """[31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    """
    data = [{'name': '1_0', 'value': 31},
            {'name': '1_1', 'value': 28},
            {'name': '1_2', 'value': 31},
            {'name': '1_3', 'value': 30},
            {'name': '1_4', 'value': 31},
            {'name': '1_5', 'value': 30},
            {'name': '1_6', 'value': 31},
            {'name': '1_7', 'value': 31},
            {'name': '1_8', 'value': 30},
            {'name': '1_9', 'value': 31},
            {'name': '1_10', 'value': 30},
            {'name': '1_11', 'value': 31}]
    return data


@fixture(scope='function')
def seasons():
    seasons = [{'name': 'winter', 'start': 'P11M', 'end': 'P2M'},
               {'name': 'spring', 'start': 'P2M', 'end': 'P5M'},
               {'name': 'summer', 'start': 'P5M', 'end': 'P8M'},
               {'name': 'autumn', 'start': 'P8M', 'end': 'P11M'}]
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


class TestInterval:

    def test_empty_interval_list(self):

        with raises(ValueError):
            Interval('test', [])

    def test_empty_interval_tuple(self):
        with raises(ValueError):
            Interval('test', ())

    def test_empty_interval_list_tuple(self):
        with raises(ValueError):
            Interval('test', [()])

    def test_load_interval(self):

        interval = Interval('test', ('PT0H', 'PT1H'))

        assert interval.name == 'test'
        assert interval.start == 'PT0H'
        assert interval.end == 'PT1H'

    def test_raise_error_load_illegal(self):
        with raises(ValueError):
            Interval('test', 'start', 'end')

    def test_convert_hour_to_hours(self):

        interval = Interval('test', ('PT0H', 'PT1H'))
        actual = interval._convert_to_hours(interval.start)
        expected = 0
        assert actual == expected

        actual = interval._convert_to_hours(interval.end)
        expected = 1
        assert actual == expected

    def test_convert_month_to_hours(self):

        interval = Interval('test', ('P1M', 'P2M'))
        actual = interval._convert_to_hours(interval.start)
        expected = 744
        assert actual == expected

        actual = interval._convert_to_hours(interval.end)
        expected = 1416
        assert actual == expected

    def test_convert_week_to_hours(self):

        interval = Interval('test', ('P2D', 'P3D'))
        actual = interval._convert_to_hours(interval.start)
        expected = 48
        assert actual == expected

        actual = interval._convert_to_hours(interval.end)
        expected = 72
        assert actual == expected

    def test_to_hours_zero(self):

        interval = Interval('test', ('PT0H', 'PT1H'))
        actual = interval.to_hours()

        assert actual == [(0, 1)]

    def test_to_hours_month(self):

        interval = Interval('test', ('P2M', 'P3M'))
        actual = interval.to_hours()

        assert actual == [(1416, 2160)]

    def test_to_hours_list(self):

        interval = Interval('test', [('PT0H', 'PT1H'),
                                     ('PT2H', 'PT3H'),
                                     ('PT5H', 'PT7H')])
        actual = interval.to_hours()

        assert actual == [(0, 1), (2, 3), (5, 7)]

    def test_str_one_interval(self):
        interval = Interval('test', ('P2M', 'P3M'))
        actual = str(interval)
        assert actual == \
            "Interval 'test' maps to:\n  hour 1416 to hour 2160\n"

    def test_str_multiple_intervals(self):
        interval = Interval('test', [('PT0H', 'PT1H'),
                                     ('PT2H', 'PT3H'),
                                     ('PT5H', 'PT7H')])
        actual = str(interval)
        assert actual == \
            "Interval 'test' maps to:\n  hour 0 to hour 1\n  hour 2 to hour 3\n  "\
            "hour 5 to hour 7\n"

    def test_repr(self):
        interval = Interval('test', ('P2M', 'P3M'), 2011)
        actual = repr(interval)
        assert actual == "Interval('test', [('P2M', 'P3M')], base_year=2011)"

    def test_load_remap_timeslices(self):
        interval = Interval('1', [('P2M', 'P3M'),
                                  ('P3M', 'P4M'),
                                  ('P4M', 'P5M')])
        actual = interval.to_hours()
        assert actual == [(1416, 2160), (2160, 2880), (2880, 3624)]


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

        timeseries = TimeSeries(data)
        actual = timeseries.names
        expected = ['1_0', '1_1', '1_2', '1_3', '1_4', '1_5',
                    '1_6', '1_7', '1_8', '1_9', '1_10', '1_11']
        assert actual == expected

        actual = timeseries.values
        expected = [1] * 12
        assert actual == expected

        register._convert_to_hourly_buckets(timeseries)
        actual = timeseries._hourly_values

        month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        month_hours = list(map(lambda x: x*24, month_days))
        expected_results = list(map(lambda x: 1/x, month_hours))

        start = 0
        for hours, expected in zip(month_hours, expected_results):
            expected_array = np.zeros(hours, dtype=np.float)
            expected_array.fill(expected)
            assert_equal(actual[start:start + hours], expected_array)
            start += hours


class TestTimeRegisterConversion:

    def test_raises_error_on_no_definition(self):

        register = TimeIntervalRegister()
        with raises(ValueError):
            register.get_intervals_in_set('blobby')

    def test_convert_from_month_to_seasons(self,
                                           months,
                                           seasons,
                                           monthly_data):

        data = monthly_data

        register = TimeIntervalRegister()
        register.add_interval_set(months, 'months')
        register.add_interval_set(seasons, 'seasons')

        timeseries = TimeSeries(data)

        actual = register.convert(timeseries, 'months', 'seasons')
        expected = [{'name': 'winter', 'value': 31. + 31 + 28},
                    {'name': 'spring', 'value': 31. + 30 + 31},
                    {'name': 'summer', 'value': 30. + 31 + 31},
                    {'name': 'autumn', 'value': 30. + 31 + 30}]

        for act, exp in zip(actual, expected):
            assert act['name'] == exp['name']
            assert act['value'] == approx(exp['value'])

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

        timeseries = TimeSeries(data)

        actual = register.convert(timeseries, 'hourly_day', 'one_day')
        expected = [{'name': 'one_day', 'value': 24}]

        assert actual == expected


class TestIntervalRegister:

    def test_interval_loads(self):
        """Pass a time-interval definition into the register

        """
        data = [{'name': '1_1',
                 'start': 'PT0H',
                 'end': 'PT1H'}]

        register = TimeIntervalRegister()
        register.add_interval_set(data, 'energy_supply_hourly')

        actual = register.get_intervals_in_set('energy_supply_hourly')

        element = Interval('1_1', ('PT0H', 'PT1H'), base_year=2010)

        expected = OrderedDict()
        expected['1_1'] = element

        assert actual == expected

    def test_months_load(self, months):
        """Pass a monthly time-interval definition into the register

        """
        register = TimeIntervalRegister()
        register.add_interval_set(months, 'months')

        actual = register.get_intervals_in_set('months')

        expected_names = \
            ['1_0', '1_1', '1_2', '1_3', '1_4', '1_5',
             '1_6', '1_7', '1_8', '1_9', '1_10', '1_11']

        expected = [Interval('1_0', ('P0M', 'P1M')),
                    Interval('1_1', ('P1M', 'P2M')),
                    Interval('1_2', ('P2M', 'P3M')),
                    Interval('1_3', ('P3M', 'P4M')),
                    Interval('1_4', ('P4M', 'P5M')),
                    Interval('1_5', ('P5M', 'P6M')),
                    Interval('1_6', ('P6M', 'P7M')),
                    Interval('1_7', ('P7M', 'P8M')),
                    Interval('1_8', ('P8M', 'P9M')),
                    Interval('1_9', ('P9M', 'P10M')),
                    Interval('1_10', ('P10M', 'P11M')),
                    Interval('1_11', ('P11M', 'P12M'))]

        for name, interval in zip(expected_names, expected):
            assert actual[name] == interval

    def test_remap_interval_load(self, remap_months):
        register = TimeIntervalRegister()
        register.add_interval_set(remap_months, 'remap_months')

        actual = register.get_intervals_in_set('remap_months')

        expected_names = ['1', '2', '3', '4']

        expected = [Interval('1', [('P10M', 'P11M'),
                                   ('P11M', 'P12M'),
                                   ('P0M', 'P1M')]),
                    Interval('2', [('P1M', 'P2M'),
                                   ('P2M', 'P3M'),
                                   ('P3M', 'P4M')]),
                    Interval('3', [('P4M', 'P5M'),
                                   ('P5M', 'P6M'),
                                   ('P6M', 'P7M')]),
                    Interval('4', [('P7M', 'P8M'),
                                   ('P8M', 'P9M'),
                                   ('P9M', 'P10M')])]

        for name, interval in zip(expected_names, expected):
            assert actual[name] == interval


class TestRemapConvert:

    def test_remap_timeslices_to_months(self,
                                        months,
                                        expected_data_remap,
                                        remap_months,
                                        data_remap):
        """
        """
        timeslice_data = data_remap

        register = TimeIntervalRegister()
        register.add_interval_set(months, 'months')
        register.add_interval_set(remap_months, 'remap_months')

        timeseries = TimeSeries(timeslice_data)

        actual = register.convert(timeseries, 'remap_months', 'months')
        expected = expected_data_remap

        for act, exp in zip(actual, expected):
            print(act['name'], act['value'])
            assert act['name'] == exp['name']
            assert act['value'] == approx(exp['value'])

    def test_remap_months_to_timeslices(self,
                                        months,
                                        monthly_data,
                                        remap_months,
                                        data_remap):
        """
        """
        timeslice_data = data_remap
        monthly_data = monthly_data

        register = TimeIntervalRegister()
        register.add_interval_set(months, 'months')
        register.add_interval_set(remap_months, 'remap_months')

        timeseries = TimeSeries(monthly_data)

        actual = register.convert(timeseries, 'months', 'remap_months')
        expected = timeslice_data

        for act, exp in zip(actual, expected):
            assert act['name'] == exp['name']
            assert act['value'] == approx(exp['value'])


class TestValidation:

    def test_validate_get_hourly_array(self, remap_months):

        data = remap_months
        register = TimeIntervalRegister()
        register.add_interval_set(data, 'remap_months')

        actual = register._get_hourly_array('remap_months')
        expected = np.ones(8760, dtype=np.int)
        assert_equal(actual, expected)

    def test_validate_intervals_passes(self, remap_months):

        data = remap_months
        register = TimeIntervalRegister()
        register.add_interval_set(data, 'remap_months')

    def test_validate_intervals_fails(self, remap_months):

        data = remap_months
        data.append({'name': '5', 'start': 'PT0H', 'end': 'PT1H'})
        register = TimeIntervalRegister()
        with raises(ValueError) as excinfo:
            register.add_interval_set(data, 'remap_months')
        assert "Duplicate entry for hour 0 in interval set remap_months." in str(excinfo.value)
