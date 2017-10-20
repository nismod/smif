
from collections import OrderedDict

import numpy as np
from numpy.testing import assert_equal
from pytest import fixture, raises
from smif.convert.interval import Interval, IntervalSet, TimeIntervalRegister


@fixture(scope='function')
def months():
    data = [
        {'id': '1_0', 'start': 'P0M', 'end': 'P1M'},
        {'id': '1_1', 'start': 'P1M', 'end': 'P2M'},
        {'id': '1_2', 'start': 'P2M', 'end': 'P3M'},
        {'id': '1_3', 'start': 'P3M', 'end': 'P4M'},
        {'id': '1_4', 'start': 'P4M', 'end': 'P5M'},
        {'id': '1_5', 'start': 'P5M', 'end': 'P6M'},
        {'id': '1_6', 'start': 'P6M', 'end': 'P7M'},
        {'id': '1_7', 'start': 'P7M', 'end': 'P8M'},
        {'id': '1_8', 'start': 'P8M', 'end': 'P9M'},
        {'id': '1_9', 'start': 'P9M', 'end': 'P10M'},
        {'id': '1_10', 'start': 'P10M', 'end': 'P11M'},
        {'id': '1_11', 'start': 'P11M', 'end': 'P12M'}
    ]

    return data


@fixture(scope='function')
def monthly_data():
    """[31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    """
    data = np.array([
        31,
        28,
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ])
    return data


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
    data = [{'id': '1', 'start': 'P0M', 'end': 'P1M'},
            {'id': '2', 'start': 'P1M', 'end': 'P2M'},
            {'id': '2', 'start': 'P2M', 'end': 'P3M'},
            {'id': '2', 'start': 'P3M', 'end': 'P4M'},
            {'id': '3', 'start': 'P4M', 'end': 'P5M'},
            {'id': '3', 'start': 'P5M', 'end': 'P6M'},
            {'id': '3', 'start': 'P6M', 'end': 'P7M'},
            {'id': '4', 'start': 'P7M', 'end': 'P8M'},
            {'id': '4', 'start': 'P8M', 'end': 'P9M'},
            {'id': '4', 'start': 'P9M', 'end': 'P10M'},
            {'id': '1', 'start': 'P10M', 'end': 'P11M'},
            {'id': '1', 'start': 'P11M', 'end': 'P12M'}]
    return data


@fixture(scope='function')
def remap_month_data():
    data = np.array([
        30+31+31,
        28+31+30,
        31+31+30,
        30+31+31
    ], dtype=float)

    return data


@fixture(scope='function')
def remap_month_data_as_months():
    data = np.array([
        30.666666666,
        29.666666666,
        29.666666666,
        29.666666666,
        30.666666666,
        30.666666666,
        30.666666666,
        30.666666666,
        30.666666666,
        30.666666666,
        30.666666666,
        30.666666666
    ])
    return data


@fixture(scope='function')
def seasons():
    # NB "winter" is split into two pieces around the year end
    data = [{'id': 'winter', 'start': 'P0M', 'end': 'P2M'},
            {'id': 'spring', 'start': 'P2M', 'end': 'P5M'},
            {'id': 'summer', 'start': 'P5M', 'end': 'P8M'},
            {'id': 'autumn', 'start': 'P8M', 'end': 'P11M'},
            {'id': 'winter', 'start': 'P11M', 'end': 'P12M'}]
    return data


@fixture(scope='function')
def monthly_data_as_seasons():
    return np.array([
        31 + 31 + 28,
        31 + 30 + 31,
        30 + 31 + 31,
        30 + 31 + 30
    ], dtype=float)


@fixture(scope='function')
def twenty_four_hours():
    data = [
        {'id': '1_0', 'start': 'PT0H', 'end': 'PT1H'},
        {'id': '1_1', 'start': 'PT1H', 'end': 'PT2H'},
        {'id': '1_2', 'start': 'PT2H', 'end': 'PT3H'},
        {'id': '1_3', 'start': 'PT3H', 'end': 'PT4H'},
        {'id': '1_4', 'start': 'PT4H', 'end': 'PT5H'},
        {'id': '1_5', 'start': 'PT5H', 'end': 'PT6H'},
        {'id': '1_6', 'start': 'PT6H', 'end': 'PT7H'},
        {'id': '1_7', 'start': 'PT7H', 'end': 'PT8H'},
        {'id': '1_8', 'start': 'PT8H', 'end': 'PT9H'},
        {'id': '1_9', 'start': 'PT9H', 'end': 'PT10H'},
        {'id': '1_10', 'start': 'PT10H', 'end': 'PT11H'},
        {'id': '1_11', 'start': 'PT11H', 'end': 'PT12H'},
        {'id': '1_12', 'start': 'PT12H', 'end': 'PT13H'},
        {'id': '1_13', 'start': 'PT13H', 'end': 'PT14H'},
        {'id': '1_14', 'start': 'PT14H', 'end': 'PT15H'},
        {'id': '1_15', 'start': 'PT15H', 'end': 'PT16H'},
        {'id': '1_16', 'start': 'PT16H', 'end': 'PT17H'},
        {'id': '1_17', 'start': 'PT17H', 'end': 'PT18H'},
        {'id': '1_18', 'start': 'PT18H', 'end': 'PT19H'},
        {'id': '1_19', 'start': 'PT19H', 'end': 'PT20H'},
        {'id': '1_20', 'start': 'PT20H', 'end': 'PT21H'},
        {'id': '1_21', 'start': 'PT21H', 'end': 'PT22H'},
        {'id': '1_22', 'start': 'PT22H', 'end': 'PT23H'},
        {'id': '1_23', 'start': 'PT23H', 'end': 'PT24H'}
    ]
    return data


@fixture(scope='function')
def one_day():
    data = [{'id': 'one_day', 'start': 'P0D', 'end': 'P1D'}]
    return data


class TestInterval:

    def test_empty_interval_list(self):

        with raises(ValueError):
            Interval('test', [])

    def test_equality(self):
        a = Interval('test', ('PT0H', 'PT1H'))
        b = Interval('test', ('PT0H', 'PT1H'))
        c = Interval('another_test', ('PT0H', 'PT1H'))
        d = Interval('another_test', ('PT1H', 'PT2H'))

        assert a == b
        assert a != c
        assert b != c
        assert c != d

    def test_invalid_set_interval(self):
        interval = Interval('test', ('PT0H', 'PT1H'))
        with raises(ValueError) as excinfo:
            interval.interval = None
        msg = "A time interval must add either a single tuple or a list of tuples"
        assert msg in str(excinfo)

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

    def test_load_multiple_interval(self):

        interval = Interval('test', [('PT0H', 'PT1H'), ('PT1H', 'PT2H')])

        assert interval.name == 'test'
        assert interval.start == ['PT0H', 'PT1H']
        assert interval.end == ['PT1H', 'PT2H']

    def test_add_list_of_interval(self):

        interval = Interval('test', ('PT0H', 'PT1H'))
        interval.interval = [('PT1H', 'PT2H')]

        assert interval.name == 'test'
        assert interval.start == ['PT0H', 'PT1H']
        assert interval.end == ['PT1H', 'PT2H']

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


class TestTimeRegisterConversion:

    def test_raises_error_on_no_definition(self):

        register = TimeIntervalRegister()
        with raises(ValueError):
            register.get_entry('blobby')

    def test_convert_from_month_to_seasons(self,
                                           months,
                                           seasons,
                                           monthly_data,
                                           monthly_data_as_seasons):
        register = TimeIntervalRegister()
        register.register(IntervalSet('months', months))
        register.register(IntervalSet('seasons', seasons))

        actual = register.convert(monthly_data, 'months', 'seasons')
        expected = monthly_data_as_seasons
        assert np.allclose(actual, expected, rtol=1e-05, atol=1e-08)

    def test_convert_from_hour_to_day(self, twenty_four_hours, one_day):

        data = np.ones(24)

        register = TimeIntervalRegister()
        register.register(IntervalSet('hourly_day', twenty_four_hours))
        register.register(IntervalSet('one_day', one_day))

        actual = register.convert(data, 'hourly_day', 'one_day')
        expected = np.array([24])

        assert np.allclose(actual, expected, rtol=1e-05, atol=1e-08)


class TestIntervalSet:

    def test_get_names(self, months):

        expected_names = \
            ['1_0', '1_1', '1_2', '1_3', '1_4', '1_5',
             '1_6', '1_7', '1_8', '1_9', '1_10', '1_11']
        interval_set = IntervalSet('months', months)
        actual_names = interval_set.get_entry_names()
        assert expected_names == actual_names

    def test_attributes(self, months):
        interval_set = IntervalSet('months', months)
        interval_set.description = 'a descriptions'

        actual = interval_set.as_dict()
        expected = {'name': 'months',
                    'description': 'a descriptions'}
        assert actual == expected


class TestIntervalRegister:

    def test_interval_loads(self):
        """Pass a time-interval definition into the register

        """
        data = [{'id': '1_1',
                 'start': 'PT0H',
                 'end': 'PT1H'}]

        register = TimeIntervalRegister()
        register.register(IntervalSet('energy_supply_hourly', data))
        assert register.names == ['energy_supply_hourly']

        actual = register.get_entry('energy_supply_hourly')

        element = Interval('1_1', ('PT0H', 'PT1H'), base_year=2010)
        expected = OrderedDict()
        expected['1_1'] = element

        assert actual.data == expected

    def test_interval_load_duplicate_name_raises(self, months):
        """Tests that error is raised if a duplicate name is used
        for an interval set
        """
        register = TimeIntervalRegister()
        register.register(IntervalSet('months', months))
        with raises(ValueError):
            register.register(IntervalSet('months', months))

    def test_months_load(self, months):
        """Pass a monthly time-interval definition into the register

        """
        register = TimeIntervalRegister()
        register.register(IntervalSet('months', months))

        actual = register.get_entry('months')

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
            assert actual.data[name] == interval

    def test_remap_interval_load(self, remap_months):
        register = TimeIntervalRegister()

        intervals = IntervalSet('remap_months', remap_months)

        register.register(intervals)

        actual = register.get_entry('remap_months')

        assert actual == intervals


class TestRemapConvert:

    def test_remap_timeslices_to_months(self,
                                        months,
                                        remap_month_data_as_months,
                                        remap_months,
                                        remap_month_data):
        register = TimeIntervalRegister()
        register.register(IntervalSet('months', months))
        register.register(IntervalSet('remap_months', remap_months))

        actual = register.convert(remap_month_data, 'remap_months', 'months')
        expected = remap_month_data_as_months

        assert np.allclose(actual, expected, rtol=1e-05, atol=1e-08)

    def test_remap_months_to_timeslices(self,
                                        months,
                                        monthly_data,
                                        remap_months,
                                        remap_month_data):
        register = TimeIntervalRegister()
        register.register(IntervalSet('months', months))
        register.register(IntervalSet('remap_months', remap_months))

        actual = register.convert(monthly_data, 'months', 'remap_months')
        expected = remap_month_data

        assert np.allclose(actual, expected, rtol=1e-05, atol=1e-08)


class TestValidation:

    def test_validate_get_hourly_array(self, remap_months):
        intervals = IntervalSet('remap_months', remap_months)

        actual = intervals._get_hourly_array()
        expected = np.ones(8760, dtype=np.int)
        assert_equal(actual, expected)

    def test_validate_intervals_passes(self, remap_months):

        register = TimeIntervalRegister()
        register.register(IntervalSet('remap_months', remap_months))

    def test_validate_intervals_fails(self, remap_months):
        data = remap_months
        data.append({'id': '5', 'start': 'PT0H', 'end': 'PT1H'})
        with raises(ValueError) as excinfo:
            IntervalSet('remap_months', data)
        assert "Duplicate entry for hour 0 in interval set remap_months." in str(excinfo.value)

    def test_time_interval_start_before_end(get_time_intervals):
        with raises(ValueError) as excinfo:
            Interval('backwards', ('P1Y', 'P3M'))
        assert "A time interval must not end before it starts" in str(excinfo)

        interval = Interval('starts_ok', ('P0Y', 'P1M'))
        with raises(ValueError) as excinfo:
            interval.interval = ('P2M', 'P1M')
        assert "A time interval must not end before it starts" in str(excinfo)
