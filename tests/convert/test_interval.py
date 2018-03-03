import numpy as np
from numpy.testing import assert_equal
from pytest import raises
from smif.convert.interval import Interval, IntervalSet, TimeIntervalRegister


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
        expected = []
        expected.append(element)

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

        for idx, interval in enumerate(expected):
            assert actual.data[idx] == interval

    def test_remap_interval_load(self, remap_months):
        register = TimeIntervalRegister()

        intervals = IntervalSet('remap_months', remap_months)

        register.register(intervals)

        actual = register.get_entry('remap_months')

        assert actual == intervals


class TestRemapConvert:

    def test_remap_coefficients(self, months, remap_months, month_to_season_coefficients):
        register = TimeIntervalRegister()
        register.register(IntervalSet('months', months))
        register.register(IntervalSet('remap_months', remap_months))
        actual = register.get_coefficients('months', 'remap_months')

        expected = month_to_season_coefficients

        np.testing.assert_equal(actual, expected)

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

        np.testing.assert_allclose(actual, expected, rtol=1e-05, atol=1e-08)

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

        np.testing.assert_allclose(actual, expected, rtol=1e-05, atol=1e-08)


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

    def test_time_interval_start_before_end(self):
        with raises(ValueError) as excinfo:
            Interval('backwards', ('P1Y', 'P3M'))
        assert "A time interval must not end before it starts" in str(excinfo)

        interval = Interval('starts_ok', ('P0Y', 'P1M'))
        with raises(ValueError) as excinfo:
            interval.interval = ('P2M', 'P1M')
        assert "A time interval must not end before it starts" in str(excinfo)
