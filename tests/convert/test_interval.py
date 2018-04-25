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
        actual = interval.bounds

        assert actual == [(0, 1)]

    def test_to_hours_month(self):

        interval = Interval('test', ('P2M', 'P3M'))
        actual = interval.bounds

        assert actual == [(1416, 2160)]

    def test_to_hours_list(self):

        interval = Interval('test', [('PT0H', 'PT1H'),
                                     ('PT2H', 'PT3H'),
                                     ('PT5H', 'PT7H')])
        actual = interval.bounds

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
        actual = interval.bounds
        assert actual == [(1416, 2160), (2160, 2880), (2880, 3624)]


class TestIntervalSet:

    def test_get_names(self, months):

        expected_names = \
            ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
             'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
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


class TestIntervalRegister():

    def test_interval_loads(self):
        """Pass a time-interval definition into the register

        """
        data = [('1_1', [('PT0H', 'PT1H')])]

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

        expected = [Interval('jan', ('P0M', 'P1M')),
                    Interval('feb', ('P1M', 'P2M')),
                    Interval('mar', ('P2M', 'P3M')),
                    Interval('apr', ('P3M', 'P4M')),
                    Interval('may', ('P4M', 'P5M')),
                    Interval('jun', ('P5M', 'P6M')),
                    Interval('jul', ('P6M', 'P7M')),
                    Interval('aug', ('P7M', 'P8M')),
                    Interval('sep', ('P8M', 'P9M')),
                    Interval('oct', ('P9M', 'P10M')),
                    Interval('nov', ('P10M', 'P11M')),
                    Interval('dec', ('P11M', 'P12M'))]

        for idx, interval in enumerate(expected):
            assert actual.data[idx] == interval

    def test_remap_interval_load(self, remap_months):
        register = TimeIntervalRegister()

        intervals = IntervalSet('remap_months', remap_months)

        register.register(intervals)

        actual = register.get_entry('remap_months')

        assert actual == intervals

    def test_conversion_with_different_coverage_fails(self, one_year, one_day, caplog):

        register = TimeIntervalRegister()
        register.register(IntervalSet("one_year", one_year))
        register.register(IntervalSet("one_day", one_day))

        data = np.array([[1]])
        register.convert(data, 'one_year', 'one_day')

        expected = "Coverage for 'one_year' is 8760 and does not match " \
                   "coverage for 'one_day' which is 24"

        assert expected in caplog.text


class TestTimeRegisterCoefficients:

    def test_coeff(self, months, seasons, month_to_season_coefficients):

        register = TimeIntervalRegister()
        register.register(IntervalSet('months', months))
        register.register(IntervalSet('seasons', seasons))

        actual = register.get_coefficients('months', 'seasons')
        expected = month_to_season_coefficients
        assert np.allclose(actual, expected, rtol=1e-05, atol=1e-08)

    def test_time_only_conversion(self, months, seasons):

        register = TimeIntervalRegister()
        register.register(IntervalSet('months', months))
        register.register(IntervalSet('seasons', seasons))
        data = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        actual = register.convert(data, 'months', 'seasons')
        expected = np.array([[3, 3, 3, 3]])
        np.testing.assert_array_equal(actual, expected)

    def test_time_only_conversion_disagg(self, months, seasons):

        register = TimeIntervalRegister()
        register.register(IntervalSet('months', months))
        register.register(IntervalSet('seasons', seasons))
        data = np.array([[3, 3, 3, 3]])
        actual = register.convert(data, 'seasons', 'months')
        expected = np.array([[1.033333, 0.933333, 1.01087, 0.978261,
                              1.01087, 0.978261, 1.01087, 1.01087,
                              0.989011, 1.021978, 0.989011, 1.033333]])
        np.testing.assert_allclose(actual, expected, rtol=1e-3)


class TestTimeRegisterConversion:

    def test_raises_error_on_no_definition(self):

        register = TimeIntervalRegister()
        with raises(ValueError):
            register.get_entry('blobby')

    def test_aggregate_from_month_to_seasons(self,
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

    def test_agggregate_from_hour_to_day(self, twenty_four_hours, one_day):

        data = np.ones((1, 24))

        register = TimeIntervalRegister()
        register.register(IntervalSet('hourly_day', twenty_four_hours))
        register.register(IntervalSet('one_day', one_day))

        actual = register.convert(data, 'hourly_day', 'one_day')
        expected = np.array([24])

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
        data.append(('5', [('PT0H', 'PT1H')]))
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


class TestIntersection:

    def test_intersection(self, months, seasons):

        month_set = IntervalSet('months', months)
        season_set = IntervalSet('seasons', seasons)

        actual = month_set.intersection(season_set.data[0])
        expected = [0, 1, 11]
        assert actual == expected

    def test_intersection_seasons(self, months, seasons):

        month_set = IntervalSet('months', months)
        season_set = IntervalSet('seasons', seasons)
        actual = season_set.intersection(month_set.data[0])
        expected = [0]
        assert actual == expected


class TestBounds:

    def test_bounds_checker(self, months):

        bounds = [(0, 1), (1, 2)]

        expected = True

        time_set = IntervalSet('months', months)
        actual = time_set.check_interval_bounds_equal(bounds)

        assert actual == expected

    def test_bounds_checker_fails(self, months):

        bounds = [(0, 8759), (8759, 8760)]

        expected = False

        time_set = IntervalSet('months', months)
        actual = time_set.check_interval_bounds_equal(bounds)

        assert actual == expected


class TestProportions:

    def test_get_proportion_month_in_season(self, months, seasons):

        month_set = IntervalSet('months', months)
        season_set = IntervalSet('seasons', seasons)

        winter = season_set.data[0]

        actual = month_set.get_proportion(0, winter)
        expected = 1
        assert actual == expected

    def test_get_proportion_season_in_month(self, months, seasons):

        month_set = IntervalSet('months', months)
        season_set = IntervalSet('seasons', seasons)

        january = month_set.data[0]

        actual = season_set.get_proportion(0, january)
        expected = 31 * 1 / (31+31+28)
        np.testing.assert_allclose(actual, expected)

    def test_remap_disagg_proportions(self, months, remap_months):
        """Find proportion of January in cold_month, split over year
        """

        month_set = IntervalSet('months', months)
        remap_set = IntervalSet('remap_months', remap_months)

        to_interval = remap_set.data[0]  # cold month

        actual = month_set.get_proportion(0, to_interval)
        expected = 0.33333
        np.testing.assert_allclose(actual, expected, rtol=1e-3)

    def test_remap_agg_proportions(self, months, remap_months):

        month_set = IntervalSet('months', months)
        remap_set = IntervalSet('remap_months', remap_months)

        to_interval = month_set.data[0]  # january

        actual = remap_set.get_proportion(0, to_interval)
        expected = 1.0333333333333
        np.testing.assert_allclose(actual, expected, rtol=1e-3)


class TestRemapConvert:

    def test_remap_coefficients(self, months, remap_months):
        """Twelve months are remapped (averaged) into four
        representative months
        """
        register = TimeIntervalRegister()
        register.register(IntervalSet('months', months))
        register.register(IntervalSet('remap_months', remap_months))
        actual = register.get_coefficients('months', 'remap_months')

        expected = np.array([[0.333333, 0, 0, 0],
                             [0.333333, 0, 0, 0],
                             [0, 0.333333, 0, 0],
                             [0, 0.333333, 0, 0],
                             [0, 0.333333, 0, 0],
                             [0, 0, 0.333333, 0],
                             [0, 0, 0.333333, 0],
                             [0, 0, 0.333333, 0],
                             [0, 0, 0, 0.333333],
                             [0, 0, 0, 0.333333],
                             [0, 0, 0, 0.333333],
                             [0.333333, 0, 0, 0]]
                            )

        np.testing.assert_allclose(actual, expected, rtol=1e-3)

    def test_resample_mapped_months_to_months(self, months, remap_months):
        """Converts from remapped month data (where one average month is used
        for each season) back to months
        """
        register = TimeIntervalRegister()
        register.register(IntervalSet('remap_months', remap_months))
        register.register(IntervalSet('months', months))

        data = np.array([[1, 1, 1, 1]])
        actual = register.convert(data,
                                  'remap_months',
                                  'months')
        expected = np.array([[1.033333, 0.933333, 1.01087, 0.978261, 1.01087,
                              0.978261, 1.01087, 1.01087, 0.989011, 1.021978,
                              0.989011, 1.033333]])

        np.testing.assert_allclose(actual, expected, rtol=1e-3)

    def test_remap_months_to_mapped_months(self,
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
