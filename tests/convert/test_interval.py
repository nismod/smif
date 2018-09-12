"""Test interval adaptor
"""
from unittest.mock import Mock

import numpy as np
from numpy.testing import assert_equal
from pytest import mark, raises
from smif.convert.interval import Interval, IntervalAdaptor, IntervalSet
from smif.convert.register import NDimensionalRegister
from smif.metadata import Spec


class TestTimeRegisterConversion:

    def test_aggregate_from_month_to_seasons(self,
                                             months,
                                             seasons,
                                             monthly_data,
                                             monthly_data_as_seasons):
        """Aggregate months to values for each season
        """
        adaptor = IntervalAdaptor('test-month-season')
        from_spec = Spec(
            name='test-var',
            dtype='float',
            dims=['months'],
            coords={
                'months':  months
            }
        )
        adaptor.add_input(from_spec)
        to_spec = Spec(
            name='test-var',
            dtype='float',
            dims=['seasons'],
            coords={
                'seasons':  seasons
            }
        )
        adaptor.add_output(to_spec)
        actual_coefficients = adaptor.generate_coefficients(from_spec, to_spec)

        data_handle = Mock()
        data_handle.get_data = Mock(return_value=monthly_data)
        data_handle.read_coefficients = Mock(return_value=actual_coefficients)

        adaptor.simulate(data_handle)
        actual = data_handle.set_results.call_args[0][1]
        expected = monthly_data_as_seasons

        assert np.allclose(actual, expected, rtol=1e-05, atol=1e-08)

    def test_agggregate_from_hour_to_day(self, twenty_four_hours, one_day):
        """Aggregate hours to a single value for a day
        """
        data = np.ones((24,))

        adaptor = IntervalAdaptor('test-hourly-day')
        from_spec = Spec(
            name='test-var',
            dtype='float',
            dims=['hourly_day'],
            coords={
                'hourly_day': twenty_four_hours
            }
        )
        adaptor.add_input(from_spec)
        to_spec = Spec(
            name='test-var',
            dtype='float',
            dims=['one_day'],
            coords={
                'one_day': one_day
            }
        )
        adaptor.add_output(to_spec)
        actual_coefficients = adaptor.generate_coefficients(from_spec, to_spec)

        data_handle = Mock()
        data_handle.get_data = Mock(return_value=data)
        data_handle.read_coefficients = Mock(return_value=actual_coefficients)

        adaptor.simulate(data_handle)
        actual = data_handle.set_results.call_args[0][1]
        expected = np.array([24])

        assert np.allclose(actual, expected, rtol=1e-05, atol=1e-08)

    def test_time_only_conversion(self, months, seasons):
        """Aggregate from months to seasons, summing groups of months
        """
        adaptor = IntervalAdaptor('test-month-season')
        from_spec = Spec(
            name='test-var',
            dtype='float',
            dims=['months'],
            coords={
                'months': months
            }
        )
        adaptor.add_input(from_spec)
        to_spec = Spec(
            name='test-var',
            dtype='float',
            dims=['seasons'],
            coords={
                'seasons': seasons
            }
        )
        adaptor.add_output(to_spec)
        actual_coefficients = adaptor.generate_coefficients(from_spec, to_spec)

        data = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        data_handle = Mock()
        data_handle.get_data = Mock(return_value=data)
        data_handle.read_coefficients = Mock(return_value=actual_coefficients)

        adaptor.simulate(data_handle)
        actual = data_handle.set_results.call_args[0][1]
        expected = np.array([3, 3, 3, 3])
        np.testing.assert_array_equal(actual, expected)

    def test_time_only_conversion_disagg(self, months, seasons):
        """Disaggregate from seasons to months based on duration of each month/season
        """
        adaptor = IntervalAdaptor('test-season-month')
        from_spec = Spec(
            name='test-var',
            dtype='float',
            dims=['seasons'],
            coords={
                'seasons': seasons
            }
        )
        adaptor.add_input(from_spec)
        to_spec = Spec(
            name='test-var',
            dtype='float',
            dims=['months'],
            coords={
                'months': months
            }
        )
        adaptor.add_output(to_spec)
        actual_coefficients = adaptor.generate_coefficients(from_spec, to_spec)

        data = np.array([3, 3, 3, 3])
        data_handle = Mock()
        data_handle.get_data = Mock(return_value=data)
        data_handle.read_coefficients = Mock(return_value=actual_coefficients)

        adaptor.simulate(data_handle)
        actual = data_handle.set_results.call_args[0][1]
        expected = np.array([1.033333, 0.933333, 1.01087, 0.978261,
                             1.01087, 0.978261, 1.01087, 1.01087,
                             0.989011, 1.021978, 0.989011, 1.033333])
        np.testing.assert_allclose(actual, expected, rtol=1e-3)

    def test_remap_coefficients(self, months, remap_months):
        """Twelve months are remapped (averaged) into four representative months
        """
        adaptor = IntervalAdaptor('test-month-remap')
        from_spec = Spec(
            name='test-var',
            dtype='float',
            dims=['months'],
            coords={
                'months': months
            }
        )
        adaptor.add_input(from_spec)
        to_spec = Spec(
            name='test-var',
            dtype='float',
            dims=['remap_months'],
            coords={
                'remap_months': remap_months
            }
        )
        adaptor.add_output(to_spec)

        actual = adaptor.generate_coefficients(from_spec, to_spec)
        expected = np.array(
            [
                [0.333333, 0, 0, 0],
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
                [0.333333, 0, 0, 0]
            ]
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-3)

    def test_resample_mapped_months_to_months(self, months, remap_months):
        """Converts from remapped month data (where one average month is used
        for each season) back to months
        """
        register = NDimensionalRegister()
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
        register = NDimensionalRegister()
        register.register(IntervalSet('months', months))
        register.register(IntervalSet('remap_months', remap_months))

        actual = register.convert(monthly_data, 'months', 'remap_months')
        expected = remap_month_data

        np.testing.assert_allclose(actual, expected, rtol=1e-05, atol=1e-08)


@mark.xfail()
class TestConvertor:
    """Integration tests of convertor - these should be updated to test IntervalAdaptor.
    """
    def test_one_region_pass_through_time(self, convertor):
        """Only one region, 12 months, neither space nor time conversion is required
        """
        data = np.array([[
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
            31
        ]], dtype=float)
        actual = convertor.convert(
            data,
            'half_squares',
            'half_squares',
            'months',
            'months',
            's',
            's'
        )
        assert np.allclose(actual, data)

    def test_one_region_time_aggregation(self, convertor):
        """Only one region, time aggregation is required
        """
        data = np.array([[
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
            31
        ]], dtype=float)  # area a, months 1-12

        expected = np.array([[
            31 + 31 + 28,
            31 + 30 + 31,
            30 + 31 + 31,
            30 + 31 + 30
        ]], dtype=float)  # area a, seasons 1-4

        actual = convertor.convert(
            data,
            'half_squares',
            'half_squares',
            'months',
            'seasons',
            's',
            's'
        )
        assert np.allclose(actual, expected)

    def test_two_region_time_aggregation(self, convertor):
        """Two regions, time aggregation by region is required
        """
        data = np.array([
            # area a, months 1-12
            [
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
            ],
            # area b, months 1-12
            [
                31+1,
                28+1,
                31+1,
                30+1,
                31+1,
                30+1,
                31+1,
                31+1,
                30+1,
                31+1,
                30+1,
                31+1,
            ]
        ], dtype=float)

        actual = convertor.convert(
            data,
            'half_squares',
            'half_squares',
            'months',
            'seasons',
            'm',
            'm'
        )

        expected = np.array([
            [
                31 + 31 + 28,
                31 + 30 + 31,
                30 + 31 + 31,
                30 + 31 + 30,
            ],
            [
                31 + 31 + 28 + 3,
                31 + 30 + 31 + 3,
                30 + 31 + 31 + 3,
                30 + 31 + 30 + 3,
            ]
        ], dtype=float)

        assert np.allclose(actual, expected)

    def test_one_region_convert_from_hour_to_day(self, convertor):
        """One region, time aggregation required
        """
        data = np.ones((1, 24))  # area a, hours 0-23
        actual = convertor.convert(
            data,
            'half_squares',
            'half_squares',
            'hourly_day',
            'one_day',
            'm',
            'm'
        )
        expected = np.array([[24]])  # area a, day 0
        assert np.allclose(actual, expected)

    def test_remap_timeslices_to_months(self, convertor):
        """One region, time remapping required
        """
        data = np.array([[
            1,  # winter month
            1,  # spring month
            1,  # summer month
            1  # autumn month
        ]], dtype=float)
        actual = convertor.convert(
            data,
            'half_squares',
            'half_squares',
            'remap_months',
            'months',
            'm',
            'm'
        )
        expected = np.array([[1.03333333, 0.93333333, 1.01086957, 0.97826087,
                              1.01086957, 0.97826087, 1.01086957, 1.01086957,
                              0.98901099, 1.02197802, 0.98901099, 1.03333333]])
        assert np.allclose(actual, expected)

    def test_remap_months_to_timeslices(self, convertor, monthly_data, remap_month_data):
        """One region, time remapping required
        """
        data = monthly_data
        actual = convertor.convert(
            data,
            'half_squares',
            'half_squares',
            'months',
            'remap_months',
            'm',
            'm'
        )
        expected = remap_month_data
        assert np.allclose(actual, expected)


class TestInterval:
    """Test interval object representation
    """
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
    """IntervalSet should map to a Coordinates definition
    """
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
    """
    """
    def test_interval_loads(self):
        """Pass a time-interval definition into the register

        """
        data = [{'name': '1_1', 'interval': [('PT0H', 'PT1H')]}]

        register = NDimensionalRegister()
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
        register = NDimensionalRegister()
        register.register(IntervalSet('months', months))
        with raises(ValueError):
            register.register(IntervalSet('months', months))

    def test_months_load(self, months):
        """Pass a monthly time-interval definition into the register
        """
        register = NDimensionalRegister()
        register.register(IntervalSet('months', months))

        actual = register.get_entry('months')

        expected = [
            Interval('jan', ('P0M', 'P1M')),
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
            Interval('dec', ('P11M', 'P12M'))
        ]

        for idx, interval in enumerate(expected):
            assert actual.data[idx] == interval

    def test_remap_interval_load(self, remap_months):
        register = NDimensionalRegister()

        intervals = IntervalSet('remap_months', remap_months)

        register.register(intervals)

        actual = register.get_entry('remap_months')

        assert actual == intervals

    def test_conversion_with_different_coverage_fails(self, one_year, one_day, caplog):

        register = NDimensionalRegister()
        register.register(IntervalSet("one_year", one_year))
        register.register(IntervalSet("one_day", one_day))

        data = np.array([[1]])
        register.convert(data, 'one_year', 'one_day')

        expected = "Coverage for 'one_year' is 8760 and does not match " \
                   "coverage for 'one_day' which is 24"

        assert expected in caplog.text


class TestTimeRegisterCoefficients:

    def test_coeff(self, months, seasons, month_to_season_coefficients):

        register = NDimensionalRegister()
        register.register(IntervalSet('months', months))
        register.register(IntervalSet('seasons', seasons))

        actual = register.get_coefficients('months', 'seasons')
        expected = month_to_season_coefficients
        assert np.allclose(actual, expected, rtol=1e-05, atol=1e-08)


class TestValidation:

    def test_validate_get_hourly_array(self, remap_months):
        intervals = IntervalSet('remap_months', remap_months)

        actual = intervals._get_hourly_array()
        expected = np.ones(8760, dtype=np.int)
        assert_equal(actual, expected)

    def test_validate_intervals_passes(self, remap_months):
        register = Mock()
        register.register(IntervalSet('remap_months', remap_months))

    def test_validate_intervals_fails(self, remap_months):
        data = remap_months
        data.append({'name': '5', 'interval': [('PT0H', 'PT1H')]})
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
