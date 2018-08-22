import numpy as np
from pytest import fixture
from smif.convert import SpaceTimeUnitConvertor
from smif.convert.interval import IntervalSet


@fixture(scope='function')
def convertor(regions_rect, regions_half_squares, months, seasons, twenty_four_hours, one_day,
              remap_months):
    """Convertor with dimensions pre-registered for use
    """
    convertor = SpaceTimeUnitConvertor()
    convertor.regions.register(regions_rect)
    convertor.regions.register(regions_half_squares)
    convertor.intervals.register(IntervalSet('months', months))
    convertor.intervals.register(IntervalSet('seasons', seasons))
    convertor.intervals.register(IntervalSet('hourly_day', twenty_four_hours))
    convertor.intervals.register(IntervalSet('one_day', one_day))
    convertor.intervals.register(IntervalSet('remap_months', remap_months))
    return convertor


class TestSpaceTimeUnitConvertor_TimeOnly:
    """No change in regions
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


class TestRemapConversion:
    """Repeating values
    """
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


class TestSpaceTimeUnitConvertor_RegionOnly:
    """No change in intervals
    """
    def test_half_to_one_region_pass_through_time(self, convertor):
        """Convert from half a region to one region, pass through time
        """
        data = np.ones((2, 12)) / 2  # area a,b, months 1-12
        actual = convertor.convert(
            data,
            'half_squares',
            'rect',
            'months',
            'months',
            'm',
            'm'
        )
        expected = np.ones((1, 12))  # area zero, months 1-12
        assert np.allclose(actual, expected)

    def test_one_region_convert_from_hour_to_day(self, convertor):
        """Two regions aggregated to one, one interval
        """

        data = np.array([[24], [24]])  # area a,b, single interval
        actual = convertor.convert(
            data,
            'half_squares',
            'rect',
            'one_day',
            'one_day',
            'm',
            'm'
        )
        expected = np.array([[48]])  # area zero, single interval
        assert np.allclose(actual, expected)


class TestSpaceTimeUnitConvertorBoth:
    """Change both regions and intervals
    """
    def test_convert_both_space_and_time(self, convertor):
        data = np.ones((2, 12)) / 2  # area a,b, months 1-12
        actual = convertor.convert(
            data,
            'half_squares',
            'rect',
            'months',
            'seasons',
            'm',
            'm'
        )
        expected = np.ones((1, 4)) * 3  # area zero, seasons 1-4
        assert np.allclose(actual, expected)
