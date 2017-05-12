import numpy as np
from pytest import approx, fixture
from smif.convert import SpaceTimeConvertor
from smif.convert.area import RegionRegister
from smif.convert.interval import TimeIntervalRegister
from test_area import regions_half_squares, regions_rect
from test_interval import (months, one_day, remap_months, seasons,
                           twenty_four_hours)


@fixture(scope='function')
def data_remap():
    data = np.array([
        [
            30+31+31,
            28+31+30,
            28+31+30,
            30+31+31,
        ]
    ])
    return data


@fixture(scope='function')
def expected_stv_remap():
    data = np.array([
        [
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
            30.666666666,
        ]
    ])
    return data


@fixture(scope='function')
def monthly_data():
    """[31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    """
    data = np.array([
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
        ]
    ])
    return data


class TestSpaceTimeConvertor_TimeOnly:

    def test_one_region_pass_through_time(self, months, seasons, regions_half_squares, monthly_data):
        """Only one region, 12 months, neither space nor time conversion is required
        """
        data = monthly_data

        intervals = TimeIntervalRegister()
        intervals.register(months, 'months')

        regions = RegionRegister()
        regions.register(regions_half_squares)

        convertor = SpaceTimeConvertor(regions, intervals)

        actual = convertor.convert(
            data,
            'half_squares',
            'half_squares',
            'months',
            'months'
        )
        assert actual == data

    def test_one_region_time_aggregation(self, months, seasons, regions_half_squares, monthly_data, data_remap):
        """Only one region, time aggregation is required
        """
        data = monthly_data  # area a, months 1-12
        expected = data_remap  # area a, seasons 1-4

        intervals = TimeIntervalRegister()
        intervals.register(months, 'months')
        intervals.register(seasons, 'seasons')

        regions = RegionRegister()
        regions.register(regions_half_squares)

        convertor = SpaceTimeConvertor(regions, intervals)

        actual = convertor.convert(
            data,
            'half_squares',
            'half_squares',
            'months',
            'seasons'
        )
        assert actual == expected

    def test_two_region_time_aggregation(self, months, seasons, regions_half_squares):
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
        ])


        intervals = TimeIntervalRegister()
        intervals.register(months, 'months')
        intervals.register(seasons, 'seasons')

        regions = RegionRegister()
        regions.register(regions_half_squares)

        convertor = SpaceTimeConvertor(regions, intervals)
        actual = convertor.convert(
            data,
            'half_squares',
            'half_squares',
            'months',
            'seasons'
        )

        expected = np.array([
            [
                31. + 31 + 28,
                31. + 30 + 31,
                30. + 31 + 31,
                30. + 31 + 30,
            ],
            [
                31. + 31 + 28 + 3,
                31. + 30 + 31 + 3,
                30. + 31 + 31 + 3,
                30. + 31 + 30 + 3,
            ]
        ])

        for act, exp in zip(actual, expected):
            assert act.region == exp.region
            assert act.interval == exp.interval
            assert act.value == approx(exp.value)
            assert act.units == exp.units

    def test_one_region_convert_from_hour_to_day(self, regions_half_squares,
                                                 twenty_four_hours, one_day):
        """One region, time aggregation required
        """

        data = np.ones((1, 24))  # area a, hours 0-23

        regions = RegionRegister()
        regions.register(regions_half_squares)

        intervals = TimeIntervalRegister()
        intervals.register(twenty_four_hours, 'hourly_day')
        intervals.register(one_day, 'one_day')

        convertor = SpaceTimeConvertor(regions, intervals)
        actual = convertor.convert(
            data,
            'half_squares',
            'half_squares',
            'hourly_day',
            'one_day'
        )
        expected = np.array([[24]])  # area a, day 0
        assert actual == expected

    def test_remap_timeslices_to_months(self,
                                        months,
                                        expected_stv_remap,
                                        remap_months,
                                        data_remap,
                                        regions_half_squares):
        """One region, time remapping required
        """
        intervals = TimeIntervalRegister()
        intervals.register(months, 'months')
        intervals.register(remap_months, 'remap_months')

        regions = RegionRegister()
        regions.register(regions_half_squares)

        convertor = SpaceTimeConvertor(regions, intervals)
        actual = convertor.convert(
            data_remap,
            'half_squares',
            'half_squares',
            'remap_months',
            'months',
        )
        expected = expected_stv_remap
        assert actual == expected


    def test_remap_months_to_timeslices(self,
                                        months,
                                        monthly_data,
                                        remap_months,
                                        data_remap,
                                        regions_half_squares):
        """One region, time remapping required
        """
        intervals = TimeIntervalRegister()
        intervals.register(months, 'months')
        intervals.register(remap_months, 'remap_months')

        regions = RegionRegister()
        regions.register(regions_half_squares)

        convertor = SpaceTimeConvertor(regions, intervals)
        actual = convertor.convert(
            monthly_data,
            'half_squares',
            'half_squares',
            'months',
            'remap_months'
        )
        expected = data_remap
        assert actual == expected


class TestSpaceTimeConvertor_RegionOnly:

    def test_half_to_one_region_pass_through_time(self, months,
                                                  regions_half_squares,
                                                  regions_rect):
        """Convert from half a region to one region, pass through time

        """

        data = np.ones((2, 12)) / 2  # area a,b, months 1-12

        intervals = TimeIntervalRegister()
        intervals.register(months, 'months')

        regions = RegionRegister()
        regions.register(regions_half_squares)
        regions.register(regions_rect)

        convertor = SpaceTimeConvertor(regions, intervals)

        actual = convertor.convert(
            data,
            'half_squares',
            'rect',
            'months',
            'months'
        )
        expected = np.ones((1, 12))  # area zero, months 1-12
        assert actual == expected

    def test_one_region_convert_from_hour_to_day(self, regions_half_squares,
                                                 regions_rect,
                                                 one_day):
        """Two regions aggregated to one, one interval
        """

        data = np.array([[24], [24]])  # area a,b, single interval

        intervals = TimeIntervalRegister()
        intervals.register(one_day, 'one_day')

        regions = RegionRegister()
        regions.register(regions_half_squares)
        regions.register(regions_rect)

        convertor = SpaceTimeConvertor(regions, intervals)
        actual = convertor.convert(
            data,
            'half_squares',
            'rect',
            'one_day',
            'one_day'
        )
        expected = np.array([[48]])  # area zero, single interval
        assert actual == expected


class TestSpaceTimeConvertorBoth:

    def test_convert_both_space_and_time(self, seasons, months,
                                         regions_half_squares,
                                         regions_rect):

        data = np.ones((2, 12)) / 2  # area a,b, months 1-12

        intervals = TimeIntervalRegister()
        intervals.register(months, 'months')
        intervals.register(seasons, 'seasons')

        regions = RegionRegister()
        regions.register(regions_half_squares)
        regions.register(regions_rect)

        convertor = SpaceTimeConvertor(regions, intervals)

        actual = convertor.convert(
            data,
            'half_squares',
            'rect',
            'months',
            'seasons'
        )
        expected = np.ones((1, 4)) * 3  # area zero, seasons 1-4
        assert actual == expected
