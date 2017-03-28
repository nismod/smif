from pytest import approx, fixture, raises
from smif import SpaceTimeValue
from smif.convert import SpaceTimeConvertor
from smif.convert.area import RegionRegister
from smif.convert.interval import TimeIntervalRegister
from test_area import regions_half_squares, regions_rect
from test_interval import (months, one_day, remap_months, seasons,
                           twenty_four_hours)


@fixture(scope='function')
def data_remap():
    data = [SpaceTimeValue('a', '1', 30+31+31, 'days'),
            SpaceTimeValue('a', '2', 28+31+30, 'days'),
            SpaceTimeValue('a', '3', 31+31+30, 'days'),
            SpaceTimeValue('a', '4', 30+31+31, 'days')]
    return data


@fixture(scope='function')
def expected_stv_remap():
    data = [SpaceTimeValue('a', '1_0', 30.666666666, 'days'),
            SpaceTimeValue('a', '1_1', 29.666666666, 'days'),
            SpaceTimeValue('a', '1_2', 29.666666666, 'days'),
            SpaceTimeValue('a', '1_3', 29.666666666, 'days'),
            SpaceTimeValue('a', '1_4', 30.666666666, 'days'),
            SpaceTimeValue('a', '1_5', 30.666666666, 'days'),
            SpaceTimeValue('a', '1_6', 30.666666666, 'days'),
            SpaceTimeValue('a', '1_7', 30.666666666, 'days'),
            SpaceTimeValue('a', '1_8', 30.666666666, 'days'),
            SpaceTimeValue('a', '1_9', 30.666666666, 'days'),
            SpaceTimeValue('a', '1_10', 30.666666666, 'days'),
            SpaceTimeValue('a', '1_11', 30.666666666, 'days')]
    return data


@fixture(scope='function')
def monthly_data():
    """[31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    """
    data = [SpaceTimeValue('a', '1_0', 31, 'days'),
            SpaceTimeValue('a', '1_1', 28, 'days'),
            SpaceTimeValue('a', '1_2', 31, 'days'),
            SpaceTimeValue('a', '1_3', 30, 'days'),
            SpaceTimeValue('a', '1_4', 31, 'days'),
            SpaceTimeValue('a', '1_5', 30, 'days'),
            SpaceTimeValue('a', '1_6', 31, 'days'),
            SpaceTimeValue('a', '1_7', 31, 'days'),
            SpaceTimeValue('a', '1_8', 30, 'days'),
            SpaceTimeValue('a', '1_9', 31, 'days'),
            SpaceTimeValue('a', '1_10', 30, 'days'),
            SpaceTimeValue('a', '1_11', 31, 'days')]
    return data


class TestSpaceTimeConvertor_Utils:

    def test_instantiation(self):
        SpaceTimeConvertor([], None, None, None, None, None, None)

    def test_conversion_not_required(self):
        convertor = SpaceTimeConvertor([], 'a', 'a', 'x', 'x', None, None)
        assert convertor._convert_intervals_required() is False
        assert convertor._convert_regions_required() is False

    def test_conversion_is_required(self):
        convertor = SpaceTimeConvertor([], 'a', 'b', 'x', 'y', None, None)
        assert convertor._convert_intervals_required() is True
        assert convertor._convert_regions_required() is True

    def test_multiple_units_raises_notimplemented(self):
        data = [SpaceTimeValue('a', '1_0', 31, 'days'),
                SpaceTimeValue('a', '1_2', 31, 'minutes'),
                SpaceTimeValue('b', '1_1', 28, 'seconds')]
        with raises(NotImplementedError):
            SpaceTimeConvertor(data, 'a', 'a', 'x', 'x', None, None)

    def test_data_by_regions(self):
        data = [SpaceTimeValue('a', '1_0', 31, 'days'),
                SpaceTimeValue('a', '1_2', 31, 'days'),
                SpaceTimeValue('b', '1_1', 28, 'days')]
        convertor = SpaceTimeConvertor(data, 'a', 'a', 'x', 'x', None, None)

        expected = {'a': [SpaceTimeValue('a', '1_0', 31, 'days'),
                          SpaceTimeValue('a', '1_2', 31, 'days')],
                    'b': [SpaceTimeValue('b', '1_1', 28, 'days')]}

        assert convertor.data_by_region == expected


class TestSpaceTimeConvertor_TimeOnly:

    def test_one_region_pass_through_time(self, months, seasons, regions_half_squares):
        """Only one region, 12 months, neither space nor time conversion is required

        """

        data = [SpaceTimeValue('a', '1_0', 31, 'days'),
                SpaceTimeValue('a', '1_1', 28, 'days'),
                SpaceTimeValue('a', '1_2', 31, 'days'),
                SpaceTimeValue('a', '1_3', 30, 'days'),
                SpaceTimeValue('a', '1_4', 31, 'days'),
                SpaceTimeValue('a', '1_5', 30, 'days'),
                SpaceTimeValue('a', '1_6', 31, 'days'),
                SpaceTimeValue('a', '1_7', 31, 'days'),
                SpaceTimeValue('a', '1_8', 30, 'days'),
                SpaceTimeValue('a', '1_9', 31, 'days'),
                SpaceTimeValue('a', '1_10', 30, 'days'),
                SpaceTimeValue('a', '1_11', 31, 'days')]

        intervals = TimeIntervalRegister()
        intervals.register(months, 'months')

        regions = RegionRegister()
        regions.register(regions_half_squares)

        convertor = SpaceTimeConvertor(data,
                                       'half_squares',
                                       'half_squares',
                                       'months',
                                       'months',
                                       regions,
                                       intervals)

        assert convertor.data_regions == set(['a'])
        assert convertor.data_by_region == {'a': data}

        actual = convertor.convert()

        expected = [SpaceTimeValue('a', '1_0', 31, 'days'),
                    SpaceTimeValue('a', '1_1', 28, 'days'),
                    SpaceTimeValue('a', '1_2', 31, 'days'),
                    SpaceTimeValue('a', '1_3', 30, 'days'),
                    SpaceTimeValue('a', '1_4', 31, 'days'),
                    SpaceTimeValue('a', '1_5', 30, 'days'),
                    SpaceTimeValue('a', '1_6', 31, 'days'),
                    SpaceTimeValue('a', '1_7', 31, 'days'),
                    SpaceTimeValue('a', '1_8', 30, 'days'),
                    SpaceTimeValue('a', '1_9', 31, 'days'),
                    SpaceTimeValue('a', '1_10', 30, 'days'),
                    SpaceTimeValue('a', '1_11', 31, 'days')]

        assert actual == expected

    def test_one_region_time_aggregation(self, months, seasons, regions_half_squares):
        """Only one region, time aggregation is required
        """

        data = [SpaceTimeValue('a', '1_0', 31, 'days'),
                SpaceTimeValue('a', '1_1', 28, 'days'),
                SpaceTimeValue('a', '1_2', 31, 'days'),
                SpaceTimeValue('a', '1_3', 30, 'days'),
                SpaceTimeValue('a', '1_4', 31, 'days'),
                SpaceTimeValue('a', '1_5', 30, 'days'),
                SpaceTimeValue('a', '1_6', 31, 'days'),
                SpaceTimeValue('a', '1_7', 31, 'days'),
                SpaceTimeValue('a', '1_8', 30, 'days'),
                SpaceTimeValue('a', '1_9', 31, 'days'),
                SpaceTimeValue('a', '1_10', 30, 'days'),
                SpaceTimeValue('a', '1_11', 31, 'days')]

        intervals = TimeIntervalRegister()
        intervals.register(months, 'months')
        intervals.register(seasons, 'seasons')

        regions = RegionRegister()
        regions.register(regions_half_squares)

        convertor = SpaceTimeConvertor(data,
                                       'half_squares',
                                       'half_squares',
                                       'months',
                                       'seasons',
                                       regions,
                                       intervals)
        assert convertor.data_regions == set(['a'])
        assert convertor.data_by_region == {'a': data}

        actual = convertor.convert()

        expected = [SpaceTimeValue('a', 'winter', 31. + 31 + 28, 'days'),
                    SpaceTimeValue('a', 'spring', 31. + 30 + 31, 'days'),
                    SpaceTimeValue('a', 'summer', 30. + 31 + 31, 'days'),
                    SpaceTimeValue('a', 'autumn', 30. + 31 + 30, 'days')]
        assert isinstance(actual, list)
        for entry in actual:
            assert isinstance(entry, SpaceTimeValue)

        for act, exp in zip(actual, expected):
            assert act.region == exp.region
            assert act.interval == exp.interval
            assert act.value == approx(exp.value)
            assert act.units == exp.units

    def test_two_region_time_aggregation(self, months, seasons, regions_half_squares):
        """Two regions, time aggregation by region is required
        """

        data = [SpaceTimeValue('a', '1_0', 31, 'days'),
                SpaceTimeValue('a', '1_1', 28, 'days'),
                SpaceTimeValue('a', '1_2', 31, 'days'),
                SpaceTimeValue('a', '1_3', 30, 'days'),
                SpaceTimeValue('a', '1_4', 31, 'days'),
                SpaceTimeValue('a', '1_5', 30, 'days'),
                SpaceTimeValue('a', '1_6', 31, 'days'),
                SpaceTimeValue('a', '1_7', 31, 'days'),
                SpaceTimeValue('a', '1_8', 30, 'days'),
                SpaceTimeValue('a', '1_9', 31, 'days'),
                SpaceTimeValue('a', '1_10', 30, 'days'),
                SpaceTimeValue('a', '1_11', 31, 'days'),
                SpaceTimeValue('b', '1_0', 31+1, 'days'),
                SpaceTimeValue('b', '1_1', 28+1, 'days'),
                SpaceTimeValue('b', '1_2', 31+1, 'days'),
                SpaceTimeValue('b', '1_3', 30+1, 'days'),
                SpaceTimeValue('b', '1_4', 31+1, 'days'),
                SpaceTimeValue('b', '1_5', 30+1, 'days'),
                SpaceTimeValue('b', '1_6', 31+1, 'days'),
                SpaceTimeValue('b', '1_7', 31+1, 'days'),
                SpaceTimeValue('b', '1_8', 30+1, 'days'),
                SpaceTimeValue('b', '1_9', 31+1, 'days'),
                SpaceTimeValue('b', '1_10', 30+1, 'days'),
                SpaceTimeValue('b', '1_11', 31+1, 'days')]

        expected_a = [SpaceTimeValue('a', '1_0', 31, 'days'),
                      SpaceTimeValue('a', '1_1', 28, 'days'),
                      SpaceTimeValue('a', '1_2', 31, 'days'),
                      SpaceTimeValue('a', '1_3', 30, 'days'),
                      SpaceTimeValue('a', '1_4', 31, 'days'),
                      SpaceTimeValue('a', '1_5', 30, 'days'),
                      SpaceTimeValue('a', '1_6', 31, 'days'),
                      SpaceTimeValue('a', '1_7', 31, 'days'),
                      SpaceTimeValue('a', '1_8', 30, 'days'),
                      SpaceTimeValue('a', '1_9', 31, 'days'),
                      SpaceTimeValue('a', '1_10', 30, 'days'),
                      SpaceTimeValue('a', '1_11', 31, 'days')]

        expected_b = [SpaceTimeValue('b', '1_0', 31+1, 'days'),
                      SpaceTimeValue('b', '1_1', 28+1, 'days'),
                      SpaceTimeValue('b', '1_2', 31+1, 'days'),
                      SpaceTimeValue('b', '1_3', 30+1, 'days'),
                      SpaceTimeValue('b', '1_4', 31+1, 'days'),
                      SpaceTimeValue('b', '1_5', 30+1, 'days'),
                      SpaceTimeValue('b', '1_6', 31+1, 'days'),
                      SpaceTimeValue('b', '1_7', 31+1, 'days'),
                      SpaceTimeValue('b', '1_8', 30+1, 'days'),
                      SpaceTimeValue('b', '1_9', 31+1, 'days'),
                      SpaceTimeValue('b', '1_10', 30+1, 'days'),
                      SpaceTimeValue('b', '1_11', 31+1, 'days')]

        intervals = TimeIntervalRegister()
        intervals.register(months, 'months')
        intervals.register(seasons, 'seasons')

        regions = RegionRegister()
        regions.register(regions_half_squares)

        convertor = SpaceTimeConvertor(data,
                                       'half_squares',
                                       'half_squares',
                                       'months',
                                       'seasons',
                                       regions,
                                       intervals)
        assert convertor.data_regions == set(['a', 'b'])
        assert convertor.data_by_region['a'] == expected_a
        assert convertor.data_by_region['b'] == expected_b

        actual = convertor.convert()

        expected = [SpaceTimeValue('a', 'winter', 31. + 31 + 28, 'days'),
                    SpaceTimeValue('a', 'spring', 31. + 30 + 31, 'days'),
                    SpaceTimeValue('a', 'summer', 30. + 31 + 31, 'days'),
                    SpaceTimeValue('a', 'autumn', 30. + 31 + 30, 'days'),
                    SpaceTimeValue('b', 'winter', 31. + 31 + 28 + 3, 'days'),
                    SpaceTimeValue('b', 'spring', 31. + 30 + 31 + 3, 'days'),
                    SpaceTimeValue('b', 'summer', 30. + 31 + 31 + 3, 'days'),
                    SpaceTimeValue('b', 'autumn', 30. + 31 + 30 + 3, 'days')]

        for act, exp in zip(actual, expected):
            assert act.region == exp.region
            assert act.interval == exp.interval
            assert act.value == approx(exp.value)
            assert act.units == exp.units

    def test_one_region_convert_from_hour_to_day(self, regions_half_squares,
                                                 twenty_four_hours, one_day):
        """One region, time aggregation required
        """

        data = [SpaceTimeValue('a', '1_0', 1, 'days'),
                SpaceTimeValue('a', '1_1', 1, 'days'),
                SpaceTimeValue('a', '1_2', 1, 'days'),
                SpaceTimeValue('a', '1_3', 1, 'days'),
                SpaceTimeValue('a', '1_4', 1, 'days'),
                SpaceTimeValue('a', '1_5', 1, 'days'),
                SpaceTimeValue('a', '1_6', 1, 'days'),
                SpaceTimeValue('a', '1_7', 1, 'days'),
                SpaceTimeValue('a', '1_8', 1, 'days'),
                SpaceTimeValue('a', '1_9', 1, 'days'),
                SpaceTimeValue('a', '1_10', 1, 'days'),
                SpaceTimeValue('a', '1_11', 1, 'days'),
                SpaceTimeValue('a', '1_12', 1, 'days'),
                SpaceTimeValue('a', '1_13', 1, 'days'),
                SpaceTimeValue('a', '1_14', 1, 'days'),
                SpaceTimeValue('a', '1_15', 1, 'days'),
                SpaceTimeValue('a', '1_16', 1, 'days'),
                SpaceTimeValue('a', '1_17', 1, 'days'),
                SpaceTimeValue('a', '1_18', 1, 'days'),
                SpaceTimeValue('a', '1_19', 1, 'days'),
                SpaceTimeValue('a', '1_20', 1, 'days'),
                SpaceTimeValue('a', '1_21', 1, 'days'),
                SpaceTimeValue('a', '1_22', 1, 'days'),
                SpaceTimeValue('a', '1_23', 1, 'days')]

        regions = RegionRegister()
        regions.register(regions_half_squares)

        intervals = TimeIntervalRegister()
        intervals.register(twenty_four_hours, 'hourly_day')
        intervals.register(one_day, 'one_day')

        convertor = SpaceTimeConvertor(data,
                                       'half_squares',
                                       'half_squares',
                                       'hourly_day',
                                       'one_day',
                                       regions,
                                       intervals)

        assert convertor.data_regions == set(['a'])
        assert convertor.data_by_region['a'] == data

        actual = convertor.convert()

        expected = [SpaceTimeValue('a', 'one_day', 24, 'days')]

        assert actual == expected

    def test_remap_timeslices_to_months(self,
                                        months,
                                        expected_stv_remap,
                                        remap_months,
                                        data_remap,
                                        regions_half_squares):
        """One region, time remapping required
        """
        timeslice_data = data_remap

        intervals = TimeIntervalRegister()
        intervals.register(months, 'months')
        intervals.register(remap_months, 'remap_months')

        regions = RegionRegister()
        regions.register(regions_half_squares)

        convertor = SpaceTimeConvertor(timeslice_data,
                                       'half_squares',
                                       'half_squares',
                                       'remap_months',
                                       'months',
                                       regions,
                                       intervals)

        assert convertor.data_regions == set(['a'])
        assert convertor.data_by_region['a'] == timeslice_data

        actual = convertor.convert()
        expected = expected_stv_remap

        assert len(actual) == len(expected)

        for act, exp in zip(actual, expected):
            assert act.region == exp.region
            assert act.interval == exp.interval
            assert act.value == approx(exp.value)
            assert act.units == exp.units

    def test_remap_months_to_timeslices(self,
                                        months,
                                        monthly_data,
                                        remap_months,
                                        data_remap,
                                        regions_half_squares):
        """One region, time remapping required
        """
        timeslice_data = monthly_data

        intervals = TimeIntervalRegister()
        intervals.register(months, 'months')
        intervals.register(remap_months, 'remap_months')

        regions = RegionRegister()
        regions.register(regions_half_squares)

        convertor = SpaceTimeConvertor(timeslice_data,
                                       'half_squares',
                                       'half_squares',
                                       'months',
                                       'remap_months',
                                       regions,
                                       intervals)

        actual = convertor.convert()
        expected = data_remap

        assert len(actual) == len(expected)

        for act, exp in zip(actual, expected):
            assert act.region == exp.region
            assert act.interval == exp.interval
            assert act.value == approx(exp.value)
            assert act.units == exp.units


class TestSpaceTimeConvertor_RegionOnly:

    def test_half_to_one_region_pass_through_time(self, months,
                                                  regions_half_squares,
                                                  regions_rect):
        """Convert from half a region to one region, pass through time

        """

        data = [SpaceTimeValue('a', '1_0', 0.5, 'days'),
                SpaceTimeValue('a', '1_1', 0.5, 'days'),
                SpaceTimeValue('a', '1_2', 0.5, 'days'),
                SpaceTimeValue('a', '1_3', 0.5, 'days'),
                SpaceTimeValue('a', '1_4', 0.5, 'days'),
                SpaceTimeValue('a', '1_5', 0.5, 'days'),
                SpaceTimeValue('a', '1_6', 0.5, 'days'),
                SpaceTimeValue('a', '1_7', 0.5, 'days'),
                SpaceTimeValue('a', '1_8', 0.5, 'days'),
                SpaceTimeValue('a', '1_9', 0.5, 'days'),
                SpaceTimeValue('a', '1_10', 0.5, 'days'),
                SpaceTimeValue('a', '1_11', 0.5, 'days'),
                SpaceTimeValue('b', '1_0', 0.5, 'days'),
                SpaceTimeValue('b', '1_1', 0.5, 'days'),
                SpaceTimeValue('b', '1_2', 0.5, 'days'),
                SpaceTimeValue('b', '1_3', 0.5, 'days'),
                SpaceTimeValue('b', '1_4', 0.5, 'days'),
                SpaceTimeValue('b', '1_5', 0.5, 'days'),
                SpaceTimeValue('b', '1_6', 0.5, 'days'),
                SpaceTimeValue('b', '1_7', 0.5, 'days'),
                SpaceTimeValue('b', '1_8', 0.5, 'days'),
                SpaceTimeValue('b', '1_9', 0.5, 'days'),
                SpaceTimeValue('b', '1_10', 0.5, 'days'),
                SpaceTimeValue('b', '1_11', 0.5, 'days')]

        intervals = TimeIntervalRegister()
        intervals.register(months, 'months')

        regions = RegionRegister()
        regions.register(regions_half_squares)
        regions.register(regions_rect)

        convertor = SpaceTimeConvertor(data,
                                       'half_squares',
                                       'rect',
                                       'months',
                                       'months',
                                       regions,
                                       intervals)

        actual = convertor.convert()

        expected = [SpaceTimeValue('zero', '1_0', 1, 'days'),
                    SpaceTimeValue('zero', '1_1', 1, 'days'),
                    SpaceTimeValue('zero', '1_2', 1, 'days'),
                    SpaceTimeValue('zero', '1_3', 1, 'days'),
                    SpaceTimeValue('zero', '1_4', 1, 'days'),
                    SpaceTimeValue('zero', '1_5', 1, 'days'),
                    SpaceTimeValue('zero', '1_6', 1, 'days'),
                    SpaceTimeValue('zero', '1_7', 1, 'days'),
                    SpaceTimeValue('zero', '1_8', 1, 'days'),
                    SpaceTimeValue('zero', '1_9', 1, 'days'),
                    SpaceTimeValue('zero', '1_10', 1, 'days'),
                    SpaceTimeValue('zero', '1_11', 1, 'days')]

        for act, exp in zip(actual, expected):
            print("Actual: {}\nExpected: {}".format(actual, expected))
            assert act.region == exp.region
            assert act.interval == exp.interval
            assert act.value == approx(exp.value)
            assert act.units == exp.units

    def test_one_region_convert_from_hour_to_day(self, regions_half_squares,
                                                 regions_rect,
                                                 one_day):
        """Two regions aggregated to one, one interval
        """

        data = [SpaceTimeValue('a', 'one_day', 24, 'days'),
                SpaceTimeValue('b', 'one_day', 24, 'days')]

        intervals = TimeIntervalRegister()
        intervals.register(one_day, 'one_day')

        regions = RegionRegister()
        regions.register(regions_half_squares)
        regions.register(regions_rect)

        convertor = SpaceTimeConvertor(data,
                                       'half_squares',
                                       'rect',
                                       'one_day',
                                       'one_day',
                                       regions,
                                       intervals)

        actual = convertor.convert()

        expected = [SpaceTimeValue('zero', 'one_day', 48, 'days')]

        assert actual == expected


class TestSpaceTimeConvertorBoth:

    def test_convert_both_space_and_time(self, seasons, months,
                                         regions_half_squares,
                                         regions_rect):

        data = [SpaceTimeValue('a', '1_0', 0.5, 'days'),
                SpaceTimeValue('a', '1_1', 0.5, 'days'),
                SpaceTimeValue('a', '1_2', 0.5, 'days'),
                SpaceTimeValue('a', '1_3', 0.5, 'days'),
                SpaceTimeValue('a', '1_4', 0.5, 'days'),
                SpaceTimeValue('a', '1_5', 0.5, 'days'),
                SpaceTimeValue('a', '1_6', 0.5, 'days'),
                SpaceTimeValue('a', '1_7', 0.5, 'days'),
                SpaceTimeValue('a', '1_8', 0.5, 'days'),
                SpaceTimeValue('a', '1_9', 0.5, 'days'),
                SpaceTimeValue('a', '1_10', 0.5, 'days'),
                SpaceTimeValue('a', '1_11', 0.5, 'days'),
                SpaceTimeValue('b', '1_0', 0.5, 'days'),
                SpaceTimeValue('b', '1_1', 0.5, 'days'),
                SpaceTimeValue('b', '1_2', 0.5, 'days'),
                SpaceTimeValue('b', '1_3', 0.5, 'days'),
                SpaceTimeValue('b', '1_4', 0.5, 'days'),
                SpaceTimeValue('b', '1_5', 0.5, 'days'),
                SpaceTimeValue('b', '1_6', 0.5, 'days'),
                SpaceTimeValue('b', '1_7', 0.5, 'days'),
                SpaceTimeValue('b', '1_8', 0.5, 'days'),
                SpaceTimeValue('b', '1_9', 0.5, 'days'),
                SpaceTimeValue('b', '1_10', 0.5, 'days'),
                SpaceTimeValue('b', '1_11', 0.5, 'days')]

        intervals = TimeIntervalRegister()
        intervals.register(months, 'months')
        intervals.register(seasons, 'seasons')

        regions = RegionRegister()
        regions.register(regions_half_squares)
        regions.register(regions_rect)

        convertor = SpaceTimeConvertor(data,
                                       'half_squares',
                                       'rect',
                                       'months',
                                       'seasons',
                                       regions,
                                       intervals)

        actual = convertor.convert()

        expected = [SpaceTimeValue('zero', 'winter', 3, 'days'),
                    SpaceTimeValue('zero', 'spring', 3, 'days'),
                    SpaceTimeValue('zero', 'summer', 3, 'days'),
                    SpaceTimeValue('zero', 'autumn', 3, 'days')]

        for act, exp in zip(actual, expected):
            print("Actual: {}\nExpected: {}".format(actual, expected))
            assert act.region == exp.region
            assert act.interval == exp.interval
            assert act.value == approx(exp.value)
            assert act.units == exp.units
