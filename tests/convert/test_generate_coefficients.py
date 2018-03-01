import numpy as np
from pytest import fixture, mark
from smif.convert import Convertor


@fixture(scope='module')
def year_to_month_coefficients():
    """From one year to 12 months

    (apportions)
    """

    month_lengths = np.array([[31, 28, 31, 30, 31, 31, 30, 30, 31, 31, 30, 31]],
                             dtype=np.float).T
    return month_lengths / 365


@fixture(scope='module')
def month_to_year_coefficients():
    """
    """
    return np.ones((1, 12), dtype=np.float)


@fixture(scope='module')
def month_to_season_coefficients():
    """
    12 months to four seasons (winter is December, January, Feb)

    Sum value for each month into season
    """
    coef = np.array(
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # winter
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # spring
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],  # summer
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0])  # autumn

    return coef


@fixture(scope='module')
def season_to_month_coefficients():
    """
    12 months to four seasons (winter is December, January, Feb)

    To convert from seasons to months, find the proportion of each season that
    corresponds to the relevant month.

    E.g. winter to january is (duration of Jan / total duration of winter)
    """
    coef = np.array(
        # winter
        #     spring
        #        summer
        #           autumn
        [[31, 0, 0, 0],  # January
         [28, 0, 0, 0],  # Feb
         [0, 31, 0, 0],  # March
         [0, 30, 0, 0],  # April
         [0, 31, 0, 0],  # May
         [0, 0, 30, 0],  # June
         [0, 0, 31, 0],  # July
         [0, 0, 31, 0],  # August
         [0, 0, 0, 30],  # September
         [0, 0, 0, 31],  # October
         [0, 0, 0, 30],  # November
         [31, 0, 0, 0]]   # December
    )

    days_in_seasons = np.array([
        31+31+28,  # winter
        31+30+31,  # spring
        30+31+31,  # summer
        30+31+30  # autumn
    ], dtype=float)

    return coef / days_in_seasons


class TestComputeCoefficients:

    def compute_duration(self):
        pass

    def test_compute_intersection(self):
        pass


class TestChooseOperation:

    def test_conversion(self):

        source = np.array([2190, 2190, 2190, 2190])
        destination = np.array([2920, 2920, 2920])

        expected = np.array(
            [[0.25, 1./12, 0, 0],
             [0, 2./12, 2./12, 0],
             [0, 0, 1./12, 0.25]], dtype=np.float)

        convertor = Convertor()
        actual = convertor.compute_intersection(source, destination)
        np.testing.assert_equal(actual, expected)


class TestPerformConversion:

    @mark.parametrize('space, time, expected', [
         # Space disaggregation only
         (np.array([[0.333, 0.333, 0.333]]),
             np.array([[1]]),
             np.array([[333], [333], [333]])
          ),
         # Time disaggregation only
         (np.array([[1.0]]),
             np.array([[0.333, 0.333, 0.333]]),
             np.array([[333, 333, 333]])
          ),
         # Space and time disaggregation
         (np.array([[0.333333, 0.333333, 0.333333]]),
             np.array([[0.333333, 0.333333, 0.333333]]),
             np.array([[111, 111, 111],
                       [111, 111, 111],
                       [111, 111, 111]]))
        ])
    def test_disaggregation_operation(self, space, time, expected):

        convertor = Convertor()

        data = np.array([[1]])
        unit_coefficients = 1000

        actual = convertor.perform_conversion(data,
                                              space,
                                              time,
                                              unit_coefficients)
        np.testing.assert_allclose(actual, expected, rtol=1e-1)

    @mark.parametrize('space, time, expected', [
        # Space aggregation only
        (np.array([[1], [1]]),
            np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]]),
            np.array([[0.666, 0.666, 0.666]]),
         ),
        # Time aggregation only
        (np.array([[1, 0], [0, 1]]),
            np.array([[1], [1], [1]]),
            np.array([[1], [1]])
         ),
        # Space and time aggregation
        (np.array([[1], [1]]),
            np.array([[1], [1], [1]]),
            np.array([[2]]))
        ])
    def test_aggregation_operation(self, space, time, expected):

        convertor = Convertor()

        # Two regions, three intervals
        data = np.array([[333.333, 333.333, 333.333],
                         [333.333, 333.333, 333.333]])
        unit_coefficients = 1e-3

        actual = convertor.perform_conversion(data,
                                              space,
                                              time,
                                              unit_coefficients)
        np.testing.assert_allclose(actual, expected, rtol=1e-1)


class TestAggregation:
    """Tests aggregation of data

    When the number of source regions or intervals is greater than the
    number of target regions or intervals and the total area or duration
    remains the same, then the data is aggregated
    """

    def test_aggregation(self):

        pass
