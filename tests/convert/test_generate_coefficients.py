import numpy as np
from pytest import mark
from smif.convert import Convertor
from smif.convert.interval import IntervalSet, TimeIntervalRegister


class TestComputeCoefficients:

    def compute_duration(self):
        pass

    def test_compute_intersection(self):
        pass


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
        data = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        actual = register.convert(data, 'months', 'seasons')
        expected = np.array([3, 3, 3, 3])
        np.testing.assert_array_equal(actual, expected)

    def test_time_only_conversion_disagg(self, months, seasons):

        register = TimeIntervalRegister()
        register.register(IntervalSet('months', months))
        register.register(IntervalSet('seasons', seasons))
        data = np.array([3, 3, 3, 3])
        actual = register.convert(data, 'seasons', 'months')
        expected = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        np.testing.assert_array_equal(actual, expected)


class TestAggregation:
    """Tests aggregation of data

    When the number of source regions or intervals is greater than the
    number of target regions or intervals and the total area or duration
    remains the same, then the data is aggregated
    """

    def test_aggregation(self, month_to_season_coefficients):

        convertor = Convertor()
        actual = convertor._convertor.intervals.get_coefficients('months', 'seasons')
        expected = month_to_season_coefficients
        np.testing.assert_allclose(actual, expected)
