"""Tests functionality of NDimensionalRegister class that computes coefficients for
different operations
"""
from unittest.mock import Mock

import numpy as np
from pytest import fixture, mark
from smif.convert.interval import IntervalSet
from smif.convert.register import NDimensionalRegister


@fixture(scope='function')
def mock_interface():
    mocked_interface = Mock()
    NDimensionalRegister.data_interface = mocked_interface
    yield mocked_interface
    NDimensionalRegister.data_interface = None


@fixture(scope='function')
def register(months, seasons):
    register = NDimensionalRegister()
    register.data_interface = Mock()
    register.data_interface.read_coefficients = Mock(return_value=None)
    register.data_interface.write_coefficients = Mock()
    months = IntervalSet('months', months)
    register.register(months)
    seasons = IntervalSet('seasons', seasons)
    register.register(seasons)
    return register


class TestPerformConversion:
    """Convert between dimension resolutions
    """
    @mark.parametrize('space, time, expected', [
        # Space disaggregation only
        (
            np.array([[0.333, 0.333, 0.333]]),
            np.array([[1.0]]),
            np.array([[0.333], [0.333], [0.333]])
        ),
        # Time disaggregation only
        (
            np.array([[1.0]]),
            np.array([[0.333, 0.333, 0.333]]),
            np.array([[0.333, 0.333, 0.333]])
        ),
        # Space and time disaggregation
        (
            np.array([[0.333333, 0.333333, 0.333333]]),
            np.array([[0.333333, 0.333333, 0.333333]]),
            np.array([[0.111, 0.111, 0.111],
                     [0.111, 0.111, 0.111],
                     [0.111, 0.111, 0.111]])
        )
    ])
    def test_disaggregation_operation(self, space, time, expected):
        data = np.array([[1]])

        intermediate = NDimensionalRegister.convert_with_coefficients(data, space, 0)
        actual = NDimensionalRegister.convert_with_coefficients(intermediate, time, 1)
        np.testing.assert_allclose(actual, expected, rtol=1e-2)

    @mark.parametrize('space, time, expected', [
        # Space aggregation only
        (
            np.array([[1], [1]]),
            np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]]),
            np.array([[666.666, 666.666, 666.666]]),
         ),
        # Time aggregation only
        (
            np.array([[1, 0], [0, 1]]),
            np.array([[1], [1], [1]]),
            np.array([[1000], [1000]])
         ),
        # Space and time aggregation
        (
            np.array([[1], [1]]),
            np.array([[1], [1], [1]]),
            np.array([[2000]]))
    ])
    def test_aggregation_operation(self, space, time, expected):

        # Two regions, three intervals
        data = np.array([[333.333, 333.333, 333.333],
                         [333.333, 333.333, 333.333]])

        regions = NDimensionalRegister()
        regions.axis = 0
        intervals = NDimensionalRegister()
        intervals.axis = 1

        regions.get_coefficients = Mock(return_value=space)
        intervals.get_coefficients = Mock(return_value=time)
        print(data)
        print(space)
        intermediate = regions.convert(data, 'half_rect', 'half_rect')
        print(intermediate)
        print(time)
        actual = intervals.convert(intermediate, 'seasons', 'seasons')

        np.testing.assert_allclose(actual, expected, rtol=1e-2)


class TestAggregationCoefficient:
    """Tests aggregation of data

    When the number of source regions or intervals is greater than the
    number of target regions or intervals and the total area or duration
    remains the same, then the data is aggregated
    """

    def test_aggregation(self, month_to_season_coefficients, register):
        """Get coefficients
        """
        actual = register.get_coefficients('months', 'seasons')
        expected = month_to_season_coefficients
        np.testing.assert_allclose(actual, expected)


class TestNDimensionalRegisterCaching:
    """Tests that coefficients are cached if a data handle is present
    """
    def test_read_coefficients(self, register):
        """Reading will check cache
        """
        register.get_coefficients('months', 'seasons')
        register.data_interface.read_coefficients.assert_called_once_with('months', 'seasons')

    def test_write_coefficients(self, month_to_season_coefficients, register):
        """Reading will update cache if empty
        """
        data = np.array(month_to_season_coefficients, dtype=np.float)
        register.get_coefficients('months', 'seasons')

        actual = register.data_interface.write_coefficients.call_args[0]
        expected = ('months', 'seasons', data)
        np.testing.assert_equal(actual, expected)
