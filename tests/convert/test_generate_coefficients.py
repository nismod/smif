"""Tests functionality of Register class that computes coefficients for
different operations
"""
from unittest.mock import Mock

import numpy as np
from pytest import fixture, mark
from smif.convert.area import RegionRegister
from smif.convert.interval import TimeIntervalRegister
from smif.convert.interval import get_register as get_interval_register
from smif.convert.register import Register


class TestPerformConversion:

    @mark.parametrize('space, time, expected', [
         # Space disaggregation only
         (np.array([[0.333, 0.333, 0.333]]),
             np.array([[1]]),
             np.array([[0.333], [0.333], [0.333]])
          ),
         # Time disaggregation only
         (np.array([[1.0]]),
             np.array([[0.333, 0.333, 0.333]]),
             np.array([[0.333, 0.333, 0.333]])
          ),
         # Space and time disaggregation
         (np.array([[0.333333, 0.333333, 0.333333]]),
             np.array([[0.333333, 0.333333, 0.333333]]),
             np.array([[0.111, 0.111, 0.111],
                       [0.111, 0.111, 0.111],
                       [0.111, 0.111, 0.111]]))
        ])
    def test_disaggregation_operation(self, space, time, expected):

        regions = RegionRegister()
        intervals = TimeIntervalRegister()

        data = np.array([[1]])

        regions.get_coefficients = Mock(return_value=space)
        intervals.get_coefficients = Mock(return_value=time)

        space = regions.convert(data,
                                'half_rect',
                                'half_rect')
        actual = intervals.convert(space,
                                   'seasons',
                                   'seasons')
        np.testing.assert_allclose(actual, expected, rtol=1e-1)

    @mark.parametrize('space, time, expected', [
        # Space aggregation only
        (np.array([[1], [1]]),
            np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]]),
            np.array([[666.666, 666.666, 666.666]]),
         ),
        # Time aggregation only
        (np.array([[1, 0], [0, 1]]),
            np.array([[1], [1], [1]]),
            np.array([[1000], [1000]])
         ),
        # Space and time aggregation
        (np.array([[1], [1]]),
            np.array([[1], [1], [1]]),
            np.array([[2000]]))
        ])
    def test_aggregation_operation(self, space, time, expected):

        # Two regions, three intervals
        data = np.array([[333.333, 333.333, 333.333],
                         [333.333, 333.333, 333.333]])

        regions = RegionRegister()
        intervals = TimeIntervalRegister()

        regions.get_coefficients = Mock(return_value=space)
        intervals.get_coefficients = Mock(return_value=time)

        space = regions.convert(data,
                                'half_rect',
                                'half_rect')
        actual = intervals.convert(space,
                                   'seasons',
                                   'seasons')

        np.testing.assert_allclose(actual, expected, rtol=1e-1)


class TestAggregationCoefficient:
    """Tests aggregation of data

    When the number of source regions or intervals is greater than the
    number of target regions or intervals and the total area or duration
    remains the same, then the data is aggregated
    """

    def test_aggregation(self, month_to_season_coefficients):

        intervals = get_interval_register()
        actual = intervals.get_coefficients('months', 'seasons')
        expected = month_to_season_coefficients
        np.testing.assert_allclose(actual, expected)


@fixture(scope='function')
def patch_register_data_interface():
    mocked_interface = Mock()
    Register.data_interface = mocked_interface
    yield mocked_interface

    Register.data_interface = None


class TestRegisterCaching:
    """Tests that coefficients are cached if a data handle is present
    """

    def test_read_coefficients(self, month_to_season_coefficients,
                               patch_register_data_interface):

        interval = get_interval_register()

        mock_interface = patch_register_data_interface
        mock_interface.read_coefficients = Mock()

        assert interval.data_interface == mock_interface

        interval.get_coefficients('months', 'seasons')

        interval.data_interface.read_coefficients.assert_called_once_with('months', 'seasons')

    def test_write_coefficients(self, month_to_season_coefficients,
                                patch_register_data_interface):
        """Checks that coefficients are
        """

        data = np.array(month_to_season_coefficients, dtype=np.float)

        interval = get_interval_register()

        mock_data_interface = patch_register_data_interface

        mock_data_interface.read_coefficients = Mock(return_value=None)
        mock_data_interface.write_coefficients = Mock()

        assert interval.data_interface == mock_data_interface

        interval.get_coefficients('months', 'seasons')

        actual = interval.data_interface.write_coefficients.call_args[0]
        expected = ('months', 'seasons', data)

        np.testing.assert_equal(actual, expected)
