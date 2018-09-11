"""Tests functionality of NDimensionalRegister class that computes coefficients for
different operations
"""
import numpy as np
from pytest import mark
from smif.convert.adaptor import Adaptor


class TestPerformConversion:
    """Convert between dimension resolutions
    """
    @mark.parametrize('space, time, expected', [
        # Space disaggregation only
        (
            np.array([[0.333, 0.333, 0.333]]),
            np.array([[1.0]]),
            np.array([[0.333],
                      [0.333],
                      [0.333]])
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
        """Disggregations should split data along the transformed dimension
        """
        data = np.array([[1]])

        intermediate = Adaptor.convert_with_coefficients(data, space, 0)
        actual = Adaptor.convert_with_coefficients(intermediate, time, 1)
        np.testing.assert_allclose(actual, expected, rtol=1e-2)

    @mark.parametrize('space, time, expected', [
        # Space aggregation only
        (
            np.array([[1],
                      [1]]),
            np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]]),
            np.array([[666.666, 666.666, 666.666]]),
        ),
        # Time aggregation only
        (
            np.array([[1, 0],
                      [0, 1]]),
            np.array([[1],
                      [1],
                      [1]]),
            np.array([[1000],
                      [1000]])
        ),
        # Space and time aggregation
        (
            np.array([[1],
                      [1]]),
            np.array([[1],
                      [1],
                      [1]]),
            np.array([[2000]]))
    ])
    def test_aggregation_operation(self, space, time, expected):
        """Aggregations should collect data along the transformed dimension
        """
        # Two regions, three intervals
        data = np.array([[333.333, 333.333, 333.333],
                         [333.333, 333.333, 333.333]])
        intermediate = Adaptor.convert_with_coefficients(data, space, 0)
        actual = Adaptor.convert_with_coefficients(intermediate, time, 1)
        np.testing.assert_allclose(actual, expected, rtol=1e-2)

    def test_multidimensional_operation(self):
        """Operations over 3-dimensional data
        """
        # start with something (1, 2, 3)
        data = np.array(
            [[[0.0, 1.0, 2.0],
              [3.0, 4.0, 5.0]]]
        )

        # split 1st dim (2, 2, 3)
        coefficients = np.ones((1, 2)) / 2
        expected = np.array(
            [[[0.0, 0.5, 1.0],
              [1.5, 2.0, 2.5]],

             [[0.0, 0.5, 1.0],
              [1.5, 2.0, 2.5]]]
        )
        actual = Adaptor.convert_with_coefficients(data, coefficients, 0)
        np.testing.assert_allclose(actual, expected)

        # sum 3rd dim (2, 2, 1)
        coefficients = np.ones((3, 1))
        expected = np.array(
            [[[1.5],
              [6.0]],

             [[1.5],
              [6.0]]]
        )
        actual = Adaptor.convert_with_coefficients(actual, coefficients, 2)
        np.testing.assert_allclose(actual, expected)
