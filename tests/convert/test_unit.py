"""Test unit adaptor
"""
from unittest.mock import Mock

import numpy as np
from smif.convert.unit import UnitAdaptor
from smif.metadata import Spec


def test_convert_unit():
    """Convert SI units
    """
    data_handle = Mock()
    input_data = np.array([[1, 2], [3, 4]], dtype=float)
    data_handle.get_data = Mock(return_value=input_data)

    adaptor = UnitAdaptor('test-ml-l')
    adaptor.add_input(Spec(
        name='test_variable',
        dtype='float',
        unit='liter'
    ))
    adaptor.add_output(Spec(
        name='test_variable',
        dtype='float',
        unit='milliliter'
    ))
    adaptor.simulate(data_handle)

    actual = data_handle.set_results.call_args[0][1]
    expected = np.array([[1000, 2000], [3000, 4000]], dtype=float)
    np.testing.assert_allclose(actual, expected)
