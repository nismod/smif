"""Test unit adaptor
"""
from unittest.mock import Mock

import numpy as np
from smif.convert.unit import UnitAdaptor
from smif.data_layer.data_array import DataArray
from smif.metadata import Spec


def test_convert_unit():
    """Convert SI units
    """
    data_handle = Mock()
    data = np.array([1], dtype=float)

    from_spec = Spec(
        name='test_variable',
        dtype='float',
        unit='liter'
    )

    data_array = DataArray(from_spec, data)

    data_handle.get_data = Mock(return_value=data_array)

    adaptor = UnitAdaptor('test-ml-l')
    adaptor.add_input(from_spec)
    adaptor.add_output(Spec(
        name='test_variable',
        dtype='float',
        unit='milliliter'
    ))
    adaptor.simulate(data_handle)

    actual = data_handle.set_results.call_args[0][1]
    expected = np.array([1000], dtype=float)
    np.testing.assert_allclose(actual, expected)


def test_convert_custom():
    """Convert custom units
    """
    data_handle = Mock()
    data = np.array([0.18346346], dtype=float)

    from_spec = Spec(
        name='test_variable',
        dtype='float',
        unit='mcm'
    )

    to_spec = Spec(
        name='test_variable',
        dtype='float',
        unit='GW'
    )

    data_array = DataArray(from_spec, data)

    data_handle.get_data = Mock(return_value=data_array)
    data_handle.read_unit_definitions = Mock(return_value=['mcm = 10.901353 * GW'])

    adaptor = UnitAdaptor('test-mcm-GW')
    adaptor.add_input(from_spec)
    adaptor.add_output(to_spec)
    adaptor.before_model_run(data_handle)  # must have run before_model_run to register units
    adaptor.simulate(data_handle)

    actual = data_handle.set_results.call_args[0][1]
    expected = np.array([2], dtype=float)
    np.testing.assert_allclose(actual, expected)
