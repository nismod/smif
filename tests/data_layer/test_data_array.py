"""Test DataArray
"""
import numpy
import numpy.testing
# import pandas
# import xarray
from pytest import fixture
from smif.data_layer.data_array import DataArray
from smif.metadata import Spec


@fixture
def spec():
    return Spec(
        name='test_data',
        dims=['a', 'b', 'c'],
        coords={
            'a': ['a1', 'a2'],
            'b': ['b1', 'b2', 'b3'],
            'c': ['c1', 'c2', 'c3', 'c4'],
        },
        dtype='float',
        default=0,
        abs_range=(0, 1),
        exp_range=(0, 0.5)
    )


@fixture
def data():
    return numpy.arange(24, dtype='float').reshape((2, 3, 4))


@fixture
def small_da(spec, data):
    return DataArray(spec, data)


class TestDataArray():
    def test_init(self, spec, data):
        """Should initialise from spec and ndarray of data
        """
        da = DataArray(spec, data)
        numpy.testing.assert_equal(da.data, data)
        assert spec == da.spec

    def test_from_spec(self, spec):
        """Should create default from spec.default
        """
        da = DataArray.default_from_spec(spec)
        numpy.testing.assert_equal(da.data, numpy.zeros((2, 3, 4)))

    def test_as_df(self, small_da):
        small_da.as_df()

    def test_as_xarray(self, small_da):
        small_da.as_xarray()
