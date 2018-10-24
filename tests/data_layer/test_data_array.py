"""Test DataArray
"""
import numpy
import pandas as pd
import xarray as xa
from numpy.testing import assert_array_equal
from pytest import fixture
from smif.data_layer.data_array import DataArray
from smif.metadata import Spec


@fixture
def dims():
    return ['a', 'b', 'c']


@fixture
def coords():
    return[['a1', 'a2'], ['b1', 'b2', 'b3'], ['c1', 'c2', 'c3', 'c4']]


@fixture
def spec(dims, coords):
    return Spec(
        name='test_data',
        dims=dims,
        coords={
            'a': coords[0],
            'b': coords[1],
            'c': coords[2],
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
        assert_array_equal(da.data, numpy.zeros((2, 3, 4)))

    def test_as_df(self, small_da, data, dims, coords):
        expected_index = pd.MultiIndex.from_product(coords, names=dims)
        expected = pd.Series(numpy.reshape(data, data.size), index=expected_index)

        actual = small_da.as_df()
        pd.testing.assert_series_equal(actual, expected)

    def test_as_xarray(self, small_da, data, dims, coords):
        actual = small_da.as_xarray()
        expected = xa.DataArray(data, coords, dims)

        xa.testing.assert_equal(actual, expected)

    def test_as_ndarray(self, small_da, data):
        actual = small_da.as_ndarray()
        expected = data
        assert_array_equal(actual, expected)

    def test_equality(self, small_da, spec, data):

        expected = DataArray(spec, data)
        assert small_da.__eq__(expected) is True

    def test_repr(self, small_da, spec, data):

        assert repr(small_da) == "<DataArray('{}', '{}')>".format(spec, data)
