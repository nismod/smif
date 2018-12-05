"""Test DataArray
"""
import numpy
import pandas as pd
import xarray as xr
from numpy.testing import assert_array_equal
from pytest import fixture
from smif.data_layer.data_array import DataArray
from smif.metadata import Spec


@fixture
def dims():
    return ['a', 'b', 'c']


@fixture
def coords():
    return [['a1', 'a2'], ['b1', 'b2', 'b3'], ['c1', 'c2', 'c3', 'c4']]


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
        abs_range=(0, 1),
        exp_range=(0, 0.5)
    )


@fixture
def data():
    return numpy.arange(24, dtype='float').reshape((2, 3, 4))


@fixture
def small_da(spec, data):
    return DataArray(spec, data)


@fixture
def small_da_df(spec, data):
    index = pd.MultiIndex.from_product([c.ids for c in spec.coords], names=spec.dims)
    return pd.DataFrame({spec.name: numpy.reshape(data, data.size)}, index=index)


@fixture
def small_da_xr(spec, data):
    return xr.DataArray(data, [(c.name, c.ids) for c in spec.coords])


class TestDataArray():
    def test_init(self, spec, data):
        """Should initialise from spec and ndarray of data
        """
        da = DataArray(spec, data)
        numpy.testing.assert_equal(da.data, data)
        assert spec == da.spec

    def test_as_df(self, small_da, small_da_df):
        """Should create a pandas.DataFrame from a DataArray
        """
        actual = small_da.as_df()
        pd.testing.assert_frame_equal(actual, small_da_df)

    def test_from_df(self, small_da, small_da_df):
        """Should create a DataArray from a pandas.DataFrame
        """
        actual = DataArray.from_df(small_da.spec, small_da_df)
        assert actual == small_da

    def test_from_df_partial(self, spec):
        """Should create a DataArray that can handle missing data, returning nan/null
        """
        df = pd.DataFrame({
            'a': ['a1'],
            'b': ['b1'],
            'c': ['c2'],
            'test_data': [1]
        }).set_index(['a', 'b', 'c'])
        expected_data = numpy.full(spec.shape, numpy.nan)
        expected_data[0, 0, 1] = 1.0
        expected = DataArray(spec, expected_data)

        actual = DataArray.from_df(spec, df)

        assert_array_equal(actual.data, expected.data)
        assert actual == expected

    def test_combine(self, small_da, data):
        """Should override values where present (use case: full array of default values,
        overridden by a partial array of specific values).

        See variously:
        - http://xarray.pydata.org/en/stable/combining.html#merging-with-no-conflicts
        - https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.update.html
        """
        partial_data = numpy.full(small_da.shape, numpy.nan)
        partial_data[0, 0, 1] = 99
        partial = DataArray(small_da.spec, partial_data)

        # update in-place
        small_da.update(partial)

        expected_data = data
        expected_data[0, 0, 1] = 99
        expected = DataArray(small_da.spec, expected_data)

        assert small_da == expected
        assert_array_equal(small_da.data, expected.data)

    def test_as_xarray(self, small_da, small_da_xr):
        actual = small_da.as_xarray()
        xr.testing.assert_equal(actual, small_da_xr)

    def test_from_xarray(self, small_da, small_da_xr):
        actual = DataArray.from_xarray(small_da.spec, small_da_xr)
        assert actual == small_da

    def test_as_ndarray(self, small_da, data):
        actual = small_da.as_ndarray()
        expected = data
        assert_array_equal(actual, expected)

    def test_equality(self, small_da, spec, data):
        expected = DataArray(spec, data)
        assert small_da == expected

    def test_repr(self, small_da, spec, data):
        assert repr(small_da) == "<DataArray('{}', '{}')>".format(spec, data)

    def test_dim_coords(self, small_da, spec):
        actual = small_da.dim_coords('a')
        expected = spec.dim_coords('a')
        assert actual == expected

    def test_dim_names(self, small_da, spec):
        actual = small_da.dim_names('a')
        expected = spec.dim_names('a')
        assert actual == expected

    def test_dim_elements(self, small_da, spec):
        actual = small_da.dim_elements('a')
        expected = spec.dim_elements('a')
        assert actual == expected

    def test_coords(self, small_da, spec):
        actual = small_da.coords
        expected = spec.coords
        assert actual == expected
