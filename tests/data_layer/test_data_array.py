"""Test DataArray
"""
import numpy
import pandas as pd
import xarray as xr
from numpy.testing import assert_array_equal
from pytest import fixture, raises
from smif.data_layer.data_array import DataArray
from smif.exception import SmifDataNotFoundError
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
def non_numeric_spec(dims, coords):
    return Spec(
        name='test_data',
        dims=dims,
        coords={
            'a': coords[0],
            'b': coords[1],
            'c': coords[2],
        },
        dtype='str',
        abs_range=('0', '1'),
        exp_range=('0', '0.5')
    )


@fixture
def non_numeric_data():
    strings = [str(x) for x in range(24)]
    return numpy.array(strings, dtype=numpy.object).reshape((2, 3, 4))


@fixture
def small_da(spec, data):
    return DataArray(spec, data)


@fixture
def small_da_non_numeric(non_numeric_spec, non_numeric_data):
    return DataArray(non_numeric_spec, non_numeric_data)


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

        # match fixture data
        expected_data = numpy.arange(24, dtype='float').reshape((2, 3, 4))
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


class TestDataFrameInterop():
    def test_to_from_df(self):
        df = pd.DataFrame([
            {
                'test': 3,
                'region': 'oxford',
                'interval': 1
            }
        ]).set_index(['region', 'interval'])

        spec = Spec(
            name='test',
            dims=['region', 'interval'],
            coords={'region': ['oxford'], 'interval': [1]},
            dtype='int64'
        )

        da = DataArray(spec, numpy.array([[3.]], dtype='int64'))
        da_from_df = DataArray.from_df(spec, df)
        assert da_from_df == da

        da_to_df = da.as_df()
        pd.testing.assert_frame_equal(da_to_df, df)

    def test_multi_dim_order(self):
        spec = Spec(
            name='test',
            coords={'lad': ['c', 'a', 'b'], 'interval': [4, 2]},
            dims=['lad', 'interval'],
            dtype='float'
        )
        data = numpy.array([
            # 4 2
            [1, 2],  # c
            [5, 6],  # a
            [9, 0]  # b
        ], dtype='float')
        da = DataArray(spec, data)

        df = pd.DataFrame([
            {'test': 6.0, 'lad': 'a', 'interval': 2},
            {'test': 0.0, 'lad': 'b', 'interval': 2},
            {'test': 2.0, 'lad': 'c', 'interval': 2},
            {'test': 5.0, 'lad': 'a', 'interval': 4},
            {'test': 9.0, 'lad': 'b', 'interval': 4},
            {'test': 1.0, 'lad': 'c', 'interval': 4},
        ]).set_index(['lad', 'interval'])
        da_from_df = DataArray.from_df(spec, df)
        assert da_from_df == da

        da_to_df = da.as_df().sort_index()
        df = df.sort_index()
        pd.testing.assert_frame_equal(da_to_df, df)

    def test_match_metadata(self):
        spec = Spec(
            name='test',
            dims=['region'],
            coords={'region': ['oxford']},
            dtype='int64'
        )

        # must have a column named the same as the spec.name
        df = pd.DataFrame([
            {'region': 'oxford', 'other': 'else'}
        ]).set_index(['region'])
        msg = "missing variable key (test)"
        with raises(KeyError) as ex:
            DataArray.from_df(spec, df)
        assert msg in str(ex)

        # must have an index level for each spec dimension
        df = pd.DataFrame([
            {'test': 3.14}
        ])
        msg = "missing dimension keys ['region']"
        with raises(KeyError) as ex:
            DataArray.from_df(spec, df)
        assert msg in str(ex)

        # must have dimension labels that exist in the spec dimension
        df = pd.DataFrame([
            {'test': 3.14, 'region': 'missing'}
        ]).set_index(['region'])
        msg = "Unknown region values ['missing']"
        with raises(ValueError) as ex:
            DataArray.from_df(spec, df)
        assert msg in str(ex)

        # must not have dimension labels outside of the spec dimension
        df = pd.DataFrame([
            {'test': 3.14, 'region': 'oxford'},
            {'test': 3.14, 'region': 'extra'}
        ]).set_index(['region'])
        msg = "Unknown region values ['extra']"
        with raises(ValueError) as ex:
            DataArray.from_df(spec, df)
        assert msg in str(ex)

    def test_scalar(self):
        # should handle zero-dimensional case (numpy array as scalar)
        data = numpy.array(2.0)
        spec = Spec(
            name='test',
            dims=[],
            coords={},
            dtype='float'
        )
        da = DataArray(spec, data)
        df = pd.DataFrame([{'test': 2.0}])
        da_from_df = DataArray.from_df(spec, df)
        assert da_from_df == da

        df_from_da = da.as_df()
        pd.testing.assert_frame_equal(df_from_da, df)


class TestMissingData:

    def test_missing_data_raises(self, small_da):
        """Should check for NaNs and raise SmifDataError
        """
        da = small_da
        da.validate_as_full()
        da.data[1, 1] = numpy.NaN

        with raises(SmifDataNotFoundError):
            da.validate_as_full()

    def test_missing_data_message(self, small_da):
        """Should check for NaNs and raise SmifDataError
        """
        da = small_da
        da.validate_as_full()
        da.data[1, 1, 1] = numpy.nan
        da.data[0, 0, 3] = numpy.nan
        with raises(SmifDataNotFoundError) as ex:
            da.validate_as_full()

        expected = "There are missing data points in 'test_data'"
        assert expected in str(ex)

    def test_missing_data_message_non_numeric(self, small_da_non_numeric):
        """Should check for NaNs and raise SmifDataError
        """
        da = small_da_non_numeric
        da.validate_as_full()
        da.data[1, 1, 1] = None
        da.data[0, 0, 3] = None
        with raises(SmifDataNotFoundError) as ex:
            da.validate_as_full()

        expected = "There are missing data points in 'test_data'"
        assert expected in str(ex)

    def test_no_missing_data(self, small_da):

        df = small_da.as_df()
        actual = small_da._show_null(df)
        expected = pd.DataFrame(columns=['test_data'], dtype=float)
        expected.index = pd.MultiIndex(
            levels=[['a1', 'a2'], ['b1', 'b2', 'b3'], ['c1', 'c2', 'c3', 'c4']],
            labels=[[], [], []],
            names=['a', 'b', 'c'])

        pd.testing.assert_frame_equal(actual, expected)

    def test_no_missing_data_non_numeric(self, small_da_non_numeric):

        df = small_da_non_numeric.as_df()
        actual = small_da_non_numeric._show_null(df)
        expected = pd.DataFrame(columns=['test_data'], dtype=str)
        expected.index = pd.MultiIndex(
            levels=[['a1', 'a2'], ['b1', 'b2', 'b3'], ['c1', 'c2', 'c3', 'c4']],
            labels=[[], [], []],
            names=['a', 'b', 'c'])

        pd.testing.assert_frame_equal(actual, expected)

    def test_missing_data_non_numeric(self, small_da_non_numeric):

        small_da_non_numeric.data[1, 1, 1] = None
        df = small_da_non_numeric.as_df()
        actual = small_da_non_numeric._show_null(df)
        index = pd.MultiIndex(
            levels=[['a1', 'a2'], ['b1', 'b2', 'b3'], ['c1', 'c2', 'c3', 'c4']],
            labels=[[1], [1], [1]],
            names=['a', 'b', 'c'])
        expected = pd.DataFrame(data=numpy.array([[None]], dtype=numpy.object),
                                index=index,
                                columns=['test_data'])
        pd.testing.assert_frame_equal(actual, expected)
