"""DataArray provides a thin wrapper around multidimensional arrays and metadata
"""
from logging import getLogger

import numpy as np  # type: ignore
from smif.exception import (SmifDataError, SmifDataMismatchError,
                            SmifDataNotFoundError)
from smif.metadata.spec import Spec

# Import pandas, xarray if available (optional dependencies)
try:
    import pandas  # type: ignore
    import xarray  # type: ignore
except ImportError:
    pass


INSTALL_WARNING = """\
Please install pandas and xarray to access smif.DataArray
data as pandas.DataFrame or xarray.DataArray. Try running:
    pip install smif[data]
or:
    conda install pandas xarray
"""


class DataArray():
    """DataArray provides access to input/parameter/results data, with conversions to common
    python data libraries (for example: numpy, pandas, xarray).

    Attributes
    ----------
    spec: smif.metadata.spec.Spec
    data: numpy.ndarray
    """
    def __init__(self, spec: Spec, data: np.ndarray):
        self.logger = getLogger(__name__)

        if not hasattr(data, 'shape'):
            self.logger.debug("Data is not an numpy.ndarray")
            data = np.array(data)

        if not hasattr(spec, 'shape'):
            self.logger.error("spec argument is not a Spec")
            raise TypeError("spec argument is not a Spec")

        if not data.shape == spec.shape:
            # special case for scalar - allow a single-value 1D array, here coerced to single
            # value 0D array. Then simpler to create from DataFrame or xarray.DataArray
            if data.shape == (1,) and spec.shape == ():
                data = np.array(data[0])
            else:
                msg = "Data shape {} does not match spec {}"
                raise SmifDataMismatchError(msg.format(data.shape, spec.shape))

        self.spec = spec
        self.data = data

    def __eq__(self, other):
        return self.spec == other.spec and \
            _array_equal_nan(self.data, other.data)

    def __repr__(self):
        return "<DataArray('{}', '{}')>".format(self.spec, self.data)

    def __str__(self):
        return "<DataArray('{}', '{}')>".format(self.spec, self.data)

    def as_dict(self):
        """
        """
        return self.spec.as_dict()

    @property
    def name(self):
        """The name of the data that this spec describes.
        """
        return self.spec.name

    @name.setter
    def name(self, value):
        self.spec.name = value

    @property
    def description(self):
        """A human-friendly description
        """
        return self.spec.description

    @property
    def dims(self):
        """Names for each dimension
        """
        return self.spec.dims

    @property
    def coords(self):
        """Coordinate labels for each dimension.
        """
        return self.spec.coords

    def dim_coords(self, dim):
        """Coordinates for a given dimension
        """
        return self.spec.dim_coords(dim)

    def dim_names(self, dim):
        """Coordinate names for a given dimension
        """
        return self.spec.dim_names(dim)

    def dim_elements(self, dim):
        """Coordinate elements for a given dimension
        """
        return self.spec.dim_elements(dim)

    @property
    def unit(self):
        """The unit for all data points.
        """
        return self.spec.unit

    @property
    def shape(self):
        """The shape of the data array
        """
        return self.data.shape

    def as_ndarray(self) -> np.ndarray:
        """Access as a :class:`numpy.ndarray`
        """
        return self.data

    def as_df(self) -> pandas.DataFrame:
        """Access DataArray as a :class:`pandas.DataFrame`
        """
        dims = self.dims
        coords = [c.ids for c in self.coords]

        try:
            if dims and coords:
                index = pandas.MultiIndex.from_product(coords, names=dims)
                return pandas.DataFrame(
                    {self.name: np.reshape(self.data, self.data.size)}, index=index)
            else:
                # with no dims or coords, should be in the zero-dimensional case
                if self.data.shape != ():
                    msg = "Expected zero-dimensional data, got %s" % self.data.shape
                    raise SmifDataMismatchError(msg)
                return pandas.DataFrame([{self.name: self.data[()]}])
        except NameError as ex:
            raise SmifDataError(INSTALL_WARNING) from ex

    @classmethod
    def from_df(cls, spec, dataframe):
        """Create a DataArray from a :class:`pandas.DataFrame`
        """
        name = spec.name
        dims = spec.dims

        data_columns = dataframe.columns.values.tolist()
        index_names = dataframe.index.names

        if dims and len(index_names) == 1 and index_names[0] is None:
            # case when an unindexed dataframe was passed in, try to recover automagically
            if set(dims).issubset(set(data_columns)):
                dataframe = dataframe.set_index(dims)
                data_columns = dataframe.columns.values.tolist()
                index_names = dataframe.index.names

        if name not in data_columns or (dims and set(dims) != set(index_names)):
            msg = "Data for '{name}' expected a data column called '{name}' and index " + \
                  "names {dims}, instead got data columns {data_columns} and index names " + \
                  "{index_names}"
            raise SmifDataMismatchError(msg.format(
                name=name,
                dims=dims,
                data_columns=data_columns,
                index_names=index_names))

        try:
            # convert to dataset
            xr_dataset = dataframe.to_xarray()

            # extract xr.DataArray
            xr_data_array = xr_dataset[spec.name]

            # reindex to ensure data order and fill out NaNs
            xr_data_array = _reindex_xr_data_array(spec, xr_data_array)

        # xarray raises Exception in v0.10 (narrowed to ValueError in v0.11)
        except Exception as ex:  # pylint: disable=broad-except
            dups = find_duplicate_indices(dataframe)
            if dups:
                msg = "Data for '{name}' contains duplicate values at {dups}"
                raise SmifDataMismatchError(msg.format(name=name, dups=dups)) from ex
            else:
                raise ex

        return cls(spec, xr_data_array.data)

    def as_xarray(self):
        """Access DataArray as a :class:`xarray.DataArray`
        """
        metadata = self.spec.as_dict()
        del metadata['dims']
        del metadata['coords']

        dims = self.dims
        coords = {c.name: c.ids for c in self.coords}

        try:
            return xarray.DataArray(
                self.data,
                coords=coords,
                dims=dims,
                name=self.name,
                attrs=metadata
            )
        except NameError as ex:
            raise SmifDataError(INSTALL_WARNING) from ex

    @classmethod
    def from_xarray(cls, spec, xr_data_array):
        """Create a DataArray from a :class:`xarray.DataArray`
        """
        # reindex to ensure data order and fill out NaNs
        xr_data_array = _reindex_xr_data_array(spec, xr_data_array)
        return cls(spec, xr_data_array.data)

    def update(self, other):
        """Update data values with any from other which are non-null
        """
        assert self.spec == other.spec, "Specs must match when updating DataArray"
        # convert self and other to xarray representation
        self_xr = self.as_xarray()
        other_xr = other.as_xarray()
        # use xarray.combine_first convenience function
        overridden = other_xr.combine_first(self_xr)
        # assign result back to self
        self.data = overridden.data

    def validate_as_full(self):
        """Check that the data array contains no NaN values
        """
        dataframe = self.as_df()
        if np.any(dataframe.isnull()):
            expected_len = len(dataframe)
            missing_data = show_null(dataframe)
            actual_len = expected_len - len(missing_data)
            dim_lens = "{" + ", ".join(
                "{}: {}".format(dim, len_) for dim, len_ in zip(self.dims, self.shape)
            ) + "}"
            self.logger.debug("Missing data:\n\n    %s", missing_data)
            msg = "Data for '{name}' had missing values - read {actual_len} but expected " + \
                  "{expected_len} in total, from dims of length {dim_lens}"
            raise SmifDataMismatchError(msg.format(
                name=self.name,
                actual_len=actual_len,
                expected_len=expected_len,
                dim_lens=dim_lens))


def show_null(dataframe) -> pandas.DataFrame:
    """Shows missing data

    Returns
    -------
    pandas.DataFrame
    """
    try:
        missing_data = dataframe[dataframe.isnull().values]
    except NameError as ex:
        raise SmifDataError(INSTALL_WARNING) from ex
    return missing_data


def find_duplicate_indices(dataframe):
    """Find duplicate indices in a DataFrame

    Returns
    -------
    list[dict]
    """
    # find duplicate index entries
    dups_df = dataframe[dataframe.index.duplicated()]
    # drop data columns, reset index to promote index to values
    dups_index_df = dups_df.drop(dups_df.columns, axis=1).reset_index()
    return dups_index_df.to_dict('records')


def _array_equal_nan(a, b):
    """Compare numpy arrays for equality, allowing NaN to be considerd equal to itself
    """
    if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
        return np.all((a == b) | (np.isnan(a) & np.isnan(b)))
    else:
        return np.all(a == b)


def _reindex_xr_data_array(spec, xr_data_array):
    """Reindex to ensure full data, order
    """
    # all index values must exist in dimension - extras would otherwise be silently dropped
    for dim in spec.dims:
        index_values = set(xr_data_array.coords[dim].values)
        dim_names = set(spec.dim_names(dim))
        in_index_but_not_dim_names = index_values - dim_names
        if in_index_but_not_dim_names:
            msg = "Data for '{name}' contained unexpected values in the set of " + \
                  "coordinates for dimension '{dim}': {extras}"
            raise SmifDataMismatchError(msg.format(
                dim=dim, extras=list(in_index_but_not_dim_names), name=spec.name))

    coords = {c.name: c.ids for c in spec.coords}
    xr_data_array = xr_data_array.reindex(indexers=coords)

    return xr_data_array
