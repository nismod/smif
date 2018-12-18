"""DataArray provides a thin wrapper around multidimensional arrays and metadata
"""
from logging import getLogger

import numpy as np
from smif.exception import (SmifDataError, SmifDataMismatchError,
                            SmifDataNotFoundError)
from smif.metadata.spec import Spec

# Import pandas, xarray if available (optional dependencies)
try:
    import pandas
    import xarray
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

    @property
    def description(self):
        """A human-friendly description
        """
        return self.spec._description

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
        xr_dataset = dataframe.to_xarray()  # convert to dataset

        try:
            xr_data_array = xr_dataset[spec.name]  # extract xr.DataArray
        except KeyError:
            # must have a column for the data variable (spec.name)
            raise KeyError(
                "Data missing variable key ({}), got {}".format(
                    spec.name, dataframe.columns))

        # reindex to ensure data order and fill out NaNs
        xr_data_array = _reindex_xr_data_array(spec, xr_data_array)
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
        df = self.as_df()
        if np.any(df.isnull()):
            missing_data = self._show_null(df)
            self.logger.debug("Missing data:\n\n    %s", missing_data)
            msg = "There are missing data points in '{}'"
            raise SmifDataNotFoundError(msg.format(self.name))

    def _show_null(self, df) -> pandas.DataFrame:
        """Shows missing data

        Returns
        -------
        pandas.DataFrame
        """
        try:
            missing_data = df[df.isnull().values]
        except NameError as ex:
            raise SmifDataError(INSTALL_WARNING) from ex
        return missing_data


def _array_equal_nan(a, b):
    """Compare numpy arrays for equality, allowing NaN to be considerd equal to itself
    """
    if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
        return np.all((a == b) | (np.isnan(a) & np.isnan(b)))
    else:
        return np.all(a == b)


def _reindex_xr_data_array(spec, xr_data_array):
    """Reindex, raise clear errors, error if extra dimension names
    """
    # must have an index level for each dimension
    missing_dims = [d for d in spec.dims if d not in xr_data_array.dims]
    if missing_dims:
        raise KeyError(
            "Data missing dimension keys {}, got {}".format(
                missing_dims, xr_data_array.dims))

    for dim in spec.dims:
        # all index values must exist in dimension
        index_values = xr_data_array.coords[dim].values
        # cast list to np.array then do set operation - alternative to python set() ops
        dim_names = np.array(spec.dim_names(dim))
        in_index_but_not_dim_names = np.setdiff1d(index_values, dim_names)
        if in_index_but_not_dim_names.size > 0:
            raise ValueError(
                "Unknown {} values {} in {}".format(
                    dim, in_index_but_not_dim_names, spec.name))

    # reindex to ensure data order
    coords = {c.name: c.ids for c in spec.coords}
    xr_data_array = xr_data_array.reindex(indexers=coords)

    return xr_data_array
