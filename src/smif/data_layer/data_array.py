"""DataArray provides a thin wrapper around multidimensional arrays and metadata
"""
from logging import getLogger

import numpy as np
from smif.exception import SmifDataError, SmifDataMismatchError
from smif.metadata.spec import Spec

# Import pandas, xarray if available (optional dependencies)
try:
    import pandas
    import xarray
except ImportError:
    pass


INSTALL_WARNING = """\
Please install pandas and/or xarray to access smif.DataArray
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
        self.spec = spec
        self.data = data

        if not hasattr(data, 'shape'):
            self.logger.debug("Data is not an numpy.ndarray")
            data = np.array(data)

        if not hasattr(spec, 'shape'):
            self.logger.error("spec argument is not a Spec")
            raise TypeError("spec argument is not a Spec")

        if not data.shape == self.spec.shape:
            msg = "Data shape {} does not match spec {}"
            raise SmifDataMismatchError(msg.format(data.shape, spec.shape))

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
            index = pandas.MultiIndex.from_product(coords, names=dims)
            return pandas.DataFrame(
                {self.name: np.reshape(self.data, self.data.size)}, index=index)
        except NameError as ex:
            raise SmifDataError(INSTALL_WARNING) from ex

    @classmethod
    def from_df(cls, spec, dataframe):
        """Create a DataArray from a :class:`pandas.DataFrame`
        """
        xr_dataset = dataframe.to_xarray()  # convert to dataset
        xr_data_array = xr_dataset[spec.name]  # extract xr.DataArray
        return cls.from_xarray(spec, xr_data_array)

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
        # set up empty xr.DataArray to override
        empty_xr_data_array = xarray.DataArray(
            np.full(spec.shape, np.nan),
            coords={c.name: c.ids for c in spec.coords},
            dims=spec.dims,
            name=spec.name
        )
        # fill out to size (values in the array before `.combine_first` win)
        xr_data_array = xr_data_array.combine_first(empty_xr_data_array)
        # check we do match shape (could be bigger than spec)
        assert xr_data_array.shape == spec.shape
        data = xr_data_array.data
        return cls(spec, data)


def _array_equal_nan(a, b):
    """Compare numpy arrays for equality, allowing NaN to be considerd equal to itself
    """
    return np.all((a == b) | (np.isnan(a) & np.isnan(b)))
