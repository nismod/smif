"""DataArray provides a thin wrapper around multidimensional arrays and metadata
"""
import numpy
from smif.exception import SmifDataError

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
    spec : smif.metadata.spec.Spec
    data : numpy.ndarray
    """
    def __init__(self, spec, data):
        self.spec = spec
        self.data = data

    @classmethod
    def default_from_spec(cls, spec):
        """Create DataArray from a spec's default
        """
        data = numpy.full(spec.shape, spec.default)
        return cls(spec, data)

    @property
    def name(self):
        """The name of the data that this spec describes.
        """
        return self.spec._name

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

    @property
    def dim_coords(self, dim):
        """Coordinates for a given dimension
        """
        return self.spec.dim_coords(dim)

    @property
    def unit(self):
        """The unit for all data points.
        """
        return self.spec.unit

    def as_ndarray(self):
        """Access as a :class:`numpy.ndarray`
        """
        return self.data

    def as_df(self):
        """Access DataArray as a :class:`pandas.DataFrame`
        """
        dims = self.dims
        coords = [c.ids for c in self.coords]

        try:
            index = pandas.MultiIndex.from_product(coords, names=dims)
            return pandas.Series(numpy.reshape(self.data, self.data.size), index=index)
        except NameError as ex:
            raise SmifDataError(INSTALL_WARNING) from ex

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
