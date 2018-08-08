"""Data is typically multi-dimensional, a Spec provides metadata to label it.
"""
from smif.metadatum.coordinates import Coordinates


class Spec(object):
    """A Spec is a data contract, it describes the shape of inputs and parameters to be
    provided to or output from each model or process.

    In practice, this looks a lot like an xarray.DataArray with no data.
    """
    def __init__(self, name=None, coords=None, dtype=None, default=None, abs_range=None,
                 exp_range=None, unit=None):
        self._name = name

        if not coords:
            raise ValueError("Spec.coords must be provided")

        if isinstance(coords, dict):
            coords = [
                Coordinates(dim, elements)
                for dim, elements in coords.items()
            ]
        else:
            for coord in coords:
                if not isinstance(coord, Coordinates):
                    raise ValueError("Spec.coords may be a dict of {dim: elements} or a " +
                                     "list of Coordinates")

        self._coords = coords

        if dtype is None:
            raise ValueError("Spec.dtype must be provided")
        self._dtype = dtype

        self._default = default
        self._abs_range = abs_range
        self._exp_range = exp_range
        self._unit = unit

    @property
    def name(self):
        """The name of the data that this spec describes.
        """
        return self._name

    @property
    def dtype(self):
        """The dtype of the data that this spec describes.
        """
        return self._dtype

    @property
    def shape(self):
        """The shape of the data that this spec describes.
        """
        return tuple(len(c.ids) for c in self._coords)

    @property
    def ndim(self):
        """The number of dimensions of the data that this spec describes.
        """
        return len(self._coords)
