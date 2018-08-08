"""Data is typically multi-dimensional, a Spec provides metadata to label it.
"""
from collections import defaultdict

from smif.metadata.coordinates import Coordinates


class Spec(object):
    """A Spec is a data contract, it describes the shape of inputs and parameters to be
    provided to or output from each model or process.

    In practice, this looks a lot like an xarray.DataArray with no data.
    """
    def __init__(self, name=None, dims=None, coords=None, dtype=None, default=None,
                 abs_range=None, exp_range=None, unit=None):
        self._name = name

        if not coords:
            raise ValueError("Spec.coords must be provided")

        # Either as a dict, in which case dims must be provided to define order
        if isinstance(coords, dict):
            if dims is None:
                msg = "Spec.dims must be specified if coords are provided as a dict"
                raise ValueError(msg)
            coords = [
                Coordinates(dim, coords[dim])
                for dim in dims
            ]
        # Or as a list of Coordinates, in which case dims must not be provided
        else:
            for coord in coords:
                if not isinstance(coord, Coordinates):
                    raise ValueError("Spec.coords may be a dict of {dim: elements} or a " +
                                     "list of Coordinates")
            if dims is not None:
                raise ValueError("Spec.dims are derived from Spec.coords if provided as a " +
                                 "list of Coordinates")
            dims = [coord.dim for coord in coords]

        self._dims = dims
        self._coords = coords

        if dtype is None:
            raise ValueError("Spec.dtype must be provided")
        self._dtype = dtype

        self._default = default
        self._abs_range = abs_range
        self._exp_range = exp_range
        self._unit = unit

    @classmethod
    def from_dict(cls, data_provided):
        """Create a Spec from a dict representation
        """
        # default anything to None, let constructor handle essential missing values
        data = defaultdict(lambda: None)
        data.update(data_provided)
        spec = Spec(
            name=data['name'],
            dims=data['dims'],
            coords=data['coords'],
            dtype=data['dtype'],
            default=data['dtype'],
            abs_range=data['abs_range'],
            exp_range=data['exp_range'],
            unit=data['unit']
        )
        return spec

    def as_dict(self):
        """Serialise to dict representation
        """
        return {
            'name': self.name,
            'dims': self._dims,
            'coords': {c.name: c.ids for c in self._coords},
            'dtype': self._dtype,
            'default': self._default,
            'abs_range': self._abs_range,
            'exp_range': self._exp_range,
            'unit': self._unit
        }

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
