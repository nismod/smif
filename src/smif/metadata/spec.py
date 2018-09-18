"""Data is typically multi-dimensional. :class:`~smif.metadata.spec.Spec` is used to describe
each dataset which is supplied to - or output from - each :class:`~smif.model.model.Model` in a
:class:`~smif.model.model.CompositeModel`
"""
from collections import defaultdict

from smif.metadata.coordinates import Coordinates


class Spec(object):
    """N-dimensional metadata.

    Spec labels each dimension with coordinates and enforces data type, units and absolute and
    expected ranges.

    The API here is modelled on :class:`xarray.DataArray`: dtype and shape describe a
    :class:`numpy.ndarray`; dims and coords follow the xarray conventions for labelled axes;
    and unit, default, abs_range and exp_range are introduced as supplementary metadata to
    help validate connections between models.

    Attributes
    ----------
    name : str
        The name of the data that this spec describes
    description : str
        A human-friendly description
    dtype : str
        Data type for data values
    default : object
        Default data value
    abs_range : tuple
        Absolute range of data values
    exp_range : tuple
        Expected range of data values
    shape : tuple[int]
        Tuple of dimension sizes
    ndim : int
        Number of dimensions
    dims : list[str]
        Dimension names
    coords : list[Coordinates]
        Dimension coordinate labels
    unit : str
        Units of data values

    Parameters
    ----------
    name : str, optional
        Name to identifiy the variable described (typically an input, output or parameter)
    description : str, optional
        Short description
    dims : list[str], optional
        List of dimension names, must be provided if coords is a dict
    coords : list[Coordinates] or dict[str, list], optional
        A list of :class`Coordinates` or a dict mapping each dimension name to a list of names
        which label that dimension.
    dtype : str
        String suitable for contructing a simple :class:`numpy.dtype`
    default : object, optional
        Value to be used as default
    abs_range : tuple, optional
        (min, max) absolute range for numeric values - can be used to raise errors
    exp_range : tuple, optional
        (min, max) expected range for numeric values - can be used to raise warnings
    unit : str, optional
        Unit to be used for data values
    """
    def __init__(self, name=None, dims=None, coords=None, dtype=None, default=None,
                 abs_range=None, exp_range=None, unit=None, description=None):
        self._name = name
        self._description = description

        # Coords may come as a dict, in which case dims must be provided to define order
        if isinstance(coords, dict):
            try:
                coords, dims = self._coords_from_dict(coords, dims)
            except (ValueError, KeyError) as error:
                msg = "Coordinate metadata incorrectly formatted for variable '{}': {}"
                raise ValueError(msg.format(self.name, error))

        # Or as a list of Coordinates, in which case dims must not be provided
        elif isinstance(coords, list):
            coords, dims = self._coords_from_list(coords, dims)

        # Or if None, this spec describes a zero-dimensional parameter - single value
        else:
            coords, dims = [], []

        self._dims = dims
        self._coords = coords

        if dtype is None:
            raise ValueError("Spec.dtype must be provided, in {}".format(self._name))
        self._dtype = dtype

        self._default = default

        if abs_range is not None:
            self._check_range(abs_range)
        self._abs_range = abs_range

        if exp_range is not None:
            self._check_range(exp_range)
        self._exp_range = exp_range

        self._unit = unit

    def _coords_from_list(self, coords, dims):
        """Set up coords and dims, checking for consistency
        """
        for coord in coords:
            if not isinstance(coord, Coordinates):
                msg = "Spec.coords may be a dict[str,list] or a list[Coordinates], in {}"
                raise ValueError(msg.format(self._name))

        if dims is not None:
            msg = "Spec.dims are derived from Spec.coords if provided as a list of " + \
                  "Coordinates, in {}"
            raise ValueError(msg.format(self._name))

        dims = [coord.dim for coord in coords]

        if len(dims) != len(set(dims)):
            msg = "Spec cannot be created with duplicate dims, in {}"
            raise ValueError(msg.format(self._name))

        return coords, dims

    def _coords_from_dict(self, coords, dims):
        """Set up coords and dims, checking for consistency
        """
        if dims is None:
            msg = "Spec.dims must be specified if coords are provided as a dict, in {}"
            raise ValueError(msg.format(self._name))

        if len(dims) != len(set(dims)):
            msg = "Spec cannot be created with duplicate dims, in {}"
            raise ValueError(msg.format(self._name))

        if sorted(dims) != sorted(coords.keys()):
            msg = "Spec.dims must match the keys in coords, in {}"
            raise ValueError(msg.format(self._name))

        coords = [Coordinates(dim, coords[dim]) for dim in dims]

        return coords, dims

    @classmethod
    def from_dict(cls, data_provided):
        """Create a Spec from a dict representation
        """
        # default anything to None, let constructor handle essential missing values
        data = defaultdict(lambda: None)
        data.update(data_provided)
        spec = Spec(
            name=data['name'],
            description=data['description'],
            dims=data['dims'],
            coords=data['coords'],
            dtype=data['dtype'],
            default=data['default'],
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
            'description': self.description,
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
    def description(self):
        """A human-friendly description
        """
        return self._description

    @property
    def dtype(self):
        """The dtype of the data that this spec describes.
        """
        return self._dtype

    @property
    def default(self):
        """The default value of data that this spec describes.
        """
        return self._default

    @property
    def abs_range(self):
        """The absolute range of data values that this spec describes.
        """
        return self._abs_range

    @property
    def exp_range(self):
        """The expected range of data values that this spec describes.
        """
        return self._exp_range

    @property
    def shape(self):
        """Tuple of dimension sizes. The shape of the data that this spec describes.
        """
        return tuple(len(c.ids) for c in self._coords)

    @property
    def ndim(self):
        """The number of dimensions of the data that this spec describes.
        """
        return len(self._coords)

    @property
    def dims(self):
        """Names for each dimension
        """
        return list(self._dims)

    @property
    def coords(self):
        """Coordinate labels for each dimension.
        """
        return list(self._coords)

    def dim_coords(self, dim):
        """Coordinates for a given dimension
        """
        for coord in self._coords:
            if coord.dim == dim:
                return coord
        raise KeyError("Coords not found for dim {}, in {}".format(dim, self._name))

    @property
    def unit(self):
        """The unit for all data points.
        """
        return self._unit

    def __eq__(self, other):
        return self.dtype == other.dtype \
            and self.dims == other.dims \
            and self.coords == other.coords \
            and self.unit == other.unit

    def __hash__(self):
        return hash((
            self.dtype,
            tuple(self.dims),
            tuple(self.coords),
            self.unit
        ))

    def __repr__(self):
        return "<Spec name='{}' dims='{}' unit='{}'>".format(self.name, self.dims, self.unit)

    def _check_range(self, range_):
        """Error if range is not a [min, max] list or tuple
        """
        if not _is_sequence(range_):
            msg = "Spec range must be a list or tuple, got {} for {}"
            raise TypeError(msg.format(range_, self._name))
        if len(range_) != 2:
            msg = "Spec range must have min and max values only, got {} for {}"
            raise ValueError(msg.format(range_, self._name))
        min_, max_ = range_
        if max_ < min_:
            msg = "Spec range min value must be smaller than max value, got {} for {}"
            raise ValueError(msg.format(range_, self._name))


def _is_sequence(obj):
    """Check for iterable object that is not a string ('strip' is a method on str)
    """
    return not hasattr(obj, "strip") \
        and (hasattr(obj, "__getitem__") or hasattr(obj, "__iter__"))
