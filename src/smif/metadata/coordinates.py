"""Each dimension of a multi-dimensional dataset can be indexed by a set of coordinates.

Dimensions might be represented as Euclidean coordinates over a grid::

>>> x = Coordinates('x', range(0, 100))
>>> y = Coordinates('y', range(0, 100))

A subset of Local Authority Districts in England would be a region-based spatial dimension::

>>> local_authority_districts = Coordinates('LADs', ['E07000128', 'E07000180'])

The hours of an average annual day would be a temporal dimension::

        >>> hours = Coordinates('daily_hours', [
        ...     {'name': 0, 'represents': '00:00-00:59:59'}
        ...     {'name': 1, 'represents': '01:00-01:59:59'}
        ...     {'name': 2, 'represents': '02:00-02:59:59'}
        ... ])

The economic sectors from the International Standard Industrial Classification of All Economic
Activities (ISIC), revision 4 would be a categorical dimension::

        >>> economic_sector = Coordinates('ISICrev4', [
        ...     {'name': 'A', 'desc': 'Agriculture, forestry and fishing'},
        ...     {'name': 'B', 'desc': 'Mining and quarrying'},
        ...     {'name': 'C', 'desc': 'Manufacturing'},
        ...     {'name': 'D', 'desc': 'Electricity, gas, steam and air conditioning supply'}
        ... ])

"""


class Coordinates(object):
    """Coordinates provide the labels to index a dimension, along with metadata that may be
    useful to describe, visualise or convert between dimensions.

    Coordinate element names are used to label each position along a finite dimension.

    A dict mapping dimension name to list of coordinate elements can be passed to a
    :class:`~smif.metadata.spec.Spec` (or :class:`xarray.DataArray`) as `coords`.

    Attributes
    ----------
    name : str
        Dimension name
    dim : str
        Alias for dimension name
    ids : list
        List of labels
    elements : list[dict]
        List of labels with metadata

    Parameters
    ----------
    name : str
        Name to identify the dimension that these coordinates index
    elements : list
        List of simple data types (used to identify elements), or a list of dicts with 'name'
        key and other metadata

    Raises
    ------
    ValueError
        If the list of elements is empty, or infinite
    KeyError
        If the elements are not a list of simple data types
        or a list of dicts with a 'name' key
    """
    def __init__(self, name, elements):
        self.name = name
        self._ids = None
        self._elements = None
        self._set_elements(elements)

    def __eq__(self, other):
        return self.name == other.name \
            and self.elements == other.elements

    def __hash__(self):
        return hash(tuple(frozenset(e.items()) for e in self._elements))

    def __repr__(self):
        return "<Coordinates name='{}' elements={}>".format(self.name, self.ids)

    @property
    def elements(self):
        """Elements are a list of dicts with at least a 'name' key

        Coordinate elements should not be changed.
        """
        return self._elements

    @property
    def ids(self):
        """Element ids is a list of coordinate identifiers
        """
        return self._ids

    def _set_elements(self, elements):
        """Set elements with a list of ids (string or numeric) or dicts (including key 'id')
        """
        if not elements:
            raise ValueError("Coordinates.elements must not be empty")

        try:
            len(elements)
        except TypeError:
            raise ValueError("Coordinate.elements must be finite in length")

        if isinstance(elements[0], dict):
            if "name" not in elements[0]:
                msg = "Elements in dimension '{}' must have a name field, " \
                      "or be a simple list of identifiers"
                raise KeyError(msg.format(self.name))

            self._ids = [e['name'] for e in elements]
            self._elements = elements
        else:
            self._ids = elements
            self._elements = [{"name": e} for e in elements]

    @property
    def dim(self):
        """Dim (dimension) is an alias for Coordinates.name
        """
        return self.name

    @dim.setter
    def dim(self, dim):
        """Set name as dim
        """
        self.name = dim
