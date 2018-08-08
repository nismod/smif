"""Dimensions are always finite, and data which has a given dimension can be indexed by a
set of coordinates
"""


class Coordinates(object):
    """Coordinates index a dimension

    A dict of {Coordinates.dim: Coordinates.ids} can be passed to a Spec (or
    xarray.DataArray)
    """
    def __init__(self, name, elements):
        self.name = name
        self.ids = None
        self._elements = None
        self.elements = elements

    @property
    def elements(self):
        """Elements are a list of dicts
        """
        return self._elements

    @elements.setter
    def elements(self, elements):
        """Set elements with a list of ids (string or numeric) or dicts (including key 'id')
        """
        if not elements:
            raise ValueError("Coordinates.elements must not be empty")

        try:
            len(elements)
        except TypeError:
            raise ValueError("Coordinate.elements must be finite in length")

        if isinstance(elements[0], dict):
            if "id" not in elements[0]:
                raise KeyError("Coordinates.elements must have an id, or be a simple list " +
                               "of identifiers")

            self.ids = [e['id'] for e in elements]
            self._elements = elements
        else:
            self.ids = elements
            self._elements = [{"id": e} for e in elements]

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
