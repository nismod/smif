"""A metadata store holds shared metadata for smif model scenarios, inputs, parameters and
outputs:
- units
- dimension definitions
"""
from abc import ABCMeta, abstractmethod


class MetadataStore(metaclass=ABCMeta):
    """A MetaDataStore must implement each of the abstract methods defined in this interface
    """
    # region Units
    @abstractmethod
    def read_unit_definitions(self):
        """Reads custom unit definitions

        Returns
        -------
        list[str]
            Pint-compatible unit definitions
        """

    def write_unit_definitions(self, definitions):
        """Reads custom unit definitions

        Parameters
        ----------
        list[str]
            Pint-compatible unit definitions
        """
    # endregion

    # region Dimensions
    @abstractmethod
    def read_dimensions(self, skip_coords=False):
        """Read dimensions

        Parameters
        ----------
        skip_coords : bool, default False
            If True, skip reading dimension elements (names and metadata)

        Returns
        -------
        list[~smif.metadata.coords.Coords]
        """

    @abstractmethod
    def read_dimension(self, dimension_name, skip_coords=False):
        """Return dimension

        Parameters
        ----------
        dimension_name : str
        skip_coords : bool, default False
            If True, skip reading dimension elements (names and metadata)

        Returns
        -------
        ~smif.metadata.coords.Coords
            A dimension definition (including elements)
        """

    @abstractmethod
    def write_dimension(self, dimension):
        """Write dimension to project configuration

        Parameters
        ----------
        dimension : ~smif.metadata.coords.Coords
        """

    @abstractmethod
    def update_dimension(self, dimension_name, dimension):
        """Update dimension

        Parameters
        ----------
        dimension_name : str
        dimension : ~smif.metadata.coords.Coords
        """

    @abstractmethod
    def delete_dimension(self, dimension_name):
        """Delete dimension

        Parameters
        ----------
        dimension_name : str
        """
    # endregion
