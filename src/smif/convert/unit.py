"""Handles conversion between units used in the `SosModel`

First implementation delegates to pint.
"""
import logging

from pint import DimensionalityError, UndefinedUnitError, UnitRegistry
from smif.convert.register import Register


class UnitRegister(Register):

    def __init__(self):
        self._register = UnitRegistry(on_redefinition='raise')
        self.LOGGER = logging.getLogger()
        self.names = []

    @property
    def names(self):
        return list(self._register)

    def register(self, unit):
        pass

    def get_entry(self, name):
        pass

    def convert(self, data, from_unit, to_unit):
        """Convert the data from one unit to another unit

        Parameters
        ----------
        data: numpy.ndarray
            An array of values with dimensions regions x intervals
        from_unit: str
            The name of the unit of the data
        to_unit: str
            The name of the required unit

        Returns
        -------
        numpy.ndarray
            An array of data with dimensions regions x intervals

        Raises
        ------
        ValueError
            If the units are not in the unit register or conversion is not possible
        """
        try:
            Q_ = self._register.Quantity(data, from_unit)
        except UndefinedUnitError:
            raise ValueError('Cannot convert from undefined unit ' + from_unit)

        try:
            result = Q_.to(to_unit).magnitude
        except UndefinedUnitError:
            raise ValueError('Cannot convert to undefined unit ' + to_unit)
        except DimensionalityError:
            raise ValueError('Cannot convert from ' + from_unit + ' to ' + to_unit)

        return result

    def parse_unit(self, unit_string):
        """Parse a unit string (abbreviation or full) into a Unit object

        Parameters
        ----------
        unit : str

        Returns
        -------
        quantity : :class:`pint.Unit`
        """
        try:
            unit = self._register.parse_units(unit_string)
        except UndefinedUnitError:
            self.LOGGER.warning("Unrecognised unit: %s", unit_string)
            unit = None
        return unit


def get_register():
    """Returns a reference to the unit registry
    """
    __UNIT_REGISTRY = UnitRegister()
    return __UNIT_REGISTRY
