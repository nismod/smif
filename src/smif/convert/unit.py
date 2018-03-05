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
        self.axis = None

    @property
    def names(self):
        return list(self._register.__dict__['_units'].keys())

    def register(self, unit_file):
        """Load unit definitions into the registry
        """
        self._register.load_definitions(unit_file)
        self.LOGGER.info("Finished registering user defined units")

        with open(unit_file, 'r') as readonlyfile:
            self.LOGGER.info("Imported user units:")
            for line in readonlyfile:
                self.LOGGER.info("    %s", line.split('=')[0])

    def get_entry(self, name):
        pass

    def get_coefficients(self, source, destination):
        return self.convert_old(1, source, destination)

    def convert_old(self, data, from_unit, to_unit):
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


__REGISTER = UnitRegister()


def get_register():
    """Returns a reference to the unit registry
    """
    return __REGISTER
