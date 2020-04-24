"""Handles conversion between units used in the `SosModel`
"""
from pint import (DimensionalityError, UndefinedUnitError,  # type: ignore
                  UnitRegistry)
from smif.convert.adaptor import Adaptor
from smif.data_layer.data_handle import DataHandle


class UnitAdaptor(Adaptor):
    """Scalar conversion of units
    """
    def __init__(self, name):
        self._register = UnitRegistry()
        super().__init__(name)

    def simulate(self, data_handle: DataHandle):
        """Register unit definitions in registry for model run
        """
        units = data_handle.read_unit_definitions()
        for unit in units:
            self._register.define(unit)
        super().simulate(data_handle)

    def convert(self, data_array, to_spec, coefficients):
        data = data_array.data
        from_spec = data_array.spec

        try:
            quantity = self._register.Quantity(data, from_spec.unit)
        except UndefinedUnitError:
            raise ValueError('Cannot convert from undefined unit {}'.format(from_spec.unit))

        try:
            converted_quantity = quantity.to(to_spec.unit)
        except UndefinedUnitError as ex:
            raise ValueError('Cannot convert undefined unit {}'.format(to_spec.unit)) from ex
        except DimensionalityError as ex:
            msg = 'Cannot convert unit from {} to {}'
            raise ValueError(msg.format(from_spec.unit, to_spec.unit)) from ex

        return converted_quantity.magnitude

    def get_coefficients(self, data_handle, from_spec, to_spec):
        # override with no-op - all the work is done in convert with scalar operations
        pass

    def generate_coefficients(self, from_spec, to_spec):
        # override with no-op - all the work is done in convert with scalar operations
        pass

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
            self.logger.warning("Unrecognised unit: %s", unit_string)
            unit = None
        return unit
