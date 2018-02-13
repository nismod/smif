"""Handles conversion between units used in the `SosModel`

First implementation delegates to pint.
"""
import logging

from pint import UndefinedUnitError, UnitRegistry

LOGGER = logging.getLogger()
__UNIT_REGISTRY = UnitRegistry(on_redefinition='raise')


def parse_unit(unit_string):
    """Parse a unit string (abbreviation or full) into a Unit object

    Parameters
    ----------
    unit : str

    Returns
    -------
    quantity : :class:`pint.Unit`
    """
    try:
        unit = __UNIT_REGISTRY.parse_units(unit_string)
    except UndefinedUnitError:
        LOGGER.warning("Unrecognised unit: %s", unit_string)
        unit = None
    return unit


def get_register():
    """Returns a reference to the unit registry
    """
    return __UNIT_REGISTRY
