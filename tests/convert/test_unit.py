from unittest.mock import Mock, patch

import numpy as np
from smif.convert import SpaceTimeUnitConvertor
from smif.convert.unit import get_register as get_unit_register


def test_parse_unit_valid():
    """Parse a valid unit
    """
    register = get_unit_register()
    meter = register.parse_unit('m')
    assert str(meter) == 'meter'


@patch('smif.convert.unit.LOGGER.warning')
def test_parse_unit_invalid(warning_logger):
    """Warn if unit not recognised
    """
    unit = 'unrecognisable'
    register = get_unit_register()
    register.parse_unit(unit)
    msg = "Unrecognised unit: %s"
    warning_logger.assert_called_with(msg, unit)


def test_convert_unit():

    data = np.array([[1, 2], [3, 4]], dtype=float)

    convertor = SpaceTimeUnitConvertor()
    actual = convertor.convert(data, Mock, Mock, Mock, Mock, 'liter', 'milliliter')

    expected = np.array([[1000, 2000], [3000, 4000]], dtype=float)

    np.allclose(actual, expected)
