import numpy as np
from unittest.mock import patch
from smif.convert.unit import parse_unit
from smif.convert import UnitConvertor


def test_parse_unit_valid():
    """Parse a valid unit
    """
    meter = parse_unit('m')
    assert str(meter) == 'meter'


@patch('smif.convert.unit.LOGGER.warning')
def test_parse_unit_invalid(warning_logger):
    """Warn if unit not recognised
    """
    unit = 'unrecognisable'
    parse_unit(unit)
    msg = "Unrecognised unit: %s"
    warning_logger.assert_called_with(msg, unit)


def test_convert_unit():

    data = np.array([[1, 2], [3, 4]], dtype=float)
    
    convertor = UnitConvertor()
    actual = convertor.convert(data, 'liter', 'milliliter')
    
    expected = np.array([[1000, 2000], [3000, 4000]], dtype=float)

    np.allclose(actual, expected) 
