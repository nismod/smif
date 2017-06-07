from unittest.mock import patch
from smif.convert.unit import parse_unit


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
