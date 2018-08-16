from unittest.mock import Mock

import numpy as np
from smif.convert.unit import UnitRegister


def test_register_user_unit(setup_folder_structure):
    """Test that we can load in user defined units
    """
    test_folder = setup_folder_structure
    units_file = test_folder.join('data', 'user_units.txt')

    register = UnitRegister()
    register.register(str(units_file))

    assert register.parse_unit('people') == 'people'


def test_parse_unit_valid():
    """Parse a valid unit
    """
    register = UnitRegister()
    meter = register.parse_unit('m')
    assert str(meter) == 'meter'


def test_parse_unit_invalid():
    """Warn if unit not recognised
    """
    unit = 'unrecognisable'
    register = UnitRegister()
    register.logger.warning = Mock()
    register.parse_unit(unit)
    msg = "Unrecognised unit: %s"
    register.logger.warning.assert_called_with(msg, unit)


def test_convert_unit():

    data = np.array([[1, 2], [3, 4]], dtype=float)

    register = UnitRegister()
    actual = register.convert(data, 'liter', 'milliliter')

    expected = np.array([[1000, 2000], [3000, 4000]], dtype=float)

    np.allclose(actual, expected)
