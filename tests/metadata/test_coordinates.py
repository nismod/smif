"""Test Coordinates metadata
"""
from collections import OrderedDict

from pytest import mark, raises
from smif.metadata import Coordinates


class CustomMapping():
    """Custom mapping that doesn't inherit dict
    """
    def __init__(self, key_values):
        self._data = {}
        for key, value in key_values:
            self._data[key] = value

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)


class TestCoordinates():
    """Coordinates has name and elements

    Equivalent to an Element Set (OpenMI)
    Can be passed as coords to DataSpec or DataArray
    """
    def test_construct_with_element_names(self):
        """Create a Coordinates with name and elements (list of ids)
        """
        name = 'building_categories'
        element_ids = ['residential', 'commercial', 'industrial']

        building_sectors = Coordinates(name, element_ids)

        assert building_sectors.name == name
        assert building_sectors.ids == element_ids
        assert building_sectors.names == element_ids
        assert building_sectors.elements == [
            {'name': 'residential'},
            {'name': 'commercial'},
            {'name': 'industrial'}
        ]

    @mark.parametrize("elements", [
        [
            {'name': 'residential', 'sector': True},
            {'name': 'commercial', 'sector': True},
            {'name': 'industrial', 'sector': True},
        ],
        [
            OrderedDict([('name', 'residential'), ('sector', True)]),
            OrderedDict([('name', 'commercial'), ('sector', True)]),
            OrderedDict([('name', 'industrial'), ('sector', True)]),
        ],
        [
            CustomMapping([('name', 'residential'), ('sector', True)]),
            CustomMapping([('name', 'commercial'), ('sector', True)]),
            CustomMapping([('name', 'industrial'), ('sector', True)]),
        ]
    ])
    def test_construct_with_elements(self, elements):
        """Create a Coordinates with name and elements (list of dicts)
        """
        name = 'building_categories'

        building_sectors = Coordinates(name, elements)

        assert building_sectors.name == name
        assert building_sectors.ids == ['residential', 'commercial', 'industrial']
        assert building_sectors.names == ['residential', 'commercial', 'industrial']
        assert building_sectors.elements == elements

    def test_name_dim_alias(self):
        """Name and dim refer to the same label
        """
        coords = Coordinates('name', ['test'])
        assert coords.name == 'name'
        assert coords.dim == 'name'

        coords.name = 'name2'
        assert coords.name == 'name2'
        assert coords.dim == 'name2'

        coords.dim = 'name3'
        assert coords.name == 'name3'
        assert coords.dim == 'name3'

    def test_coordinates_must_have_elements(self):
        """A Coordinates must have one or more elements
        """
        with raises(ValueError) as ex:
            Coordinates('zero_d', [])
        assert "must not be empty" in str(ex)

    def test_elements_must_have_name(self):
        """Coordinates elements must have "name"
        """
        elements = [
            {"description": "Petrol", "state": "liquid"},
            {"description": "Diesel", "state": "liquid"},
            {"description": "Coal", "state": "solid"},
        ]
        with raises(KeyError) as ex:
            Coordinates('fossil_fuels', elements)
        assert "must have a name" in str(ex)

    def test_elements_must_be_finite(self):
        """Only accept finite Coordinatess
        """
        def natural_numbers():
            i = 0
            while True:
                yield i
                i += 1

        elements = natural_numbers()

        with raises(ValueError) as ex:
            Coordinates('natural_numbers', elements)
        assert "must be finite" in str(ex)

    def test_eq(self):
        """Equality based on equivalent name and elements
        """
        a = Coordinates('name', [1, 2, 3])
        b = Coordinates('name', [
            {'name': 1},
            {'name': 2},
            {'name': 3}
        ])
        c = Coordinates('another', [1, 2, 3])
        d = Coordinates('name', [2, 3, 4])
        e = Coordinates('name', [
            {'name': 1, 'note': 'meta'},
            {'name': 2, 'note': 'meta'},
            {'name': 3, 'note': 'meta'}
        ])
        assert a == b
        assert a != c
        assert a != d
        assert a != e
