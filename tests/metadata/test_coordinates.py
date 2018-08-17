"""Test Coordinates metadata
"""
from pytest import raises
from smif.metadata import Coordinates


class TestCoordinates():
    """Coordinates has name and elements

    Equivalent to an Element Set (OpenMI)
    Can be passed as coords to DataSpec or DataArray
    """
    def test_construct_with_element_ids(self):
        """Create a Coordinates with name and elements (list of ids)
        """
        name = 'building_categories'
        element_ids = ['residential', 'commercial', 'industrial']

        building_sectors = Coordinates(name, element_ids)

        assert building_sectors.name == name
        assert building_sectors.ids == element_ids
        assert building_sectors.elements == [
            {'id': 'residential'},
            {'id': 'commercial'},
            {'id': 'industrial'}
        ]

    def test_construct_with_elements(self):
        """Create a Coordinates with name and elements (list of dicts)
        """
        name = 'building_categories'
        elements = [
            {'id': 'residential', 'sector': True},
            {'id': 'commercial', 'sector': True},
            {'id': 'industrial', 'sector': True}
        ]

        building_sectors = Coordinates(name, elements)

        assert building_sectors.name == name
        assert building_sectors.ids == ['residential', 'commercial', 'industrial']
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

    def test_elements_must_have_id(self):
        """Coordinates elements must have "id"
        """
        elements = [
            {"name": "Petrol", "state": "liquid"},
            {"name": "Diesel", "state": "liquid"},
            {"name": "Coal", "state": "solid"},
        ]
        with raises(KeyError) as ex:
            Coordinates('fossil_fuels', elements)
        assert "must have an id" in str(ex)

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
            {'id': 1},
            {'id': 2},
            {'id': 3}
        ])
        c = Coordinates('another', [1, 2, 3])
        d = Coordinates('name', [2, 3, 4])
        e = Coordinates('name', [
            {'id': 1, 'note': 'meta'},
            {'id': 2, 'note': 'meta'},
            {'id': 3, 'note': 'meta'}
        ])
        assert a == b
        assert a != c
        assert a != d
        assert a != e