"""Test ElementSet metadata
"""
from pytest import raises

# from smif.metadata import ElementSet

class TestElementSet():
    """A ElementSet has name and elements

    Equivalent to an Element set (OpenMI)
    """
    def test_construct_with_elements(self):
        """Create a ElementSet with name and elements
        """
        name = 'building_categories'
        element_ids = ['residential', 'commercial', 'industrial']

        building_sectors = ElementSet(name, element_ids)

        assert building_sectors.name == name
        assert building_sectors.ids == element_ids

    def test_ElementSet_must_have_elements(self):
        """A ElementSet must have one or more elements
        """
        with raises(ValueError) as ex:
            ElementSet('zero_d', [])

    def test_elements_must_have_id(self):
        """ElementSet elements must have "id"
        """
        elements = [
            {"name": "Petrol", "state": "liquid"},
            {"name": "Diesel", "state": "liquid"},
            {"name": "Coal", "state": "solid"},
        ]
        with raises(KeyError) as ex:
            ElementSet('fossil_fuels', elements)

    def test_elements_must_be_finite(self):
        """Only accept finite ElementSets
        """
        def natural_numbers():
            i = 0
            while True:
                yield i
                i += 1

        elements = natural_numbers()

        with raises(ValueError) as ex:
            ElementSet('natural_numbers', elements)
