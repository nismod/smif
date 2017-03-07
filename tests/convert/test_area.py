"""Test aggregation/disaggregation of data between sets of areas
"""
from pytest import fixture
from shapely.geometry import shape
from smif.convert.area import (RegionRegister, RegionSet,
                               proportion_of_a_intersecting_b)


@fixture(scope='function')
def regions():
    """Return data structure for test regions/shapes
    """
    return [
        {
            'type': 'Feature',
            'properties': {'name': 'unit'},
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[0, 0], [0, 1], [1, 1], [1, 0]]]
            }
        },
        {
            'type': 'Feature',
            'properties': {'name': 'half'},
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[0, 0], [0, 0.5], [1, 0.5], [1, 0]]]
            }
        },
        {
            'type': 'Feature',
            'properties': {'name': 'two'},
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[0, 0], [0, 2], [1, 2], [1, 0]]]
            }
        }
    ]


@fixture(scope='function')
def regions_half_squares():
    """Return two adjacent square regions::

        |```|```|
        | A | B |
        |...|...|

    """
    return RegionSet('half_squares', [
        {
            'type': 'Feature',
            'properties': {'name': 'a'},
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[0, 0], [0, 1], [1, 1], [1, 0]]]
            }
        },
        {
            'type': 'Feature',
            'properties': {'name': 'b'},
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[0, 1], [0, 2], [1, 2], [1, 1]]]
            }
        },
    ])


@fixture(scope='function')
def regions_rect():
    """Return single region covering 2x1 area::

        |```````|
        |   0   |
        |.......|

    """
    return RegionSet('rect', [
        {
            'type': 'Feature',
            'properties': {'name': 'zero'},
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[0, 0], [0, 2], [1, 2], [1, 0]]]
            }
        }
    ])


@fixture(scope='function')
def regions_half_triangles():
    """Return regions split diagonally::

        |``````/|
        | 0 / 1 |
        |/......|

    """
    return RegionSet('rect', [
        {
            'type': 'Feature',
            'properties': {'name': 'zero'},
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[0, 0], [0, 2], [2, 0]]]
            }
        },
        {
            'type': 'Feature',
            'properties': {'name': 'one'},
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[0, 2], [2, 2], [2, 0]]]
            }
        },
    ])


def test_proportion(regions):
    """Sense-check proportion calculator
    """
    unit = shape(regions[0]['geometry'])
    half = shape(regions[1]['geometry'])
    two = shape(regions[2]['geometry'])

    assert proportion_of_a_intersecting_b(unit, unit) == 1

    assert proportion_of_a_intersecting_b(unit, half) == 0.5
    assert proportion_of_a_intersecting_b(half, unit) == 1

    assert proportion_of_a_intersecting_b(unit, two) == 1
    assert proportion_of_a_intersecting_b(two, unit) == 0.5
    assert proportion_of_a_intersecting_b(half, two) == 1
    assert proportion_of_a_intersecting_b(two, half) == 0.25


class TestRegionSet():
    """Test creating, looking up, retrieving region sets
    """
    def test_create(self, regions):
        rset = RegionSet('test', regions)
        assert rset.name == 'test'
        assert len(rset.regions) == 3


class TestRegionRegister():
    """Test creating registers, registering region sets, converting data
    """
    def test_create(self):
        rreg = RegionRegister()
        assert rreg.registered_sets == []
