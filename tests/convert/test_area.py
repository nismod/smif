"""Test aggregation/disaggregation of data between sets of areas
"""
from copy import copy
from pytest import fixture, raises
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
def regions_single_half_square():
    """Return single half-size square region::

        |```|
        | A |
        |...|

    """
    return RegionSet('single_half_square', [
        {
            'type': 'Feature',
            'properties': {'name': 'a'},
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[0, 0], [0, 1], [1, 1], [1, 0]]]
            }
        }
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
    return RegionSet('half_triangles', [
        {
            'type': 'Feature',
            'properties': {'name': 'zero'},
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[0, 0], [0, 2], [1, 0]]]
            }
        },
        {
            'type': 'Feature',
            'properties': {'name': 'one'},
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[0, 2], [1, 2], [1, 0]]]
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
        assert len(rset) == 3
        assert rset[0].name == 'unit'
        assert rset[1].name == 'half'
        assert rset[2].name == 'two'


class TestRegionRegister():
    """Test creating registers, registering region sets, converting data
    """
    def test_create(self):
        rreg = RegionRegister()
        assert rreg.region_set_names == []

        with raises(ValueError) as ex:
            rreg.get_regions_in_set('nonexistent')
        assert "Region set nonexistent not registered" in str(ex)

    def test_convert_equal(self, regions_rect):
        rreg = RegionRegister()
        # register rect
        rreg.register(regions_rect)
        # register alt rect (same area)
        regions_rect_alt = copy(regions_rect)
        regions_rect_alt.name = 'rect_alt'
        rreg.register(regions_rect_alt)

        data = {'zero': 1}
        converted = rreg.convert(data, 'rect', 'rect_alt')
        assert data == converted

    def test_convert_to_half(self, regions_rect, regions_half_squares):
        rreg = RegionRegister()
        rreg.register(regions_rect)
        rreg.register(regions_half_squares)

        data = {'zero': 1}
        converted = rreg.convert(data, 'rect', 'half_squares')
        expected = {'a': 0.5, 'b': 0.5}
        assert converted == expected

    def test_convert_from_half(self, regions_rect, regions_half_squares):
        rreg = RegionRegister()
        rreg.register(regions_rect)
        rreg.register(regions_half_squares)

        data = {'a': 0.5, 'b': 0.5}
        converted = rreg.convert(data, 'half_squares', 'rect')
        expected = {'zero': 1}
        assert converted == expected

        data = {'a': 2, 'b': 3}
        converted = rreg.convert(data, 'half_squares', 'rect')
        expected = {'zero': 5}
        assert converted == expected

    def test_convert_to_half_not_covered(self, regions_rect, regions_single_half_square):
        rreg = RegionRegister()
        rreg.register(regions_rect)
        rreg.register(regions_single_half_square)

        data = {'zero': 3}
        converted = rreg.convert(data, 'rect', 'single_half_square')
        expected = {'a': 1.5}
        assert converted == expected

    def test_convert_from_half_not_covered(self, regions_rect, regions_single_half_square):
        rreg = RegionRegister()
        rreg.register(regions_rect)
        rreg.register(regions_single_half_square)

        data = {'a': 3}
        converted = rreg.convert(data, 'single_half_square', 'rect')
        expected = {'zero': 3}
        assert converted == expected

    def test_convert_square_to_triangle(self, regions_half_squares, regions_half_triangles):
        rreg = RegionRegister()
        rreg.register(regions_half_squares)
        rreg.register(regions_half_triangles)

        data = {'a': 1, 'b': 1}
        converted = rreg.convert(data, 'half_squares', 'half_triangles')
        expected = {'zero': 1, 'one': 1}
        assert converted == expected

        data = {'a': 0, 'b': 1}
        converted = rreg.convert(data, 'half_squares', 'half_triangles')
        expected = {'zero': 0.25, 'one': 0.75}
        assert converted == expected

    def test_convert_triangle_to_square(self, regions_half_squares, regions_half_triangles):
        rreg = RegionRegister()
        rreg.register(regions_half_squares)
        rreg.register(regions_half_triangles)

        data = {'zero': 1, 'one': 1}
        converted = rreg.convert(data, 'half_triangles', 'half_squares')
        expected = {'a': 1, 'b': 1}
        assert converted == expected

        data = {'zero': 0.25, 'one': 0.75}
        converted = rreg.convert(data, 'half_triangles', 'half_squares')
        expected = {'a': 0.375, 'b': 0.625}
        assert converted == expected
