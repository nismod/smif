"""Test aggregation/disaggregation of data between sets of areas
"""
from copy import copy
import numpy as np
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
        assert rreg.names == []

        with raises(ValueError) as ex:
            rreg.get_entry('nonexistent')
        assert "Region set 'nonexistent' not registered" in str(ex)

    def test_convert_equal(self, regions_rect):
        rreg = RegionRegister()
        # register rect
        rreg.register(regions_rect)
        # register alt rect (same area)
        regions_rect_alt = copy(regions_rect)
        regions_rect_alt.name = 'rect_alt'
        rreg.register(regions_rect_alt)

        data = np.ones(1)
        converted = rreg.convert(data, 'rect', 'rect_alt')
        np.testing.assert_equal(data, converted)

    def test_convert_to_half(self, regions_rect, regions_half_squares):
        rreg = RegionRegister()
        rreg.register(regions_rect)
        rreg.register(regions_half_squares)

        data = np.ones(1)
        converted = rreg.convert(data, 'rect', 'half_squares')
        expected = np.array([0.5, 0.5])
        np.testing.assert_equal(converted, expected)

    def test_convert_from_half(self, regions_rect, regions_half_squares):
        rreg = RegionRegister()
        rreg.register(regions_rect)
        rreg.register(regions_half_squares)

        data = np.ones(2) / 2
        converted = rreg.convert(data, 'half_squares', 'rect')
        expected = np.ones(1)
        np.testing.assert_equal(converted, expected)

        data = np.array([2, 3])
        converted = rreg.convert(data, 'half_squares', 'rect')
        expected = np.array([5])
        np.testing.assert_equal(converted, expected)

    def test_convert_to_half_not_covered(self, regions_rect, regions_single_half_square):
        rreg = RegionRegister()
        rreg.register(regions_rect)
        rreg.register(regions_single_half_square)

        data = np.array([3])
        converted = rreg.convert(data, 'rect', 'single_half_square')
        expected = np.array([1.5])
        np.testing.assert_equal(converted, expected)

    def test_convert_from_half_not_covered(self, regions_rect, regions_single_half_square):
        rreg = RegionRegister()
        rreg.register(regions_rect)
        rreg.register(regions_single_half_square)

        data = np.array([3])
        converted = rreg.convert(data, 'single_half_square', 'rect')
        expected = np.array([3])
        np.testing.assert_equal(converted, expected)

    def test_convert_square_to_triangle(self, regions_half_squares, regions_half_triangles):
        rreg = RegionRegister()
        rreg.register(regions_half_squares)
        rreg.register(regions_half_triangles)

        data = np.array([1, 1])
        converted = rreg.convert(data, 'half_squares', 'half_triangles')
        expected = np.array([1, 1])
        np.testing.assert_equal(converted, expected)

        data = np.array([0, 1])
        converted = rreg.convert(data, 'half_squares', 'half_triangles')
        expected = np.array([0.25, 0.75])
        np.testing.assert_equal(converted, expected)

    def test_convert_triangle_to_square(self, regions_half_squares, regions_half_triangles):
        rreg = RegionRegister()
        rreg.register(regions_half_squares)
        rreg.register(regions_half_triangles)

        data = np.array([1, 1])
        converted = rreg.convert(data, 'half_triangles', 'half_squares')
        expected = np.array([1, 1])
        np.testing.assert_equal(converted, expected)

        data = np.array([0.25, 0.75])
        converted = rreg.convert(data, 'half_triangles', 'half_squares')
        expected = np.array([0.375, 0.625])
        np.testing.assert_equal(converted, expected)
