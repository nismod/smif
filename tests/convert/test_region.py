"""Test aggregation/disaggregation of data between sets of areas
"""
from copy import copy
from unittest.mock import Mock

import numpy as np
from pytest import fixture, raises
from smif.convert.region import RegionAdaptor, RegionSet
from smif.convert.register import NDimensionalRegister
from smif.metadata import Spec


@fixture(scope='function')
def register(regions_half_triangles, regions_half_squares, regions_single_half_square,
             regions_rect):
    """Region register with regions pre-registered
    """
    register = NDimensionalRegister()
    register.register(RegionSet('half_triangles', regions_half_triangles))
    register.register(RegionSet('half_squares', regions_half_squares))
    register.register(RegionSet('single_half_square', regions_single_half_square))
    register.register(RegionSet('rect', regions_rect))
    alt = copy(regions_rect)
    register.register(RegionSet('rect_alt', alt))
    return register


class TestRegionAdaptor:
    """Converting between region sets, assuming uniform distribution as necessary
    """
    def test_aggregate_region(self, regions_rect, regions_half_squares):
        """Two regions aggregated to one, one interval
        """
        adaptor = RegionAdaptor('test-square-half')
        from_spec = Spec(
            name='test-var',
            dtype='float',
            dims=['half_squares'],
            coords={
                'half_squares': [
                    {'name': f['properties']['name'], 'feature': f}
                    for f in regions_half_squares
                ]
            }
        )
        adaptor.add_input(from_spec)
        to_spec = Spec(
            name='test-var',
            dtype='float',
            dims=['rect'],
            coords={
                'rect': [
                    {'name': f['properties']['name'], 'feature': f}
                    for f in regions_rect
                ]
            }
        )
        adaptor.add_output(to_spec)

        actual_coefficients = adaptor.generate_coefficients(from_spec, to_spec)
        expected = np.ones((2, 1))  # aggregating coefficients
        np.testing.assert_allclose(actual_coefficients, expected, rtol=1e-3)

        data = np.array([24, 24])  # area a,b
        data_handle = Mock()
        data_handle.get_data = Mock(return_value=data)
        data_handle.read_coefficients = Mock(return_value=actual_coefficients)
        adaptor.simulate(data_handle)

        actual = data_handle.set_results.call_args[0][1]
        expected = np.array([48])  # area zero
        assert np.allclose(actual, expected)

    def test_half_to_one_region_pass_through_time(self, regions_rect, regions_half_squares):
        """Convert from half a region to one region, pass through time
        """
        adaptor = RegionAdaptor('test-square-half')
        from_spec = Spec(
            name='test-var',
            dtype='float',
            dims=['rect', 'months'],
            coords={
                'rect': [
                    {'name': f['properties']['name'], 'feature': f}
                    for f in regions_rect
                ],
                'months': list(range(12))
            }
        )
        adaptor.add_input(from_spec)
        to_spec = Spec(
            name='test-var',
            dtype='float',
            dims=['half_squares', 'months'],
            coords={
                'half_squares': [
                    {'name': f['properties']['name'], 'feature': f}
                    for f in regions_half_squares
                ],
                'months': list(range(12))
            }
        )
        adaptor.add_output(to_spec)

        actual_coefficients = adaptor.generate_coefficients(from_spec, to_spec)
        expected = np.ones((1, 2)) / 2  # disaggregating coefficients
        np.testing.assert_allclose(actual_coefficients, expected, rtol=1e-3)

        data = np.ones((1, 12))  # area zero, months 1-12
        data_handle = Mock()
        data_handle.get_data = Mock(return_value=data)
        data_handle.read_coefficients = Mock(return_value=actual_coefficients)
        adaptor.simulate(data_handle)

        actual = data_handle.set_results.call_args[0][1]
        expected = np.ones((2, 12)) / 2  # areas a-b, months 1-12
        assert np.allclose(actual, expected)


def test_proportion(regions):
    """Sense-check proportion calculator
    """
    region_set = RegionSet('regions', regions)

    assert region_set.get_proportion(0, region_set[0]) == 1

    assert region_set.get_proportion(0, region_set[1]) == 0.5
    assert region_set.get_proportion(1, region_set[0]) == 1

    assert region_set.get_proportion(0, region_set[2]) == 1
    assert region_set.get_proportion(2, region_set[0]) == 0.5
    assert region_set.get_proportion(1, region_set[2]) == 1
    assert region_set.get_proportion(2, region_set[1]) == 0.25


class TestRegionSet():
    """Test creating, looking up, retrieving region sets
    """

    def test_serialise(self, regions):
        rset = RegionSet('test', regions)
        rset.description = 'my description'
        actual = rset.as_dict()
        expected = {'name': 'test',
                    'description': 'my description'}

        assert actual == expected

    def test_create(self, regions):
        rset = RegionSet('test', regions)
        assert rset.name == 'test'
        assert len(rset) == 3
        assert rset[0].name == 'unit'
        assert rset[1].name == 'half'
        assert rset[2].name == 'two'

    def test_get_names(self, regions):
        rset = RegionSet('test', regions)
        actual = rset.get_entry_names()
        expected = ['unit', 'half', 'two']
        assert actual == expected

    def test_as_features(self, regions_half_squares):
        """Retrieve regions as feature dicts
        """
        rset = RegionSet('test', regions_half_squares)
        actual = rset.as_features()
        expected = [
            {
                'type': 'Feature',
                'properties': {'name': 'a'},
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': (
                        ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)),
                    )
                }
            },
            {
                'type': 'Feature',
                'properties': {'name': 'b'},
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': (
                        ((0.0, 1.0), (0.0, 2.0), (1.0, 2.0), (1.0, 1.0), (0.0, 1.0)),
                    )
                }
            }
        ]
        assert actual == expected

    def test_centroids_as_features(self, regions_half_squares):
        """Retrieve centroids
        """
        actual = RegionSet('test', regions_half_squares).centroids_as_features()
        expected = [
            {
                'type': 'Feature',
                'properties': {'name': 'a'},
                'geometry': {
                    'type': 'Point',
                    'coordinates': (0.5, 0.5)
                }
            },
            {
                'type': 'Feature',
                'properties': {'name': 'b'},
                'geometry': {
                    'type': 'Point',
                    'coordinates': (0.5, 1.5)
                }
            },
        ]
        assert actual == expected

    def test_must_have_unique_names(self):
        with raises(AssertionError) as ex:
            RegionSet('test', [
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
                    'properties': {'name': 'a'},
                    'geometry': {
                        'type': 'Polygon',
                        'coordinates': [[[0, 1], [0, 2], [1, 2], [1, 1]]]
                    }
                }
            ])

        assert 'Region set must have uniquely named regions' in str(ex)


class TestRegionRegister():
    """Test creating registers, registering region sets, converting data
    """
    def test_create(self):
        register = NDimensionalRegister()
        assert register.names == []

        with raises(ValueError) as ex:
            register.get_entry('nonexistent')
        assert "ResolutionSet 'nonexistent' not registered" in str(ex)

    def test_convert_equal(self, register):
        data = np.ones(1)
        converted = register.convert(data, 'rect', 'rect_alt')
        np.testing.assert_equal(data, converted)

    def test_convert_to_half(self, register):
        data = np.ones(1)
        converted = register.convert(data, 'rect', 'half_squares')
        expected = np.array([0.5, 0.5])
        np.testing.assert_equal(converted, expected)

    def test_convert_from_half(self, register):
        data = np.ones(2) / 2
        converted = register.convert(data, 'half_squares', 'rect')
        expected = np.ones(1)
        np.testing.assert_equal(converted, expected)

        data = np.array([2, 3])
        converted = register.convert(data, 'half_squares', 'rect')
        expected = np.array([5])
        np.testing.assert_equal(converted, expected)

    def test_coverage_half(self, register):
        half_covered = register.get_entry('single_half_square')
        actual = half_covered.coverage
        expected = 1.0
        assert actual == expected

    def test_coverage_whole(self, register):
        rect = register.get_entry('rect')
        actual = rect.coverage
        expected = 2.0
        assert actual == expected

    def test_convert_to_half_not_covered(self, register, caplog):
        data = np.array([3])
        register.convert(data, 'rect', 'single_half_square')
        expected = "Coverage for 'rect' is 2 and does not match " \
                   "coverage for 'single_half_square' which is 1"

        assert expected in caplog.text

    def test_convert_from_half_not_covered(self, register, caplog):
        data = np.array([3])
        register.convert(data, 'single_half_square', 'rect')

        expected = "Coverage for 'single_half_square' is 1 and does not " \
                   "match coverage for 'rect' which is 2"

        assert expected in caplog.text

    def test_convert_square_to_triangle(self, register):
        data = np.array([1, 1])
        converted = register.convert(data, 'half_squares', 'half_triangles')
        expected = np.array([1, 1])
        np.testing.assert_equal(converted, expected)

        data = np.array([0, 1])
        converted = register.convert(data, 'half_squares', 'half_triangles')
        expected = np.array([0.25, 0.75])
        np.testing.assert_equal(converted, expected)

    def test_convert_triangle_to_square(self, register):
        data = np.array([1, 1])
        converted = register.convert(data, 'half_triangles', 'half_squares')
        expected = np.array([1, 1])
        np.testing.assert_equal(converted, expected)

        data = np.array([0.25, 0.75])
        converted = register.convert(data, 'half_triangles', 'half_squares')
        expected = np.array([0.375, 0.625])
        np.testing.assert_equal(converted, expected)


class TestGetCoefficients:

    def test_get_coefficients(self, register):
        actual = register.get_coefficients('half_triangles', 'half_squares')
        expected = np.array([[0.75, 0.25],
                             [0.25, 0.75]])
        np.testing.assert_equal(actual, expected)

    def test_coefs_should_map_to_themselves(self, register):
        actual = register.get_coefficients('half_triangles', 'half_triangles')
        expected = np.array([[1, 0],
                             [0, 1]])
        np.testing.assert_equal(actual, expected)
