"""Test all MetadataStore implementations
"""
from pytest import fixture, mark, param
from smif.data_layer.database_interface import DbMetadataStore
from smif.data_layer.file.file_metadata_store import FileMetadataStore
from smif.data_layer.memory_interface import MemoryMetadataStore


@fixture(
    params=[
        'memory',
        'file',
        param('database', marks=mark.skip)
    ])
def init_handler(request, setup_empty_folder_structure):
    if request.param == 'memory':
        handler = MemoryMetadataStore()
    elif request.param == 'file':
        base_folder = setup_empty_folder_structure
        handler = FileMetadataStore(base_folder)
    elif request.param == 'database':
        handler = DbMetadataStore()
        raise NotImplementedError

    return handler


@fixture
def handler(init_handler, unit_definitions, dimension, sample_dimensions):
    handler = init_handler

    # metadata
    handler.write_unit_definitions(unit_definitions)
    handler.write_dimension(dimension)
    for dim in sample_dimensions:
        handler.write_dimension(dim)

    return handler


class TestUnits():
    """Read units definitions
    """
    def test_read_units(self, handler, unit_definitions):
        expected = unit_definitions
        actual = handler.read_unit_definitions()
        assert actual == expected


class TestDimensions():
    """Read/write/update/delete dimensions
    """
    def test_read_dimensions(self, handler, dimension, sample_dimensions):
        actual = handler.read_dimensions()
        expected = [dimension] + sample_dimensions
        assert sorted_by_name(actual) == sorted_by_name(expected)

    def test_read_dimension(self, handler, dimension):
        assert handler.read_dimension('category') == dimension

    def test_write_dimension(self, handler, dimension, sample_dimensions):
        another_dimension = {'name': '3rd', 'elements': ['a', 'b']}
        handler.write_dimension(another_dimension)
        actual = handler.read_dimensions()
        expected = [dimension, another_dimension] + sample_dimensions
        assert sorted_by_name(actual) == sorted_by_name(expected)

    def test_update_dimension(self, handler, dimension, sample_dimensions):
        another_dimension = {'name': 'category', 'elements': [4, 5, 6]}
        handler.update_dimension('category', another_dimension)
        actual = handler.read_dimensions()
        expected = [another_dimension] + sample_dimensions
        assert sorted_by_name(actual) == sorted_by_name(expected)

    def test_delete_dimension(self, handler, sample_dimensions):
        handler.delete_dimension('category')
        actual = handler.read_dimensions()
        expected = sample_dimensions
        assert sorted_by_name(actual) == sorted_by_name(expected)


def sorted_by_name(list_):
    """Helper to sort lists-of-dicts
    """
    return sorted(list_, key=lambda d: d['name'])
