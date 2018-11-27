"""Test file metadata store
"""
from pytest import fixture
from smif.data_layer.datafile_interface import FileMetadataStore


@fixture(scope='function')
def config_handler(setup_folder_structure, sample_dimensions):
    handler = FileMetadataStore(str(setup_folder_structure))
    for dimension in sample_dimensions:
        handler.write_dimension(dimension)
    return handler
