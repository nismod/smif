import os

from pytest import fixture
from smif.data_layer import DatafileInterface, MemoryInterface
from smif.data_layer.load import dump
from smif.metadata import Spec


@fixture(scope='function')
def get_memory_handler():
    return MemoryInterface()


@fixture(scope='function')
def get_handler_csv(setup_folder_structure, project_config):
    basefolder = setup_folder_structure
    project_config_path = os.path.join(
        str(basefolder), 'config', 'project.yml')
    dump(project_config, project_config_path)
    return DatafileInterface(str(basefolder), 'local_csv')


@fixture(scope='function')
def get_handler_binary(setup_folder_structure, project_config):
    basefolder = setup_folder_structure
    project_config_path = os.path.join(
        str(basefolder), 'config', 'project.yml')
    dump(project_config, project_config_path)
    return DatafileInterface(str(basefolder), 'local_binary')


@fixture(scope='function')
def get_remapped_scenario_data():
    """Return sample scenario_data
    """
    data = [
        {
            'value': 100,
            'units': 'people',
            'region': 'oxford',
            'interval': 'cold_month',
            'year': 2015
        },
        {
            'value': 150,
            'units': 'people',
            'region': 'oxford',
            'interval': 'spring_month',
            'year': 2015
        },
        {
            'value': 200,
            'units': 'people',
            'region': 'oxford',
            'interval': 'hot_month',
            'year': 2015
        },
        {
            'value': 210,
            'units': 'people',
            'region': 'oxford',
            'interval': 'fall_month',
            'year': 2015
        },
        {
            'value': 100,
            'units': 'people',
            'region': 'oxford',
            'interval': 'cold_month',
            'year': 2016
        },
        {
            'value': 150,
            'units': 'people',
            'region': 'oxford',
            'interval': 'spring_month',
            'year': 2016
        },
        {
            'value': 200,
            'units': 'people',
            'region': 'oxford',
            'interval': 'hot_month',
            'year': 2016
        },
        {
            'value': 200,
            'units': 'people',
            'region': 'oxford',
            'interval': 'fall_month',
            'year': 2016
        }
    ]
    spec = Spec(
        name='people',
        unit='people',
        dtype='int',
        dims=['county', 'season'],
        coords={
            'county': ['oxford'],
            'season': ['cold_month', 'spring_month', 'hot_month', 'fall_month']
        }
    )
    return data, spec
