import os

from pytest import fixture
from smif.data_layer import DatafileInterface
from smif.data_layer.load import dump


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
