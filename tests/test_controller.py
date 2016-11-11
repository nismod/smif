import os
from tempfile import TemporaryDirectory

import yaml

from pytest import fixture
from smif.controller import Controller


@fixture(scope='function')
def setup_project_folder():
    """Sets up a temporary folder with the required project folder structure

        /assets/
        /assets/assets1.yaml
        /config/
        /config/model.yaml
        /config/timesteps.yaml
        /planning/
        /planning/pre-specified.yaml

    """

    project_folder = TemporaryDirectory()
    folder_list = ['assets', 'config', 'planning']
    for folder in folder_list:
        os.mkdir(os.path.join(project_folder.name, folder))

    filename = os.path.join(project_folder.name, 'config', 'model.yaml')

    file_contents = {'sector_models': ['water_supply'],
                     'timesteps': ['timesteps.yaml'],
                     'assets': ['assets1.yaml'],
                     'planning': {'rule_based': {'use': False,
                                                 'files': None},
                                  'optimisation': {'use': False,
                                                   'files': None},
                                  'pre_spec': {'use': True,
                                               'files': ['pre-specified.yaml']
                                               }
                                  }
                     }

    with open(filename, 'w') as config_file:
        yaml.dump(file_contents, config_file)

    filename = os.path.join(project_folder.name, 'config', 'timesteps.yaml')
    with open(filename, 'w') as config_file:
        timesteps_contents = [2010, 2011, 2012]
        yaml.dump(timesteps_contents, config_file)

    filename = os.path.join(project_folder.name, 'assets', 'assets1.yaml')
    with open(filename, 'w') as config_file:
        timesteps_contents = ['asset_a', 'asset_b', 'asset_c']
        yaml.dump(timesteps_contents, config_file)

    return project_folder


class TestController():

    def test_model_list(self, setup_project_folder):

        with setup_project_folder as folder_name:
            cont = Controller(folder_name)

        expected = ['water_supply']
        actual = cont.model_list
        assert actual == expected

    def test_timesteps(self, setup_project_folder):

        with setup_project_folder as folder_name:
            cont = Controller(folder_name)

        expected = [2010, 2011, 2012]
        actual = cont._timesteps
        assert actual == expected

    def test_assets(self, setup_project_folder):

        with setup_project_folder as folder_name:
            cont = Controller(folder_name)

        expected = ['asset_a', 'asset_b', 'asset_c']
        actual = cont._assets
        assert actual == expected
