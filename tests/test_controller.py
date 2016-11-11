import os
from tempfile import TemporaryDirectory

import yaml

from pytest import fixture
from smif.controller import Controller


@fixture(scope='function')
def setup_project_folder():
    """Sets up a temporary folder with the required project folder structure

        /models
        /models/water_supply/
        /models/water_supply/run.py
        /models/water_supply/assets/assets1.yaml
        /config/
        /config/model.yaml
        /config/timesteps.yaml
        /planning/
        /planning/pre-specified.yaml

    """

    project_folder = TemporaryDirectory()
    folder_list = ['config', 'planning', 'models']
    for folder in folder_list:
        os.mkdir(os.path.join(project_folder.name, folder))
    os.mkdir(os.path.join(project_folder.name, 'models', 'water_supply'))
    os.mkdir(os.path.join(project_folder.name, 'models', 'water_supply',
                          'assets'))

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

    filename = os.path.join(project_folder.name,
                            'models',
                            'water_supply',
                            'assets',
                            'assets1.yaml')
    with open(filename, 'w') as config_file:
        timesteps_contents = ['water_asset_a',
                              'water_asset_b',
                              'water_asset_c']
        yaml.dump(timesteps_contents, config_file)

    # Write a run.py file for the water_supply model
    filename = os.path.join(project_folder.name,
                            'models',
                            'water_supply',
                            'run.py')
    with open(filename, 'w') as run_file:
        contents = """from unittest.mock import MagicMock
from smif.sectormodel import SectorModel
import time

if __name__ == '__main__':
    model = SectorModel('water_supply')
    model.simulate = MagicMock(return_value=3)
    model.simulate()
    time.sleep(1) # delays for 1 seconds
"""
        for line in contents:
            run_file.write(line)

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

        expected = ['water_asset_a', 'water_asset_b', 'water_asset_c']
        actual = cont._all_assets
        assert actual == expected


class TestRunModel():

    def test_run_sector_model(self, setup_project_folder):
        with setup_project_folder as folder_name:
            cont = Controller(folder_name)

            assert os.path.exists(os.path.join(folder_name,
                                               'models',
                                               'water_supply',
                                               'run.py'))

            cont.run_sector_model('water_supply')
