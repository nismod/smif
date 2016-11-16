import os
import yaml

from pytest import fixture, raises
from smif.controller import Controller


@fixture(scope='session')
def setup_folder_structure(tmpdir_factory):
    """

    Returns
    -------
    :class:`LocalPath`
        Path to the temporary folder
    """
    folder_list = ['config', 'planning', 'models']
    test_folder = tmpdir_factory.mktemp("smif")

    for folder in folder_list:
        test_folder.mkdir(folder)

    return test_folder


@fixture(scope='function')
def setup_assets_file(setup_folder_structure):
    """Assets are associated with sector models, not the integration config

    """
    base_folder = setup_folder_structure
    filename = base_folder.join('models',
                                'water_supply',
                                'assets',
                                'assets_1.yaml')
    assets_contents = ['water_asset_a',
                       'water_asset_b',
                       'water_asset_c']
    contents = yaml.dump(assets_contents)
    filename.write(contents, ensure=True)


@fixture(scope='function')
def setup_config_file(setup_folder_structure):
    """Configuration file contains entries for sector models, timesteps and
    planning
    """
    ps_name = 'pre-specified.yaml'
    file_contents = {'sector_models': ['water_supply'],
                     'timesteps': 'timesteps.yaml',
                     'assets': ['assets1.yaml'],
                     'planning': {'rule_based': {'use': False},
                                  'optimisation': {'use': False},
                                  'pre_specified': {'use': True,
                                                    'files': [ps_name]}
                                  }
                     }

    contents = yaml.dump(file_contents)
    filepath = setup_folder_structure.join('config', 'model.yaml')
    filepath.write(contents)


@fixture(scope='function')
def setup_runpy_file(tmpdir, setup_folder_structure):
    """The run.py model should contain an instance of SectorModel which wraps
    the sector model and allows it to be run.
    """
    base_folder = setup_folder_structure
    # Write a run.py file for the water_supply model
    filename = base_folder.join('models',
                                'water_supply',
                                'run.py')
    contents = """from unittest.mock import MagicMock
import time

if __name__ == '__main__':
    class Model():
        pass
    model = Model()
    model.simulate = MagicMock(return_value=3)
    model.simulate()
    time.sleep(1) # delays for 1 seconds
"""
    filename.write(contents, ensure=True)


@fixture(scope='function')
def setup_project_folder(setup_runpy_file,
                         setup_assets_file,
                         setup_folder_structure,
                         setup_config_file,
                         setup_timesteps_file):
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
    base_folder = setup_folder_structure

    return base_folder


@fixture(scope='function')
def setup_timesteps_file(setup_folder_structure):
    base_folder = setup_folder_structure
    filename = base_folder.join('config', 'timesteps.yaml')
    timesteps_contents = [2010, 2011, 2012]
    contents = yaml.dump(timesteps_contents)
    filename.write(contents, ensure=True)


class TestController():

    def test_model_list(self, setup_project_folder):

        cont = Controller(str(setup_project_folder))

        expected = ['water_supply']
        actual = cont.model_list
        assert actual == expected

    def test_timesteps(self, setup_project_folder):

        cont = Controller(str(setup_project_folder))

        expected = [2010, 2011, 2012]
        actual = cont.timesteps
        assert actual == expected

    def test_assets(self, setup_project_folder):

        cont = Controller(str(setup_project_folder))

        expected = ['water_asset_a', 'water_asset_b', 'water_asset_c']
        actual = cont.all_assets
        assert actual == expected


class TestRunModel():

    def test_run_sector_model(self, setup_project_folder):
        cont = Controller(str(setup_project_folder))
        assert os.path.exists(os.path.join(str(setup_project_folder),
                                           'models',
                                           'water_supply',
                                           'run.py'))

        cont.run_sector_model('water_supply')

    def test_invalid_sector_model(self, setup_project_folder):
        cont = Controller(str(setup_project_folder))
        with raises(AssertionError):
            cont.run_sector_model('invalid_sector_model')
