#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Dummy conftest.py for smif.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    https://pytest.org/latest/plugins.html
"""
from __future__ import absolute_import, division, print_function

import logging

import yaml

import pytest

_log_format = '%(asctime)s %(name)-12s: %(levelname)-8s %(message)s'
logging.basicConfig(filename='test_logs.log',
                    level=logging.DEBUG,
                    format=_log_format,
                    filemode='w')


@pytest.fixture(scope='function')
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


@pytest.fixture(scope='function')
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


@pytest.fixture(scope='function')
def setup_assets_file_two(setup_folder_structure):
    """Assets are associated with sector models, not the integration config


    Defines a second assets file
    """
    base_folder = setup_folder_structure
    filename = base_folder.join('models',
                                'water_supply',
                                'assets',
                                'assets_2.yaml')
    assets_contents = ['water_asset_d']
    contents = yaml.dump(assets_contents)
    filename.write(contents, ensure=True)


@pytest.fixture(scope='function')
def setup_config_file_two(setup_folder_structure):
    """Configuration file contains entries for sector models, timesteps and
    planning

    Contains two asset files
    """
    ps_name = 'pre-specified.yaml'
    file_contents = {'sector_models': ['water_supply'],
                     'timesteps': 'timesteps.yaml',
                     'assets': ['assets1.yaml', 'assets2.yaml'],
                     'planning': {'rule_based': {'use': False},
                                  'optimisation': {'use': False},
                                  'pre_specified': {'use': True,
                                                    'files': [ps_name]}
                                  }
                     }

    contents = yaml.dump(file_contents)
    filepath = setup_folder_structure.join('config', 'model.yaml')
    filepath.write(contents)


@pytest.fixture(scope='function')
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


@pytest.fixture(scope='function')
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

class Model():
    pass
model = Model()
model.simulate = MagicMock(return_value=3)
model.simulate()
time.sleep(1) # delays for 1 seconds

wrapper = MagicMock(return_value=1)
wrapper.inputs.parameters.values = MagicMock(return_value=1)
"""
    filename.write(contents, ensure=True)


@pytest.fixture(scope='function')
def setup_project_folder(setup_runpy_file,
                         setup_assets_file,
                         setup_folder_structure,
                         setup_config_file,
                         setup_timesteps_file,
                         setup_water_attributes,
                         setup_water_inputs):
    """Sets up a temporary folder with the required project folder structure

        /models
        /models/water_supply/
        /models/water_supply/run.py
        /models/water_supply/assets/assets1.yaml
        /models/water_supply/inputs.yaml
        /config/
        /config/model.yaml
        /config/timesteps.yaml
        /planning/
        /planning/pre-specified.yaml

    """
    base_folder = setup_folder_structure

    return base_folder


@pytest.fixture(scope='function')
def setup_water_inputs(setup_folder_structure):
    base_folder = setup_folder_structure
    filename = base_folder.join('models', 'water_supply', 'inputs.yaml')
    contents = {'decision variables': ['water treatment capacity',
                                       'reservoir pumpiness'],
                'parameters': ['raininess'],
                'water treatment capacity': {'bounds': (0, 20),
                                             'index': 1,
                                             'init': 10
                                             },
                'reservoir pumpiness': {'bounds': (0, 100),
                                        'index': 0,
                                        'init': 24.583
                                        },
                'raininess': {'bounds': (0, 5),
                              'index': 0,
                              'value': 3
                              }
                }
    yaml_contents = yaml.dump(contents)
    filename.write(yaml_contents, ensure=True)


@pytest.fixture(scope='function')
def setup_timesteps_file(setup_folder_structure):
    base_folder = setup_folder_structure
    filename = base_folder.join('config', 'timesteps.yaml')
    timesteps_contents = [2010, 2011, 2012]
    contents = yaml.dump(timesteps_contents)
    filename.write(contents, ensure=True)


@pytest.fixture(scope='function')
def setup_water_asset_a(setup_folder_structure,
                        setup_config_file,
                        setup_assets_file):
    project_folder = setup_folder_structure
    filepath = 'models/water_supply/assets/water_asset_a.yaml'
    water_file_a = project_folder.join(filepath)

    content = """capital_cost:
    value: 1000
    unit: "£/kW"
economic_lifetime: 25 # assume unit of years
operational_lifetime: 25 # assume unit of years
    """
    water_file_a.write(content)

    return str(project_folder)


@pytest.fixture(scope='function')
def setup_water_asset_b(setup_folder_structure,
                        setup_config_file,
                        setup_assets_file):
    project_folder = setup_folder_structure
    filepath = 'models/water_supply/assets/water_asset_b.yaml'
    water_file_b = project_folder.join(filepath)

    content = """capital_cost:
    value: 1500
    unit: "£/kW"
    """
    water_file_b.write(content)

    return str(project_folder)


@pytest.fixture(scope='function')
def setup_water_asset_c(setup_folder_structure,
                        setup_config_file,
                        setup_assets_file):
    project_folder = setup_folder_structure
    filepath = 'models/water_supply/assets/water_asset_c.yaml'
    water_file_c = project_folder.join(filepath)

    content = """capital_cost:
  value: 3000
  unit: "£/kW"
    """
    water_file_c.write(content)

    return str(project_folder)


@pytest.fixture(scope='function')
def setup_water_asset_d(setup_folder_structure,
                        setup_config_file_two,
                        setup_assets_file_two):
    project_folder = setup_folder_structure
    filepath = 'models/water_supply/assets/water_asset_d.yaml'
    water_file_d = project_folder.join(filepath)

    content = """capital_cost:
  value: 3000
  unit: "£/kW"
    """
    water_file_d.write(content)

    return str(project_folder)


@pytest.fixture(scope='function')
def setup_minimal_water(setup_folder_structure,
                        setup_config_file,
                        setup_assets_file):
    return str(setup_folder_structure)


@pytest.fixture(scope='function')
def setup_water_attributes(setup_water_asset_a,
                           setup_water_asset_b,
                           setup_water_asset_c):

    return setup_water_asset_a
