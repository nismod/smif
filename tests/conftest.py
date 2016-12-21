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
import pytest
import yaml

logging.basicConfig(filename='test_logs.log',
                    level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s: %(levelname)-8s %(message)s',
                    filemode='w')


@pytest.fixture(scope='function')
def setup_folder_structure(tmpdir_factory):
    """

    Returns
    -------
    :class:`LocalPath`
        Path to the temporary folder
    """
    folder_list = ['config', 'data', 'models']
    test_folder = tmpdir_factory.mktemp("smif")

    for folder in folder_list:
        test_folder.mkdir(folder)

    return test_folder


@pytest.fixture(scope='function')
def setup_project_folder(setup_runpy_file,
                         setup_folder_structure,
                         setup_config_file,
                         setup_timesteps_file,
                         setup_water_attributes,
                         setup_water_inputs,
                         setup_water_outputs,
                         setup_pre_specified_planning):
    """Sets up a temporary folder with the required project folder structure

        /config
        /config/model.yaml
        /config/timesteps.yaml
        /data
        /data/water_supply/
        /data/water_supply/inputs.yaml
        /data/water_supply/outputs.yaml
        /data/water_supply/assets
        /data/water_supply/assets/assets_1.yaml
        /data/water_supply/pre-specified.yaml
        /models
        /models/water_supply/water_supply.py

    """
    base_folder = setup_folder_structure

    return base_folder


@pytest.fixture(scope='function')
def setup_assets_file(setup_folder_structure):
    """Assets are associated with sector models, not the integration config

    """
    base_folder = setup_folder_structure
    filename = base_folder.join('data',
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
    filename = base_folder.join('data',
                                'water_supply',
                                'assets',
                                'assets_2.yaml')
    assets_contents = ['water_asset_d']
    contents = yaml.dump(assets_contents)
    filename.write(contents, ensure=True)


@pytest.fixture(scope='function')
def setup_config_file(setup_folder_structure):
    """Configuration file contains entries for sector models, timesteps and
    planning
    """
    file_contents = {
        'sector_models': [
            {
                "name": "water_supply",
                "path": "../models/water_supply/__init__.py",
                "classname": "WaterSupplySectorModel",
                "config_dir": "../data/water_supply"
            }
        ],
        'base_year': 2010,
        'timesteps': 'timesteps.yaml',
        'assets': ['../data/water_supply/assets/assets_1.yaml'],
        'planning': {
            'rule_based': {'use': False},
            'optimisation': {'use': False},
            'pre_specified': {
                'use': True,
                'files': ['../data/water_supply/pre-specified.yaml']
            }
        }
    }

    contents = yaml.dump(file_contents)
    filepath = setup_folder_structure.join('config', 'model.yaml')
    filepath.write(contents)


@pytest.fixture(scope='function')
def setup_pre_specified_planning_conflict(setup_folder_structure):
    """Sets up a configuration file for pre-specified planning

    """
    file_name = 'pre-specified_asset_d.yaml'
    file_contents = [{'new_capacity': {'unit': 'Ml/yr', 'value': 6},
                      'asset': 'water_asset_z',
                      'description': 'Existing water treatment plants',
                      'location': {'lat': 51.74556, 'lon': -1.240528},
                      'timeperiod': 2015
                      }]
    contents = yaml.dump(file_contents)
    filepath = setup_folder_structure.join('data',
                                           'water_supply',
                                           file_name)
    filepath.write(contents)


@pytest.fixture(scope='function')
def setup_config_conflict_assets(setup_folder_structure,
                                 setup_pre_specified_planning_conflict):
    """Configuration file contains entries for sector models, timesteps and
    planning

    Contains conflicting assets in the pre-specified rules and the sector model
    """
    ps_name = 'pre-specified_asset_d.yaml'
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
def setup_config_conflict_periods(setup_folder_structure,
                                  setup_timesteps_file_two):
    """Configuration file contains entries for sector models, timesteps and
    planning

    Contains conflicting assets in the pre-specified rules and the sector model
    """
    ps_name = 'pre-specified.yaml'
    file_contents = {'sector_models': ['water_supply'],
                     'timesteps': 'timesteps2.yaml',
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
def setup_pre_specified_planning(setup_folder_structure):
    """Sets up a configuration file for pre-specified planning

    """
    file_name = 'pre-specified.yaml'
    file_contents = [{'new_capacity': {'unit': 'Ml/yr', 'value': 6},
                      'asset': 'water_asset_a',
                      'description': 'Existing water treatment plants',
                      'location': {'lat': 51.74556, 'lon': -1.240528},
                      'timeperiod': 2010
                     },
                     {'new_capacity': {'unit': 'Ml/yr', 'value': 6},
                      'asset': 'water_asset_b',
                      'description': 'Existing water treatment plants',
                      'location': {'lat': 51.74556, 'lon': -1.240528},
                      'timeperiod': 2010
                     },
                     {'new_capacity': {'unit': 'Ml/yr', 'value': 6},
                      'asset': 'water_asset_c',
                      'description': 'Existing water treatment plants',
                      'location': {'lat': 51.74556, 'lon': -1.240528},
                      'timeperiod': 2010
                     }]
    contents = yaml.dump(file_contents)
    filepath = setup_folder_structure.join('data',
                                           'water_supply',
                                           file_name)
    filepath.write(contents, ensure=True)


@pytest.fixture(scope='function')
def setup_pre_specified_planning_two(setup_folder_structure):
    """Sets up a configuration file for pre-specified planning

    """
    file_name = 'pre-specified_alt.yaml'
    file_contents = [{'new_capacity': {'unit': 'Ml/yr', 'value': 6},
                      'asset': 'water_asset_a',
                      'description': 'Existing water treatment plants',
                      'location': {'lat': 51.74556, 'lon': -1.240528},
                      'timeperiod': 2015
                     },
                     {'new_capacity': {'unit': 'Ml/yr', 'value': 6},
                      'asset': 'water_asset_a',
                      'description': 'Existing water treatment plants',
                      'location': {'lat': 51.74556, 'lon': -1.240528},
                      'timeperiod': 2020
                     },
                     {'new_capacity': {'unit': 'Ml/yr', 'value': 6},
                      'asset': 'water_asset_a',
                      'description': 'Existing water treatment plants',
                      'location': {'lat': 51.74556, 'lon': -1.240528},
                      'timeperiod': 2025
                     }]
    contents = yaml.dump(file_contents)
    filepath = setup_folder_structure.join('data',
                                           'water_supply',
                                           file_name)
    filepath.write(contents)


@pytest.fixture(scope='function')
def setup_config_file_two(setup_folder_structure):
    """Configuration file contains entries for sector models, timesteps and
    planning

    Contains two asset files
    """
    file_contents = {
        'sector_models': [
            {
                "name": "water_supply",
                "path": "../models/water_supply/__init__.py",
                "classname": "WaterSupplySectorModel",
                "config_dir": "../data/water_supply"
            }
        ],
        'base_year': 2010,
        'timesteps': 'timesteps.yaml',
        'assets': [
            '../data/water_supply/assets_1.yaml',
            '../data/water_supply/assets_2.yaml'
        ],
        'planning': {
            'rule_based': {'use': False},
            'optimisation': {'use': False},
            'pre_specified': {
                'use': True,
                'files': ['../data/water_supply/pre-specified.yaml']
            }
        }
    }

    contents = yaml.dump(file_contents)
    filepath = setup_folder_structure.join('config', 'model.yaml')
    filepath.write(contents)


@pytest.fixture(scope='function')
def setup_config_file_timesteps_two(setup_folder_structure):
    """Configuration file contains entries for sector models, timesteps and
    planning
    """
    file_contents = {
        'sector_models': [
            {
                "name": "water_supply",
                "path": "../models/water_supply/__init__.py",
                "classname": "WaterSupplySectorModel",
                "config_dir": "../data/water_supply"
            }
        ],
        'base_year': 2010,
        'timesteps': 'timesteps_2.yaml',
        'assets': ['../data/water_supply/assets_1.yaml'],
        'planning': {
            'rule_based': {'use': False},
            'optimisation': {'use': False},
            'pre_specified': {
                'use': True,
                'files': ['../data/water_supply/pre-specified.yaml']
            }
        }
    }

    contents = yaml.dump(file_contents)
    filepath = setup_folder_structure.join('config', 'model.yaml')
    filepath.write(contents)


@pytest.fixture(scope='function')
def setup_runpy_file(tmpdir, setup_folder_structure):
    """The python script should contain an instance of SectorModel which wraps
    the sector model and allows it to be run.
    """
    base_folder = setup_folder_structure
    # Write a file for the water_supply model
    filename = base_folder.join('models', 'water_supply', '__init__.py')
    contents = """
from smif.sector_model import SectorModel

class WaterSupplySectorModel(SectorModel):
    def simulate(self, decisions):
        pass

    def extract_obj(self, results):
        return 0

"""
    filename.write(contents, ensure=True)


@pytest.fixture(scope='function')
def setup_water_inputs(setup_folder_structure):
    base_folder = setup_folder_structure
    filename = base_folder.join('data', 'water_supply', 'inputs.yaml')
    contents = {
        'decision variables': [
            {
                'name': 'reservoir pumpiness',
                'bounds': (0, 100),
                'value': 24.583
            },
            {
                'name': 'water treatment capacity',
                'bounds': (0, 20),
                'value': 10
            }
        ],
        'parameters': [
            {
                'name': 'raininess',
                'bounds': (0, 5),
                'value': 3
            }
        ],
        'dependencies': []
    }
    yaml_contents = yaml.dump(contents)
    filename.write(yaml_contents, ensure=True)


@pytest.fixture(scope='function')
def water_outputs_contents():
    contents = {
        'metrics': [
            {
                'name': 'storage_state',
                'description': 'Storage at end',
                'file_name': 'results.txt',
                'row_num': 26,
                'col_num': 44,
                'type': 'int'
            },
            {
                'name': 'storage_blobby',
                'description': 'Storage at end',
                'file_name': 'results.txt',
                'row_num': 33,
                'col_num': 55,
                'type': 'int'
            }
        ],
        'model outputs': [
            {
                'name': 'unshfl13',
                'description': 'TOTAL DEMAND 13 Test1',
                'file_name': 'results.txt',
                'row_num': 33,
                'col_num': 44,
                'type': 'int'
            }
        ]
    }
    return contents


@pytest.fixture(scope='function')
def setup_water_outputs(setup_folder_structure,
                        water_outputs_contents):
    base_folder = setup_folder_structure
    filename = base_folder.join('data', 'water_supply', 'outputs.yaml')
    contents = water_outputs_contents
    yaml_contents = yaml.dump(contents)
    filename.write(yaml_contents, ensure=True)
    return filename


@pytest.fixture(scope='function')
def setup_results_file(setup_folder_structure):
    contents = \
        '''
Run Statistics
Software version: 111 System file version: 149
Saved on 29/11/2016 at 17:10

System file name:


System description:
   Run description:

------------------------------------------------------------
NLP solver: RELAX pure NLP
            Total number of NLP calls =     69241
            Total sim cpu (sec)    9.9360 Script cpu
------------------------------------------------------------
Mass balance report:
   Storage at start of replicate          =            216650
   + volume in arcs at start of replicate =                 0
   + total inflow                         =     6136254644472
   + total harvest node inflow            =                 0
   - total outflow at demand nodes        =         117263073
   - total flow into waste nodes          =     6136137397761
   - total reservoir evaporation          =                 0
   + backup reservoir evaporation         =                 0
   - total arc loss                       =                 0
 = Storage at end of replicate            =            200288
   + volume in arcs at end of replicate   =                 0
   + water balance error                  =                 0

TOTAL DEMAND                 =                         122427
  Node Demand node name                         Total shortfall
  -------------------------------------------------------------
    13 Test1                                               9080
    14 Test2                                               7832
    15 Test3                                               3876
------------------------------------------------------------
'''
    base_folder = setup_folder_structure
    # Write a results.txt file for the water_supply model
    filename = base_folder.join('models',
                                'water_supply',
                                'results.txt')
    filename.write(contents, ensure=True)
    return str(base_folder.join('models', 'water_supply'))


@pytest.fixture(scope='function')
def setup_timesteps_file(setup_folder_structure):
    base_folder = setup_folder_structure
    filename = base_folder.join('config', 'timesteps.yaml')
    timesteps_contents = [2010, 2011, 2012]
    contents = yaml.dump(timesteps_contents)
    filename.write(contents, ensure=True)


@pytest.fixture(scope='function')
def setup_timesteps_file_two(setup_folder_structure):
    base_folder = setup_folder_structure
    filename = base_folder.join('config', 'timesteps_2.yaml')
    timesteps_contents = [2015, 2020, 2025]
    contents = yaml.dump(timesteps_contents)
    filename.write(contents, ensure=True)


@pytest.fixture(scope='function')
def setup_timesteps_file_invalid(setup_folder_structure):
    base_folder = setup_folder_structure
    filename = base_folder.join('config', 'timesteps.yaml')
    timesteps_contents = "invalid timesteps file"
    contents = yaml.dump(timesteps_contents)
    filename.write(contents, ensure=True)


@pytest.fixture(scope='function')
def setup_water_attributes(setup_folder_structure):
    data = [
        {
            "name": "water_asset_a",
            "capital_cost": {
                "value": 1000,
                "unit": "£/kW"
            },
            "economic_lifetime": 25, # assume unit of years
            "operational_lifetime": 25 # assume unit of years
        },
        {
            "name": "water_asset_b",
            "capital_cost": {
                "value": 1500,
                "unit": "£/kW"
            }
        },
        {
            "name": "water_asset_c",
            "capital_cost": {
                "value": 3000,
                "unit": "£/kW"
            }
        }
    ]
    content = yaml.dump(data)

    filename = setup_folder_structure.join('data', 'water_supply', 'assets', 'water_asset_abc.yaml')
    filename.write(content, ensure=True)

    return filename


@pytest.fixture(scope='function')
def setup_water_asset_d(setup_folder_structure,
                        setup_config_file_two):

    content = """
-
    name: water_asset_d
    capital_cost:
        value: 3000
        unit: "£/kW"
"""
    filename = setup_folder_structure.join('data', 'water_supply', 'assets', 'water_asset_d.yaml')
    filename.write(content, ensure=True)

    return filename


@pytest.fixture(scope='function')
def setup_minimal_water(setup_folder_structure,
                        setup_config_file):
    return str(setup_folder_structure)
