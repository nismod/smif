#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Dummy conftest.py for smif.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    https://pytest.org/latest/plugins.html
"""
from __future__ import absolute_import, division, print_function

import json
import logging

import yaml
from pytest import fixture

logging.basicConfig(filename='test_logs.log',
                    level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s: %(levelname)-8s %(message)s',
                    filemode='w')


@fixture(scope='function')
def one_dependency():
    """Returns a model input dictionary with a single (unlikely to be met)
    dependency
    """
    inputs = [
            {
                'name': 'macguffins produced',
                'spatial_resolution': 'LSOA',
                'temporal_resolution': 'annual'
            }
        ]

    return inputs


@fixture(scope='function')
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


@fixture(scope='function')
def setup_project_folder(setup_runpy_file,
                         setup_folder_structure,
                         setup_config_file,
                         setup_timesteps_file,
                         setup_water_inputs,
                         setup_water_outputs,
                         setup_time_intervals,
                         setup_regions,
                         setup_initial_conditions_file,
                         setup_pre_specified_planning,
                         setup_water_interventions_abc,
                         setup_interventions_file_one):
    """Sets up a temporary folder with the required project folder structure

        /config
        /config/model.yaml
        /config/timesteps.yaml
        /data
        /data/regions.geojson
        /data/intervals.yaml
        /data/water_supply/
        /data/water_supply/inputs.yaml
        /data/water_supply/outputs.yaml
        /data/water_supply/interventions/
        /data/water_supply/interventions/water_asset_abc.yaml
        /data/water_supply/interventions/assets_new.yaml
        /data/water_supply/pre-specified.yaml
        /models
        /models/water_supply/water_supply.py

    """
    base_folder = setup_folder_structure

    return base_folder


@fixture(scope='function')
def setup_project_missing_model_config(setup_runpy_file,
                                       setup_folder_structure,
                                       setup_config_file,
                                       setup_timesteps_file,
                                       setup_initial_conditions_file,
                                       setup_water_interventions_abc,
                                       setup_pre_specified_planning):
    """Sets up a temporary folder with the required project folder structure

        /config
        /config/model.yaml
        /config/timesteps.yaml
        /data
        /data/water_supply/
        /data/water_supply/pre-specified.yaml
        /data/water_supply/initial_conditions
        /data/water_supply/initial_conditions/assets_1.yaml
        /data/water_supply/interventions
        /data/water_supply/interventions/assets_1.yaml
        /models
        /models/water_supply/water_supply.py

        Deliberately missing:

        /data/water_supply/inputs.yaml
        /data/water_supply/outputs.yaml
        /data/intervals.yaml
        /data/water_supply/regions.geojson

    """
    base_folder = setup_folder_structure

    return base_folder


@fixture(scope='function')
def setup_project_empty_model_io(setup_runpy_file,
                                 setup_folder_structure,
                                 setup_config_file,
                                 setup_timesteps_file,
                                 setup_water_inputs,
                                 setup_water_outputs,
                                 setup_time_intervals,
                                 setup_regions,
                                 setup_initial_conditions_file,
                                 setup_pre_specified_planning,
                                 setup_water_interventions_abc,
                                 setup_interventions_file_one):
    """Sets up a temporary folder with the required project folder structure

        /config
        /config/model.yaml
        /config/timesteps.yaml
        /data
        /data/regions.geojson
        /data/intervals.yaml
        /data/water_supply/
        /data/water_supply/inputs.yaml
        /data/water_supply/outputs.yaml
        /data/water_supply/interventions/
        /data/water_supply/interventions/water_asset_abc.yaml
        /data/water_supply/interventions/assets_new.yaml
        /data/water_supply/pre-specified.yaml
        /models
        /models/water_supply/water_supply.py

    """
    base_folder = setup_folder_structure

    inputs_filename = base_folder.join('data', 'water_supply', 'inputs.yaml')
    inputs_filename.write('')

    outputs_filename = base_folder.join('data', 'water_supply', 'outputs.yaml')
    outputs_filename.write('')

    return base_folder


@fixture(scope='function')
def setup_initial_conditions_file(setup_folder_structure):
    """Assets are associated with sector models, not the integration config

    """
    base_folder = setup_folder_structure
    filename = base_folder.join('data',
                                'water_supply',
                                'initial_conditions',
                                'assets_1.yaml')
    assets_contents = [
        {
            'name': 'water_asset_a',
            'capacity': {
                'value': 5,
                'units': 'GW'
            },
            'operational_lifetime': {
                'value': 150,
                'units': "years"
            },
            'economic_lifetime': {
                'value': 50,
                'units': "years"
            },
            'capital_cost': {
                'value': 50,
                'units': "million £/km"
            },
            'location': 'oxford',
            'build_date': 2017
        },
        {
            'name': 'water_asset_b',
            'capacity': {
                'value': 15,
                'units': 'GW'
            },
            'operational_lifetime': {
                'value': 150,
                'units': "years"
            },
            'economic_lifetime': {
                'value': 50,
                'units': "years"
            },
            'capital_cost': {
                'value': 50,
                'units': "million £/km"
            },
            'location': 'oxford',
            'build_date': 2017
        },
        {
            'name': 'water_asset_c',
            'capacity': {
                'value': 25,
                'units': 'GW'
            },
            'operational_lifetime': {
                'value': 150,
                'units': "years"
            },
            'economic_lifetime': {
                'value': 50,
                'units': "years"
            },
            'capital_cost': {
                'value': 50,
                'units': "million £/km"
            },
            'location': 'oxford',
            'build_date': 2017
        }
    ]
    contents = yaml.dump(assets_contents)
    filename.write(contents, ensure=True)


@fixture(scope='function')
def setup_initial_conditions_file_two(setup_folder_structure):
    """Assets are associated with sector models, not the integration config

    Defines a second assets file
    """
    base_folder = setup_folder_structure
    filename = base_folder.join('data',
                                'water_supply',
                                'initial_conditions',
                                'assets_2.yaml')
    assets_contents = [
        {
            'name': 'water_asset_d',
            'capacity': {
                'value': 15,
                'units': 'GW'
            },
            'operational_lifetime': {
                'value': 150,
                'units': "years"
            },
            'economic_lifetime': {
                'value': 50,
                'units': "years"
            },
            'capital_cost': {
                'value': 50,
                'units': "million £/km"
            },
            'location': 'oxford',
            'build_date': 2017
        }
    ]
    contents = yaml.dump(assets_contents)
    filename.write(contents, ensure=True)


@fixture(scope='function')
def setup_interventions_file_one(setup_folder_structure):
    """Interventions are associated with sector models,
    not the integration config

    Defines an interventions file
    """
    base_folder = setup_folder_structure
    filename = base_folder.join('data',
                                'water_supply',
                                'interventions',
                                'assets_1.yaml')
    assets_contents = [
        {
            'name': 'water_asset_d',
            'capacity': {
                'value': 15,
                'units': 'GW'
            },
            'operational_lifetime': {
                'value': 150,
                'units': "years"
            },
            'economic_lifetime': {
                'value': 50,
                'units': "years"
            },
            'capital_cost': {
                'value': 50,
                'units': "million £/km"
            },
            'location': 'oxford'
        }
    ]
    contents = yaml.dump(assets_contents)
    filename.write(contents, ensure=True)


@fixture(scope='function')
def setup_interventions_file_two(setup_folder_structure):
    """Interventions are associated with sector models,
    not the integration config

    Defines an interventions file
    """
    base_folder = setup_folder_structure
    filename = base_folder.join('data',
                                'water_supply',
                                'interventions',
                                'assets_2.yaml')
    assets_contents = [
        {
            'name': 'water_asset_e',
            'capacity': {
                'value': 5,
                'units': 'Ml/day'
            },
            'operational_lifetime': {
                'value': 150,
                'units': "years"
            },
            'economic_lifetime': {
                'value': 50,
                'units': "years"
            },
            'capital_cost': {
                'value': 50,
                'units': "million £"
            },
            'location': 'oxford'
        }
    ]
    contents = yaml.dump(assets_contents)
    filename.write(contents, ensure=True)


@fixture(scope='function')
def setup_scenario_data(setup_folder_structure):
    file_contents = [
        {
            'value': 100,
            'units': 'people',
            'region': 'GB',
            'year': 2015
        },
        {
            'value': 150,
            'units': 'people',
            'region': 'GB',
            'year': 2016
        },
        {
            'value': 200,
            'units': 'people',
            'region': 'GB',
            'year': 2017
        }
    ]
    contents = yaml.dump(file_contents)
    filepath = setup_folder_structure.join('data', 'population.yaml')
    filepath.write(contents)

    # overwrite model.yaml
    file_contents = {
        'sector_models': [
            {
                "name": "water_supply",
                "path": "../models/water_supply/__init__.py",
                "classname": "WaterSupplySectorModel",
                "config_dir": "../data/water_supply",
                'initial_conditions': [
                    '../data/water_supply/initial_conditions/assets_1.yaml'
                ],
                'interventions': [
                    '../data/water_supply/interventions/water_asset_abc.yaml'
                ]
            }
        ],
        'scenario_data': [
            {
                'parameter': 'population',
                'file': '../data/population.yaml',
                'spatial_resolution': 'national',
                'temporal_resolution': 'annual'
            }
        ],
        'timesteps': 'timesteps.yaml',
        "region_sets": [{'name': 'national',
                         'file': 'regions.geojson'}],
        "interval_sets": [{'name': 'annual',
                           'file': 'intervals.yaml'}],
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


@fixture(scope='function')
def setup_no_planning(setup_folder_structure):
    file_contents = {
        'sector_models': [
            {
                "name": "water_supply",
                "path": "../models/water_supply/__init__.py",
                "classname": "WaterSupplySectorModel",
                "config_dir": "../data/water_supply",
                'initial_conditions': [
                    '../data/water_supply/initial_conditions/assets_1.yaml'
                ],
                'interventions': [
                    '../data/water_supply/interventions/water_asset_abc.yaml'
                ]
            }
        ],
        'scenario_data': [],
        'timesteps': 'timesteps.yaml',
        'planning': {
            'rule_based': {'use': False},
            'optimisation': {'use': False},
            'pre_specified': {'use': False}
        }
    }

    contents = yaml.dump(file_contents)
    filepath = setup_folder_structure.join('config', 'model.yaml')
    filepath.write(contents)


@fixture(scope='function')
def setup_planning_missing(setup_folder_structure):
    file_contents = {
        'sector_models': [
            {
                "name": "water_supply",
                "path": "../models/water_supply/__init__.py",
                "classname": "WaterSupplySectorModel",
                "config_dir": "../data/water_supply",
                'initial_conditions': [
                    '../data/water_supply/initial_conditions/assets_1.yaml'
                ],
                'interventions': [
                    '../data/water_supply/interventions/water_asset_abc.yaml'
                ]
            }
        ],
        'scenario_data': [],
        'timesteps': 'timesteps.yaml',
        'planning': {
            'rule_based': {'use': False},
            'optimisation': {'use': False},
            'pre_specified': {
                'use': True,
                'files': ['./does_not_exist.yaml']
            }
        }
    }

    contents = yaml.dump(file_contents)
    filepath = setup_folder_structure.join('config', 'model.yaml')
    filepath.write(contents)


@fixture(scope='function')
def setup_planning_empty(setup_folder_structure):
    filepath = setup_folder_structure.join('data', 'water_supply', 'pre-specified.yaml')
    filepath.write('')


@fixture(scope='function')
def setup_abs_path_to_timesteps(setup_folder_structure):
    timesteps_abs_path = str(setup_folder_structure.join(
        'config',
        'timesteps.yaml'
    ))

    file_contents = {
        'sector_models': [
            {
                "name": "water_supply",
                "path": "../models/water_supply/__init__.py",
                "classname": "WaterSupplySectorModel",
                "config_dir": "../data/water_supply",
                'initial_conditions': [
                    '../data/water_supply/initial_conditions/assets_1.yaml'
                ],
                'interventions': [
                    '../data/water_supply/interventions/water_asset_abc.yaml'
                ]
            }
        ],
        'scenario_data': [],
        'timesteps': timesteps_abs_path,
        'planning': {
            'rule_based': {'use': False},
            'optimisation': {'use': False},
            'pre_specified': {'use': False}
        }
    }

    contents = yaml.dump(file_contents)
    filepath = setup_folder_structure.join('config', 'model.yaml')
    filepath.write(contents)


@fixture(scope='function')
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
                "config_dir": "../data/water_supply",
                'initial_conditions': [
                    '../data/water_supply/initial_conditions/assets_1.yaml'
                ],
                'interventions': [
                    '../data/water_supply/interventions/water_asset_abc.yaml'
                ]
            }
        ],
        'timesteps': 'timesteps.yaml',
        'scenario_data': [],
        "region_sets": [{'name': 'national',
                         'file': 'regions.geojson'}],
        "interval_sets": [{'name': 'annual',
                           'file': 'intervals.yaml'}],
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


@fixture(scope='function')
def setup_region_shapefile():
    'uk_nations_shp/regions.shp'


@fixture(scope='function')
def setup_pre_specified_planning_conflict(setup_folder_structure):
    """Sets up a configuration file for pre-specified planning

    """
    file_name = 'pre-specified_asset_d.yaml'
    file_contents = [
        {
            'name': 'water_asset_z',
            'description': 'Existing water treatment plants',
            'capacity': 6,
            'location': {'lat': 51.74556, 'lon': -1.240528},
            'build_date': 2015
        }
    ]
    contents = yaml.dump(file_contents)
    filepath = setup_folder_structure.join('data',
                                           'water_supply',
                                           file_name)
    filepath.write(contents)


@fixture(scope='function')
def setup_config_conflict_assets(setup_folder_structure,
                                 setup_pre_specified_planning_conflict):
    """Configuration file contains entries for sector models, timesteps and
    planning

    Contains conflicting assets in the pre-specified rules and the sector model
    """
    ps_name = 'pre-specified_asset_d.yaml'
    file_contents = {
        'sector_models': [
            {
                "name": "water_supply",
                "path": "../models/water_supply/__init__.py",
                "classname": "WaterSupplySectorModel",
                "config_dir": "../data/water_supply",
                'initial_conditions': [
                    '../data/water_supply/initial_conditions/assets1.yaml'
                ],
                'interventions': [
                    '../data/water_supply/interventions/assets1.yaml'
                ]
            }
        ],
        'scenario_data': [],
        'timesteps': 'timesteps.yaml',
        'planning': {
            'rule_based': {'use': False},
            'optimisation': {'use': False},
            'pre_specified': {
                'use': True,
                'files': [ps_name]
            }
        }
    }

    contents = yaml.dump(file_contents)
    filepath = setup_folder_structure.join('config', 'model.yaml')
    filepath.write(contents)


@fixture(scope='function')
def setup_config_conflict_periods(setup_folder_structure,
                                  setup_timesteps_file_two):
    """Configuration file contains entries for sector models, timesteps and
    planning

    Contains conflicting assets in the pre-specified rules and the sector model
    """
    ps_name = 'pre-specified.yaml'
    file_contents = {
        'sector_models': [
            {
                "name": "water_supply",
                "path": "../models/water_supply/__init__.py",
                "classname": "WaterSupplySectorModel",
                "config_dir": "../data/water_supply",
                'initial_conditions': [
                    '../data/water_supply/initial_conditions/assets1.yaml'
                ],
                'interventions': [
                    '../data/water_supply/interventions/assets1.yaml'
                ]
            }
        ],
        'scenario_data': [],
        'timesteps': 'timesteps2.yaml',
        'planning': {
            'rule_based': {'use': False},
            'optimisation': {'use': False},
            'pre_specified': {
                'use': True,
                'files': [ps_name]
            }
        }
    }

    contents = yaml.dump(file_contents)
    filepath = setup_folder_structure.join('config', 'model.yaml')
    filepath.write(contents)


@fixture(scope='function')
def setup_pre_specified_planning(setup_folder_structure):
    """Sets up a configuration file for pre-specified planning

    """
    file_name = 'pre-specified.yaml'
    file_contents = [
        {
            'name': 'water_asset_a',
            'build_date': 2010,
            'attributes': [
                {
                    'key': 'new_capacity',
                    'value': 6
                },
                {
                    'key': 'description',
                    'value': 'Existing water treatment plants'
                },
                {
                    'key': 'location',
                    'value': {'lat': 51.74556, 'lon': -1.240528}
                }
            ]
        },
        {
            'name': 'water_asset_b',
            'build_date': 2010,
            'attributes': [
                {
                    'key': 'new_capacity',
                    'value': 6
                },
                {
                    'key': 'description',
                    'value': 'Existing water treatment plants'
                },
                {
                    'key': 'location',
                    'value': {'lat': 51.74556, 'lon': -1.240528}
                }
            ]
        },
        {
            'name': 'water_asset_c',
            'build_date': 2010,
            'attributes': [
                {
                    'key': 'new_capacity',
                    'value': 6
                },
                {
                    'key': 'description',
                    'value': 'Existing water treatment plants'
                },
                {
                    'key': 'location',
                    'value': {'lat': 51.74556, 'lon': -1.240528}
                }
            ]
        }
    ]
    contents = yaml.dump(file_contents)
    filepath = setup_folder_structure.join('data',
                                           'water_supply',
                                           file_name)
    filepath.write(contents, ensure=True)


@fixture(scope='function')
def setup_pre_specified_planning_two(setup_folder_structure):
    """Sets up a configuration file for pre-specified planning

    """
    file_name = 'pre-specified_alt.yaml'
    file_contents = [
        {
            'name': 'water_asset_a',
            'build_date': 2015,
            'attributes': [
                {
                    'key': 'new_capacity',
                    'value': 6
                },
                {
                    'key': 'description',
                    'value': 'Existing water treatment plants'
                },
                {
                    'key': 'location',
                    'value': {'lat': 51.74556, 'lon': -1.240528}
                }
            ]
        },
        {
            'name': 'water_asset_a',
            'build_date': 2020,
            'attributes': [
                {
                    'key': 'new_capacity',
                    'value': 6
                },
                {
                    'key': 'description',
                    'value': 'Existing water treatment plants'
                },
                {
                    'key': 'location',
                    'value': {'lat': 51.74556, 'lon': -1.240528}
                }
            ]
        },
        {
            'name': 'water_asset_a',
            'build_date': 2025,
            'attributes': [
                {
                    'key': 'new_capacity',
                    'value': 6
                },
                {
                    'key': 'description',
                    'value': 'Existing water treatment plants'
                },
                {
                    'key': 'location',
                    'value': {'lat': 51.74556, 'lon': -1.240528}
                }
            ]
        }
    ]
    contents = yaml.dump(file_contents)
    filepath = setup_folder_structure.join('data',
                                           'water_supply',
                                           file_name)
    filepath.write(contents)


@fixture(scope='function')
def setup_config_file_two(setup_folder_structure, setup_interventions_file_one):
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
                "config_dir": "../data/water_supply",
                'initial_conditions': [
                    '../data/water_supply/initial_conditions/initial_2015_oxford.yaml'
                ],
                'interventions': [
                    '../data/water_supply/interventions/assets_1.yaml',
                    '../data/water_supply/interventions/assets_2.yaml'
                ]
            }
        ],
        'scenario_data': [],
        'timesteps': 'timesteps.yaml',
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


@fixture(scope='function')
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
                "config_dir": "../data/water_supply",
                'initial_conditions': [
                    '../data/water_supply/assets/assets_1.yaml'
                ],
                'interventions': [
                    '../data/water_supply/assets/assets_1.yaml'
                ]
            }
        ],
        'scenario_data': [],
        'timesteps': 'timesteps_2.yaml',
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


@fixture(scope='function')
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
    def initialise(self, initial_conditions):
        pass

    def simulate(self, decisions, state, data):
        pass

    def extract_obj(self, results):
        return 0

"""
    filename.write(contents, ensure=True)


@fixture(scope='function')
def setup_water_inputs(setup_folder_structure):
    base_folder = setup_folder_structure
    filename = base_folder.join('data', 'water_supply', 'inputs.yaml')
    contents = [{'name': 'reservoir pumpiness',
                 'spatial_resolution': 'LSOA',
                 'temporal_resolution': 'annual'}]
    yaml_contents = yaml.dump(contents)
    filename.write(yaml_contents, ensure=True)


@fixture(scope='function')
def water_outputs_contents():
    contents = [
            {
                'name': 'storage_state',
                'spatial_resolution': 'national',
                'temporal_resolution': 'annual'
            },
            {
                'name': 'storage_blobby',
                'spatial_resolution': 'national',
                'temporal_resolution': 'annual'
            },
            {
                'name': 'total_water_demand',
                'spatial_resolution': 'national',
                'temporal_resolution': 'annual'
            }
        ]
    return contents


@fixture(scope='function')
def setup_water_outputs(setup_folder_structure,
                        water_outputs_contents):
    base_folder = setup_folder_structure
    filename = base_folder.join('data', 'water_supply', 'outputs.yaml')
    contents = water_outputs_contents
    yaml_contents = yaml.dump(contents)
    filename.write(yaml_contents, ensure=True)
    return filename


@fixture(scope='function')
def setup_time_intervals(setup_folder_structure):
    base_folder = setup_folder_structure
    filename = base_folder.join('config', 'intervals.yaml')
    contents = [
        {
            "start": "P0Y",
            "end": "P1Y",
            "id": "whole_year"
        }
    ]
    yaml_contents = yaml.dump(contents)
    filename.write(yaml_contents, ensure=True)
    return filename


@fixture(scope='function')
def setup_region_data():
    data = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {
                "name": "urn:ogc:def:crs:EPSG::27700"
            }
        },
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "oxford"
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [448180, 209366],
                            [449500, 211092],
                            [450537, 211029],
                            [450873, 210673],
                            [451250, 210793],
                            [451642, 210023],
                            [453855, 208466],
                            [454585, 208468],
                            [456077, 207967],
                            [456146, 207738],
                            [456668, 207779],
                            [456708, 207444],
                            [456278, 207122],
                            [456149, 206615],
                            [455707, 206798],
                            [455749, 204521],
                            [456773, 204488],
                            [457014, 204184],
                            [456031, 203475],
                            [456444, 202854],
                            [456087, 202044],
                            [455369, 201799],
                            [454396, 202203],
                            [453843, 201634],
                            [452499, 203209],
                            [452052, 203566],
                            [451653, 203513],
                            [450645, 205137],
                            [449497, 205548],
                            [449051, 206042],
                            [448141, 208446],
                            [448180, 209366]
                        ]
                    ]
                }
            },
        ]
    }
    return data


@fixture(scope='function')
def setup_country_data():
    data = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {
                "name": "urn:ogc:def:crs:EPSG::27700"
            }
        },
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "GB"
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [0, 1],
                            [0, 1],
                            [2, 3]
                        ]
                    ]
                }
            },
            {
                "type": "Feature",
                "properties": {
                    "name": "NI"
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [2, 3],
                            [2, 3],
                            [4, 5]
                        ]
                    ]
                }
            },
        ]
    }
    return data


@fixture(scope='function')
def setup_regions(setup_folder_structure, setup_region_data):
    base_folder = setup_folder_structure
    filename = base_folder.join('config', 'regions.geojson')
    data = setup_region_data
    filename.write(json.dumps(data), ensure=True)
    return filename


@fixture(scope='function')
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


@fixture(scope='function')
def setup_timesteps_file(setup_folder_structure):
    base_folder = setup_folder_structure
    filename = base_folder.join('config', 'timesteps.yaml')
    timesteps_contents = [2010, 2011, 2012]
    contents = yaml.dump(timesteps_contents)
    filename.write(contents, ensure=True)


@fixture(scope='function')
def setup_timesteps_file_two(setup_folder_structure):
    base_folder = setup_folder_structure
    filename = base_folder.join('config', 'timesteps_2.yaml')
    timesteps_contents = [2015, 2020, 2025]
    contents = yaml.dump(timesteps_contents)
    filename.write(contents, ensure=True)


@fixture(scope='function')
def setup_timesteps_file_invalid(setup_folder_structure):
    base_folder = setup_folder_structure
    filename = base_folder.join('config', 'timesteps.yaml')
    timesteps_contents = "invalid timesteps file"
    contents = yaml.dump(timesteps_contents)
    filename.write(contents, ensure=True)


@fixture(scope='function')
def setup_water_interventions_abc(setup_folder_structure):
    data = [
        {
            "name": "water_asset_a",
            "location": "oxford",
            "capital_cost": {
                "units": "£",
                "value": 1000
            },
            "economic_lifetime": {
                "units": "years",
                "value": 25
            },
            "operational_lifetime": {
                "units": "years",
                "value": 25
            }
        },
        {
            "name": "water_asset_b",
            "location": "oxford",
            "capital_cost": {
                "units": "£",
                "value": 1500
            },
            "economic_lifetime": {
                "units": "years",
                "value": 25
            },
            "operational_lifetime": {
                "units": "years",
                "value": 25
            }
        },
        {
            "name": "water_asset_c",
            "location": "oxford",
            "capital_cost": {
                "units": "£",
                "value": 3000
            },
            "economic_lifetime": {
                "units": "years",
                "value": 25
            },
            "operational_lifetime": {
                "units": "years",
                "value": 25
            }
        }

    ]
    content = yaml.dump(data)

    filename = setup_folder_structure.join('data',
                                           'water_supply',
                                           'interventions',
                                           'water_asset_abc.yaml')
    filename.write(content, ensure=True)

    return filename


@fixture(scope='function')
def setup_water_intervention_d(setup_folder_structure,
                               setup_config_file_two):

    content = """
- name: water_asset_d
  location: oxford
  operational_lifetime:
    units: years
    value: 50
  economic_lifetime:
    units: years
    value: 45
  capital_cost:
    units: "£"
    value: 3000
"""
    filename = setup_folder_structure.join('data',
                                           'water_supply',
                                           'interventions',
                                           'water_asset_d.yaml')
    filename.write(content, ensure=True)

    return filename


@fixture(scope='function')
def setup_minimal_water(setup_folder_structure,
                        setup_config_file):
    return str(setup_folder_structure)
