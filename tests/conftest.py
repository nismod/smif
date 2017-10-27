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
import os
from copy import copy

import yaml
from pytest import fixture
from smif.convert.area import get_register as get_region_register
from smif.convert.area import RegionSet
from smif.convert.interval import get_register as get_interval_register
from smif.convert.interval import IntervalSet

from .convert.test_area import (regions_half_squares, regions_half_triangles,
                                regions_rect, regions_single_half_square)
from .convert.test_interval import (months, one_day, remap_months, seasons,
                                    twenty_four_hours)

logging.basicConfig(filename='test_logs.log',
                    level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s: %(levelname)-8s %(message)s',
                    filemode='w')


@fixture(scope='function')
def setup_folder_structure(tmpdir_factory):
    """

    Returns
    -------
    :class:`LocalPath`
        Path to the temporary folder
    """
    folder_list = [
        'config',
        os.path.join('config', 'sos_model_runs'),
        os.path.join('config', 'sos_models'),
        os.path.join('config', 'sector_models'),
        'data',
        os.path.join('data', 'initial_conditions'),
        os.path.join('data', 'interval_definitions'),
        os.path.join('data', 'interventions'),
        os.path.join('data', 'narratives'),
        os.path.join('data', 'region_definitions'),
        os.path.join('data', 'scenarios'),
        'models'
    ]

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
                         setup_interventions_file_one,
                         setup_parameters):
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
        /data/water_supply/parameters.yaml
        /models
        /models/water_supply/water_supply.py

    """
    base_folder = setup_folder_structure

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


@fixture
def setup_parameters(setup_folder_structure):
    base_folder = setup_folder_structure
    filename = base_folder.join('data',
                                'water_supply',
                                'parameters.yaml')
    parameter_contents = [
        {'name': 'smart_meter_savings',
         'description': 'The savings from smart meters',
         'absolute_range': (0, 100),
         'suggested_range': (3, 10),
         'default_value': 3,
         'units': '%'}
         ]
    contents = yaml.dump(parameter_contents)
    filename.write(contents, ensure=True)


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
                ],
                'parameters': [
                    '../data/water_supply/parameters.yaml'
                ]
            }
        ],
        'dependencies': [],
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
        },
        'convergence_max_iterations': 1000,
        'convergence_relative_tolerance': 0.0001,
        'convergence_absolute_tolerance': 0.1,
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
                ],
                'parameters': [
                    '../data/water_supply/parameters.yaml'
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
def setup_runpy_file(tmpdir, setup_folder_structure):
    """The python script should contain an instance of SectorModel which wraps
    the sector model and allows it to be run.
    """
    base_folder = setup_folder_structure
    # Write a file for the water_supply model
    filename = base_folder.join('models', 'water_supply', '__init__.py')
    contents = """
from smif.model.sector_model import SectorModel

class WaterSupplySectorModel(SectorModel):
    def initialise(self, initial_conditions):
        pass

    def simulate(self, timestep, data=None):
        return {self.name: data}

    def extract_obj(self, results):
        return 0

"""
    filename.write(contents, ensure=True)


@fixture(scope='function')
def setup_water_inputs(setup_folder_structure):
    base_folder = setup_folder_structure
    filename = base_folder.join('data', 'water_supply', 'inputs.yaml')
    contents = [{
        'name': 'reservoir pumpiness',
        'spatial_resolution': 'LSOA',
        'temporal_resolution': 'annual',
        'units': 'magnitude'
    }]
    yaml_contents = yaml.dump(contents)
    filename.write(yaml_contents, ensure=True)


@fixture(scope='function')
def water_outputs_contents():
    contents = [
        {
            'name': 'storage_state',
            'spatial_resolution': 'national',
            'temporal_resolution': 'annual',
            'units': 'Ml'
        },
        {
            'name': 'storage_blobby',
            'spatial_resolution': 'national',
            'temporal_resolution': 'annual',
            'units': 'Ml'
        },
        {
            'name': 'total_water_demand',
            'spatial_resolution': 'national',
            'temporal_resolution': 'annual',
            'units': 'Ml'
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


@fixture(scope='session')
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
                            [0, 1.1],
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
                            [2, 3.2],
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
def setup_timesteps_file(setup_folder_structure):
    base_folder = setup_folder_structure
    filename = base_folder.join('config', 'timesteps.yaml')
    timesteps_contents = [2010, 2011, 2012]
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


@fixture(scope="session", autouse=True)
def setup_registers(setup_region_data):
    """One-time setup: load all the fixture region and interval
    sets into the module-level registers.
    """
    regions = get_region_register()
    lsoa = RegionSet('LSOA', setup_region_data['features'])
    regions.register(lsoa)
    regions.register(regions_half_squares())
    regions.register(regions_single_half_square())
    regions.register(regions_half_triangles())
    regions.register(regions_rect())

    # register alt rect (same area)
    regions_rect_alt = copy(regions_rect())
    regions_rect_alt.name = 'rect_alt'
    regions.register(regions_rect_alt)

    intervals = get_interval_register()
    annual_data = [{'id': 1, 'start': 'P0Y', 'end': 'P1Y'}]
    intervals.register(IntervalSet('annual', annual_data))
    intervals.register(IntervalSet('months', months()))
    intervals.register(IntervalSet('seasons', seasons()))
    intervals.register(IntervalSet('hourly_day', twenty_four_hours()))
    intervals.register(IntervalSet('one_day', one_day()))
    intervals.register(IntervalSet('remap_months', remap_months()))
